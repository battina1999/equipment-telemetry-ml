"""Deterministic synthetic refrigeration telemetry and causal features.

No records come from an employer, equipment vendor or real store.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 734
HORIZON_STEPS = 4  # Four six-hour samples = 24 hours.
FEATURES = [
    "cabinet_temp_c", "ambient_temp_c", "vibration_rms", "power_kw",
    "duty_fraction", "door_open_minutes", "temp_mean_48h", "temp_std_48h",
    "temp_change_24h", "vibration_mean_48h", "vibration_change_24h",
    "power_mean_48h", "duty_mean_48h",
]


def simulate(seed: int = SEED, assets: int = 48, days: int = 120) -> pd.DataFrame:
    """A toy latent-wear process; not a physical refrigeration simulation.

A fault occurs when wear crosses a limit, then synthetic maintenance resets it.
Predictors are noisy sensor transformations of wear, ambient load and door use.
Wear and time-to-fault are NEVER model features.
"""
    if assets < 1 or days < 4:
        raise ValueError("Use at least one asset and four days.")
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(assets):
        offset = rng.normal(0, .35)
        efficiency = rng.uniform(.85, 1.15)
        wear = rng.uniform(.02, .55)
        rate = rng.uniform(.008, .016)
        repair_left = 0
        for step in range(days * 4):
            hour = step * 6
            ambient = 24 + 4*np.sin(2*np.pi*(hour % 24)/24) + rng.normal(0, 1.5)
            doors = max(0, rng.normal(8 + 3*np.sin(2*np.pi*(hour % 24)/24), 3))
            if repair_left:
                repair_left -= 1
                available = False
                fault = False
            else:
                available = True
                wear += rate + rng.normal(0, .005) + rng.exponential(.002)
                fault = bool(wear >= 1)
            # Readings during fault and recovery are generated but never scored.
            stress = max(wear - .35, 0)
            temp = 3 + offset + .055*(ambient-24) + .045*doors + 3.3*stress**2 + rng.normal(0, .38)
            vib = .4 + .8*stress + rng.normal(0, .10)
            power = efficiency*(.75 + .013*(ambient-20) + .35*stress) + rng.normal(0, .065)
            duty = np.clip(.32 + .012*(ambient-20) + .41*stress + .003*doors + rng.normal(0,.055),0,1)
            rows.append((f"unit-{asset:03d}", pd.Timestamp("2025-01-01")+pd.Timedelta(hours=hour),
                         step/4, temp, ambient, vib, power, duty, doors, int(fault), available and not fault))
            if fault:
                wear = rng.uniform(.01,.12)
                rate = rng.uniform(.008,.016)
                repair_left = 3
    return pd.DataFrame(rows, columns=["asset_id","timestamp","day","cabinet_temp_c",
        "ambient_temp_c","vibration_rms","power_kw","duty_fraction","door_open_minutes",
        "fault_event","operating"])


def causal_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Use each sensor reading and its history only; no labels or future rows."""
    result = raw.sort_values(["asset_id","timestamp"]).copy()
    group = result.groupby("asset_id", sort=False)
    for original, name in [("cabinet_temp_c","temp"),("vibration_rms","vibration"),
                           ("power_kw","power"),("duty_fraction","duty")]:
        result[name+"_mean_48h"] = group[original].transform(lambda x:x.rolling(8,min_periods=8).mean())
    result["temp_std_48h"] = group["cabinet_temp_c"].transform(lambda x:x.rolling(8,min_periods=8).std())
    result["temp_change_24h"] = group["cabinet_temp_c"].diff(4)
    result["vibration_change_24h"] = group["vibration_rms"].diff(4)
    return result


def make_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    result = causal_features(raw)
    faults = result.groupby("asset_id",sort=False)["fault_event"]
    future = pd.concat([faults.shift(-h) for h in range(1,HORIZON_STEPS+1)],axis=1)
    result["fault_next_24h"] = future.max(axis=1)
    complete_horizon = future.notna().all(axis=1)
    return result.loc[result.operating & complete_horizon & result[FEATURES].notna().all(axis=1)].copy()


def split_dataset(frame: pd.DataFrame):
    """Freeze 36 development assets; test later days on seen AND unseen units.

Day gaps exceed the 24-hour label horizon and 48-hour feature window.
The last full day is dropped by make_dataset so every target is observable.
"""
    known = frame.asset_id.isin([f"unit-{n:03d}" for n in range(36)])
    return {
        "train":frame[known & (frame.day < 65)].copy(),
        "calibration":frame[known & (frame.day >= 72) & (frame.day < 86)].copy(),
        "test_known":frame[known & (frame.day >= 93)].copy(),
        "test_unseen":frame[~known & (frame.day >= 93)].copy(),
    }
