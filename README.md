# Equipment Telemetry ML Lab

I built this independent project to work through an equipment-monitoring question: can recent sensor readings flag a refrigeration unit that will fault in the next 24 hours?

The entire dataset is **synthetic**. It contains 23,040 six-hourly readings from 48 simulated units over 120 days. There is no employer, store, vendor or customer data here. The simulator is a toy wear process, not a validated physical model.

## What is implemented

- A reproducible telemetry generator for cabinet temperature, ambient temperature, vibration, power, compressor duty and door use.
- Features using current readings and past windows only, with labels strictly in the next 24 hours.
- A scaled logistic-regression baseline and a random forest with fixed model settings.
- Separate fitting, calibration and later-period tests, including 12 units excluded from fitting and calibration.
- A saved model, per-reading test scores, a CLI scorer, a FastAPI endpoint and six tests.

This is a **fault-risk classification experiment**, rather than proof of reduced maintenance cost or a deployed equipment-monitoring service.

## Results worth inspecting

Logistic regression had the higher calibration average precision, so it became the saved model. Its review threshold of **0.81** maximized F2 on calibration data. This favors recall; it does not establish calibrated probabilities.

| Later-period test | Readings | Positive readings | Average precision | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 36 previously seen units | 3,536 | 209 | 0.811 | 68.2% | 83.3% |
| 12 unseen units | 1,163 | 81 | 0.823 | 61.0% | 88.9% |

On the unseen units, the selected threshold flagged **118 readings**: 72 positives and 46 false positives. It missed nine positive readings. Those tradeoffs matter more than a single high accuracy figure. The random forest had unseen-unit average precision of 0.783; its separately selected threshold produced higher recall but lower precision.

[Full metrics](metrics.json) · [Model card](MODEL_CARD.md) · [Per-reading test scores](test_predictions.json)

Several readings can refer to the same future fault. These are correlated **reading-level** results, not independent event-level alert performance. All units follow the same synthetic generator family, so even the unseen-unit test does not estimate performance on real equipment.

## Run it

Tested with Python 3.10.11 and the pinned packages below. Python 3.10 or 3.11 is recommended for reproducing these exact package versions.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python api.py example_request.json
```

The scorer returns a score, the fixed threshold, a review flag and an explicit synthetic-data marker. [Example request](example_request.json) · [Recorded response](example_response.json)

To run the local API:

```bash
uvicorn api:app --host 127.0.0.1 --port 8001
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The API accepts **already engineered features**, rather than an arbitrary raw sensor message. Produce those windows with `causal_features` in [telemetry.py](telemetry.py), using the same six-hour cadence and units. It rejects missing, unexpected and non-finite values. It has no authentication or production traffic controls and should be run locally for this demonstration.

Only load the bundled model you trust. Joblib artifacts must not be loaded from untrusted uploads.

## Reproduce the benchmark

```bash
python train.py --output benchmark-rerun
```

The generator seed is 734. The report records the generated raw-data checksum, package versions, split sizes and candidate results. Training refuses to overwrite a directory that already contains a saved model. The checked-in model was trained only on the fitting split; it was not refit on calibration or test data.

The split boundaries are fixed: fitting before day 65; calibration on days 72–85; tests from day 93. The gaps exceed both the label horizon and feature window. The unseen units may use their own past sensor history to construct features, but none of their rows fit the model or set its threshold.

Tests check future-data mutation, strictly future labels, complete label horizons, split separation, deterministic generation, API/saved-score parity and consistency between predictions and reported counts.

## What this does not demonstrate yet

Real equipment validation, event-level alert deduplication, probability calibration, drift monitoring in a running service, Azure Databricks/ADF/AKS deployment, and CI/CD are outside this implementation. Before using such a model in a store, I would need real sensor and repair records, an operational definition of a useful warning, and a prospective evaluation with equipment owners.
