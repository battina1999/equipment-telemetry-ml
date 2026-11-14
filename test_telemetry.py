import json
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from telemetry import FEATURES,simulate,causal_features,make_dataset,split_dataset
from api import app

ROOT=Path(__file__).parent

def test_simulation_is_reproducible_and_has_faults():
    first=simulate(assets=2,days=30)
    pd.testing.assert_frame_equal(first,simulate(assets=2,days=30))
    assert first.fault_event.sum()>0
    assert not first.loc[first.fault_event.astype(bool),'operating'].any()

def test_future_mutation_cannot_change_past_features():
    raw=simulate(assets=2,days=30)
    before=causal_features(raw)
    changed=raw.copy()
    mask=changed.day>=15
    changed.loc[mask,['cabinet_temp_c','vibration_rms','power_kw','duty_fraction']]=999
    changed.loc[mask,'fault_event']=1
    after=causal_features(changed)
    pd.testing.assert_frame_equal(before.loc[before.day<15,FEATURES],after.loc[after.day<15,FEATURES])

def test_target_is_strictly_future_and_has_complete_horizon():
    raw=simulate(assets=1,days=30)
    frame=make_dataset(raw)
    for i,row in frame.iterrows():
        future=raw[(raw.asset_id==row.asset_id)&(raw.timestamp>row.timestamp)&(raw.timestamp<=row.timestamp+pd.Timedelta(hours=24))]
        assert len(future)==4
        assert row.fault_next_24h==future.fault_event.max()
        assert row.operating
        assert not row.fault_event

def test_splits_are_chronological_asset_disjoint_and_purged():
    frame=make_dataset(simulate())
    split=split_dataset(frame)
    train,cal,known,unseen=[split[x] for x in ['train','calibration','test_known','test_unseen']]
    assert train.day.max()+1<cal.day.min()
    assert cal.day.max()+1<known.day.min()
    assert set(train.asset_id).isdisjoint(unseen.asset_id)
    assert set(cal.asset_id).isdisjoint(unseen.asset_id)
    assert not set(FEATURES)&{'fault_event','fault_next_24h','day','asset_id','timestamp','operating'}
    assert all(v.fault_next_24h.nunique()==2 for v in split.values())

def test_api_matches_saved_example_and_rejects_invalid_schema():
    client=TestClient(app)
    assert client.get('/health').json()['synthetic_only'] is True
    example=json.loads((ROOT/'example_request.json').read_text())
    response=client.post('/predict',json=example)
    assert response.status_code==200
    result=response.json()
    assert 0<=result['fault_risk_score']<=1
    saved=json.loads((ROOT/'test_predictions.json').read_text())
    expected=next(row['score'] for row in saved if row['cohort']=='test_unseen')
    assert abs(result['fault_risk_score']-expected)<1e-6
    assert result['review_flag']==(result['fault_risk_score']>=result['threshold'])
    assert client.post('/predict',json={'features':{'cabinet_temp_c':4}}).status_code==422
    example['features']['extra']=1
    assert client.post('/predict',json=example).status_code==422

def test_saved_metrics_and_predictions_agree():
    metrics=json.loads((ROOT/'metrics.json').read_text())
    predictions=json.loads((ROOT/'test_predictions.json').read_text())
    for cohort in ['test_known','test_unseen']:
        rows=[p for p in predictions if p['cohort']==cohort]
        assert len(rows)==metrics['tests'][metrics['winner']][cohort]['rows']
        tp=sum(p['actual']==1 and p['score']>=metrics['threshold'] for p in rows)
        fn=sum(p['actual']==1 and p['score']<metrics['threshold'] for p in rows)
        assert tp==metrics['tests'][metrics['winner']][cohort]['confusion']['tp']
        assert fn==metrics['tests'][metrics['winner']][cohort]['confusion']['fn']
