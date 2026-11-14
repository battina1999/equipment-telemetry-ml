"""Local inference for engineered six-hourly telemetry features.

Only load the bundled trusted artifact. Never load arbitrary uploaded pickle/joblib files.
"""
from pathlib import Path
from functools import lru_cache
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from telemetry import FEATURES

app=FastAPI(title='Equipment Telemetry ML Lab',description='Independent synthetic-data experiment. Not an operational maintenance system.')

class Request(BaseModel):
    model_config=ConfigDict(extra='forbid')
    features:dict[str,float]

@lru_cache(maxsize=1)
def model_bundle():
    return joblib.load(Path(__file__).with_name('model.joblib'))

@app.get('/health')
def health():
    bundle=model_bundle()
    return {'status':'ok','model':bundle['name'],'synthetic_only':True,'horizon_hours':24}

@app.post('/predict')
def predict(request:Request):
    missing=sorted(set(FEATURES)-set(request.features))
    extra=sorted(set(request.features)-set(FEATURES))
    if missing or extra:
        raise HTTPException(422,detail={'missing_features':missing,'unexpected_features':extra})
    if any(not math.isfinite(value) for value in request.features.values()):
        raise HTTPException(422,detail='Features must be finite numbers.')
    bundle=model_bundle()
    row=pd.DataFrame([{f:request.features[f] for f in FEATURES}],columns=FEATURES)
    score=float(bundle['model'].predict_proba(row)[0,1])
    return {'fault_risk_score':score,'threshold':bundle['threshold'],'review_flag':score>=bundle['threshold'],
        'horizon_hours':24,'model':bundle['name'],'synthetic_only':True,
        'interpretation':'Model score, not a calibrated probability or maintenance instruction.'}

if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description='Score one engineered synthetic telemetry window locally.')
    parser.add_argument('request',nargs='?',default='example_request.json')
    args=parser.parse_args()
    payload=json.loads(Path(args.request).read_text())
    print(json.dumps(predict(Request(**payload)),indent=2))
