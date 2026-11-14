"""Fit fixed candidates, select on calibration, then evaluate untouched tests."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score,roc_auc_score,precision_score,
    recall_score,f1_score,fbeta_score,confusion_matrix,brier_score_loss)
from telemetry import SEED,FEATURES,simulate,make_dataset,split_dataset


def measures(y, probability, threshold):
    pred = probability >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"rows":len(y),"positive_rows":int(sum(y)),"prevalence":float(np.mean(y)),
        "average_precision":float(average_precision_score(y,probability)),
        "roc_auc":float(roc_auc_score(y,probability)),
        "brier_score":float(brier_score_loss(y,probability)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "alerts_per_100_readings":float(np.mean(pred)*100),
        "confusion":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}


def run(output: Path):
    if (output/'model.joblib').exists():
        raise SystemExit('Output already has a model. Choose a new --output to preserve the reported benchmark.')
    output.mkdir(parents=True,exist_ok=True)
    raw=simulate()
    dataset=make_dataset(raw)
    split=split_dataset(dataset)
    train,cal=split['train'],split['calibration']
    candidates={
        'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,class_weight='balanced',random_state=SEED)),
        'random_forest':RandomForestClassifier(n_estimators=180,max_depth=7,min_samples_leaf=12,class_weight='balanced',n_jobs=1,random_state=SEED),
    }
    calibration={}
    for name,model in candidates.items():
        model.fit(train[FEATURES],train.fault_next_24h)
        probability=model.predict_proba(cal[FEATURES])[:,1]
        choices=[(float(fbeta_score(cal.fault_next_24h,probability>=t,beta=2,zero_division=0)),float(t)) for t in np.arange(.05,.951,.01)]
        best_score,threshold=max(choices,key=lambda x:(x[0],x[1]))
        calibration[name]={**measures(cal.fault_next_24h,probability,threshold),'threshold':threshold,'f2':best_score}
    winner=max(calibration,key=lambda k:calibration[k]['average_precision'])
    threshold=calibration[winner]['threshold']
    bundle={'model':candidates[winner],'features':FEATURES,'threshold':threshold,'name':winner,
            'synthetic_only':True,'horizon_hours':24,'seed':SEED,'schema_version':1}
    joblib.dump(bundle,output/'model.joblib',compress=3)
    tests={}
    examples=[]
    for name,model in candidates.items():
        tests[name]={}
        for cohort in ['test_known','test_unseen']:
            frame=split[cohort]
            probs=model.predict_proba(frame[FEATURES])[:,1]
            tests[name][cohort]=measures(frame.fault_next_24h,probs,calibration[name]['threshold'])
            if name==winner:
                for (_,row),prob in zip(frame.iterrows(),probs):
                    examples.append({'asset_id':row.asset_id,'timestamp':row.timestamp.isoformat(),
                        'cohort':cohort,'actual':int(row.fault_next_24h),'score':round(float(prob),6),
                        'cabinet_temp_c':round(row.cabinet_temp_c,3)})
    raw_csv=raw.to_csv(index=False,date_format='%Y-%m-%dT%H:%M:%S')
    report={'project':'Equipment Telemetry ML Lab','data':'100% deterministic synthetic refrigeration telemetry; no real store or employer data.',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
        'seed':SEED,'raw_rows':len(raw),'raw_sha256':hashlib.sha256(raw_csv.encode()).hexdigest(),
        'assets':48,'sample_interval_hours':6,'days':120,'target':'Fault event in the NEXT 24 hours; current faults and recovery excluded.',
        'selection':'Candidate with greatest calibration average precision. Candidate thresholds maximize calibration F2; ties choose higher threshold. Hyperparameters fixed before evaluation.',
        'winner':winner,'threshold':threshold,'features':FEATURES,
        'splits':{k:{'rows':len(v),'assets':v.asset_id.nunique(),'min_day':float(v.day.min()),'max_day':float(v.day.max()),'positive_rows':int(v.fault_next_24h.sum())} for k,v in split.items()},
        'calibration':calibration,'tests':tests,
        'limitations':['Synthetic generator defines all patterns; results do not estimate real equipment performance.',
            'Reading-level labels overlap within fault horizons; metrics are not independent event-level alarms or confidence intervals.',
            'Unseen assets follow the same generator family; no external distribution-shift validation.',
            'No probability calibration, operational deployment, Azure execution, or real maintenance savings are established.']}
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'test_predictions.json').write_text(json.dumps(examples,separators=(',',':'))+'\n')
    sample=split['test_unseen'].iloc[0]
    (output/'example_request.json').write_text(json.dumps({'features':{f:float(sample[f]) for f in FEATURES}},indent=2)+'\n')
    print(json.dumps({'winner':winner,'threshold':threshold,'splits':report['splits'],'tests':tests[winner]},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('benchmark-rerun'))
    run(parser.parse_args().output)
