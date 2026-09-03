#!/usr/bin/env python3
"""Classical-baseline AUPRC (attack + benign-class AP) for ATBench and Combined,
to complete Table 7's AUPRC column. Environment-independent (sklearn only).
Reuses labeleff_multidataset.build_graphs for content-feature graph construction,
then pools (mean||max) exactly like the MLP/classical baselines.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevent XGBoost/libomp segfault on macOS
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import labeleff_multidataset as LM

SEEDS=[7,42,123]

def pooled(dataset):
    g=LM.build_graphs(dataset)
    X=np.stack([np.concatenate([x.x.numpy().mean(0), x.x.numpy().max(0)]) for x in g]).astype(np.float32)
    y=np.array([x.y.item() for x in g])
    return X,y

def run(dataset):
    X,y=pooled(dataset)
    out={}
    for name in ["XGBoost","RandomForest","LogisticRegression","LinearSVM"]:
        aus,aps,apbs=[],[],[]
        for s in SEEDS:
            tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=s)
            ytr,yte=y[tr],y[te]; nb=int((ytr==0).sum()); na=int((ytr==1).sum())
            if name=="LogisticRegression": clf=LogisticRegression(class_weight="balanced",max_iter=1000,solver="lbfgs",C=1.0,random_state=s)
            elif name=="LinearSVM": clf=SGDClassifier(loss="hinge",class_weight="balanced",max_iter=5000,alpha=1e-4,tol=1e-4,random_state=s)
            elif name=="RandomForest": clf=RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=1,random_state=s)
            else: clf=XGBClassifier(n_estimators=200,scale_pos_weight=nb/max(na,1),eval_metric="logloss",verbosity=0,nthread=1,tree_method="hist",random_state=s)
            if name in ("LogisticRegression","LinearSVM"):
                sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
            else: Xtr,Xte=X[tr],X[te]
            clf.fit(Xtr,ytr)
            sco=clf.decision_function(Xte) if hasattr(clf,"decision_function") else clf.predict_proba(Xte)[:,1]
            aus.append(roc_auc_score(yte,sco)); aps.append(average_precision_score(yte,sco)); apbs.append(average_precision_score(1-yte,-sco))
        out[name]={"auroc":round(float(np.mean(aus)),3),"auprc":round(float(np.mean(aps)),3),"auprc_benign":round(float(np.mean(apbs)),3),"prevalence":round(float(y.mean()),3)}
    return out

res={d:run(d) for d in ["atbench","combined"]}
json.dump(res,open(os.path.join(_HERE,"results","classical_auprc_extra.json"),"w"),indent=2)
for d,r in res.items():
    print(f"== {d} (attack prevalence {list(r.values())[0]['prevalence']}) ==")
    for c,m in r.items(): print(f"  {c:18s} AUROC={m['auroc']} AUPRC={m['auprc']} AP_benign={m['auprc_benign']}")
