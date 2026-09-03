#!/usr/bin/env python3
"""AUPRC for the architecture and classical-baseline tables.

RAS-Eval is imbalanced but ATTACK is the MAJORITY positive class
(3797/4402 ~ 0.863 prevalence), so the AUPRC baseline is ~0.86 -- we therefore
also report benign-class AP (the informative minority) and the prevalence.

Local reproduction with fixed per-seed torch init; AUROC will track (not exactly
match) the Colab tables. Intended so the AUPRC columns are computed in one
consistent environment alongside their AUROC.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
import common as C
import graph_supervised as GS

SEEDS = [7, 42, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval_arch(train_g, test_g, seed, arch):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_g)); nv = max(1, len(idx) // 5)
    trn = [train_g[i] for i in idx[nv:]]; val = [train_g[i] for i in idx[:nv]]
    in_dim = train_g[0].x.shape[1]
    model = (GS.MLPClassifier(in_dim, 128) if arch == "mlp"
             else GS.GNNClassifier(in_dim, 128, arch=arch)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nb = sum(1 for g in trn if g.y.item() == 0); na = sum(1 for g in trn if g.y.item() == 1)
    w = torch.tensor([na/(nb+na), nb/(nb+na)], dtype=torch.float).to(DEVICE) if nb and na else None
    tl = DataLoader(trn, batch_size=64, shuffle=True); vl = DataLoader(val, batch_size=64)
    best, best_state, pc = 0, None, 0
    for ep in range(1, 201):
        GS.train_epoch(model, tl, opt, DEVICE, w)
        if ep % 10 == 0:
            vp, vy = GS.evaluate(model, vl, DEVICE)
            if len(np.unique(vy)) > 1:
                a = roc_auc_score(vy, vp)
                if a > best: best, pc, best_state = a, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else: pc += 1
            if pc >= 5: break
    if best_state: model.load_state_dict(best_state)
    p, y = GS.evaluate(model, DataLoader(test_g, batch_size=64), DEVICE)
    return p, y


def agg(folds, keys):
    return {k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))]
            for k in keys}


def main():
    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")
    graphs, _, _ = C.build_graphs(sessions, feats, edge_mode="full")
    labels = np.array([s["label"] for s in sessions])
    task_ids = np.array([s["task_id"] for s in sessions])
    keys = ["auroc", "auprc", "auprc_benign", "f1", "recall", "fpr"]

    results = {"prevalence_attack": float(labels.mean()), "architectures": {}, "classical": {}}

    print("\n=== E-D arch table with AUPRC (RAS-Eval, task-strat, local) ===")
    for arch in ["gat", "gcn", "sage", "mlp"]:
        folds = []
        for seed in SEEDS:
            tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
            train_g = [graphs[i] for i in tr]; test_g = [graphs[i] for i in te]
            p, y = train_eval_arch(train_g, test_g, seed, arch)
            folds.append(C.metrics_from_probs(p, y))
        results["architectures"][arch] = agg(folds, keys)
        m = results["architectures"][arch]
        print(f"  {arch.upper():5s} AUROC={m['auroc'][0]:.4f} AUPRC={m['auprc'][0]:.4f} "
              f"AUPRC_benign={m['auprc_benign'][0]:.4f} recall={m['recall'][0]:.3f} fpr={m['fpr'][0]:.3f}")

    # pooled session features for classical
    X = np.stack([np.concatenate([f.mean(0), f.max(0)]) for f in feats]).astype(np.float32)

    print("\n=== E-D classical baselines with AUPRC (RAS-Eval, task-strat, local) ===")
    for name in ["XGBoost", "RandomForest", "LogisticRegression", "LinearSVM"]:
        folds = []
        for seed in SEEDS:
            tr, va, te = GS.task_stratified_split(len(X), task_ids, seed)
            ytr, yte = labels[tr], labels[te]
            nb = int((ytr == 0).sum()); na = int((ytr == 1).sum())
            if name == "LogisticRegression":
                clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", C=1.0, random_state=seed)
            elif name == "LinearSVM":
                clf = SGDClassifier(loss="hinge", class_weight="balanced", max_iter=5000, alpha=1e-4, tol=1e-4, random_state=seed)
            elif name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=1, random_state=seed)
            else:
                clf = XGBClassifier(n_estimators=200, scale_pos_weight=nb/max(na,1), eval_metric="logloss",
                                    verbosity=0, nthread=1, tree_method="hist", random_state=seed)
            # Standardize features for linear models (LR/SVM), raw for trees —
            # matches scripts/classical_baselines.py (fit scaler on train only).
            if name in ("LogisticRegression", "LinearSVM"):
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
            else:
                Xtr, Xte = X[tr], X[te]
            clf.fit(Xtr, ytr)
            scores = clf.decision_function(Xte) if hasattr(clf, "decision_function") else clf.predict_proba(Xte)[:, 1]
            preds = clf.predict(Xte)
            m = {
                "auroc": float(roc_auc_score(yte, scores)),
                "auprc": float(average_precision_score(yte, scores)),
                "auprc_benign": float(average_precision_score(1 - yte, -scores)),
                "recall": float(((preds == 1) & (yte == 1)).sum() / max((yte == 1).sum(), 1)),
                "fpr": float(((preds == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1)),
                "f1": float(2 * ((preds == 1) & (yte == 1)).sum() /
                            max(2 * ((preds == 1) & (yte == 1)).sum() + ((preds == 1) & (yte == 0)).sum() + ((preds == 0) & (yte == 1)).sum(), 1)),
            }
            folds.append(m)
        results["classical"][name] = agg(folds, keys)
        m = results["classical"][name]
        print(f"  {name:18s} AUROC={m['auroc'][0]:.4f} AUPRC={m['auprc'][0]:.4f} "
              f"AUPRC_benign={m['auprc_benign'][0]:.4f} recall={m['recall'][0]:.3f} fpr={m['fpr'][0]:.3f}")

    out = os.path.join(C.RESULTS_DIR, "e_d_auprc.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPrevalence(attack)={results['prevalence_attack']:.3f} -> AUPRC baseline ~{results['prevalence_attack']:.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
