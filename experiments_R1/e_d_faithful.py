#!/usr/bin/env python3
"""AUPRC for the architecture table, computed under the EXACT seeding protocol
of scripts/graph_supervised.py:main() so the AUROC reproduces the primary
architecture table and AUPRC is added from the SAME predictions.

Fidelity requirements (verified against graph_supervised.py):
  * global torch/np seed set ONCE (default seed = 42), never per-fold;
  * arch order gat -> gcn -> sage -> mlp;
  * per arch: task-stratified 3 seeds, THEN label-stratified 3 seeds — the
    label-stratified loop is run purely to advance the global RNG identically to
    main() (its results are discarded), otherwise the next arch's init differs;
  * run_fold_capture is a line-for-line copy of graph_supervised.run_fold with
    only an added (probs, labels) return — no RNG-consuming line is changed.

Classical baselines (Table 7) are re-run here too (already reproduce the paper,
scaler-fixed) so one JSON holds the final Table 6 + Table 7.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, average_precision_score)
from xgboost import XGBClassifier
import common as C
import graph_supervised as GS

SEEDS = [7, 42, 123]
GLOBAL_SEED = 42  # graph_supervised.parse_args() default --seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_fold_capture(train_graphs, test_graphs, args, device, fold_idx, arch="gat"):
    """EXACT copy of graph_supervised.run_fold (RNG consumption unchanged),
    returning (probs, labels) in addition so AUPRC can be computed."""
    rng = np.random.RandomState(args.seed + fold_idx)
    indices = rng.permutation(len(train_graphs))
    n_val = max(1, len(indices) // 5)
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]
    trn_graphs = [train_graphs[i] for i in trn_idx]
    val_graphs = [train_graphs[i] for i in val_idx]

    train_loader = DataLoader(trn_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    if arch == "mlp":
        model = GS.MLPClassifier(in_dim, args.hidden_dim).to(device)
    else:
        model = GS.GNNClassifier(in_dim, args.hidden_dim, arch=arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n_benign = sum(1 for g in trn_graphs if g.y.item() == 0)
    n_attack = sum(1 for g in trn_graphs if g.y.item() == 1)
    if n_benign > 0 and n_attack > 0:
        w = torch.tensor([n_attack / (n_benign + n_attack),
                          n_benign / (n_benign + n_attack)], dtype=torch.float).to(device)
    else:
        w = None

    best_val_auroc = 0
    best_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(1, args.epochs + 1):
        loss = GS.train_epoch(model, train_loader, optimizer, device, w)
        if epoch % 10 == 0:
            val_probs, val_labels = GS.evaluate(model, val_loader, device)
            if len(np.unique(val_labels)) > 1:
                val_auroc = roc_auc_score(val_labels, val_probs)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    probs, labels = GS.evaluate(model, test_loader, device)
    return probs, labels


def agg(folds, keys):
    return {k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))]
            for k in keys}


def main():
    # global seed ONCE, exactly like graph_supervised.main()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")
    graphs, _, _ = C.build_graphs(sessions, feats, edge_mode="full")
    labels = np.array([g.y.item() for g in graphs])
    task_ids = np.array([g.task_id for g in graphs])
    keys = ["auroc", "auprc", "auprc_benign", "f1", "recall", "fpr"]

    results = {"protocol": "faithful (graph_supervised.main global-seed replay)",
               "prevalence_attack": float(labels.mean()),
               "architectures": {}, "classical": {}}

    print("\n=== E-D (faithful) architecture table — reproduces Table 6 AUROC + AUPRC ===")
    for arch in ["gat", "gcn", "sage", "mlp"]:
        ts_folds = []
        for seed in SEEDS:                                  # task-stratified (captured)
            tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
            train_g = [graphs[i] for i in tr]; test_g = [graphs[i] for i in te]
            args = argparse.Namespace(seed=seed, batch_size=64, epochs=200, lr=1e-3, hidden_dim=128)
            probs, labs = run_fold_capture(train_g, test_g, args, DEVICE, 0, arch=arch)
            ts_folds.append(C.metrics_from_probs(probs, labs))
        for seed in SEEDS:                                  # label-stratified (RNG advance only)
            tr, va, te = GS.label_stratified_split(len(graphs), labels, seed)
            train_g = [graphs[i] for i in tr]; test_g = [graphs[i] for i in te]
            args = argparse.Namespace(seed=seed, batch_size=64, epochs=200, lr=1e-3, hidden_dim=128)
            run_fold_capture(train_g, test_g, args, DEVICE, 0, arch=arch)
        results["architectures"][arch] = agg(ts_folds, keys)
        m = results["architectures"][arch]
        print(f"  {arch.upper():5s} AUROC={m['auroc'][0]:.4f}±{m['auroc'][1]:.3f} "
              f"AUPRC={m['auprc'][0]:.4f} AP_benign={m['auprc_benign'][0]:.4f} "
              f"recall={m['recall'][0]:.3f} fpr={m['fpr'][0]:.3f}")

    # classical (unchanged, scaler-fixed; reproduces Table 7)
    X = np.stack([np.concatenate([f.mean(0), f.max(0)]) for f in feats]).astype(np.float32)
    print("\n=== E-D classical baselines with AUPRC (Table 7) ===")
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
            if name in ("LogisticRegression", "LinearSVM"):
                scaler = StandardScaler(); Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
            else:
                Xtr, Xte = X[tr], X[te]
            clf.fit(Xtr, ytr)
            scores = clf.decision_function(Xte) if hasattr(clf, "decision_function") else clf.predict_proba(Xte)[:, 1]
            preds = clf.predict(Xte)
            folds.append({
                "auroc": float(roc_auc_score(yte, scores)),
                "auprc": float(average_precision_score(yte, scores)),
                "auprc_benign": float(average_precision_score(1 - yte, -scores)),
                "recall": float(recall_score(yte, preds)),
                "fpr": float(((preds == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1)),
                "f1": float(f1_score(yte, preds)),
            })
        results["classical"][name] = agg(folds, keys)
        m = results["classical"][name]
        print(f"  {name:18s} AUROC={m['auroc'][0]:.4f} AUPRC={m['auprc'][0]:.4f} "
              f"AP_benign={m['auprc_benign'][0]:.4f} recall={m['recall'][0]:.3f} fpr={m['fpr'][0]:.3f}")

    out = os.path.join(C.RESULTS_DIR, "e_d_faithful.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPrevalence(attack)={results['prevalence_attack']:.3f}. Saved -> {out}")


if __name__ == "__main__":
    main()
