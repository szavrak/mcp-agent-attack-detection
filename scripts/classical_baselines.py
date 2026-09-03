#!/usr/bin/env python3
"""Classical ML baselines on pooled SBERT features for Paper 1.

Trains logistic regression, linear SVM, XGBoost, and random forest on the
same mean+max pooled content embeddings (1536-dim) used by the MLP baseline.

Usage:
    python scripts/classical_baselines.py --dataset raseval
    python scripts/classical_baselines.py --dataset atbench
    python scripts/classical_baselines.py --dataset combined
"""
import argparse
import json
import os
import sys
import time

# Prevent XGBoost/libomp segfault on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier

# Reuse Paper 1's data loading and splitting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from graph_supervised import (
    load_sessions,
    build_tool_vocab,
    get_content_embedder,
    sessions_to_graphs,
    task_stratified_split,
    label_stratified_split,
)
from load_datasets import load_ras_eval, load_atbench, load_mcpbench


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["raseval", "atbench", "combined"],
                   default="raseval")
    p.add_argument("--raw-dir-ras", default="data/raw/ras_eval")
    p.add_argument("--raw-dir-atb", default="data/raw/atbench")
    p.add_argument("--raw-dir-mcp", default="data/raw/cx_cmu")
    p.add_argument("--seeds", type=str, default="7,42,123")
    p.add_argument("--pooled", action="store_true", default=True,
                   help="Use all benign models (matches Paper 1 canonical)")
    p.add_argument("--out", default=None,
                   help="Output path (default: results/classical_baselines_{dataset}.json)")
    return p.parse_args()


def pool_graph_features(graphs):
    """Mean+max pool node features per session -> (N, 2*feat_dim) array.

    Matches the MLP baseline's pooling: global_mean_pool + global_max_pool,
    concatenated.
    """
    features = []
    for g in graphs:
        x = g.x.numpy()  # (n_nodes, feat_dim)
        mean = x.mean(axis=0)
        mx = x.max(axis=0)
        features.append(np.concatenate([mean, mx]))
    return np.stack(features)


def get_classifiers(n_benign, n_attack, seed=42):
    """Return dict of classifier name -> fitted classifier.

    All use class weighting to handle the benign/attack imbalance.
    """
    scale_pos = n_benign / max(n_attack, 1)
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", C=1.0,
            random_state=seed,
        ),
        "LinearSVM": SGDClassifier(
            loss="hinge", class_weight="balanced", max_iter=5000,
            alpha=1e-4, tol=1e-4, random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=scale_pos,
            eval_metric="logloss", verbosity=0, nthread=1,
            tree_method="hist", random_state=seed,
        ),
    }


def evaluate_classifier(clf, X_test, y_test, name):
    """Compute AUROC, F1, recall, precision, FPR for a fitted classifier."""
    # Score: use decision_function for SVM, predict_proba for others
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
    else:
        scores = clf.predict_proba(X_test)[:, 1]

    preds = clf.predict(X_test)

    results = {}
    if len(np.unique(y_test)) > 1:
        results["auroc"] = float(roc_auc_score(y_test, scores))
    results["f1"] = float(f1_score(y_test, preds))
    results["recall"] = float(recall_score(y_test, preds))
    results["precision"] = float(precision_score(y_test, preds, zero_division=0))
    fpr = ((preds == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)
    results["fpr"] = float(fpr)
    return results


def load_and_encode(dataset, args):
    """Load sessions for the given dataset and encode to graphs."""
    embedder = get_content_embedder()

    if dataset in ("raseval", "combined"):
        ras_b, ras_a = load_ras_eval(args.raw_dir_ras, pooled=args.pooled)
    if dataset in ("atbench", "combined"):
        atb_b, atb_a = load_atbench(args.raw_dir_atb)
    if dataset == "combined":
        mcp_b = load_mcpbench(args.raw_dir_mcp)

    if dataset == "raseval":
        sessions = ras_b + ras_a
    elif dataset == "atbench":
        sessions = atb_b + atb_a
    else:
        sessions = ras_b + atb_b + mcp_b + ras_a + atb_a

    tool_vocab = build_tool_vocab(sessions)
    n_tools = len(tool_vocab)
    graphs = sessions_to_graphs(sessions, "content", tool_vocab, n_tools, embedder)
    return graphs


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_path = args.out or f"results/classical_baselines_{args.dataset}.json"

    use_task_strat = (args.dataset == "raseval")
    split_name = "task_stratified" if use_task_strat else "label_stratified"
    print(f"=== Classical ML Baselines ({args.dataset}, {split_name}) ===\n")

    # Try cache first, fall back to encoding
    cache_map = {
        "raseval": "paper2/cache/ras_eval_content_pooled1.pt",
        "atbench": "paper2/cache/atbench_content_pooled1.pt",
        "combined": "paper2/cache/combined_content_pooled1.pt",
    }
    cache_path = cache_map.get(args.dataset, "")
    if os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        payload = torch.load(cache_path, weights_only=False)
        graphs = payload["graphs"]
    else:
        print("No cache found. Loading and encoding sessions...")
        graphs = load_and_encode(args.dataset, args)

    labels = np.array([g.y.item() for g in graphs])
    task_ids = np.array([getattr(g, 'task_id', -1) for g in graphs])
    print(f"Sessions: {len(graphs)} ({(labels==0).sum()} benign, {(labels==1).sum()} attack)")

    # Pool node features -> (N, 1536) matrix
    print("Pooling node features (mean + max)...")
    X = pool_graph_features(graphs)
    print(f"Feature matrix: {X.shape}")

    # Run across seeds and classifiers
    all_results = {}
    for name in ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]:
        print(f"\n{'='*50}")
        print(f"Classifier: {name}")
        print(f"{'='*50}")
        seed_results = []

        for seed in seeds:
            if use_task_strat:
                train_idx, val_idx, test_idx = task_stratified_split(
                    len(graphs), task_ids, seed
                )
                train_tasks = set(task_ids[train_idx])
                test_tasks = set(task_ids[test_idx])
                assert len(train_tasks & test_tasks) == 0, "Task leakage!"
            else:
                train_idx, val_idx, test_idx = label_stratified_split(
                    len(graphs), labels, seed
                )

            # Train on train split only (not train+val, matching MLP protocol)
            X_train, y_train = X[train_idx], labels[train_idx]
            X_test, y_test = X[test_idx], labels[test_idx]

            # Standardize features (fit on train only)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            n_benign = (y_train == 0).sum()
            n_attack = (y_train == 1).sum()
            clfs = get_classifiers(n_benign, n_attack, seed=seed)
            clf = clfs[name]

            # Use scaled features for LR/SVM, raw for tree-based
            if name in ("LogisticRegression", "LinearSVM"):
                X_tr, X_te = X_train_s, X_test_s
            else:
                X_tr, X_te = X_train, X_test

            t0 = time.time()
            clf.fit(X_tr, y_train)
            fit_time = time.time() - t0

            results = evaluate_classifier(clf, X_te, y_test, name)
            results["fit_time_s"] = round(fit_time, 2)
            seed_results.append(results)

            print(f"  seed={seed}: AUROC={results.get('auroc',0):.4f}  "
                  f"F1={results['f1']:.4f}  Recall={results['recall']:.4f}  "
                  f"FPR={results['fpr']:.4f}  ({fit_time:.1f}s)")

        # Aggregate
        agg = {"per_seed": seed_results}
        for metric in ["auroc", "f1", "recall", "precision", "fpr"]:
            vals = [r[metric] for r in seed_results if metric in r]
            if vals:
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals))

        print(f"\n  MEAN +/- STD:")
        for metric in ["auroc", "f1", "recall", "fpr"]:
            m = agg.get(f"{metric}_mean", 0)
            s = agg.get(f"{metric}_std", 0)
            print(f"    {metric}: {m:.4f} +/- {s:.4f}")

        all_results[name] = agg

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Classical Baselines (content, task-stratified, seeds={seeds})")
    print(f"{'='*70}")
    print(f"{'Classifier':<22} {'AUROC':>14} {'F1':>14} {'Recall':>14} {'FPR':>14}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<22} "
              f"{r.get('auroc_mean',0):.4f}+/-{r.get('auroc_std',0):.3f} "
              f"{r.get('f1_mean',0):.4f}+/-{r.get('f1_std',0):.3f} "
              f"{r.get('recall_mean',0):.4f}+/-{r.get('recall_std',0):.3f} "
              f"{r.get('fpr_mean',0):.4f}+/-{r.get('fpr_std',0):.3f}")

    # Save results
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    output = {
        "dataset": args.dataset,
        "mode": "content",
        "pooled": args.pooled,
        "seeds": seeds,
        "split": split_name,
        "feature_dim": int(X.shape[1]),
        "n_sessions": int(len(graphs)),
        "n_benign": int((labels == 0).sum()),
        "n_attack": int((labels == 1).sum()),
        "classifiers": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
