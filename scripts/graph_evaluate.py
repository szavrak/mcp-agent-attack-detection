#!/usr/bin/env python3
"""Multi-dataset graph-based evaluation.

Three protocols:
1. RAS-Eval only (70/10/20 label-stratified)
2. ATBench only (70/10/20 label-stratified)
3. Combined: RAS-Eval + ATBench + mcpbench-benign (70/10/20 label-stratified)

Reports: AUROC, AUPRC, macro F1, precision, recall, FPR, per-attack-type and per-source breakdown.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_supervised import (
    build_tool_vocab, sessions_to_graphs,
    get_content_embedder, GNNClassifier, MLPClassifier, train_epoch, evaluate,
)
from load_datasets import load_ras_eval, load_atbench, load_mcpbench


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["metadata", "content", "both"], default="content")
    p.add_argument("--dataset", choices=["ras_eval", "atbench", "combined"], default="combined")
    p.add_argument("--arch", choices=["gat", "gcn", "sage", "mlp", "all"], default="all")
    p.add_argument("--raw-dir-ras", default="data/raw/ras_eval")
    p.add_argument("--raw-dir-atb", default="data/raw/atbench")
    p.add_argument("--raw-dir-mcp", default="data/raw/cx_cmu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=str, default="7,42,123",
                   help="Comma-separated seeds for multi-seed evaluation")
    p.add_argument("--pooled", action="store_true")
    args = p.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(",")]
    return args


def label_stratified_split(n, labels, seed=42):
    """Return train/val/test indices: 70/10/20, stratified by label."""
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels,
                                            random_state=seed)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, stratify=temp_labels,
                                          random_state=seed)
    return train_idx, val_idx, test_idx


def task_stratified_split(n, task_ids, seed=42):
    """Return train/val/test indices: 70/10/20, tasks disjoint across splits."""
    unique_tasks = sorted(set(task_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_tasks)

    nt = len(unique_tasks)
    n_train = int(0.7 * nt)
    n_val = max(1, int(0.1 * nt))

    train_tasks = set(unique_tasks[:n_train])
    val_tasks = set(unique_tasks[n_train:n_train + n_val])
    test_tasks = set(unique_tasks[n_train + n_val:])

    train_idx = [i for i in range(n) if task_ids[i] in train_tasks]
    val_idx = [i for i in range(n) if task_ids[i] in val_tasks]
    test_idx = [i for i in range(n) if task_ids[i] in test_tasks]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def compute_metrics(probs, labels):
    """Compute all classification metrics."""
    results = {}
    if len(np.unique(labels)) < 2:
        return results

    preds = (probs > 0.5).astype(int)
    results["auroc"] = float(roc_auc_score(labels, probs))
    results["auprc"] = float(average_precision_score(labels, probs))
    results["macro_f1"] = float(f1_score(labels, preds, average="macro"))
    results["weighted_f1"] = float(f1_score(labels, preds, average="weighted"))
    results["precision"] = float(precision_score(labels, preds, zero_division=0))
    results["recall"] = float(recall_score(labels, preds))
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    results["fpr"] = float(fp / max(fp + tn, 1))
    results["accuracy"] = float((preds == labels).mean())
    return results


def per_group_metrics(probs, labels, groups):
    """Compute recall and FPR per group."""
    preds = (probs > 0.5).astype(int)
    breakdown = {}
    for gid in sorted(set(groups)):
        mask = np.array(groups) == gid
        if mask.sum() == 0:
            continue
        g_labels = labels[mask]
        g_preds = preds[mask]
        g_probs = probs[mask]
        n_attack = int((g_labels == 1).sum())
        n_benign = int((g_labels == 0).sum())
        stats = {"n_total": int(mask.sum()), "n_attack": n_attack, "n_benign": n_benign}
        if n_attack > 0:
            stats["recall"] = float(((g_preds == 1) & (g_labels == 1)).sum() / n_attack)
        if n_benign > 0:
            stats["fpr"] = float(((g_preds == 1) & (g_labels == 0)).sum() / n_benign)
        if len(np.unique(g_labels)) > 1:
            stats["auroc"] = float(roc_auc_score(g_labels, g_probs))
        breakdown[gid] = stats
    return breakdown


def train_and_evaluate(train_g, val_g, test_g, arch, args, device, seed):
    """Train model and return predictions on test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_g, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_g, batch_size=args.batch_size)
    test_loader = DataLoader(test_g, batch_size=args.batch_size)

    in_dim = train_g[0].x.shape[1]
    if arch == "mlp":
        model = MLPClassifier(in_dim, args.hidden_dim).to(device)
    else:
        model = GNNClassifier(in_dim, args.hidden_dim, arch=arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n_b = sum(1 for g in train_g if g.y.item() == 0)
    n_a = sum(1 for g in train_g if g.y.item() == 1)
    w = None
    if n_b > 0 and n_a > 0:
        w = torch.tensor([n_a / (n_b + n_a), n_b / (n_b + n_a)],
                         dtype=torch.float).to(device)

    best_val_auroc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, w)

        if epoch % 10 == 0:
            val_probs, val_labels = evaluate(model, val_loader, device)
            if len(np.unique(val_labels)) > 1:
                val_auroc = roc_auc_score(val_labels, val_probs)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
            if patience_counter >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_probs, test_labels = evaluate(model, test_loader, device)
    return test_probs, test_labels, best_val_auroc


def run_protocol(sessions, protocol_name, args, device, split="label_stratified"):
    """Run full evaluation for one dataset configuration across multiple seeds.

    split: "label_stratified" or "task_stratified"
    Returns aggregated results (mean ± std) across seeds.
    """
    print(f"\n{'='*70}")
    print(f"PROTOCOL: {protocol_name} (split={split}, seeds={args.seed_list})")
    print(f"{'='*70}")

    tool_vocab = build_tool_vocab(sessions)
    n_tools = len(tool_vocab)
    print(f"  Tools: {n_tools}")

    embedder = None
    if args.mode in ("content", "both"):
        print("  Loading sentence transformer...")
        embedder = get_content_embedder()

    print("  Building graphs...")
    graphs = sessions_to_graphs(sessions, args.mode, tool_vocab, n_tools, embedder)

    labels = np.array([g.y.item() for g in graphs])
    attack_types = [s["attack_type"] for s in sessions]
    sources = [s["source"] for s in sessions]
    agents = [s["agent"] for s in sessions]
    task_ids = [s["task_id"] for s in sessions]

    archs = ["gat", "gcn", "sage", "mlp"] if args.arch == "all" else [args.arch]
    protocol_results = {}

    for arch in archs:
        print(f"\n  --- {arch.upper()} ---")
        seed_metrics = []

        for seed in args.seed_list:
            # Split with this seed
            if split == "task_stratified":
                train_idx, val_idx, test_idx = task_stratified_split(
                    len(graphs), task_ids, seed)
                train_tasks = set(task_ids[i] for i in train_idx)
                test_tasks = set(task_ids[i] for i in test_idx)
                assert len(train_tasks & test_tasks) == 0, "Task leakage detected"
            else:
                train_idx, val_idx, test_idx = label_stratified_split(
                    len(graphs), labels, seed)

            train_g = [graphs[i] for i in train_idx]
            val_g = [graphs[i] for i in val_idx]
            test_g = [graphs[i] for i in test_idx]

            test_probs, test_labels, val_auroc = train_and_evaluate(
                train_g, val_g, test_g, arch, args, device, seed)

            metrics = compute_metrics(test_probs, test_labels)
            metrics["val_auroc"] = float(val_auroc)
            seed_metrics.append(metrics)

            print(f"    seed={seed}: AUROC={metrics.get('auroc',0):.4f}  "
                  f"MacF1={metrics.get('macro_f1',0):.4f}  "
                  f"Recall={metrics.get('recall',0):.4f}  FPR={metrics.get('fpr',0):.4f}")

        # Aggregate across seeds
        agg = {}
        metric_keys = ["auroc", "auprc", "macro_f1", "weighted_f1",
                       "precision", "recall", "fpr", "accuracy"]
        for m in metric_keys:
            vals = [s[m] for s in seed_metrics if m in s]
            if vals:
                agg[f"{m}_mean"] = float(np.mean(vals))
                agg[f"{m}_std"] = float(np.std(vals))
        agg["per_seed"] = seed_metrics

        # Per-attack-type breakdown (from last seed for breakdown detail)
        last_seed = args.seed_list[-1]
        if split == "task_stratified":
            train_idx, val_idx, test_idx = task_stratified_split(
                len(graphs), task_ids, last_seed)
        else:
            train_idx, val_idx, test_idx = label_stratified_split(
                len(graphs), labels, last_seed)
        test_attack_types = [attack_types[i] for i in test_idx]
        test_sources = [sources[i] for i in test_idx]
        test_agents = [agents[i] for i in test_idx]

        # Re-run last seed to get predictions for breakdown
        train_g = [graphs[i] for i in train_idx]
        val_g = [graphs[i] for i in val_idx]
        test_g = [graphs[i] for i in test_idx]
        test_probs, test_labels, _ = train_and_evaluate(
            train_g, val_g, test_g, arch, args, device, last_seed)

        agg["per_attack_type"] = per_group_metrics(test_probs, test_labels, test_attack_types)
        agg["per_source"] = per_group_metrics(test_probs, test_labels, test_sources)
        agg["per_agent"] = per_group_metrics(test_probs, test_labels, test_agents)

        print(f"\n    MEAN±STD: AUROC={agg.get('auroc_mean',0):.4f}±{agg.get('auroc_std',0):.4f}  "
              f"AUPRC={agg.get('auprc_mean',0):.4f}±{agg.get('auprc_std',0):.4f}  "
              f"MacF1={agg.get('macro_f1_mean',0):.4f}±{agg.get('macro_f1_std',0):.4f}  "
              f"Recall={agg.get('recall_mean',0):.4f}±{agg.get('recall_std',0):.4f}  "
              f"FPR={agg.get('fpr_mean',0):.4f}±{agg.get('fpr_std',0):.4f}")

        # Per-attack-type
        print(f"\n    Per-attack-type (seed={last_seed}):")
        for name, stats in sorted(agg["per_attack_type"].items()):
            parts = []
            if "recall" in stats:
                parts.append(f"recall={stats['recall']:.3f}")
            if "fpr" in stats:
                parts.append(f"FPR={stats['fpr']:.3f}")
            parts.append(f"n={stats['n_total']}")
            print(f"      {name:40s} {', '.join(parts)}")

        if len(set(test_sources)) > 1:
            print(f"\n    Per-source (seed={last_seed}):")
            for name, stats in sorted(agg["per_source"].items()):
                parts = []
                if "auroc" in stats:
                    parts.append(f"AUROC={stats['auroc']:.3f}")
                if "recall" in stats:
                    parts.append(f"recall={stats['recall']:.3f}")
                if "fpr" in stats:
                    parts.append(f"FPR={stats['fpr']:.3f}")
                parts.append(f"n={stats['n_total']}")
                print(f"      {name:15s} {', '.join(parts)}")

        protocol_results[arch] = agg

    # Summary table
    print(f"\n  {'Arch':<8} {'AUROC':>14} {'AUPRC':>14} {'MacF1':>14} {'Recall':>14} {'FPR':>14}")
    print(f"  {'-'*72}")
    for arch in archs:
        r = protocol_results[arch]
        print(f"  {arch.upper():<8} "
              f"{r.get('auroc_mean',0):.4f}±{r.get('auroc_std',0):.3f} "
              f"{r.get('auprc_mean',0):.4f}±{r.get('auprc_std',0):.3f} "
              f"{r.get('macro_f1_mean',0):.4f}±{r.get('macro_f1_std',0):.3f} "
              f"{r.get('recall_mean',0):.4f}±{r.get('recall_std',0):.3f} "
              f"{r.get('fpr_mean',0):.4f}±{r.get('fpr_std',0):.3f}")

    return protocol_results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Multi-Dataset Evaluation (mode={args.mode}, dataset={args.dataset}) ===")
    print(f"    Seeds: {args.seed_list}")

    all_protocol_results = {}

    if args.dataset in ("ras_eval", "combined"):
        ras_b, ras_a = load_ras_eval(args.raw_dir_ras, pooled=args.pooled)
    if args.dataset in ("atbench", "combined"):
        atb_b, atb_a = load_atbench(args.raw_dir_atb)
    if args.dataset == "combined":
        mcp_b = load_mcpbench(args.raw_dir_mcp)

    if args.dataset == "ras_eval":
        sessions = ras_b + ras_a
        # Task-stratified (primary — no task leakage)
        results = run_protocol(sessions, "RAS-Eval", args, device, split="task_stratified")
        all_protocol_results["ras_eval_task_strat"] = results
        # Label-stratified (for comparison)
        results = run_protocol(sessions, "RAS-Eval", args, device, split="label_stratified")
        all_protocol_results["ras_eval_label_strat"] = results

    elif args.dataset == "atbench":
        sessions = atb_b + atb_a
        results = run_protocol(sessions, "ATBench", args, device, split="label_stratified")
        all_protocol_results["atbench"] = results

    elif args.dataset == "combined":
        # RAS-Eval: both splits
        ras_sessions = ras_b + ras_a
        results = run_protocol(ras_sessions, "RAS-Eval", args, device, split="task_stratified")
        all_protocol_results["ras_eval_task_strat"] = results
        results = run_protocol(ras_sessions, "RAS-Eval", args, device, split="label_stratified")
        all_protocol_results["ras_eval_label_strat"] = results

        # ATBench: label-stratified only (no task structure)
        atb_sessions = atb_b + atb_a
        results = run_protocol(atb_sessions, "ATBench", args, device, split="label_stratified")
        all_protocol_results["atbench"] = results

        # Combined: label-stratified (mixed sources, no shared task structure)
        combined_sessions = ras_b + atb_b + mcp_b + ras_a + atb_a
        results = run_protocol(combined_sessions, "Combined (RAS+ATBench+mcpbench)",
                              args, device, split="label_stratified")
        all_protocol_results["combined"] = results

    # Save results
    out_dir = f"results_eval_{args.dataset}_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    results_out = {
        "mode": args.mode,
        "dataset": args.dataset,
        "pooled": args.pooled,
        "seeds": args.seed_list,
        "protocols": all_protocol_results,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
