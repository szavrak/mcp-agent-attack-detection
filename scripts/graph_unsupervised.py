#!/usr/bin/env python3
"""Graph-based UNSUPERVISED anomaly detection for RAS-Eval.

Trains a graph autoencoder on BENIGN-ONLY data, scores by reconstruction error.
No attack labels used during training — only benign graphs.

Evaluation: task-stratified GroupKFold.
  - Train AE on benign from train-tasks only
  - Set threshold on val benign (from train-tasks, held out from training)
  - Evaluate on test-tasks (benign + attacked)
  - NO test data touches training or threshold selection.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from graph_supervised import (
    load_sessions, build_tool_vocab, metadata_features,
    content_features, build_edges, get_content_embedder,
    sessions_to_graphs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["metadata", "content", "both"], default="content")
    p.add_argument("--raw-dir", default="data/raw/ras_eval")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooled", action="store_true")
    return p.parse_args()


class GraphAutoencoder(torch.nn.Module):
    """Encode graph → bottleneck → reconstruct mean node features."""
    def __init__(self, in_dim, hidden_dim, bottleneck_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.pool_proj = torch.nn.Linear(hidden_dim * 2, bottleneck_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        z = self.pool_proj(torch.cat([x_mean, x_max], dim=1))
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, data):
        z = self.encode(data)
        x_hat = self.decode(z)
        return x_hat, z


def compute_target(data):
    """Compute per-graph mean node features as reconstruction target."""
    targets = []
    batch = data.batch
    for gid in range(data.num_graphs):
        mask = batch == gid
        targets.append(data.x[mask].mean(dim=0))
    return torch.stack(targets)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        x_hat, _ = model(batch)
        target = compute_target(batch)
        loss = F.mse_loss(x_hat, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def score_graphs(model, loader, device):
    """Score each graph by reconstruction MSE."""
    model.eval()
    scores = []
    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch)
        target = compute_target(batch)
        for i in range(batch.num_graphs):
            mse = F.mse_loss(x_hat[i], target[i]).item()
            scores.append(mse)
    return np.array(scores)


def run_fold(train_graphs, test_graphs, args, device, fold_idx):
    """Train on benign-only from train set, evaluate on test set."""
    rng = np.random.RandomState(args.seed + fold_idx)

    # Split train into benign-only for AE training + val
    train_benign = [g for g in train_graphs if g.y.item() == 0]
    indices = rng.permutation(len(train_benign))
    n_val = max(1, len(indices) // 5)
    val_benign = [train_benign[i] for i in indices[:n_val]]
    trn_benign = [train_benign[i] for i in indices[n_val:]]

    if len(trn_benign) < 5:
        print(f"  Fold {fold_idx}: too few benign ({len(trn_benign)}), skipping")
        return None

    train_loader = DataLoader(trn_benign, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_benign, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_benign[0].x.shape[1]
    model = GraphAutoencoder(in_dim, args.hidden_dim, args.bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % 10 == 0:
            val_scores = score_graphs(model, val_loader, device)
            val_loss = val_scores.mean()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Set threshold from val benign scores (NOT test)
    val_scores = score_graphs(model, val_loader, device)

    # Score test set
    test_scores = score_graphs(model, test_loader, device)
    test_labels = np.array([g.y.item() for g in test_graphs])

    results = {}
    if len(np.unique(test_labels)) > 1:
        results["auroc"] = roc_auc_score(test_labels, test_scores)
        for pct_name, pct in [("p95", 95), ("p99", 99)]:
            thresh = np.percentile(val_scores, pct)
            preds = (test_scores > thresh).astype(int)
            tp = ((preds == 1) & (test_labels == 1)).sum()
            fp = ((preds == 1) & (test_labels == 0)).sum()
            fn = ((preds == 0) & (test_labels == 1)).sum()
            tn = ((preds == 0) & (test_labels == 0)).sum()
            recall = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            results[f"f1_{pct_name}"] = float(f1)
            results[f"recall_{pct_name}"] = float(recall)
            results[f"fpr_{pct_name}"] = float(fpr)

    n_b = (test_labels == 0).sum()
    n_a = (test_labels == 1).sum()
    print(f"  Fold {fold_idx}: AUROC={results.get('auroc', 0):.4f}  "
          f"F1@p95={results.get('f1_p95', 0):.4f}  Recall@p95={results.get('recall_p95', 0):.4f}  "
          f"FPR@p95={results.get('fpr_p95', 0):.4f}  "
          f"(train: {len(trn_benign)}b, val: {len(val_benign)}b, test: {n_b}b/{n_a}a)")
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Graph Unsupervised AE (mode={args.mode}) ===\n")

    benign, attacked = load_sessions(args.raw_dir, pooled=args.pooled)
    all_sessions = benign + attacked

    tool_vocab = build_tool_vocab(all_sessions)
    n_tools = len(tool_vocab)
    print(f"Tool vocabulary: {n_tools} tools")

    embedder = None
    if args.mode in ("content", "both"):
        print("Loading sentence transformer...")
        embedder = get_content_embedder()

    print("Building graphs...")
    graphs = sessions_to_graphs(all_sessions, args.mode, tool_vocab, n_tools, embedder)

    labels = np.array([g.y.item() for g in graphs])
    task_ids = np.array([g.task_id for g in graphs])

    # Task-stratified 5-fold
    print(f"\n{'='*60}")
    print(f"Task-Stratified 5-Fold UNSUPERVISED (mode={args.mode})")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(graphs, labels, task_ids)):
        train_tasks = set(task_ids[train_idx])
        test_tasks = set(task_ids[test_idx])
        assert len(train_tasks & test_tasks) == 0

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        results = run_fold(train_graphs, test_graphs, args, device, fold)
        if results is not None:
            fold_results.append(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY UNSUPERVISED (mode={args.mode})")
    print(f"{'='*60}")
    for metric in ["auroc", "f1_p95", "recall_p95", "fpr_p95", "f1_p99", "recall_p99", "fpr_p99"]:
        vals = [r[metric] for r in fold_results if metric in r]
        if vals:
            print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out_dir = f"results_graph_unsupervised_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    results_out = {
        "mode": args.mode,
        "method": "graph_autoencoder",
        "training": "benign-only (unsupervised)",
        "task_stratified": {
            "folds": fold_results,
            "mean": {m: float(np.mean([r[m] for r in fold_results if m in r]))
                     for m in ["auroc", "f1_p95", "recall_p95", "fpr_p95"]},
            "std": {m: float(np.std([r[m] for r in fold_results if m in r]))
                    for m in ["auroc", "f1_p95", "recall_p95", "fpr_p95"]},
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
