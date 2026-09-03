#!/usr/bin/env python3
"""Graph-based SSL (contrastive learning) + fine-tuning for RAS-Eval.

Phase 1 — Pre-train: GraphCL-style contrastive learning on BENIGN-ONLY graphs.
  Augmentations: node feature masking + edge dropping.
  Loss: NT-Xent (InfoNCE).

Phase 2 — Fine-tune: add classification head, train with labels.

Phase 3 — Evaluate: task-stratified GroupKFold.
  - Pre-train on benign from train-tasks
  - Fine-tune on labeled train-tasks (train split)
  - Select best model on val (from train-tasks)
  - Evaluate on test-tasks
  - NO test data touches training, pre-training, or model selection.

Also reports pre-train-only anomaly detection (centroid distance) for comparison.
"""
import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from graph_supervised import (
    load_sessions, build_tool_vocab, sessions_to_graphs, get_content_embedder,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["metadata", "content", "both"], default="content")
    p.add_argument("--raw-dir", default="data/raw/ras_eval")
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooled", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--mask-rate", type=float, default=0.2)
    p.add_argument("--drop-edge-rate", type=float, default=0.2)
    return p.parse_args()


# ---- Encoder ----

class GATEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return torch.cat([x_mean, x_max], dim=1)


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.ELU(),
            torch.nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---- Augmentations ----

def augment_graph(data, mask_rate=0.2, drop_edge_rate=0.2):
    """Apply node feature masking + edge dropping."""
    aug = data.clone()

    # Node feature masking
    if mask_rate > 0:
        mask = torch.rand(aug.x.shape) > mask_rate
        aug.x = aug.x * mask.float().to(aug.x.device)

    # Edge dropping
    if drop_edge_rate > 0 and aug.edge_index.shape[1] > 1:
        n_edges = aug.edge_index.shape[1]
        keep = torch.rand(n_edges) > drop_edge_rate
        if keep.sum() > 0:
            aug.edge_index = aug.edge_index[:, keep]
        # Ensure at least one edge (self-loop)
        if aug.edge_index.shape[1] == 0:
            aug.edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=aug.x.device)

    return aug


# ---- NT-Xent Loss ----

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross-entropy loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_sim = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N),
    ])

    # Loss = -log(exp(pos) / sum(exp(all)))
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


# ---- Pre-training ----

def pretrain_epoch(encoder, proj_head, loader, optimizer, device, args):
    encoder.train()
    proj_head.train()
    total_loss = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        # Two augmented views
        aug1 = augment_graph(batch, args.mask_rate, args.drop_edge_rate)
        aug2 = augment_graph(batch, args.mask_rate, args.drop_edge_rate)

        h1 = encoder(aug1.x, aug1.edge_index, aug1.batch)
        h2 = encoder(aug2.x, aug2.edge_index, aug2.batch)

        z1 = proj_head(h1)
        z2 = proj_head(h2)

        loss = nt_xent_loss(z1, z2, args.temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(proj_head.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs

    return total_loss / n


# ---- Fine-tuning ----

def finetune_epoch(encoder, cls_head, loader, optimizer, device, class_weights=None,
                   freeze_encoder=False):
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()
    cls_head.train()
    total_loss = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        if freeze_encoder:
            with torch.no_grad():
                h = encoder(batch.x, batch.edge_index, batch.batch)
        else:
            h = encoder(batch.x, batch.edge_index, batch.batch)

        out = cls_head(h)
        loss = F.cross_entropy(out, batch.y.view(-1), weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(cls_head.parameters()) + ([] if freeze_encoder else list(encoder.parameters())),
            1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs

    return total_loss / n


@torch.no_grad()
def evaluate_classifier(encoder, cls_head, loader, device):
    encoder.eval()
    cls_head.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        h = encoder(batch.x, batch.edge_index, batch.batch)
        out = cls_head(h)
        probs = F.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


@torch.no_grad()
def evaluate_centroid(encoder, train_benign_loader, test_loader, device):
    """Unsupervised: score by distance from benign centroid in embedding space."""
    encoder.eval()

    # Compute benign centroid
    embeddings = []
    for batch in train_benign_loader:
        batch = batch.to(device)
        h = encoder(batch.x, batch.edge_index, batch.batch)
        embeddings.append(h.cpu())
    centroid = torch.cat(embeddings, dim=0).mean(dim=0)

    # Score test by distance to centroid
    scores, labels = [], []
    for batch in test_loader:
        batch = batch.to(device)
        h = encoder(batch.x, batch.edge_index, batch.batch)
        dists = torch.norm(h.cpu() - centroid.unsqueeze(0), dim=1)
        scores.extend(dists.numpy())
        labels.extend(batch.y.view(-1).cpu().numpy())

    return np.array(scores), np.array(labels)


def run_fold(train_graphs, test_graphs, args, device, fold_idx):
    rng = np.random.RandomState(args.seed + fold_idx)

    # Split train into train/val (val from train-tasks for model selection)
    indices = rng.permutation(len(train_graphs))
    n_val = max(1, len(indices) // 5)
    val_graphs = [train_graphs[i] for i in indices[:n_val]]
    trn_graphs = [train_graphs[i] for i in indices[n_val:]]

    trn_benign = [g for g in trn_graphs if g.y.item() == 0]

    if len(trn_benign) < 5:
        print(f"  Fold {fold_idx}: too few benign ({len(trn_benign)}), skipping")
        return None

    benign_loader = DataLoader(trn_benign, batch_size=args.batch_size, shuffle=True)
    trn_loader = DataLoader(trn_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    encoder = GATEncoder(in_dim, args.hidden_dim).to(device)
    proj_head = ProjectionHead(args.hidden_dim * 2, 64).to(device)

    # Phase 1: Pre-train on benign only (contrastive)
    pretrain_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.pretrain_epochs + 1):
        loss = pretrain_epoch(encoder, proj_head, benign_loader, pretrain_optimizer, device, args)

    # Evaluate pre-train only (centroid distance — unsupervised)
    centroid_scores, centroid_labels = evaluate_centroid(
        encoder, benign_loader, test_loader, device)
    centroid_auroc = roc_auc_score(centroid_labels, centroid_scores) if len(np.unique(centroid_labels)) > 1 else 0.0

    # Phase 2: Fine-tune with labels
    cls_head = ClassificationHead(args.hidden_dim * 2, args.hidden_dim).to(device)

    n_benign = sum(1 for g in trn_graphs if g.y.item() == 0)
    n_attack = sum(1 for g in trn_graphs if g.y.item() == 1)
    if n_benign > 0 and n_attack > 0:
        w = torch.tensor([n_attack / (n_benign + n_attack),
                          n_benign / (n_benign + n_attack)], dtype=torch.float).to(device)
    else:
        w = None

    ft_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(cls_head.parameters()),
        lr=args.lr * 0.1, weight_decay=1e-4)

    best_val_auroc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.finetune_epochs + 1):
        finetune_epoch(encoder, cls_head, trn_loader, ft_optimizer, device, w,
                       freeze_encoder=False)

        if epoch % 5 == 0:
            val_probs, val_labels = evaluate_classifier(encoder, cls_head, val_loader, device)
            if len(np.unique(val_labels)) > 1:
                val_auroc = roc_auc_score(val_labels, val_probs)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {
                        "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                        "cls_head": {k: v.cpu().clone() for k, v in cls_head.state_dict().items()},
                    }
                else:
                    patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        cls_head.load_state_dict(best_state["cls_head"])

    # Final evaluation on test (never seen before)
    test_probs, test_labels = evaluate_classifier(encoder, cls_head, test_loader, device)

    results = {}
    if len(np.unique(test_labels)) > 1:
        results["auroc_centroid"] = float(centroid_auroc)
        results["auroc_finetuned"] = float(roc_auc_score(test_labels, test_probs))
        preds = (test_probs > 0.5).astype(int)
        tp = ((preds == 1) & (test_labels == 1)).sum()
        fp = ((preds == 1) & (test_labels == 0)).sum()
        fn = ((preds == 0) & (test_labels == 1)).sum()
        tn = ((preds == 0) & (test_labels == 0)).sum()
        results["f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        results["recall"] = float(tp / max(tp + fn, 1))
        results["fpr"] = float(fp / max(fp + tn, 1))

    n_b = (test_labels == 0).sum()
    n_a = (test_labels == 1).sum()
    print(f"  Fold {fold_idx}: centroid={results.get('auroc_centroid',0):.4f}  "
          f"finetuned={results.get('auroc_finetuned',0):.4f}  "
          f"F1={results.get('f1',0):.4f}  Recall={results.get('recall',0):.4f}  "
          f"FPR={results.get('fpr',0):.4f}  "
          f"(train: {len(trn_benign)}b+{n_attack}a, test: {n_b}b/{n_a}a)")
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Graph SSL Contrastive + Fine-tune (mode={args.mode}) ===\n")

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

    print(f"\n{'='*60}")
    print(f"Task-Stratified 5-Fold SSL (mode={args.mode})")
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
    print(f"SUMMARY SSL (mode={args.mode})")
    print(f"{'='*60}")
    for metric in ["auroc_centroid", "auroc_finetuned", "f1", "recall", "fpr"]:
        vals = [r[metric] for r in fold_results if metric in r]
        if vals:
            print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out_dir = f"results_graph_ssl_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    results_out = {
        "mode": args.mode,
        "method": "graph_contrastive_learning + finetune",
        "pretraining": "benign-only contrastive (NT-Xent)",
        "finetuning": "labeled (cross-entropy)",
        "task_stratified": {
            "folds": fold_results,
            "mean": {m: float(np.mean([r[m] for r in fold_results if m in r]))
                     for m in ["auroc_centroid", "auroc_finetuned", "f1", "recall", "fpr"]},
            "std": {m: float(np.std([r[m] for r in fold_results if m in r]))
                    for m in ["auroc_centroid", "auroc_finetuned", "f1", "recall", "fpr"]},
        },
        "hyperparameters": {
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "temperature": args.temperature,
            "mask_rate": args.mask_rate,
            "drop_edge_rate": args.drop_edge_rate,
            "batch_size": args.batch_size,
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
