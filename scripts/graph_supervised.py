#!/usr/bin/env python3
"""Graph-based supervised anomaly detection for RAS-Eval.

Builds per-session graphs from agent trajectories and trains a GNN classifier.
Two feature modes: metadata-only and content-aware (sentence embeddings).

Usage:
    python scripts/graph_supervised.py --mode metadata
    python scripts/graph_supervised.py --mode content
    python scripts/graph_supervised.py --mode both
"""
import argparse
import json
import os
import sys
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["metadata", "content", "both"], default="both")
    p.add_argument("--arch", choices=["gat", "gcn", "sage", "mlp", "all"], default="all",
                   help="GNN architecture, 'mlp' for no-graph baseline, or 'all'")
    p.add_argument("--raw-dir", default="data/raw/ras_eval")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="7,42,123",
                   help="Comma-separated seeds for repeated evaluation")
    p.add_argument("--pooled", action="store_true",
                   help="Use all models for benign (not just glm-4-flash)")
    return p.parse_args()


# ---- Data Loading ----

def load_attack_task_map(raw_dir):
    path = os.path.join(raw_dir, "_repo", "data", "tasks", "attack_tasks.json")
    with open(path) as f:
        tasks = json.load(f)
    return {t["index"]: t["target_index"] for t in tasks}


def extract_tool_calls(record):
    """Extract tool calls with full content from a RAS-Eval record."""
    calls = []
    pending = []
    for msg in record.get("response", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "AIMessage" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                pending.append({
                    "name": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "args_str": json.dumps(tc.get("args", {})),
                })
        elif msg.get("type") == "ToolMessage" and pending:
            ci = pending.pop(0)
            content = str(msg.get("content", ""))
            calls.append({
                "tool_name": ci["name"],
                "args_str": ci["args_str"],
                "response_str": content,
                "response_length": len(content),
                "params_hash": int(hashlib.md5(ci["args_str"].encode()).hexdigest(), 16) % 10000,
            })
    return calls


def load_sessions(raw_dir, pooled=False):
    """Load benign + attacked sessions with task IDs."""
    attack_map = load_attack_task_map(raw_dir)

    benign = []
    logs_dir = os.path.join(raw_dir, "logs")
    exclude = {"guard_response.jsonl"}
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl") or fname in exclude:
            continue
        if not pooled and fname != "glm-4-flash.jsonl":
            continue
        with open(os.path.join(logs_dir, fname)) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                calls = extract_tool_calls(r)
                if not calls:
                    continue
                benign.append({
                    "calls": calls,
                    "label": 0,
                    "task_id": r.get("index", r.get("id", -1)),
                    "model": fname.replace(".jsonl", ""),
                })

    attacked = []
    with open(os.path.join(raw_dir, "attacked", "glm-4-flash.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            calls = extract_tool_calls(r)
            if not calls:
                continue
            target = attack_map.get(r["index"], -1)
            attacked.append({
                "calls": calls,
                "label": 1,
                "task_id": target,
                "model": "glm-4-flash",
            })

    print(f"Loaded {len(benign)} benign + {len(attacked)} attacked sessions")
    return benign, attacked


# ---- Feature Extraction ----

def build_tool_vocab(sessions):
    tools = set()
    for s in sessions:
        for c in s["calls"]:
            tools.add(c["tool_name"])
    return {t: i for i, t in enumerate(sorted(tools))}


def metadata_features(call, tool_vocab, n_tools):
    """Per-node metadata features: one-hot tool + normalized params_hash + response_length."""
    onehot = np.zeros(n_tools, dtype=np.float32)
    idx = tool_vocab.get(call["tool_name"], -1)
    if idx >= 0:
        onehot[idx] = 1.0
    return np.concatenate([
        onehot,
        [call["params_hash"] / 10000.0],
        [min(call["response_length"], 10000) / 10000.0],
    ])


def get_content_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return model


def content_features(call, embedder):
    """Per-node content features: sentence embeddings of args + response."""
    args_emb = embedder.encode(call["args_str"][:512], show_progress_bar=False)
    resp_text = call["response_str"][:512] if call["response_str"] else "empty"
    resp_emb = embedder.encode(resp_text, show_progress_bar=False)
    return np.concatenate([args_emb, resp_emb]).astype(np.float32)


def build_edges(calls):
    """Build edge index: sequential + data-flow edges."""
    n = len(calls)
    src, dst = [], []

    # Sequential edges (bidirectional)
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])

    # Data flow: if response of call_i appears (substring) in args of call_j (j > i)
    for i in range(n):
        resp = calls[i]["response_str"]
        if not resp or len(resp) > 1000:
            continue
        for j in range(i + 1, n):
            args = calls[j]["args_str"]
            if resp[:50] in args or any(
                v in args for v in resp.split()[:5] if len(v) > 4
            ):
                src.extend([i, j])
                dst.extend([j, i])

    if not src:
        src, dst = [0], [0]

    return torch.tensor([src, dst], dtype=torch.long)


def sessions_to_graphs(sessions, mode, tool_vocab, n_tools, embedder=None):
    """Convert sessions to PyG Data objects."""
    graphs = []
    for idx, s in enumerate(sessions):
        calls = s["calls"]
        n = len(calls)

        node_feats = []
        for c in calls:
            parts = []
            if mode in ("metadata", "both"):
                parts.append(metadata_features(c, tool_vocab, n_tools))
            if mode in ("content", "both"):
                parts.append(content_features(c, embedder))
            node_feats.append(np.concatenate(parts))

        x = torch.tensor(np.stack(node_feats), dtype=torch.float)
        edge_index = build_edges(calls)
        y = torch.tensor([s["label"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.task_id = s["task_id"]
        data.session_idx = idx
        graphs.append(data)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(sessions)} sessions")

    print(f"  Built {len(graphs)} graphs, feature dim = {graphs[0].x.shape[1]}")
    return graphs


# ---- GNN Models ----

def build_conv(arch, in_dim, out_dim, heads=4):
    if arch == "gat":
        return GATConv(in_dim, out_dim, heads=heads, concat=False)
    elif arch == "gcn":
        return GCNConv(in_dim, out_dim)
    elif arch == "sage":
        return SAGEConv(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown arch: {arch}")


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, arch="gat", heads=4):
        super().__init__()
        self.conv1 = build_conv(arch, in_dim, hidden_dim, heads)
        self.conv2 = build_conv(arch, hidden_dim, hidden_dim, heads)
        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.elu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class MLPClassifier(torch.nn.Module):
    """No-graph baseline: mean+max pool raw node features, then MLP."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, batch = data.x, data.batch
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = F.elu(self.lin1(x))
        x = self.dropout(x)
        x = F.elu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        return x


# Backward compatible alias
GATClassifier = lambda in_dim, hidden_dim, heads=4: GNNClassifier(in_dim, hidden_dim, arch="gat", heads=heads)


# ---- Training + Evaluation ----

def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.view(-1), weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


def run_fold(train_graphs, test_graphs, args, device, fold_idx, arch="gat"):
    """Train and evaluate one fold. Splits train into train/val for early stopping."""
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
        model = MLPClassifier(in_dim, args.hidden_dim).to(device)
    else:
        model = GNNClassifier(in_dim, args.hidden_dim, arch=arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Class weights for imbalanced data
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
        loss = train_epoch(model, train_loader, optimizer, device, w)

        if epoch % 10 == 0:
            # Early stopping on VALIDATION set (not test)
            val_probs, val_labels = evaluate(model, val_loader, device)
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

    # Final eval on held-out TEST set with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    probs, labels = evaluate(model, test_loader, device)

    results = {}
    if len(np.unique(labels)) > 1:
        results["auroc"] = roc_auc_score(labels, probs)
        preds = (probs > 0.5).astype(int)
        results["f1"] = f1_score(labels, preds)
        results["precision"] = precision_score(labels, preds, zero_division=0)
        results["recall"] = recall_score(labels, preds)
        fpr = ((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        results["fpr"] = float(fpr)

    n_b = (labels == 0).sum()
    n_a = (labels == 1).sum()
    print(f"  Fold {fold_idx}: AUROC={results.get('auroc', 0):.4f}  "
          f"F1={results.get('f1', 0):.4f}  Recall={results.get('recall', 0):.4f}  "
          f"FPR={results.get('fpr', 0):.4f}  (test: {n_b}b/{n_a}a)")
    return results


def task_stratified_split(n, task_ids, seed=42):
    """70/10/20 split with tasks disjoint across train/val/test."""
    unique_tasks = sorted(set(task_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_tasks)
    nt = len(unique_tasks)
    n_train = int(0.7 * nt)
    n_val = max(1, int(0.1 * nt))
    train_tasks = set(unique_tasks[:n_train])
    val_tasks = set(unique_tasks[n_train:n_train + n_val])
    test_tasks = set(unique_tasks[n_train + n_val:])
    train_idx = np.array([i for i in range(n) if task_ids[i] in train_tasks])
    val_idx = np.array([i for i in range(n) if task_ids[i] in val_tasks])
    test_idx = np.array([i for i in range(n) if task_ids[i] in test_tasks])
    return train_idx, val_idx, test_idx


def label_stratified_split(n, labels, seed=42):
    """70/10/20 stratified by label."""
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, stratify=temp_labels, random_state=seed)
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Graph Supervised Detection (mode={args.mode}) ===\n")

    # Load data
    benign, attacked = load_sessions(args.raw_dir, pooled=args.pooled)
    all_sessions = benign + attacked

    # Build tool vocabulary
    tool_vocab = build_tool_vocab(all_sessions)
    n_tools = len(tool_vocab)
    print(f"Tool vocabulary: {n_tools} tools")

    # Load embedder if needed
    embedder = None
    if args.mode in ("content", "both"):
        print("Loading sentence transformer...")
        embedder = get_content_embedder()

    # Convert to graphs
    print("Building graphs...")
    graphs = sessions_to_graphs(all_sessions, args.mode, tool_vocab, n_tools, embedder)

    labels = np.array([g.y.item() for g in graphs])
    task_ids = np.array([g.task_id for g in graphs])

    archs = ["gat", "gcn", "sage", "mlp"] if args.arch == "all" else [args.arch]
    seeds = [int(s) for s in args.seeds.split(",")] if hasattr(args, "seeds") and args.seeds else [args.seed]
    all_results = {}

    for arch in archs:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch.upper()} | mode={args.mode}")
        print(f"{'='*60}")

        # Task-stratified 70/10/20 split, repeated across seeds
        print(f"\n--- Task-Stratified 70/10/20 (seeds={seeds}) ---")
        seed_results = []
        for seed in seeds:
            train_idx, val_idx, test_idx = task_stratified_split(len(graphs), task_ids, seed)
            assert len(set(task_ids[train_idx]) & set(task_ids[test_idx])) == 0
            train_g = [graphs[i] for i in train_idx]
            test_g = [graphs[i] for i in test_idx]
            args_copy = argparse.Namespace(**vars(args))
            args_copy.seed = seed
            results = run_fold(train_g, test_g, args_copy, device, 0, arch=arch)
            seed_results.append(results)

        print(f"\n  SUMMARY {arch.upper()} (task-stratified):")
        for metric in ["auroc", "f1", "recall", "fpr"]:
            vals = [r[metric] for r in seed_results if metric in r]
            if vals:
                print(f"    {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Random label-stratified split for leakage comparison
        print(f"\n--- Label-Stratified 70/10/20 (seeds={seeds}) ---")
        rand_results = []
        for seed in seeds:
            train_idx, val_idx, test_idx = label_stratified_split(len(graphs), labels, seed)
            train_g = [graphs[i] for i in train_idx]
            test_g = [graphs[i] for i in test_idx]
            args_copy = argparse.Namespace(**vars(args))
            args_copy.seed = seed
            results = run_fold(train_g, test_g, args_copy, device, 0, arch=arch)
            rand_results.append(results)

        print(f"\n  SUMMARY {arch.upper()} (label-stratified):")
        for metric in ["auroc", "f1", "recall", "fpr"]:
            vals = [r[metric] for r in rand_results if metric in r]
            if vals:
                print(f"    {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        all_results[arch] = {
            "task_stratified": {
                "seeds": seed_results,
                "mean": {m: float(np.mean([r[m] for r in seed_results if m in r]))
                         for m in ["auroc", "f1", "recall", "fpr"]},
                "std": {m: float(np.std([r[m] for r in seed_results if m in r]))
                        for m in ["auroc", "f1", "recall", "fpr"]},
            },
            "label_stratified": {
                "seeds": rand_results,
                "mean": {m: float(np.mean([r[m] for r in rand_results if m in r]))
                         for m in ["auroc", "f1", "recall", "fpr"]},
            },
        }

    # Final comparison table
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE COMPARISON (mode={args.mode})")
    print(f"{'='*60}")
    print(f"{'Arch':<8} {'TaskStrat AUROC':>16} {'LabelStrat AUROC':>18}")
    print("-" * 46)
    for arch, r in all_results.items():
        ts = r["task_stratified"]["mean"]["auroc"]
        ts_s = r["task_stratified"]["std"]["auroc"]
        ls = r["label_stratified"]["mean"]["auroc"]
        print(f"{arch.upper():<8} {ts:>8.4f}±{ts_s:.3f}    {ls:>8.4f}")

    # Save
    out_dir = f"results_graph_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    results_out = {"mode": args.mode, "pooled": args.pooled, "architectures": all_results}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
