"""
Ablation: effect of data-flow substring window size on detection.
Tests resp[:W] for W in {50, 100, 200, 500, None(full)} and reports
edge counts + AUROC on RAS-Eval (task-stratified, seed=42, SAGE, content).
"""
import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_supervised import (
    load_sessions, sessions_to_graphs,
    GNNClassifier, train_epoch, evaluate
)

SEED = 42
WINDOWS = [50, 100, 200, 500, None]

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "ras_eval"
benign, attacked = load_sessions(str(raw_dir), pooled=True)
all_sessions = benign + attacked

# Task IDs for GroupKFold
task_ids = np.array([s.get("target_index", s.get("task_id", -1)) for s in all_sessions])

gkf = GroupKFold(n_splits=5)
train_idx, test_idx = list(gkf.split(np.zeros(len(all_sessions)), groups=task_ids))[0]

# Encode node features ONCE with the default build_edges (window=50)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

tool_vocab = {}
for s in all_sessions:
    for c in s["calls"]:
        t = c["tool_name"]
        if t not in tool_vocab:
            tool_vocab[t] = len(tool_vocab)
n_tools = len(tool_vocab)

print("Encoding node features (one time)...")
base_graphs = sessions_to_graphs(all_sessions, "content", tool_vocab, n_tools, embedder)
print(f"Encoded {len(base_graphs)} graphs")


def rebuild_edges_with_window(session, window):
    calls = session["calls"]
    n = len(calls)
    src, dst = [], []
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    df_count = 0
    for i in range(n):
        resp = calls[i]["response_str"]
        if not resp or len(resp) > 1000:
            continue
        for j in range(i + 1, n):
            args = calls[j]["args_str"]
            resp_sub = resp[:window] if window else resp
            if resp_sub in args or any(
                v in args for v in resp.split()[:5] if len(v) > 4
            ):
                src.extend([i, j])
                dst.extend([j, i])
                df_count += 1
    if not src:
        src, dst = [0], [0]
    return torch.tensor([src, dst], dtype=torch.long), df_count


print(f"\n{'Window':<10} {'DF edges':<12} {'AUROC':<10} {'Time(s)':<10}")
print("─" * 45)

for window in WINDOWS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    total_df = 0
    graphs = []
    for i, s in enumerate(all_sessions):
        g = base_graphs[i]
        edge_index, df = rebuild_edges_with_window(s, window)
        total_df += df
        graphs.append(Data(x=g.x, edge_index=edge_index, y=g.y))

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64)

    in_dim = graphs[0].x.size(1)
    model = GNNClassifier(in_dim, hidden_dim=128, arch="sage").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_auroc = 0
    for epoch in range(80):
        train_epoch(model, train_loader, optimizer, device)
        probs, labels = evaluate(model, test_loader, device)
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = 0.5
        if auroc > best_auroc:
            best_auroc = auroc

    elapsed = time.time() - t0
    w_str = str(window) if window else "full"
    print(f"{w_str:<10} {total_df:<12} {best_auroc:<10.4f} {elapsed:<10.1f}")
