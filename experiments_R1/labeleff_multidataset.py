#!/usr/bin/env python3
"""Table 10 (label efficiency) for ATBench / Combined — the columns the original
graph_label_efficiency.py cannot produce (it is hard-wired to RAS-Eval).

Reuses graph_label_efficiency.run_fold VERBATIM (same SSL pre-train / fine-tune /
supervised recipe) under StratifiedKFold (label-stratified — the protocol the
paper uses for ATBench and Combined, which have no shared task structure).
RAS-Eval stays on the original script (GroupKFold). Content features only.
"""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json
import sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import graph_label_efficiency as GLE
from graph_supervised import build_tool_vocab, sessions_to_graphs, get_content_embedder
from load_datasets import load_atbench, load_all


def build_graphs(dataset):
    if dataset == "atbench":
        b, a = load_atbench(os.path.join(_ROOT, "data/raw/atbench"))
        sessions = b + a
    elif dataset == "combined":
        b, a = load_all(
            ras_dir=os.path.join(_ROOT, "data/raw/ras_eval"),
            atb_dir=os.path.join(_ROOT, "data/raw/atbench"),
            mcp_dir=os.path.join(_ROOT, "data/raw/cx_cmu"),
            pooled=True)
        sessions = b + a
    else:
        raise ValueError(dataset)
    vocab = build_tool_vocab(sessions)
    embedder = get_content_embedder()
    graphs = sessions_to_graphs(sessions, "content", vocab, len(vocab), embedder)
    return graphs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["atbench", "combined"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    # graph_label_efficiency defaults, so run_fold behaves identically
    args = argparse.Namespace(seed=a.seed, hidden_dim=128, batch_size=64, lr=1e-3,
                              pretrain_epochs=100, finetune_epochs=50, supervised_epochs=100,
                              temperature=0.5, mask_rate=0.2, drop_edge_rate=0.2)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Label efficiency ({a.dataset}, label-stratified) ===")
    graphs = build_graphs(a.dataset)
    in_dim = graphs[0].x.shape[1]
    labels = np.array([g.y.item() for g in graphs])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    frac_results = {}
    for fraction in GLE.LABEL_FRACTIONS:
        print(f"\n  --- {fraction*100:.0f}% ---")
        folds = []
        for fold, (tr, te) in enumerate(skf.split(graphs, labels)):
            train_g = [graphs[i] for i in tr]; test_g = [graphs[i] for i in te]
            benign_in_fold = [g for g in train_g if g.y.item() == 0]
            folds.append(GLE.run_fold(train_g, test_g, benign_in_fold, in_dim,
                                      args, device, fold, fraction))
        ft = [r["ssl_finetune"].get("auroc", 0) for r in folds]
        sup = [r["supervised"].get("auroc", 0) for r in folds]
        frac_results[f"{fraction*100:.0f}%"] = {
            "ft_auroc_mean": float(np.mean(ft)), "ft_auroc_std": float(np.std(ft)),
            "sup_auroc_mean": float(np.mean(sup)), "sup_auroc_std": float(np.std(sup)),
            "folds": folds,
        }
        print(f"  Summary: FT={np.mean(ft):.4f}±{np.std(ft):.3f}  Sup={np.mean(sup):.4f}±{np.std(sup):.3f}")

    out_dir = a.out_dir or os.path.join(_ROOT, f"results_label_efficiency_{a.dataset}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"stratified_no_grouping": frac_results}, f, indent=2)
    print(f"\nSaved -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
