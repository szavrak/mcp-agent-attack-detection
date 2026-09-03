#!/usr/bin/env python3
"""Reproduce the SAGE/content/task-stratified AUROC locally to establish
comparability with the reference number (0.917 +/- .018 on GPU; ~0.909 on CPU).
Uses graph_supervised.run_fold verbatim.
"""
import argparse
import numpy as np
import torch
import common as C
import graph_supervised as GS


def main():
    device = torch.device("cpu")
    seeds = [7, 42, 123]

    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")
    graphs, _ = C.build_graphs(sessions, feats, edge_mode="full")
    task_ids = np.array([g.task_id for g in graphs])

    aurocs = []
    for seed in seeds:
        tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
        assert len(set(task_ids[tr]) & set(task_ids[te])) == 0
        train_g = [graphs[i] for i in tr]
        test_g = [graphs[i] for i in te]
        args = argparse.Namespace(seed=seed, batch_size=64, epochs=200,
                                  lr=1e-3, hidden_dim=128)
        res = GS.run_fold(train_g, test_g, args, device, 0, arch="sage")
        aurocs.append(res["auroc"])

    print(f"\nSAGE/content/task-stratified local: "
          f"{np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  (paper Colab 0.917)")


if __name__ == "__main__":
    main()
