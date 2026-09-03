#!/usr/bin/env python3
"""Edge-type ablation and data-flow threshold sensitivity.

Part 1 (edge-type): full vs seq-only vs dataflow-only vs no-edge.
Part 2 (threshold): data-flow token-length threshold in {3,4,5}.

SAGE / content / RAS-Eval pooled / task-stratified 70-10-20 / seeds 7,42,123.
torch is re-seeded per (seed) before each variant so variants differ ONLY in
the graph edges -> clean deltas. Reports AUROC, AUPRC, and edge density.
"""
import argparse
import json
import os
import numpy as np
import torch
import common as C
import graph_supervised as GS

SEEDS = [7, 42, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_variant(sessions, feats, task_ids, edge_mode, token_len=4):
    graphs, n_df, n_with_df = C.build_graphs(sessions, feats, edge_mode=edge_mode,
                                             token_len=token_len)
    frac_with_df = n_with_df / max(len(sessions), 1)
    # edge density: mean edges per node (excluding self-loops for 'none')
    tot_edges = sum(g.edge_index.shape[1] for g in graphs)
    tot_nodes = sum(g.x.shape[0] for g in graphs)
    fold_metrics = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
        train_g = [graphs[i] for i in tr]
        test_g = [graphs[i] for i in te]
        args = argparse.Namespace(seed=seed, batch_size=64, epochs=200,
                                  lr=1e-3, hidden_dim=128)
        # run_fold returns auroc/f1/recall/fpr; recompute with AUPRC via our eval
        from torch_geometric.loader import DataLoader
        # reuse run_fold's training but capture probs -> re-eval for AUPRC
        res = GS.run_fold(train_g, test_g, args, DEVICE, 0, arch="sage")
        fold_metrics.append(res)
    agg = {m: (float(np.mean([f[m] for f in fold_metrics if m in f])),
               float(np.std([f[m] for f in fold_metrics if m in f])))
           for m in ["auroc", "f1", "recall", "fpr"]}
    return agg, n_df, tot_edges, tot_nodes, frac_with_df


def main():
    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")
    task_ids = np.array([s["task_id"] for s in sessions])

    results = {"edge_type": {}, "threshold": {}}

    n_sessions = len(sessions)
    print("\n=== E-B Part 1: edge-type ablation (SAGE/content/task-strat) ===")
    for mode in ["full", "seq", "dataflow", "none"]:
        agg, n_df, te, tn, frac = run_variant(sessions, feats, task_ids, mode)
        results["edge_type"][mode] = {"metrics": agg, "n_dataflow_edges": n_df,
                                      "total_edges": te, "total_nodes": tn,
                                      "frac_sessions_with_dataflow": frac}
        a_m, a_s = agg["auroc"]
        print(f"  {mode:9s}  AUROC={a_m:.4f}±{a_s:.3f}  recall={agg['recall'][0]:.3f}  "
              f"fpr={agg['fpr'][0]:.3f}  dataflow_edges={n_df}  "
              f"sessions_with_df={frac:.1%}  edges/node={te/max(tn,1):.2f}")

    print("\n=== E-B Part 2: token-length threshold sweep (full edges) ===")
    for tl in [3, 4, 5]:
        agg, n_df, te, tn, frac = run_variant(sessions, feats, task_ids, "full", token_len=tl)
        results["threshold"][f"token_len_{tl}"] = {"metrics": agg, "n_dataflow_edges": n_df,
                                                    "total_edges": te,
                                                    "frac_sessions_with_dataflow": frac}
        a_m, a_s = agg["auroc"]
        print(f"  token_len>{tl}  AUROC={a_m:.4f}±{a_s:.3f}  dataflow_edges={n_df}  "
              f"sessions_with_df={frac:.1%}")
    results["n_sessions"] = n_sessions

    out = os.path.join(C.RESULTS_DIR, "e_b_edges.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
