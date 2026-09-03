#!/usr/bin/env python3
"""Label efficiency experiment: SSL+fine-tune vs supervised from scratch.

For each label fraction (1%, 5%, 10%, 25%, 50%, 100%):
  - SSL: pre-train encoder on ALL benign (contrastive, no labels),
         fine-tune classifier on X% of labeled data
  - Supervised: train GAT from scratch on X% of labeled data

Both evaluated under:
  1. GroupKFold (task-stratified, inductive — unseen tasks at test time)
  2. StratifiedKFold (stratified by label, no grouping — tasks may leak)

No test leakage: val set used for early stopping, test never seen during training.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_supervised import (
    load_sessions, build_tool_vocab, sessions_to_graphs,
    get_content_embedder, GATClassifier,
)
from graph_ssl import GATEncoder, ProjectionHead, ClassificationHead, augment_graph, nt_xent_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/ras_eval")
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--supervised-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooled", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--mask-rate", type=float, default=0.2)
    p.add_argument("--drop-edge-rate", type=float, default=0.2)
    return p.parse_args()


LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def subsample_labeled(graphs, fraction, rng):
    """Subsample a fraction of labeled graphs, preserving class ratio."""
    if fraction >= 1.0:
        return graphs
    benign = [g for g in graphs if g.y.item() == 0]
    attack = [g for g in graphs if g.y.item() == 1]
    n_b = max(1, int(len(benign) * fraction))
    n_a = max(1, int(len(attack) * fraction))
    idx_b = rng.permutation(len(benign))[:n_b]
    idx_a = rng.permutation(len(attack))[:n_a]
    return [benign[i] for i in idx_b] + [attack[i] for i in idx_a]


def get_class_weights(graphs, device):
    n_b = sum(1 for g in graphs if g.y.item() == 0)
    n_a = sum(1 for g in graphs if g.y.item() == 1)
    if n_b > 0 and n_a > 0:
        return torch.tensor([n_a / (n_b + n_a), n_b / (n_b + n_a)],
                            dtype=torch.float).to(device)
    return None


# ---- SSL Pre-training ----

def pretrain_ssl(encoder, proj_head, benign_graphs, args, device):
    """Pre-train encoder on benign graphs with contrastive learning."""
    loader = DataLoader(benign_graphs, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.pretrain_epochs + 1):
        encoder.train()
        proj_head.train()
        for batch in loader:
            batch = batch.to(device)
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


# ---- Fine-tune / Train ----

def train_classifier(model_or_encoder, cls_head, train_graphs, val_graphs,
                     args, device, max_epochs, lr, is_ssl=False,
                     freeze_encoder=False):
    """Train classifier (SSL fine-tune or supervised from scratch).
    Returns best model state based on val AUROC.
    freeze_encoder=True: linear probing (only train cls_head)."""
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    w = get_class_weights(train_graphs, device)

    if is_ssl:
        if freeze_encoder:
            params = list(cls_head.parameters())
        else:
            params = list(model_or_encoder.parameters()) + list(cls_head.parameters())
    else:
        params = list(model_or_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    best_val_auroc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        if is_ssl:
            if freeze_encoder:
                model_or_encoder.eval()
            else:
                model_or_encoder.train()
            cls_head.train()
        else:
            model_or_encoder.train()

        for batch in train_loader:
            batch = batch.to(device)
            if is_ssl:
                if freeze_encoder:
                    with torch.no_grad():
                        h = model_or_encoder(batch.x, batch.edge_index, batch.batch)
                else:
                    h = model_or_encoder(batch.x, batch.edge_index, batch.batch)
                out = cls_head(h)
            else:
                out = model_or_encoder(batch)
            loss = F.cross_entropy(out, batch.y.view(-1), weight=w)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        if epoch % 5 == 0:
            if is_ssl:
                model_or_encoder.eval()
                cls_head.eval()
            else:
                model_or_encoder.eval()

            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    if is_ssl:
                        h = model_or_encoder(batch.x, batch.edge_index, batch.batch)
                        out = cls_head(h)
                    else:
                        out = model_or_encoder(batch)
                    probs = F.softmax(out, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(batch.y.view(-1).cpu().numpy())

            if len(np.unique(all_labels)) > 1:
                val_auroc = roc_auc_score(all_labels, np.array(all_probs))
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    if is_ssl:
                        best_state = {
                            "encoder": {k: v.cpu().clone() for k, v in model_or_encoder.state_dict().items()},
                            "cls_head": {k: v.cpu().clone() for k, v in cls_head.state_dict().items()},
                        }
                    else:
                        best_state = {k: v.cpu().clone() for k, v in model_or_encoder.state_dict().items()}
                else:
                    patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        if is_ssl:
            model_or_encoder.load_state_dict(best_state["encoder"])
            cls_head.load_state_dict(best_state["cls_head"])
        else:
            model_or_encoder.load_state_dict(best_state)

    return best_val_auroc


def evaluate_model(model_or_encoder, cls_head, test_graphs, device, is_ssl=False):
    """Evaluate on test set. Returns AUROC, F1, Recall, FPR."""
    loader = DataLoader(test_graphs, batch_size=64)
    if is_ssl:
        model_or_encoder.eval()
        cls_head.eval()
    else:
        model_or_encoder.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if is_ssl:
                h = model_or_encoder(batch.x, batch.edge_index, batch.batch)
                out = cls_head(h)
            else:
                out = model_or_encoder(batch)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    results = {}
    if len(np.unique(labels)) > 1:
        results["auroc"] = float(roc_auc_score(labels, probs))
        preds = (probs > 0.5).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        results["f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        results["recall"] = float(tp / max(tp + fn, 1))
        results["fpr"] = float(fp / max(fp + tn, 1))
    return results


def run_fold(train_graphs, test_graphs, all_benign_in_fold, in_dim, args, device,
             fold_idx, fraction):
    """Run one fold for one label fraction. Returns results for both SSL and supervised."""
    rng = np.random.RandomState(args.seed + fold_idx + int(fraction * 1000))

    # Split train into train/val (80/20)
    indices = rng.permutation(len(train_graphs))
    n_val = max(1, len(indices) // 5)
    val_graphs = [train_graphs[i] for i in indices[:n_val]]
    trn_graphs_full = [train_graphs[i] for i in indices[n_val:]]

    # Subsample labeled data
    trn_labeled = subsample_labeled(trn_graphs_full, fraction, rng)
    n_b_labeled = sum(1 for g in trn_labeled if g.y.item() == 0)
    n_a_labeled = sum(1 for g in trn_labeled if g.y.item() == 1)

    # ---- SSL: pre-train on ALL benign ----
    encoder_ssl = GATEncoder(in_dim, args.hidden_dim).to(device)
    proj_head = ProjectionHead(args.hidden_dim * 2, 64).to(device)
    pretrain_ssl(encoder_ssl, proj_head, all_benign_in_fold, args, device)

    # ---- SSL + linear probing (freeze encoder, train classifier only) ----
    import copy
    encoder_frozen = copy.deepcopy(encoder_ssl)
    cls_head_lp = ClassificationHead(args.hidden_dim * 2, args.hidden_dim).to(device)
    train_classifier(encoder_frozen, cls_head_lp, trn_labeled, val_graphs,
                     args, device, args.finetune_epochs, args.lr, is_ssl=True,
                     freeze_encoder=True)
    lp_results = evaluate_model(encoder_frozen, cls_head_lp, test_graphs, device, is_ssl=True)

    # ---- SSL + full fine-tune (unfreeze encoder) ----
    cls_head_ft = ClassificationHead(args.hidden_dim * 2, args.hidden_dim).to(device)
    train_classifier(encoder_ssl, cls_head_ft, trn_labeled, val_graphs,
                     args, device, args.finetune_epochs, args.lr * 0.1, is_ssl=True,
                     freeze_encoder=False)
    ssl_results = evaluate_model(encoder_ssl, cls_head_ft, test_graphs, device, is_ssl=True)

    # ---- Supervised: train from scratch on subsampled labels ----
    model_sup = GATClassifier(in_dim, args.hidden_dim).to(device)
    train_classifier(model_sup, None, trn_labeled, val_graphs,
                     args, device, args.supervised_epochs, args.lr, is_ssl=False)
    sup_results = evaluate_model(model_sup, None, test_graphs, device, is_ssl=False)

    n_test_b = sum(1 for g in test_graphs if g.y.item() == 0)
    n_test_a = sum(1 for g in test_graphs if g.y.item() == 1)

    print(f"    Fold {fold_idx} ({fraction*100:.0f}%): "
          f"LP={lp_results.get('auroc',0):.4f}  "
          f"FT={ssl_results.get('auroc',0):.4f}  "
          f"Sup={sup_results.get('auroc',0):.4f}  "
          f"(labeled: {n_b_labeled}b+{n_a_labeled}a, test: {n_test_b}b/{n_test_a}a)")

    return {"linear_probe": lp_results, "ssl_finetune": ssl_results,
            "supervised": sup_results,
            "n_labeled": n_b_labeled + n_a_labeled,
            "n_labeled_benign": n_b_labeled, "n_labeled_attack": n_a_labeled}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Label Efficiency Experiment ===\n")

    benign, attacked = load_sessions(args.raw_dir, pooled=args.pooled)
    all_sessions = benign + attacked

    tool_vocab = build_tool_vocab(all_sessions)
    n_tools = len(tool_vocab)

    print("Loading sentence transformer...")
    embedder = get_content_embedder()

    print("Building graphs...")
    graphs = sessions_to_graphs(all_sessions, "content", tool_vocab, n_tools, embedder)
    in_dim = graphs[0].x.shape[1]

    labels = np.array([g.y.item() for g in graphs])
    task_ids = np.array([g.task_id for g in graphs])

    all_results = {}

    # ---- 1. Task-Stratified GroupKFold ----
    print(f"\n{'='*70}")
    print("PROTOCOL 1: Task-Stratified GroupKFold (inductive)")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=5)
    gkf_results = {}

    for fraction in LABEL_FRACTIONS:
        print(f"\n  --- Label fraction: {fraction*100:.0f}% ---")
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(graphs, labels, task_ids)):
            assert len(set(task_ids[train_idx]) & set(task_ids[test_idx])) == 0

            train_graphs = [graphs[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]
            all_benign_in_fold = [g for g in train_graphs if g.y.item() == 0]

            results = run_fold(train_graphs, test_graphs, all_benign_in_fold,
                               in_dim, args, device, fold, fraction)
            fold_results.append(results)

        lp_aurocs = [r["linear_probe"].get("auroc", 0) for r in fold_results]
        ft_aurocs = [r["ssl_finetune"].get("auroc", 0) for r in fold_results]
        sup_aurocs = [r["supervised"].get("auroc", 0) for r in fold_results]
        gkf_results[f"{fraction*100:.0f}%"] = {
            "folds": fold_results,
            "lp_auroc_mean": float(np.mean(lp_aurocs)),
            "lp_auroc_std": float(np.std(lp_aurocs)),
            "ft_auroc_mean": float(np.mean(ft_aurocs)),
            "ft_auroc_std": float(np.std(ft_aurocs)),
            "sup_auroc_mean": float(np.mean(sup_aurocs)),
            "sup_auroc_std": float(np.std(sup_aurocs)),
        }
        print(f"  Summary: LP={np.mean(lp_aurocs):.4f}±{np.std(lp_aurocs):.4f}  "
              f"FT={np.mean(ft_aurocs):.4f}±{np.std(ft_aurocs):.4f}  "
              f"Sup={np.mean(sup_aurocs):.4f}±{np.std(sup_aurocs):.4f}")

    all_results["task_stratified"] = gkf_results

    # ---- 2. StratifiedKFold (no grouping) ----
    print(f"\n{'='*70}")
    print("PROTOCOL 2: StratifiedKFold (transductive, tasks may leak)")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    skf_results = {}

    for fraction in LABEL_FRACTIONS:
        print(f"\n  --- Label fraction: {fraction*100:.0f}% ---")
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
            train_graphs = [graphs[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]
            all_benign_in_fold = [g for g in train_graphs if g.y.item() == 0]

            results = run_fold(train_graphs, test_graphs, all_benign_in_fold,
                               in_dim, args, device, fold, fraction)
            fold_results.append(results)

        lp_aurocs = [r["linear_probe"].get("auroc", 0) for r in fold_results]
        ft_aurocs = [r["ssl_finetune"].get("auroc", 0) for r in fold_results]
        sup_aurocs = [r["supervised"].get("auroc", 0) for r in fold_results]
        skf_results[f"{fraction*100:.0f}%"] = {
            "folds": fold_results,
            "lp_auroc_mean": float(np.mean(lp_aurocs)),
            "lp_auroc_std": float(np.std(lp_aurocs)),
            "ft_auroc_mean": float(np.mean(ft_aurocs)),
            "ft_auroc_std": float(np.std(ft_aurocs)),
            "sup_auroc_mean": float(np.mean(sup_aurocs)),
            "sup_auroc_std": float(np.std(sup_aurocs)),
        }
        print(f"  Summary: LP={np.mean(lp_aurocs):.4f}±{np.std(lp_aurocs):.4f}  "
              f"FT={np.mean(ft_aurocs):.4f}±{np.std(ft_aurocs):.4f}  "
              f"Sup={np.mean(sup_aurocs):.4f}±{np.std(sup_aurocs):.4f}")

    all_results["stratified_no_grouping"] = skf_results

    # ---- Print Final Summary Table ----
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for protocol_name, protocol_results in all_results.items():
        print(f"\n--- {protocol_name} ---")
        print(f"{'Labels':>8}  {'LinProbe':>14}  {'SSL+FT':>14}  {'Supervised':>14}  {'LP-Sup':>8}")
        print("-" * 65)
        for frac_label, r in protocol_results.items():
            lp_m = r["lp_auroc_mean"]
            ft_m = r["ft_auroc_mean"]
            sup_m = r["sup_auroc_mean"]
            delta = lp_m - sup_m
            sign = "+" if delta > 0 else ""
            print(f"{frac_label:>8}  "
                  f"{lp_m:>8.4f}±{r['lp_auroc_std']:.3f}  "
                  f"{ft_m:>8.4f}±{r['ft_auroc_std']:.3f}  "
                  f"{sup_m:>8.4f}±{r['sup_auroc_std']:.3f}  "
                  f"{sign}{delta:>7.4f}")

    # Save
    out_dir = "results_label_efficiency"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
