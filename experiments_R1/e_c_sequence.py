#!/usr/bin/env python3
"""Sequence-model (GRU) baseline. Tool calls are an ordered sequence, so a
recurrent model is a more natural baseline than flat pooled tree ensembles.

A GRU consumes the ordered per-call content embeddings (768-d each), then a
mean||max pool over hidden states feeds the same 2-layer head used by the GNNs.
RAS-Eval pooled / content / task-stratified 70-10-20 / seeds 7,42,123.
Same training recipe as graph_supervised.run_fold (Adam 1e-3, wd 1e-4, dropout
0.3, class weights, early stopping every 10 epochs on val AUROC, patience 5).
Reports AUROC + AUPRC + recall + FPR, comparable to the local SAGE baseline.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import common as C
import graph_supervised as GS

SEEDS = [7, 42, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden=128):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True,
                          dropout=0.3, bidirectional=False)
        self.lin1 = nn.Linear(hidden * 2, hidden)
        self.lin2 = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, padded, lengths):
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        from torch.nn.utils.rnn import pad_packed_sequence
        seq, lens = pad_packed_sequence(out, batch_first=True)
        # masked mean + max pool over time
        mask = (torch.arange(seq.shape[1], device=seq.device)[None, :]
                < lens.to(seq.device)[:, None]).float().unsqueeze(-1)
        mean = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)
        neg = seq.masked_fill(mask == 0, float("-inf"))
        mx = neg.max(1).values
        mx[mx == float("-inf")] = 0
        h = torch.cat([mean, mx], dim=1)
        h = F.elu(self.lin1(h)); h = self.drop(h)
        return self.lin2(h)


def make_batches(seqs, labels, idxs, bs, shuffle, rng=None):
    idxs = list(idxs)
    if shuffle:
        rng.shuffle(idxs)
    for k in range(0, len(idxs), bs):
        chunk = idxs[k:k + bs]
        tens = [torch.tensor(seqs[i], dtype=torch.float) for i in chunk]
        lens = torch.tensor([t.shape[0] for t in tens])
        padded = pad_sequence(tens, batch_first=True).to(DEVICE)
        y = torch.tensor([labels[i] for i in chunk], dtype=torch.long).to(DEVICE)
        yield padded, lens, y


def run_fold(seqs, labels, tr, va, te, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = GRUClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nb = sum(1 for i in tr if labels[i] == 0); na = sum(1 for i in tr if labels[i] == 1)
    w = torch.tensor([na/(nb+na), nb/(nb+na)], dtype=torch.float).to(DEVICE) if nb and na else None

    best, best_state, pc = 0, None, 0
    for ep in range(1, 201):
        model.train()
        for padded, lens, y in make_batches(seqs, labels, tr, 64, True, rng):
            opt.zero_grad()
            out = model(padded, lens)
            loss = F.cross_entropy(out, y, weight=w)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 10 == 0:
            vp, vy = predict(model, seqs, labels, va)
            if len(np.unique(vy)) > 1:
                from sklearn.metrics import roc_auc_score
                a = roc_auc_score(vy, vp)
                if a > best: best, pc, best_state = a, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else: pc += 1
            if pc >= 5: break
    if best_state: model.load_state_dict(best_state)
    p, yv = predict(model, seqs, labels, te)
    return C.metrics_from_probs(p, yv)


@torch.no_grad()
def predict(model, seqs, labels, idxs):
    model.eval()
    probs, ys = [], []
    for padded, lens, y in make_batches(seqs, labels, idxs, 64, False):
        out = model(padded, lens)
        probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); ys.extend(y.cpu().numpy())
    return np.array(probs), np.array(ys)


def main():
    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")
    labels = np.array([s["label"] for s in sessions])
    task_ids = np.array([s["task_id"] for s in sessions])

    fold_metrics = []
    for seed in SEEDS:
        tr, va, te = GS.task_stratified_split(len(sessions), task_ids, seed)
        m = run_fold(feats, labels, tr, va, te, seed)
        fold_metrics.append(m)
        print(f"  seed {seed}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
              f"recall={m['recall']:.3f} fpr={m['fpr']:.3f}")

    keys = ["auroc", "auprc", "auprc_benign", "f1", "recall", "fpr"]
    agg = {k: [float(np.mean([f[k] for f in fold_metrics])),
               float(np.std([f[k] for f in fold_metrics]))] for k in keys}
    print("\n=== E-C GRU sequence baseline (RAS-Eval, task-strat) ===")
    for k in keys:
        print(f"  {k:14s} {agg[k][0]:.4f} ± {agg[k][1]:.3f}")

    out = os.path.join(C.RESULTS_DIR, "e_c_sequence.json")
    with open(out, "w") as f:
        json.dump({"gru": agg, "folds": fold_metrics}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
