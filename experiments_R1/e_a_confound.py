#!/usr/bin/env python3
"""Model-style confound analysis -- the top threat to validity.

All RAS-Eval attacks target glm-4-flash while benign spans 8 LLMs. Does the
detector learn "attack vs benign" or "glm-4-flash-style vs other-model-style"?

A1  Same-model detection: restrict benign to glm-4-flash only, keep glm attacks,
    rerun SAGE/content/task-strat. If AUROC holds ~ mixed-benign, signal=attack.
A2  Per-model FPR probe: run the trained pooled detector on held-out benign,
    bucketed by source model. Uniform FPR => detector is not using model identity
    as a benign/attack cue.
A3  glm-vs-rest benign classifier: on benign-only pooled SBERT session features,
    predict glm-4-flash vs other models under a TASK-DISJOINT split (controls the
    "different models cover different tasks" confound). Near-chance AUROC => SBERT
    does not linearly separate glm style. A response-length-only control shows
    whether any separability is just verbosity.
"""
import argparse
import json
import os
import numpy as np
import torch
import common as C
import graph_supervised as GS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEEDS = [7, 42, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def session_pool_feats(node_feats):
    """mean||max pool per session -> 1536-d (same as classical baselines)."""
    out = []
    for f in node_feats:
        out.append(np.concatenate([f.mean(0), f.max(0)]))
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------- A1
def a1_same_model():
    benign, attacked = C.load_ras(pooled=False)   # glm-4-flash benign only
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="glmonly")
    task_ids = np.array([s["task_id"] for s in sessions])
    graphs, _, _ = C.build_graphs(sessions, feats, edge_mode="full")
    aurocs, recalls, fprs = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
        train_g = [graphs[i] for i in tr]; test_g = [graphs[i] for i in te]
        args = argparse.Namespace(seed=seed, batch_size=64, epochs=200, lr=1e-3, hidden_dim=128)
        r = GS.run_fold(train_g, test_g, args, DEVICE, 0, arch="sage")
        aurocs.append(r["auroc"]); recalls.append(r.get("recall", 0)); fprs.append(r.get("fpr", 0))
    return {
        "n_benign_glm": len(benign), "n_attacked": len(attacked),
        "auroc": [float(np.mean(aurocs)), float(np.std(aurocs))],
        "recall": [float(np.mean(recalls)), float(np.std(recalls))],
        "fpr": [float(np.mean(fprs)), float(np.std(fprs))],
    }


# ---------------------------------------------------------------- A2
def a2_per_model_fpr(sessions, feats, n_benign):
    """Train pooled detector; report FPR on held-out benign bucketed by model."""
    task_ids = np.array([s["task_id"] for s in sessions])
    graphs, _, _ = C.build_graphs(sessions, feats, edge_mode="full")
    models = np.array([s.get("model", "unknown") for s in sessions])
    from torch_geometric.loader import DataLoader

    per_model_fp = {}   # model -> [n_false_pos, n_total]
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr, va, te = GS.task_stratified_split(len(graphs), task_ids, seed)
        train_g = [graphs[i] for i in tr]
        # train exactly like run_fold but keep the model to score arbitrary benign
        model = _train_sage(train_g, seed)
        # held-out benign = benign sessions whose index is in test split
        test_set = set(te.tolist())
        idxs = [i for i in range(len(sessions)) if i < n_benign and i in test_set]
        if not idxs:
            continue
        probs = _predict(model, [graphs[i] for i in idxs])
        preds = (probs > 0.5).astype(int)
        for k, i in enumerate(idxs):
            m = models[i]
            d = per_model_fp.setdefault(m, [0, 0])
            d[0] += int(preds[k] == 1); d[1] += 1
    return {m: {"fpr": v[0] / max(v[1], 1), "n": v[1]} for m, v in sorted(per_model_fp.items())}


def _train_sage(train_g, seed):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_g)); n_val = max(1, len(idx) // 5)
    trn = [train_g[i] for i in idx[n_val:]]; val = [train_g[i] for i in idx[:n_val]]
    in_dim = train_g[0].x.shape[1]
    model = GS.GNNClassifier(in_dim, 128, arch="sage").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nb = sum(1 for g in trn if g.y.item() == 0); na = sum(1 for g in trn if g.y.item() == 1)
    w = torch.tensor([na/(nb+na), nb/(nb+na)], dtype=torch.float).to(DEVICE) if nb and na else None
    tl = DataLoader(trn, batch_size=64, shuffle=True); vl = DataLoader(val, batch_size=64)
    best, best_state, pc = 0, None, 0
    for ep in range(1, 201):
        GS.train_epoch(model, tl, opt, DEVICE, w)
        if ep % 10 == 0:
            vp, vy = GS.evaluate(model, vl, DEVICE)
            if len(np.unique(vy)) > 1:
                a = roc_auc_score(vy, vp)
                if a > best: best, pc, best_state = a, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else: pc += 1
            if pc >= 5: break
    if best_state: model.load_state_dict(best_state)
    return model


def _predict(model, graphs):
    from torch_geometric.loader import DataLoader
    p, _ = GS.evaluate(model, DataLoader(graphs, batch_size=64), DEVICE)
    return p


# ---------------------------------------------------------------- A3
def a3_glm_vs_rest(sessions, feats, n_benign):
    b_idx = list(range(n_benign))
    X = session_pool_feats([feats[i] for i in b_idx])
    models = np.array([sessions[i].get("model", "unknown") for i in b_idx])
    y = (models == "glm-4-flash").astype(int)
    tasks = np.array([sessions[i]["task_id"] for i in b_idx])
    # length-only control feature: mean & max response length per session
    lens = np.array([[np.mean([c["response_length"] for c in sessions[i]["calls"]]),
                      np.max([c["response_length"] for c in sessions[i]["calls"]])]
                     for i in b_idx], dtype=np.float32)

    def task_disjoint(seed):
        uniq = sorted(set(tasks.tolist())); rng = np.random.RandomState(seed); rng.shuffle(uniq)
        cut = int(0.7 * len(uniq)); tr_t = set(uniq[:cut])
        tr = np.array([i for i in range(len(b_idx)) if tasks[i] in tr_t])
        te = np.array([i for i in range(len(b_idx)) if tasks[i] not in tr_t])
        return tr, te

    res = {"n_glm": int(y.sum()), "n_other": int((1 - y).sum())}
    for name, feat in [("sbert", X), ("length_only", lens)]:
        lr_a, rf_a = [], []
        for seed in SEEDS:
            tr, te = task_disjoint(seed)
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            # LR gets standardized features (matches classical_baselines.py); RF raw.
            scaler = StandardScaler()
            ftr_s = scaler.fit_transform(feat[tr]); fte_s = scaler.transform(feat[te])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(ftr_s, y[tr])
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=seed).fit(feat[tr], y[tr])
            lr_a.append(roc_auc_score(y[te], lr.predict_proba(fte_s)[:, 1]))
            rf_a.append(roc_auc_score(y[te], rf.predict_proba(feat[te])[:, 1]))
        res[name] = {
            "logreg_auroc": [float(np.mean(lr_a)), float(np.std(lr_a))] if lr_a else None,
            "rf_auroc": [float(np.mean(rf_a)), float(np.std(rf_a))] if rf_a else None,
        }
    return res


def main():
    results = {}
    out = os.path.join(C.RESULTS_DIR, "e_a_confound.json")

    def save():  # incremental persist so a later sub-step failure can't discard earlier results
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== E-A1: same-model (glm-only benign) detection ===")
    results["a1_same_model"] = a1_same_model()
    r = results["a1_same_model"]
    print(f"  benign_glm={r['n_benign_glm']}  attacked={r['n_attacked']}  "
          f"AUROC={r['auroc'][0]:.4f}±{r['auroc'][1]:.3f}  recall={r['recall'][0]:.3f}  fpr={r['fpr'][0]:.3f}")
    save()

    # pooled sessions for A2/A3
    benign, attacked = C.load_ras(pooled=True)
    sessions = benign + attacked
    feats = C.compute_content_cache(sessions, tag="pooled")

    print("\n=== E-A2: per-model FPR probe (trained pooled detector on held-out benign) ===")
    results["a2_per_model_fpr"] = a2_per_model_fpr(sessions, feats, len(benign))
    for m, d in results["a2_per_model_fpr"].items():
        print(f"  {m:28s}  FPR={d['fpr']:.3f}  (n={d['n']})")
    save()

    print("\n=== E-A3: glm-vs-rest benign style classifier (task-disjoint) ===")
    results["a3_glm_vs_rest"] = a3_glm_vs_rest(sessions, feats, len(benign))
    r = results["a3_glm_vs_rest"]
    print(f"  n_glm={r['n_glm']} n_other={r['n_other']}")
    for name in ["sbert", "length_only"]:
        if name in r:
            print(f"  {name:12s}  logreg={r[name]['logreg_auroc']}  rf={r[name]['rf_auroc']}")
    save()
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
