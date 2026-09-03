#!/usr/bin/env python3
"""Shared harness for the content and architecture ablation experiments.

Design goal: reuse the EXACT methodology of scripts/graph_supervised.py
(feature extraction, GNN/MLP models, training loop, splits) so that the
ablations are directly comparable to the primary results. This module only
ADDS: (1) on-disk caching of the expensive SBERT node features, (2) edge-type
variants for the edge ablation, (3) AUPRC in evaluation, (4) a GRU sequence
baseline, (5) helpers for the model-style-confound probe.

Nothing in scripts/ is modified. New results go under experiments_R1/results/.
"""
import os
# Must be set before sentence_transformers / transformers import (broken base env).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import hashlib
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

# Reuse the main pipeline's methodology verbatim.
import graph_supervised as GS  # noqa: E402

RAW_DIR = os.path.join(_ROOT, "data", "raw", "ras_eval")
CACHE_DIR = os.path.join(_HERE, "cache")
# Results dir is overridable (via R1_RESULTS_DIR) so a runner can redirect output
# to the manuscript's results folder. Defaults to experiments_R1/results.
RESULTS_DIR = os.environ.get("R1_RESULTS_DIR", os.path.join(_HERE, "results"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data + cached content features
# ----------------------------------------------------------------------------

def load_ras(pooled):
    """Return (benign, attacked) RAS-Eval sessions with per-session 'model' tag."""
    return GS.load_sessions(RAW_DIR, pooled=pooled)


def _session_key(session):
    """Stable hash of a session's textual content (for cache invalidation)."""
    h = hashlib.md5()
    for c in session["calls"]:
        h.update(c["args_str"].encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update(c["response_str"].encode("utf-8", "ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def compute_content_cache(sessions, tag):
    """Compute per-node SBERT content features (768-d) for every session, cached.

    Returns a list of float32 arrays, one per session, shape [n_calls, 768].
    Uses batched encoding for speed, but the per-node vector is identical to
    graph_supervised.content_features (concat[args_emb(384), resp_emb(384)]).
    """
    cache_path = os.path.join(CACHE_DIR, f"content_{tag}.npz")
    keys_path = os.path.join(CACHE_DIR, f"content_{tag}.keys.json")
    cur_keys = [_session_key(s) for s in sessions]

    if os.path.exists(cache_path) and os.path.exists(keys_path):
        with open(keys_path) as f:
            old = json.load(f)
        if old.get("keys") == cur_keys:
            data = np.load(cache_path)
            feats = [data[f"s{i}"] for i in range(len(sessions))]
            print(f"[cache] loaded content features for '{tag}' ({len(feats)} sessions)")
            return feats

    print(f"[cache] computing SBERT content features for '{tag}' ({len(sessions)} sessions)...")
    embedder = GS.get_content_embedder()

    # Flatten all texts (args + response per call) for one batched encode pass.
    texts, spans = [], []
    for s in sessions:
        start = len(texts)
        for c in s["calls"]:
            texts.append(c["args_str"][:512])
            texts.append(c["response_str"][:512] if c["response_str"] else "empty")
        spans.append((start, len(texts)))

    embs = embedder.encode(texts, batch_size=256, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float32)

    feats = []
    for (a, b) in spans:
        pair = embs[a:b].reshape(-1, 2, 384)          # [n_calls, 2, 384]
        node = pair.reshape(pair.shape[0], 768)        # concat[args, resp]
        feats.append(node)

    np.savez_compressed(cache_path, **{f"s{i}": f for i, f in enumerate(feats)})
    with open(keys_path, "w") as f:
        json.dump({"keys": cur_keys}, f)
    print(f"[cache] saved -> {cache_path}")
    return feats


# ----------------------------------------------------------------------------
# Edge variants (for the edge-type / threshold ablation)
# ----------------------------------------------------------------------------

def build_edges_variant(calls, edge_mode="full", token_len=4, substr_window=50,
                        resp_cap=1000, n_first_tokens=5):
    """Edge index with configurable edge set.

    edge_mode: 'full'  (sequential + data-flow, the default)
               'seq'   (sequential only)
               'dataflow' (data-flow only)
               'none'  (no edges; self-loops keep graphs well-defined)
    Returns (edge_index[2,E], n_dataflow_edges) where n_dataflow_edges counts
    undirected data-flow links added (each stored bidirectionally).
    """
    n = len(calls)
    src, dst = [], []
    n_df = 0

    if edge_mode in ("full", "seq"):
        for i in range(n - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

    if edge_mode in ("full", "dataflow"):
        for i in range(n):
            resp = calls[i]["response_str"]
            if not resp or len(resp) > resp_cap:
                continue
            for j in range(i + 1, n):
                args = calls[j]["args_str"]
                if resp[:substr_window] in args or any(
                    v in args for v in resp.split()[:n_first_tokens] if len(v) > token_len
                ):
                    src.extend([i, j])
                    dst.extend([j, i])
                    n_df += 1

    if edge_mode == "none":
        # self-loops on every node
        src = list(range(n))
        dst = list(range(n))
    elif not src:
        src, dst = [0], [0]

    return torch.tensor([src, dst], dtype=torch.long), n_df


# ----------------------------------------------------------------------------
# Graph assembly from cached features + variant edges
# ----------------------------------------------------------------------------

def build_graphs(sessions, node_feats, edge_mode="full", **edge_kw):
    """Assemble PyG Data objects from cached node features and variant edges.

    Returns (graphs, total_dataflow_edges, n_sessions_with_dataflow_edge). The
    last count lets the caller report what fraction of sessions actually carry a
    data-flow edge (needed to interpret the 'dataflow'-only ablation).
    """
    from torch_geometric.data import Data
    graphs = []
    total_df = 0
    n_with_df = 0
    for idx, s in enumerate(sessions):
        x = torch.tensor(node_feats[idx], dtype=torch.float)
        edge_index, n_df = build_edges_variant(s["calls"], edge_mode=edge_mode, **edge_kw)
        total_df += n_df
        n_with_df += int(n_df > 0)
        y = torch.tensor([s["label"]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.task_id = s["task_id"]
        data.session_idx = idx
        graphs.append(data)
    return graphs, total_df, n_with_df


# ----------------------------------------------------------------------------
# Evaluation with AUPRC (attack = positive; also benign-class AP)
# ----------------------------------------------------------------------------

def metrics_from_probs(probs, labels):
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score, average_precision_score)
    out = {}
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, probs))
        out["auprc"] = float(average_precision_score(labels, probs))        # attack-positive
        out["auprc_benign"] = float(average_precision_score(1 - labels, 1 - probs))  # minority
        preds = (probs > 0.5).astype(int)
        out["f1"] = float(f1_score(labels, preds))
        out["precision"] = float(precision_score(labels, preds, zero_division=0))
        out["recall"] = float(recall_score(labels, preds))
        out["fpr"] = float(((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
        out["prevalence"] = float((labels == 1).mean())
    return out
