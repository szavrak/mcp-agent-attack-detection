# Content-Aware Attack Detection for LLM Agent Tool-Call Traffic

An empirical study of content-aware attack detection for LLM agent tool-call traffic over the Model Context Protocol (MCP).

## Overview

Each agent session is encoded as a graph where nodes represent tool calls and edges capture sequential adjacency and data-flow links (response-to-argument substring matches). Node features combine sentence embeddings (SBERT, all-MiniLM-L6-v2) with structural metadata. Detectors (GNN, MLP, classical ML) classify sessions as benign or attacked.

**Key finding**: metadata-only detection plateaus at AUROC ~0.64; content embeddings push it above 0.89. Tree ensembles on pooled SBERT features (XGBoost: AUROC 0.975) outperform the GNN architectures, indicating that MCP attacks primarily alter semantic content while leaving structural metadata unchanged, and that the dominant signal is in the embeddings rather than the graph structure.

## Features

- Session-graph encoding with two edge types (sequential + data-flow)
- Three GNN architectures (GAT, GCN, GraphSAGE) + MLP baseline
- Supervised and self-supervised (contrastive pre-training) paradigms
- Task-stratified evaluation (GroupKFold) to prevent task leakage
- No model internals required -- operates on externally observable signals only

## Setup

```bash
pip install -r requirements.txt
```

Download datasets (requires `huggingface_hub`):

```bash
python scripts/download_datasets.py
```

## Usage

> **Which script produces the paper's tables?** `graph_evaluate.py` is the
> **authoritative** table-generating script for the GNN results (it computes the
> full metric set including AUPRC and is the source of the per-dataset and
> architecture tables). `graph_supervised.py` is the **shared library** every
> other script imports (data loading, feature extraction, model classes, and
> splits); its standalone `main()` is a lightweight training runner whose numbers
> can differ from the paper by ~0.02–0.03 AUROC due to a rougher training loop,
> so **do not use it to reproduce the paper's numbers** — use `graph_evaluate.py`.
> Classical baselines come from `classical_baselines.py`, and label efficiency
> from `graph_label_efficiency.py`. The exact library versions used for the
> reported numbers are pinned in `requirements.txt`.

**Reproducing the paper's tables** (RAS-Eval requires `--pooled`; seeds are `7,42,123`):
```bash
# Architecture comparison + per-dataset results (AUROC, AUPRC, F1, recall, FPR)
python scripts/graph_evaluate.py --mode content --dataset ras_eval --arch all  --pooled --seeds 7,42,123
python scripts/graph_evaluate.py --mode content --dataset combined --arch sage --pooled --seeds 7,42,123

# Classical baselines (XGBoost, RF, LogReg, LinearSVM)
python scripts/classical_baselines.py --dataset raseval   # also: atbench, combined

# Label efficiency (SSL+FT vs supervised)
python scripts/graph_label_efficiency.py --pooled --seed 42
```

**Other entry points:**
```bash
# Standalone supervised runner (component/library; NOT the table source, see note above)
python scripts/graph_supervised.py --mode content --arch sage --pooled --seeds 7,42,123

# Self-supervised pre-training + fine-tune
python scripts/graph_ssl.py --mode content --pooled
```

## Datasets

| Dataset | Source | Sessions | Tools |
|---------|--------|----------|-------|
| **RAS-Eval** | [GitHub](https://github.com/) (sparse-cloned automatically) | ~28k benign + 3.8k attacked | 75 |
| **ATBench** | [HuggingFace: AI45Research/ATBench](https://huggingface.co/datasets/AI45Research/ATBench) | ~1k (503 safe / 497 unsafe) | 2,084 |

### Automatic download

```bash
# Download all datasets into data/raw/
python scripts/download_datasets.py --dataset all

# Or individually
python scripts/download_datasets.py --dataset ras_eval
python scripts/download_datasets.py --dataset atbench
```

This places files under `data/raw/`:

```
data/raw/
├── ras_eval/
│   ├── logs/        # benign session logs
│   └── attacked/    # attacked session logs
└── atbench/
    └── atbench.json # labeled trajectories
```

### Manual download

If the script fails, you can download manually:

1. **RAS-Eval**: Clone the [RAS-Eval repo](https://github.com/) and copy the `data/logs/` and `data/attacked/` folders into `data/raw/ras_eval/`.
2. **ATBench**: Download `test.json` from [HuggingFace](https://huggingface.co/datasets/AI45Research/ATBench) and save it as `data/raw/atbench/atbench.json`.

> **Note**: ATBench requires `huggingface_hub` (`pip install huggingface_hub`). RAS-Eval uses a git sparse checkout (no extra dependencies).

## Results (RAS-Eval, task-stratified, 3 seeds)

| Model | Feature Mode | AUROC |
|-------|-------------|-------|
| GraphSAGE | Content | 0.917 +/- 0.018 |
| GAT | Content | 0.891 +/- 0.028 |
| GCN | Content | 0.893 +/- 0.022 |
| MLP (no graph) | Content | 0.896 +/- 0.010 |
| GAT | Metadata-only | 0.640 +/- 0.106 |

## Citation

Journal article: https://arxiv.org/abs/2605.11053

## License

TBD
