# MCPShield

Content-aware graph-based attack detection for LLM agent tool-call traffic over the Model Context Protocol (MCP).

## Overview

MCPShield encodes each agent session as a graph where nodes represent tool calls and edges capture sequential adjacency and data-flow links (response-to-argument substring matches). Node features combine sentence embeddings (SBERT, all-MiniLM-L6-v2) with structural metadata. A graph neural network classifies sessions as benign or attacked.

**Key finding**: metadata-only detection plateaus at AUROC ~0.64; content embeddings push it to ~0.92, demonstrating that MCP attacks primarily alter semantic content while leaving structural metadata unchanged.

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

**Supervised training**:
```bash
python scripts/graph_supervised.py --dataset raseval --mode content --arch sage --seeds 7,42,123
```

**SSL + fine-tune**:
```bash
python scripts/graph_ssl.py --dataset raseval --mode content
```

**Evaluation**:
```bash
python scripts/graph_evaluate.py --dataset combined --mode content --pooled --seeds 7,42,123
```

**Classical baselines** (RF, XGBoost, SVM, etc.):
```bash
python scripts/classical_baselines.py --dataset raseval
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

Paper under review. Citation information will be added upon publication.

## License

TBD
