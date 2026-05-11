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

- **RAS-Eval** -- MCP-native format, ~28k benign + 3.8k attacked sessions, 75 tools, 80 tasks
- **ATBench** -- 1k labeled trajectories, 2084 tools

Both are downloaded automatically via `scripts/download_datasets.py`.

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
