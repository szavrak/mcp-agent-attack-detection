# Running the ablation experiments on Colab (A100/T4)

These reproduce the content/architecture ablations (model-style confound, edge
ablation, sequence baseline, AUPRC) in a single GPU environment. Everything
reuses `scripts/graph_supervised.py` verbatim.

## Cell 1 — mount + setup
```python
from google.colab import drive; drive.mount('/content/drive')
%cd "/content/drive/MyDrive/mcp-agent-attack-detection"  # adjust to where the repo lives on your Drive
!pip -q install torch_geometric sentence-transformers xgboost >/dev/null
```

## Cell 2 — run everything
```python
import os
os.environ.update(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python", USE_TF="0",
                  USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
%cd experiments_R1
!python run_all.py
```

To run one group only: `!python e_a_confound.py` (or `e_b_edges` / `e_c_sequence` / `e_d_auprc`).

## Outputs
JSON under `experiments_R1/results/`:
- `e_a_confound.json` — same-model AUROC, per-model FPR, glm-vs-rest classifier.
- `e_b_edges.json`    — edge-type ablation + token-length sweep.
- `e_c_sequence.json` — GRU sequence baseline (AUROC/AUPRC).
- `e_d_auprc.json`    — architecture + classical tables with AUPRC (attack + benign-class).

## Notes
- Local (CPU) reproduction validates the findings; a GPU run gives the final,
  environment-consistent numbers.
- `e_d_faithful.py` reproduces the architecture-table AUROC under the exact
  global-seed protocol of `graph_supervised.main()` and adds AUPRC from the same
  predictions, so AUROC and AUPRC are computed in one consistent pass.
