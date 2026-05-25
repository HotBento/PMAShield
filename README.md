# PMAShield

Code for the paper **"When Heads Disagree: Detecting Preference Manipulation Attacks in LLM Tool Calling via Attention Head Disagreement"**.

Anonymous submission — code available at: https://anonymous.4open.science/r/PMAShield-CEEE/

---

## Overview

PMAShield is a white-box detector for **Preference Manipulation Attacks (PMAs)** on MCP-based LLM tool-calling agents. It operates in two phases:

1. **Offline (once per model):** Discover *tool-selection heads* — attention heads that consistently attend to the chosen tool — using a calibration set of benign tool-selection examples.
2. **Online (at inference time):** Compute four inter-head disagreement metrics (Agreement *A*, Entropy *E*, JS Divergence *D_JS*, Outlier Score *O*) over the selection head set and classify with a lightweight logistic regression model.

See the paper for details on §3 (mechanistic analysis) and §4–§5 (detection method and experiments).

---

## Repository Structure

```
PMAShield/
├── pma_shield/
│   ├── core/           — abstract base classes
│   ├── providers/      — HuggingFace local model provider (white-box)
│   ├── prompt/         — chat-template + tool-list formatting
│   ├── model_registry.py
│   ├── interp/         — §3 mechanistic interpretability
│   │   ├── patching.py         activation patching (layer- and head-level)
│   │   ├── head_roles.py       functional role taxonomy
│   │   ├── probe_data.py       YAML probe-scenario loader
│   │   ├── prompting.py        forced tool-call prefix construction
│   │   ├── figures/            publication-quality figure generators
│   │   └── scripts/            CLI entry points (remote GPU + local figure assembly)
│   ├── detector/       — §4–§5 PMAShield detection
│   │   ├── capture.py          Stage 1: extract per-head features
│   │   ├── selection.py        Stage 3: offline head selection (rule-based + τ sweep)
│   │   ├── disagreement.py     Stage 4: compute A / E / D_JS / O
│   │   ├── detection.py        Stage 5: LOSO logistic classifier + ablation
│   │   ├── data_mcptox.py      MCPTox dataset loader
│   │   ├── data_mpma.py        MPMA-DPMA dataset loader
│   │   └── scripts/            CLI entry points
│   └── transferability/ — RQ2 cross-attack transfer matrix
├── data/
│   ├── probe_scenarios/ — YAML probe scenarios for §3 interpretability
│   └── README.md        — dataset download instructions
└── scripts/             — bash pipeline scripts
```

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# CPU-only (figure generation, local verification)
uv sync

# GPU (model inference, patching, capture — remote server)
uv sync --group gpu
```

---

## Data Preparation

See [data/README.md](data/README.md) for download instructions.

Place datasets as follows:

```
data/
├── mcptox/
│   └── response_all.json      # MCPTox benchmark (https://github.com/MCPTox/MCPTox)
└── mpma/
    └── ...                    # MPMA benchmark (https://github.com/MPMA2025/MPMA)
```

---

## Reproducing §3: Mechanistic Interpretability

All patching experiments run on a GPU server. Figure assembly runs locally.

### On the GPU server

```bash
# Run all three patching experiments for Qwen3-8B
bash scripts/phase_interp.sh --model Qwen/Qwen3-8B --gpu 0

# Or run individual stages:
python -m pma_shield.interp.scripts.run_patching \
    --model Qwen/Qwen3-8B --limit 40 \
    --out results/interp/Qwen_Qwen3-8B/patching/

python -m pma_shield.interp.scripts.run_head_roles \
    --model Qwen/Qwen3-8B \
    --importance results/interp/Qwen_Qwen3-8B/patching/head_importance.npz \
    --out results/interp/Qwen_Qwen3-8B/

python -m pma_shield.interp.scripts.run_circuit_matrix \
    --model Qwen/Qwen3-8B --top-k 16 \
    --importance results/interp/Qwen_Qwen3-8B/patching/head_importance.npz \
    --out results/interp/Qwen_Qwen3-8B/patching/
```

### Locally (no GPU)

```bash
# Generate publication figures from saved results
python -m pma_shield.interp.scripts.make_figures
# → figures/interp/*.pdf

# Preview figures with mock data (no GPU, < 10 s)
python -m pma_shield.interp.scripts.make_mock_figures
# → figures/interp_mock/*.pdf
```

---

## Reproducing §5: Detection Experiments

### Full pipeline for one model (GPU server)

```bash
MODEL=Qwen/Qwen3-8B
SAFE=Qwen_Qwen3-8B

# Stage 0: generate pairs
python -m pma_shield.detector.scripts.run_data \
    --attack mcptox --out data/mcptox/

# Stage 1: capture attention features
python -m pma_shield.detector.scripts.run_capture \
    --model $MODEL --pairs data/mcptox/pairs.jsonl \
    --out results/mcptox/$SAFE/attn_cache/

# Stage 3: discover selection heads (with τ grid search)
python -m pma_shield.detector.scripts.run_selection \
    --capture results/mcptox/$SAFE/attn_cache/ \
    --out results/mcptox/$SAFE/ \
    --grid-search

# Stage 4: compute disagreement metrics
python -m pma_shield.detector.scripts.run_disagreement \
    --capture results/mcptox/$SAFE/attn_cache/ \
    --selection results/mcptox/$SAFE/selection_heads.json \
    --out results/mcptox/$SAFE/

# Stage 5: LOSO detection + ablation (RQ1, RQ3)
python -m pma_shield.detector.scripts.run_detection \
    --disagreement results/mcptox/$SAFE/disagreement.csv \
    --out results/mcptox/$SAFE/
```

Repeat with `--attack mpma` for MPMA-DPMA results.

### All six models

```bash
bash scripts/run_all_models.sh
```

### RQ2: Cross-attack transfer

```bash
python -m pma_shield.detector.scripts.run_transfer \
    --model Qwen/Qwen3-8B \
    --mcptox-dir   results/mcptox/Qwen_Qwen3-8B/ \
    --mpma-dir     results/mpma/Qwen_Qwen3-8B/ \
    --out          results/transfer/Qwen_Qwen3-8B/
```

---

## Expected Results

Main detection performance (AUROC, Qwen3-8B):

| Attack    | AUROC | F1    |
|-----------|-------|-------|
| MCPTox    | 0.741 | 0.733 |
| MPMA-DPMA | 0.689 | 0.632 |

See Tables 2–4 in the paper for full cross-model and ablation results.

---

## Models

Six instruction-tuned LLMs are evaluated:

| Model | HuggingFace ID |
|-------|----------------|
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Qwen3-4B | `Qwen/Qwen3-4B` |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` |
| Gemma-3-4B | `google/gemma-3-4b-it` |
| Gemma-4-E4B | `google/gemma-4-E4B-it` |
| Llama-3.1-8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

---

## License

Code released under the MIT License.
