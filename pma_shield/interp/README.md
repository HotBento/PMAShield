# `mcp_eval.interp` — Mechanistic Interpretability for Tool Selection (Paper §3)

Reproducible pipeline for the figures and table in
*§3 Mechanistic Interpretability of LLM Tool Selection* and the matching
appendix sections in [acl_latex.tex](../../latex/6a06c6f79e0a2a06a392d848/latex/acl_latex.tex).

Replaces the early-exploration notebooks under [`notebooks/`](../../notebooks/)
(`tool_selection_patching.ipynb`, `head_heatmap_patching.ipynb`,
`attn_head_viewer.ipynb`, `mcptox_attention_consistency.ipynb`, …) with a
structured package that:

* Runs on **all six paper models** (Qwen3-4B/8B, Phi-4-mini, Gemma-3-4B,
  Gemma-4-E4B, Llama-3.1-8B) via one CLI per stage.
* Produces **publication-quality PDF figures** through a single style module.
* Ships a **mock-data preview script** so the figure layout can be iterated
  *without any GPU* — useful for tuning sizes / fonts / colours before
  committing GPU hours.

---

## Pipeline

```
data/probe_scenarios/*.yaml
        │
        ▼
mcp_eval.interp.scripts.run_patching        (REMOTE / GPU)
        │
        ├── results/interp/<MODEL>/patching/layer_attn_mlp.npz      (Finding 1)
        └── results/interp/<MODEL>/patching/head_importance.npz     (Finding 2)
              │
              ▼
mcp_eval.interp.scripts.run_head_roles       (REMOTE / GPU)
        │
        └── results/interp/<MODEL>/head_roles.json                 (Finding 3)
              │
              ▼
mcp_eval.interp.scripts.run_circuit_matrix   (REMOTE / GPU; primary model only)
        │
        └── results/interp/<MODEL>/patching/circuit_matrix.npz     (Appendix A3)

results/{mcptox,mpma,injecagent}/<MODEL>/disagreement/{benign,points}.npz
        │  (existing mcp_eval.mcptox pipeline)
        ▼
mcp_eval.interp.scripts.make_figures         (LOCAL)
        │
        └── figures/interp/*.pdf
```

## Module map

| File | Role |
|---|---|
| [`config.py`](config.py) | Output paths, 6-model list, figure size constants |
| [`probe_data.py`](probe_data.py) | YAML loader for `data/probe_scenarios/` |
| [`prompting.py`](prompting.py) | Forced tool-call output prefix (notebook parity) |
| [`patching.py`](patching.py) | Layer / head / head-pair activation patching (GPU) |
| [`head_roles.py`](head_roles.py) | Attention-mass scoring + role classification |
| [`figures/style.py`](figures/style.py) | ACL-compliant matplotlib style + `save_fig` |
| [`figures/fig_*.py`](figures/) | One file per paper figure (pure plotting) |
| [`figures/table_head_roles.py`](figures/table_head_roles.py) | LaTeX booktabs for `tab:head-roles` |
| [`scripts/make_mock_figures.py`](scripts/make_mock_figures.py) | LOCAL preview pipeline |
| [`scripts/make_figures.py`](scripts/make_figures.py) | LOCAL final-figure assembly |
| [`scripts/run_patching.py`](scripts/run_patching.py) | REMOTE — layer + head patching |
| [`scripts/run_head_roles.py`](scripts/run_head_roles.py) | REMOTE — head role classification |
| [`scripts/run_circuit_matrix.py`](scripts/run_circuit_matrix.py) | REMOTE — top-K head pair matrix |

---

## Local workflow (no GPU)

```bash
# 1. Preview all figures with mock data (used to iterate on style):
python -m mcp_eval.interp.scripts.make_mock_figures
#    → figures/interp_mock/*.pdf

# 2. Unit tests:
pytest tests/interp/          # or: python -m pytest -q tests/interp/

# 3. After rsync-ing remote results, build the real figures:
python -m mcp_eval.interp.scripts.make_figures
#    → figures/interp/*.pdf  (paths used by the LaTeX source)
```

## Remote workflow (GPU server)

```bash
# Single model end-to-end:
bash scripts/multi_model/phase_interp.sh --gpu 0 --model Qwen/Qwen3-8B

# Smoke test (4 contrastive pairs):
bash scripts/multi_model/phase_interp.sh --gpu 0 --model Qwen/Qwen3-8B --limit 4

# All six paper models, sequentially:
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B microsoft/Phi-4-mini-instruct \
         google/gemma-3-4b-it google/gemma-4-E4B-it \
         meta-llama/Meta-Llama-3.1-8B-Instruct; do
    bash scripts/multi_model/phase_interp.sh --gpu 0 --model "$m"
done
```

Memory-optimised invocation (mirrors `phase1_mcptox.sh`):

```bash
bash scripts/multi_model/phase_interp.sh \
    --gpu 0 --model Qwen/Qwen3-8B \
    --torch-dtype float16 --load-in-8bit
```

The dispatcher writes everything under `results/interp/<MODEL_SAFE>/`. The
primary model used for §3 main-text figures is **`Qwen/Qwen3-8B`** (see
`config.PRIMARY_MODEL`); the other five contribute to the appendix grids.

## Figure → LaTeX map

| LaTeX label | PDF file | Source CLI |
|---|---|---|
| `fig:attn-vs-mlp` | `figures/interp/fig_attn_vs_mlp.pdf` | `run_patching` (primary) |
| `fig:head-heatmap` | `figures/interp/fig_head_heatmap.pdf` | `run_patching` (primary) |
| `tab:head-roles` | `figures/interp/tab_head_roles.tex` | `run_head_roles` |
| `fig:head-patterns` | `figures/interp/fig_head_patterns.pdf` | `run_head_roles --save-patterns` |
| `fig:disagreement-scatter` | `figures/interp/fig_disagreement_scatter.pdf` | `disagreement_from_csv` (from `mcp_eval.mcptox`) |
| `fig:circuit-matrix` (appendix) | `figures/interp/fig_circuit_matrix.pdf` | `run_circuit_matrix` |

The §3 interpretability analysis is **single-model (Qwen3-8B)** — the paper no
longer carries multi-model interpretability figures. The detector
(\S\ref{sec:experiments}) is still evaluated on all six models via the
`mcp_eval.mcptox` pipeline.

### Regenerating the head-patterns figure (`fig:head-patterns`)

This figure needs per-head attention matrices, which `run_head_roles` only saves
when asked:

```bash
# REMOTE (GPU): also dump patterns/<role>_lLhH.npz
python -m mcp_eval.interp.scripts.run_head_roles \
    --model Qwen/Qwen3-8B --top-k 24 --save-patterns
# LOCAL: assemble the figure
python -m mcp_eval.interp.scripts.make_figures
```

`--format-head L,H` (default `1,0`) picks the early-layer head shown in the
format/structure panel, since those heads do not appear in the patching top-k.

### Disagreement scatter (`fig:disagreement-scatter`)

Convert the Stage-4 `disagreement.csv` (from the `mcp_eval.mcptox` detection
pipeline) into the point layout `make_figures` expects:

```bash
python -m mcp_eval.interp.scripts.disagreement_from_csv \
    --csv results/mcptox/mcptox/disagreement.csv --attack mcptox --model Qwen/Qwen3-8B
```

## Critical: forced output prefix (notebook parity)

Tool selection is only legible at the step where the model emits the **tool
name**. Following the reference notebooks, [`prompting.py`](prompting.py) appends
a *forced output prefix* to the chat-template text so the model is teacher-forced
into committing to a tool call. For Qwen3:

```
<think>\n</think>\n\n<tool_call>\n{"name": "
```

The commit position is then the last input token, and the patching metric is the
**contrastive log-odds** between the two competing tools' first name-tokens
(computed *in JSON context*) — exactly as in `tool_selection_patching.ipynb`.
Head-level patching replaces the per-head slice of the **`o_proj` input** (not the
post-projection output), matching `head_heatmap_patching.ipynb`.

Skipping the forced prefix (an earlier bug) measures logits at the generic
generation-prompt position and produces results that disagree with the paper's
conclusions. The per-family prefixes are unit-tested in
[`tests/interp/test_prompting.py`](../../tests/interp/test_prompting.py).

## Adding a probe scenario

1. Edit the appropriate file in [`data/probe_scenarios/`](../../data/probe_scenarios/)
   (see the [README](../../data/probe_scenarios/README.md) for schema).
2. Re-run `run_patching.py` for affected models.
3. Re-run `make_figures.py`.

Stable scenario IDs are important: `make_pairs` uses a deterministic shuffle
that depends on the input scenario order.
