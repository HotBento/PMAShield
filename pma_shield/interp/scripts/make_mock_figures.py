"""Generate synthetic preview figures for paper §3 — **LOCAL, no GPU required**.

Run::

    python -m mcp_eval.interp.scripts.make_mock_figures

Output goes to ``figures/interp_mock/*.pdf``. File names match those produced
by :mod:`mcp_eval.interp.scripts.make_figures`, so the layout / typography
seen here is exactly what the final paper will show. Use this iteration loop
to lock in the figure style before running any GPU experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pma_shield.interp.config import ALL_MODELS, MOCK_FIGURES_DIR, MODEL_DISPLAY
from pma_shield.interp.figures.fig_attn_vs_mlp import plot_attn_vs_mlp
from pma_shield.interp.figures.fig_circuit_matrix import plot_circuit_matrix
from pma_shield.interp.figures.fig_disagreement import (
    plot_disagreement_multi_model,
    plot_disagreement_scatter,
)
from pma_shield.interp.figures.fig_head_heatmap import (
    plot_head_heatmap,
    plot_multi_model_heatmaps,
)
from pma_shield.interp.figures.fig_all_head_patterns import HeadPatternEntry, plot_all_head_patterns
from pma_shield.interp.figures.fig_head_patterns import PatternPanel, plot_head_patterns
from pma_shield.interp.figures.table_head_roles import render_table
from pma_shield.model_registry import get_arch


def _mock_layer_curves(n_layers: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    layers = np.arange(n_layers)
    centre = n_layers * 0.65
    attn = 5.5 * np.exp(-((layers - centre) ** 2) / (2 * (n_layers * 0.10) ** 2))
    attn += rng.normal(scale=0.12, size=n_layers).clip(0)
    attn += 0.10
    mlp = 0.65 * np.exp(-((layers - centre + 1) ** 2) / (2 * (n_layers * 0.18) ** 2))
    mlp += rng.normal(scale=0.05, size=n_layers).clip(0)
    return attn, mlp


def _mock_head_importance(n_layers: int, n_heads: int, *, rng: np.random.Generator) -> np.ndarray:
    mat = rng.exponential(scale=0.15, size=(n_layers, n_heads))
    hotspots = [
        (int(0.60 * n_layers), int(0.90 * n_heads)),
        (int(0.75 * n_layers), int(0.25 * n_heads)),
        (int(0.58 * n_layers), int(0.80 * n_heads)),
        (int(0.72 * n_layers), int(0.35 * n_heads)),
        (int(0.60 * n_layers), int(0.95 * n_heads)),
    ]
    for ly, hd in hotspots:
        mat[ly, hd] += rng.uniform(4.0, 8.5)
    return mat


def _mock_patterns(rng: np.random.Generator) -> list[PatternPanel]:
    # Realistic layout: tool name spans appear early (inside the system prompt /
    # tool-definition block); the user query occupies the tail of the sequence
    # and spans to just before the commit token (last position = seq-1).
    seq = 48
    tool_spans = [(6, 9), (18, 21)]  # tool names early in the prompt
    chosen_idx = 0                   # tool 0 is the chosen one
    query_start = 32                 # user turn begins near the end
    query_span = (query_start, seq - 2)   # extend to just before commit token

    # Intent head: commit row peaks on the user-query / user-turn region.
    intent = rng.uniform(0.0, 0.03, size=(seq, seq))
    intent[-1, query_start:seq - 1] = rng.uniform(0.04, 0.30, size=seq - 1 - query_start)
    intent[-1, 0] = rng.uniform(0.50, 0.70)   # BOS / system-prompt anchor
    intent = np.clip(intent, 0, 1)

    # Intent-matching head: commit row almost exclusively on the chosen tool span.
    matching = rng.uniform(0.0, 0.02, size=(seq, seq))
    cs, ce = tool_spans[chosen_idx]
    matching[-1, cs] = rng.uniform(0.90, 1.00)
    matching[-1, cs + 1:ce + 1] = rng.uniform(0.00, 0.05, size=ce - cs)
    matching = np.clip(matching, 0, 1)

    return [
        PatternPanel(
            "intent", "L30H7", intent,
            highlight_query_span=query_span,
            highlight_tool_spans=tool_spans,
            chosen_tool_idx=-1,
        ),
        PatternPanel(
            "intent_matching", "L29H11", matching,
            highlight_query_span=query_span,
            highlight_tool_spans=tool_spans,
            chosen_tool_idx=chosen_idx,
        ),
    ]


def _mock_all_head_entries(
    rng: np.random.Generator,
    *,
    top_heads: list[tuple[int, int]],
    roles: list[str],
) -> list[HeadPatternEntry]:
    """Synthetic (seq×seq) attention matrices for every head in *top_heads*."""
    seq = 48
    entries: list[HeadPatternEntry] = []
    for (layer, head), role in zip(top_heads, roles):
        mat = rng.exponential(scale=0.02, size=(seq, seq))
        if role == "intent_matching":
            # Commit row (bottom) peaks on a tool span.
            tool_start = rng.integers(20, 35)
            mat[-1, tool_start:tool_start + 5] = rng.uniform(0.5, 1.0, 5)
            mat[-2, tool_start:tool_start + 5] = rng.uniform(0.3, 0.7, 5)
        elif role == "intent":
            # Commit row peaks on query tokens (early positions).
            q_start = rng.integers(3, 10)
            mat[-1, q_start:q_start + 8] = rng.uniform(0.4, 0.9, 8)
        else:  # tool_summary
            for col in rng.integers(15, seq - 2, size=3):
                mat[:, col - 1:col + 2] += rng.uniform(0.2, 0.6, (seq, 3))
        entries.append(HeadPatternEntry(layer=layer, head=head, role=role, attn=mat))
    return entries


def _mock_disagreement_points(rng: np.random.Generator, *, n_benign: int = 80, n_attacked: int = 80) -> dict[str, np.ndarray]:
    benign = np.column_stack(
        [
            np.clip(rng.normal(loc=0.97, scale=0.04, size=n_benign), 0.5, 1.0),
            np.clip(rng.normal(loc=0.05, scale=0.04, size=n_benign), 0.0, 1.0),
        ]
    )
    attacked = np.column_stack(
        [
            np.clip(rng.normal(loc=0.55, scale=0.20, size=n_attacked), 0.0, 1.0),
            np.clip(rng.normal(loc=0.45, scale=0.18, size=n_attacked), 0.0, 1.5),
        ]
    )
    return {"benign": benign, "MCPTox": attacked}


def _mock_three_attacks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = _mock_disagreement_points(rng)
    mpma = np.column_stack(
        [
            np.clip(rng.normal(loc=0.62, scale=0.17, size=70), 0.0, 1.0),
            np.clip(rng.normal(loc=0.38, scale=0.16, size=70), 0.0, 1.5),
        ]
    )
    injec = np.column_stack(
        [
            np.clip(rng.normal(loc=0.45, scale=0.22, size=70), 0.0, 1.0),
            np.clip(rng.normal(loc=0.55, scale=0.20, size=70), 0.0, 1.5),
        ]
    )
    return {"benign": base["benign"], "MCPTox": base["MCPTox"], "MPMA": mpma, "InjecAgent": injec}


def _mock_circuit_matrix(n: int, *, rng: np.random.Generator) -> np.ndarray:
    # Directed path patching: non-negative, asymmetric, NaN diagonal.
    # Upstream (earlier) heads influence downstream (later) heads → upper triangle
    # carries the signal when head_labels are sorted by layer.
    mat = rng.exponential(scale=0.02, size=(n, n))
    for i in range(n):
        for j in range(n):
            if i < j:                       # A upstream of B
                mat[i, j] += rng.exponential(scale=0.06) * (1.0 - (j - i) / n)
    np.fill_diagonal(mat, np.nan)
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=MOCK_FIGURES_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_arch = get_arch("Qwen/Qwen3-8B")
    n_layers, n_heads = primary_arch.num_layers, primary_arch.num_heads

    # Finding 1
    attn, mlp = _mock_layer_curves(n_layers, rng=rng)
    plot_attn_vs_mlp(attn, mlp, out_dir=out_dir, title="Qwen3-8B")

    # Finding 2
    primary_heatmap = _mock_head_importance(n_layers, n_heads, rng=rng)
    plot_head_heatmap(primary_heatmap, out_dir=out_dir, title="Qwen3-8B")

    # Finding 3: three-panel representative examples (main text)
    plot_head_patterns(_mock_patterns(rng), out_dir=out_dir)
    render_table(
        {
            "intent": [(19, 21), (20, 5)],
            "tool_summary": [(0, 7), (35, 11)],
            "intent_matching": [(24, 29), (30, 7), (23, 26)],
        },
        out_path=out_dir / "tab_head_roles.tex",
    )

    # Finding 3: appendix grid — all top-24 head patterns
    mock_top24 = sorted(
        [(24, 29), (24, 30), (25, 14), (25, 30), (28, 12), (28, 23),
         (29, 0), (29, 11), (29, 15), (30, 5), (30, 7), (31, 13),
         (31, 28), (31, 30), (32, 2), (32, 5), (33, 11), (33, 17),
         (34, 16), (34, 17), (34, 28), (34, 29), (35, 26), (35, 27)]
    )
    mock_roles24 = (
        ["intent_matching"] * 15 + ["intent"] * 9
    )  # approximate distribution for mock
    plot_all_head_patterns(
        _mock_all_head_entries(rng, top_heads=mock_top24, roles=mock_roles24),
        out_dir=out_dir,
    )

    # Finding 4
    plot_disagreement_scatter(_mock_disagreement_points(rng), out_dir=out_dir, title="Qwen3-8B + MCPTox")

    # Appendix A1 — multi-model heatmaps
    multi_heatmaps: dict[str, np.ndarray] = {}
    for mid in ALL_MODELS:
        arch = get_arch(mid)
        multi_heatmaps[MODEL_DISPLAY[mid]] = _mock_head_importance(arch.num_layers, arch.num_heads, rng=rng)
    plot_multi_model_heatmaps(multi_heatmaps, out_dir=out_dir)

    # Appendix A2 — multi-model disagreement
    multi_points = {MODEL_DISPLAY[mid]: _mock_three_attacks(rng) for mid in ALL_MODELS}
    plot_disagreement_multi_model(multi_points, out_dir=out_dir)

    # Appendix A3 — circuit matrix (heads sorted low -> high layer)
    top_heads = sorted(
        [(24, 29), (30, 7), (23, 26), (29, 11), (24, 30), (24, 31), (32, 11),
         (35, 11), (24, 14), (19, 21), (0, 7), (3, 4), (12, 8), (20, 5), (33, 6), (34, 12)]
    )
    head_labels = [f"L{l}H{h}" for l, h in top_heads]
    plot_circuit_matrix(_mock_circuit_matrix(len(top_heads), rng=rng), head_labels, out_dir=out_dir)

    print(f"[mock] wrote PDFs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
