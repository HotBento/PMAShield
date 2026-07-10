"""Figure ``fig:attn-vs-mlp``.

Bar plot of per-layer causal contribution to tool selection, decomposed into
attention and MLP components. Data is supplied as two equal-length 1-D arrays
``attn_delta`` and ``mlp_delta`` (``|Δlog p|`` per layer).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pma_shield.interp.config import FIG_SINGLE_W
from pma_shield.interp.figures.style import PALETTE, save_fig, setup_style


def plot_attn_vs_mlp(
    attn_delta: np.ndarray,
    mlp_delta: np.ndarray,
    *,
    out_dir: Path,
    name: str = "fig_attn_vs_mlp",
    title: str | None = None,
) -> Path:
    setup_style()
    attn = np.asarray(attn_delta, dtype=float)
    mlp = np.asarray(mlp_delta, dtype=float)
    assert attn.shape == mlp.shape and attn.ndim == 1, "expected matching 1-D arrays"
    layers = np.arange(attn.shape[0])

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, 1.9))
    width = 0.42
    ax.bar(layers - width / 2, attn, width=width, color=PALETTE["attn"], label="Attention")
    ax.bar(layers + width / 2, mlp, width=width, color=PALETTE["mlp"], label="MLP")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$|\Delta \log p|$")
    ax.set_xlim(-0.6, layers[-1] + 0.6)
    ax.set_xticks(layers[:: max(1, len(layers) // 8)])
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    return save_fig(fig, out_dir, name)


def plot_attn_vs_mlp_multi_model(
    attn_share_by_model: dict[str, float],
    *,
    out_dir: Path,
    name: str = "fig_attn_vs_mlp_multi_model",
    title: str | None = None,
) -> Path:
    """Bar chart of aggregate attention share (attn / (attn+MLP)) per model.

    Cross-model companion to :func:`plot_attn_vs_mlp` (Finding 1, rebuttal
    Nyqg W3): layer counts differ across models (e.g. 40 vs. 36), so a
    per-layer overlay is not directly comparable — the single aggregate
    ratio is. ``attn_share_by_model`` maps display name -> attn / (attn+mlp)
    in [0, 1], e.g. from :func:`pma_shield.interp.concentration_stats.summarize`.
    """
    setup_style()
    models = list(attn_share_by_model.keys())
    shares = np.array([attn_share_by_model[m] for m in models], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, 1.9))
    x = np.arange(len(models))
    ax.bar(x, shares, color=PALETTE["attn"])
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.set_ylabel("Attention share")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    for xi, s in zip(x, shares):
        if not np.isnan(s):
            ax.text(xi, s + 0.02, f"{s:.2f}", ha="center", va="bottom", fontsize=7)
    if title:
        ax.set_title(title)
    return save_fig(fig, out_dir, name)
