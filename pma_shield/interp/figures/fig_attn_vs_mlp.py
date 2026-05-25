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
