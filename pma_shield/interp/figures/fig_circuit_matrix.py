"""Appendix figure: directed head-to-head path-patching circuit matrix.

Cell ``(row=A, col=B)`` is the mean absolute change in head ``B``'s commit-row
attention pattern when upstream head ``A``'s ``o_proj`` input is replaced from
the contrastive scenario (see
:func:`mcp_eval.interp.patching.head_pair_circuit_matrix`). The metric is
non-negative and asymmetric; the diagonal is NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pma_shield.interp.config import FIG_SINGLE_W
from pma_shield.interp.figures.style import HEATMAP_CMAP, save_fig, setup_style


def plot_circuit_matrix(
    matrix: np.ndarray,
    head_labels: Sequence[str],
    *,
    out_dir: Path,
    name: str = "fig_circuit_matrix",
    title: str | None = None,
) -> Path:
    setup_style()
    mat = np.asarray(matrix, dtype=float)
    assert mat.shape == (len(head_labels), len(head_labels)), (
        "matrix shape must match head_labels"
    )
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, FIG_SINGLE_W))
    # Non-negative magnitude → white->red colormap. NaN (diagonal) shown grey.
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="0.85")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.nanmax(mat)),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(head_labels)))
    ax.set_yticks(np.arange(len(head_labels)))
    ax.set_xticklabels(head_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(head_labels, fontsize=5)
    ax.set_xlabel("Downstream head (B, observed)")
    ax.set_ylabel("Upstream head (A, patched)")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label(
        r"$\overline{|\Delta\alpha_B|}$"
    )
    return save_fig(fig, out_dir, name)
