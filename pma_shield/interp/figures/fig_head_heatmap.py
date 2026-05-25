"""Figure ``fig:head-heatmap`` (single model) + appendix multi-model grid.

Both functions accept a ``(n_layers, n_heads)`` matrix of causal importance
scores (``|Δlog p|``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from pma_shield.interp.config import FIG_DOUBLE_W, FIG_SINGLE_W
from pma_shield.interp.figures.style import HEATMAP_CMAP, model_grid_dims, save_fig, setup_style


def _draw_heatmap(ax, mat: np.ndarray, *, vmax: float | None = None) -> None:
    n_layers, n_heads = mat.shape
    vmax = vmax if vmax is not None else float(np.nanmax(mat))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(0, n_heads, max(1, n_heads // 8)))
    ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 8)))
    return im


def plot_head_heatmap(
    importance: np.ndarray,
    *,
    out_dir: Path,
    name: str = "fig_head_heatmap",
    title: str | None = None,
    annotate_top_k: int = 6,
) -> Path:
    setup_style()
    mat = np.asarray(importance, dtype=float)
    assert mat.ndim == 2, "expected (n_layers, n_heads) matrix"

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, 2.4))
    im = _draw_heatmap(ax, mat)
    if title:
        ax.set_title(title)

    if annotate_top_k > 0:
        flat = mat.flatten()
        idx = np.argsort(flat)[::-1][:annotate_top_k]
        for k in idx:
            ly, hd = np.unravel_index(k, mat.shape)
            ax.add_patch(
                plt.Rectangle(
                    (hd - 0.5, ly - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#1f4e79",
                    linewidth=0.7,
                )
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$|\Delta\log p|$", rotation=90)
    return save_fig(fig, out_dir, name)


def plot_multi_model_heatmaps(
    matrices: Mapping[str, np.ndarray],
    *,
    out_dir: Path,
    name: str = "fig_head_heatmap_multi",
) -> Path:
    """Appendix figure: layer×head heatmap for every model in *matrices*.

    Keys of *matrices* are display names; values are 2-D arrays.
    """
    setup_style()
    models = list(matrices)
    nrow, ncol = model_grid_dims(len(models))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(FIG_DOUBLE_W, 2.6 * nrow),
        squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.45},
    )
    vmax = max(float(np.nanmax(m)) for m in matrices.values())
    last_im = None
    for ax_pos, model_name in zip(axes.flat, models):
        last_im = _draw_heatmap(ax_pos, np.asarray(matrices[model_name]), vmax=vmax)
        ax_pos.set_title(model_name)
    for ax_pos in axes.flat[len(models):]:
        ax_pos.set_visible(False)
    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.18, 0.012, 0.66])
        fig.colorbar(last_im, cax=cbar_ax).set_label(r"$|\Delta\log p|$")
    return save_fig(fig, out_dir, name)
