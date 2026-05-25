"""Figure ``fig:disagreement-scatter`` + appendix multi-model grid.

Each sample contributes ``(r_bar, H)``: the fraction of selection heads voting
for the correct tool and the mean per-head entropy. Benign and attacked
points are plotted in different colours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pma_shield.interp.config import FIG_DOUBLE_W, FIG_SINGLE_W
from pma_shield.interp.figures.style import PALETTE, model_grid_dims, save_fig, setup_style


def _scatter(ax, points: Mapping[str, np.ndarray], *, with_legend: bool = True) -> None:
    for label, arr in points.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        color = PALETTE.get(label.lower(), PALETTE["neutral"])
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            s=8,
            alpha=0.55,
            color=color,
            label=label,
            edgecolor="none",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"head agreement $\bar{r}$")
    ax.set_ylabel(r"mean entropy $H$")
    if with_legend:
        ax.legend(loc="upper right")


def plot_disagreement_scatter(
    points_by_group: Mapping[str, np.ndarray],
    *,
    out_dir: Path,
    name: str = "fig_disagreement_scatter",
    title: str | None = None,
) -> Path:
    """Single-panel scatter (used in §3 main text).

    ``points_by_group`` example::

        {"benign": np.array([[r1, h1], ...]), "mcptox": np.array([[r1, h1], ...])}
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, 2.4))
    _scatter(ax, points_by_group)
    if title:
        ax.set_title(title)
    return save_fig(fig, out_dir, name)


def plot_disagreement_multi_model(
    points_by_model: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    name: str = "fig_disagreement_multi",
) -> Path:
    """Appendix grid: one panel per model, each panel = benign + 3 attack groups."""
    setup_style()
    models = list(points_by_model)
    nrow, ncol = model_grid_dims(len(models))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(FIG_DOUBLE_W, 2.5 * nrow),
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )
    for ax_pos, model_name in zip(axes.flat, models):
        _scatter(ax_pos, points_by_model[model_name], with_legend=False)
        ax_pos.set_title(model_name)
    for ax_pos in axes.flat[len(models):]:
        ax_pos.set_visible(False)

    # one shared legend at the bottom
    handles, labels = [], []
    seen: set[str] = set()
    for grp in points_by_model.values():
        for label in grp:
            if label in seen:
                continue
            seen.add(label)
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", linestyle="",
                    color=PALETTE.get(label.lower(), PALETTE["neutral"]),
                )
            )
            labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.13)
    return save_fig(fig, out_dir, name)
