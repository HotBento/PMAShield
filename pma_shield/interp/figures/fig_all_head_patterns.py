"""Appendix figure: attention-pattern grid for all top-K selection heads.

Each panel shows the full (seq_len × seq_len) attention matrix for one head,
ordered left-to-right, top-to-bottom by (layer, head).  The role label and
head coordinates are shown in each panel title (small font).  A shared
PowerNorm colour scale makes panels directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pma_shield.interp.config import FIG_DOUBLE_W
from pma_shield.interp.figures.style import HEATMAP_CMAP, ROLE_COLORS, save_fig, setup_style


# Border colour per role (applied as a spine/edge colour on each axes).
_ROLE_EDGE: dict[str, str] = {
    "intent": ROLE_COLORS.get("intent", "#1f77b4"),
    "tool_summary": ROLE_COLORS.get("tool_summary", "#2ca02c"),
    "intent_matching": ROLE_COLORS.get("intent_matching", "#d62728"),
}
_DEFAULT_EDGE = "#7f7f7f"


@dataclass
class HeadPatternEntry:
    """One head's data for the grid."""

    layer: int
    head: int
    role: str
    attn: np.ndarray


def plot_all_head_patterns(
    entries: Sequence[HeadPatternEntry],
    *,
    out_dir: Path,
    name: str = "fig_all_head_patterns",
    ncols: int = 6,
    gamma: float = 0.4,
    panel_size: float = 1.05,
) -> Path:
    """Save a grid of attention-pattern heatmaps for every head in *entries*.

    Entries are sorted by (layer, head) before layout.  Each panel border is
    coloured by role (intent=blue, tool_summary=green, intent_matching=red).
    A shared PowerNorm colour scale (``gamma < 1``) makes all panels comparable.
    Returns the path to the saved PDF.
    """
    from matplotlib.colors import PowerNorm

    setup_style()
    sorted_entries = sorted(entries, key=lambda e: (e.layer, e.head))
    n = len(sorted_entries)
    nrows = (n + ncols - 1) // ncols

    vmax = max(float(np.asarray(e.attn).max()) for e in sorted_entries)
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_DOUBLE_W, panel_size * nrows),
        squeeze=False,
        gridspec_kw={"hspace": 0.6, "wspace": 0.25},
    )

    for ax_idx, entry in enumerate(sorted_entries):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        mat = np.asarray(entry.attn)
        ax.imshow(
            mat,
            origin="upper",
            cmap=HEATMAP_CMAP,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_box_aspect(1)
        ax.set_title(
            f"L{entry.layer}H{entry.head}\n{entry.role.replace('_', '-')}",
            fontsize=5,
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        edge_col = _ROLE_EDGE.get(entry.role, _DEFAULT_EDGE)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_col)
            spine.set_linewidth(1.2)

    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.70])
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    return save_fig(fig, out_dir, name)
