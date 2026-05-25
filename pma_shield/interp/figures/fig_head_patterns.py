"""Figure ``fig:head-patterns``: combined commit-step attention for two head roles.

Produces a **single** PDF with both panels stacked vertically and a shared
colorbar on the right.  Span boundaries are marked with coloured dashed
vertical lines; labels appear below the x-axis as bracket + text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from pma_shield.interp.config import FIG_DOUBLE_W
from pma_shield.interp.figures.style import HEATMAP_CMAP, ROLE_COLORS, save_fig, setup_style

_SPAN_QUERY  = ROLE_COLORS["intent"]           # blue
_SPAN_CHOSEN = ROLE_COLORS["intent_matching"]  # red
_SPAN_OTHER  = "#999999"                       # gray for non-chosen tool spans

_ROLE_TITLE: dict[str, str] = {
    "intent": "Intent head",
    "intent_matching": "Intent-matching head",
}


@dataclass
class PatternPanel:
    """Data for one (role, head) panel."""

    role: str
    head_label: str
    attn: np.ndarray                          # full (seq_len, seq_len); only [-1, :] is plotted
    token_labels: Sequence[str] | None = None
    highlight_query_span: tuple[int, int] | None = None
    highlight_tool_spans: Sequence[tuple[int, int]] = ()
    chosen_tool_idx: int = -1                 # index into highlight_tool_spans; -1 = none


def _span_lines(ax: plt.Axes, start: int, end: int, color: str) -> None:
    """Draw dashed vertical lines at the left and right boundary of a span."""
    ax.axvline(start - 0.5, color=color, ls="--", lw=0.8, alpha=0.9)
    ax.axvline(end   + 0.5, color=color, ls="--", lw=0.8, alpha=0.9)


def plot_head_patterns(
    panels: Sequence[PatternPanel],
    *,
    out_dir: Path,
    name: str = "fig_head_patterns",
    gamma: float = 0.4,
) -> list[Path]:
    """Save a single combined PDF with panels stacked and a shared colorbar.

    Returns a one-element list containing the written path.
    """
    setup_style()
    assert len(panels) >= 1

    n = len(panels)
    vmax = max(float(np.asarray(p.attn)[-1, :].max()) for p in panels)
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(FIG_DOUBLE_W, 0.72 * n + 0.45),
        gridspec_kw={"hspace": 0.6},
    )
    if n == 1:
        axes = [axes]

    images = []
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        commit  = np.asarray(panel.attn)[-1, :]
        seq_len = commit.shape[0]

        im = ax.imshow(
            commit.reshape(1, -1),
            aspect="auto",
            cmap=HEATMAP_CMAP,
            norm=norm,
            interpolation="nearest",
            extent=[-0.5, seq_len - 0.5, 0, 1],
        )
        images.append(im)

        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6, length=2, pad=1.5)
        # Hide x-tick numbers on all panels except the last.
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        # Title above the strip.
        role_str = _ROLE_TITLE.get(panel.role, panel.role)
        ax.set_title(f"{role_str}  ({panel.head_label})",
                     fontsize=7, pad=3, loc="left")

        # Dashed vertical lines at span boundaries only (no text labels).
        if panel.highlight_query_span is not None:
            _span_lines(ax, *panel.highlight_query_span, _SPAN_QUERY)

        for j, (ts, te) in enumerate(panel.highlight_tool_spans):
            color = _SPAN_CHOSEN if j == panel.chosen_tool_idx else _SPAN_OTHER
            _span_lines(ax, ts, te, color)

    # Shared colorbar spanning all panels on the right.
    cb = fig.colorbar(images[0], ax=axes, orientation="vertical",
                      shrink=0.82, aspect=18, pad=0.01)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Attention", fontsize=6, labelpad=2)

    path = save_fig(fig, out_dir, name)
    return [path]
