"""Single source of truth for figure style across §3.

The style targets the ACL 2024 LaTeX template: single-column width ≈ 3.3in,
double-column width ≈ 6.7in. PDFs are produced with embedded fonts so they
pass camera-ready font checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_APPLIED = False

PALETTE: dict[str, str] = {
    "attn": "#1f77b4",
    "mlp": "#d62728",
    "benign": "#2ca02c",
    "attacked": "#d62728",
    "mcptox": "#d62728",
    "mpma": "#9467bd",
    "injecagent": "#ff7f0e",
    "neutral": "#7f7f7f",
}

ROLE_COLORS: dict[str, str] = {
    "intent": "#1f77b4",
    "tool_summary": "#2ca02c",
    "intent_matching": "#d62728",
}

# Shared sequential heatmap colormap: low = white, high = red.
HEATMAP_CMAP: str = "Reds"

# Accent used for span overlays on top of the white->red heatmaps.
HEATMAP_ACCENT: str = "#1f4e79"


def setup_style() -> None:
    """Idempotently install matplotlib rcParams for ACL-quality figures."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,          # TrueType, embeddable
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",     # Linux freetype fallback that maps to Times
                "Liberation Serif",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",  # closest TeX-Times-like maths set
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.6,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.format": "pdf",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )
    _STYLE_APPLIED = True


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save *fig* as ``<out_dir>/<name>.pdf`` and return the resulting path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def model_grid_dims(n_models: int) -> tuple[int, int]:
    """Compact grid for appendix pile-ups of up to 6 models."""
    if n_models <= 3:
        return 1, n_models
    if n_models <= 4:
        return 2, 2
    return 2, 3


def legend_below(ax: plt.Axes, labels: Sequence[str], handles: Sequence) -> None:
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(4, len(labels)),
    )
