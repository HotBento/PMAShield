"""
Cross-model circuit-concentration statistics (rebuttal, reviewer Nyqg W3).

Nyqg asked whether the "specialized head circuit" claim generalises beyond
Qwen3-8B, and separately pointed out (W2) that the circuit is quantitatively
distributed (Gini = 0.36 for Qwen3-8B; top-6 heads ~5% of total effect;
hundreds of heads needed to reach half). This module computes the same
summary statistics — aggregate attention share, Gini coefficient, top-k
effect share, heads-needed-for-half-the-effect — from any model's
``head_importance.npz`` / ``layer_attn_mlp.npz`` (Stage-1 patching outputs),
so they can be compared side by side across models (e.g. Qwen3-8B vs.
Qwen3-4B) instead of only visually via the heatmap figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative importance array (any shape, flattened).

    0 = perfectly uniform (every head equally important), 1 = maximally
    concentrated (one head has all the effect). Matches the definition used
    for the paper's reported Gini = 0.36 on Qwen3-8B head importances.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    x = np.clip(x, 0.0, None)
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # Standard discrete Gini via the Lorenz curve trapezoid formula.
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def topk_share(values: np.ndarray, k: int) -> float:
    """Fraction of the total effect carried by the top-k entries."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if total == 0 or x.size == 0:
        return float("nan")
    k = min(k, x.size)
    top = np.sort(x)[-k:]
    return float(top.sum() / total)


def heads_for_share(values: np.ndarray, target_share: float = 0.5) -> int:
    """Minimum number of top-ranked entries needed to reach ``target_share``
    of the total effect (e.g. "heads needed to reach half the total")."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if total == 0 or x.size == 0:
        return -1
    sorted_desc = np.sort(x)[::-1]
    cum = np.cumsum(sorted_desc) / total
    idx = int(np.searchsorted(cum, target_share) + 1)
    return min(idx, x.size)


@dataclass
class ConcentrationSummary:
    model_id: str
    attn_share: float          # attention / (attention + MLP), from layer_attn_mlp.npz
    gini: float                # Gini coefficient of head_importance.npz
    top6_share: float          # fraction of total effect in the top-6 heads
    heads_for_half: int        # heads needed to reach 50% of total effect
    n_heads_total: int


def summarize(
    model_id: str,
    *,
    layer_attn: np.ndarray | None = None,
    layer_mlp: np.ndarray | None = None,
    head_importance: np.ndarray | None = None,
    top_k: int = 6,
) -> ConcentrationSummary:
    """Build a :class:`ConcentrationSummary` from Stage-1 patching arrays.

    Any of ``layer_attn``/``layer_mlp``/``head_importance`` may be omitted
    (e.g. a model where only the head-level patching was run); the
    corresponding summary fields are NaN / -1 in that case.
    """
    attn_share = float("nan")
    if layer_attn is not None and layer_mlp is not None:
        a = np.asarray(layer_attn, dtype=np.float64).sum()
        m = np.asarray(layer_mlp, dtype=np.float64).sum()
        if (a + m) > 0:
            attn_share = float(a / (a + m))

    gini = float("nan")
    top6 = float("nan")
    half = -1
    n_heads = 0
    if head_importance is not None:
        flat = np.asarray(head_importance, dtype=np.float64).ravel()
        n_heads = int(flat.size)
        gini = gini_coefficient(flat)
        top6 = topk_share(flat, top_k)
        half = heads_for_share(flat, 0.5)

    return ConcentrationSummary(
        model_id=model_id,
        attn_share=attn_share,
        gini=gini,
        top6_share=top6,
        heads_for_half=half,
        n_heads_total=n_heads,
    )
