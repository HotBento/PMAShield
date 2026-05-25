"""
Soft / hard tool votes per attention head.

Used by both Stage 3 (selection-cluster qualification) and Stage 4
(disagreement metrics).

Definitions
-----------
For a head ``(L, H)`` with attention row ``α`` over input positions::

    p_{L,H}(t) = sum_{i ∈ name_span(t)} α_i  /  sum_{t'} sum_{i ∈ name_span(t')} α_i

is the head's *soft vote* over tools (a categorical distribution). When the
denominator is zero (no attention on any tool-name span), the head produces
all-NaN soft votes for that sample.

Hard vote: ``argmax_t p_{L,H}(t)`` — represented as the tool's index into the
per-sample ``tool_order``. ``-1`` denotes "no valid vote" (all-NaN soft).

This module operates on the **compact features** produced by Stage 1
(``per_tool_name`` slot) — no need to access raw attention rows.
"""

from __future__ import annotations

import numpy as np

from .features import MAX_TOOLS, feature_layout

# ──────────────────────────────────────────────────────────────────────────
# Vote computation
# ──────────────────────────────────────────────────────────────────────────

def soft_votes_from_features(features: np.ndarray) -> np.ndarray:
    """Compute per-head soft votes from the Stage-1 feature tensor.

    Parameters
    ----------
    features
        Any array with last axis ``FEAT_DIM`` (e.g. shape
        ``(N_pairs, 2, L, H, FEAT_DIM)`` or ``(L, H, FEAT_DIM)``).

    Returns
    -------
    np.ndarray
        Same leading shape, with last axis ``MAX_TOOLS``. Unused / no-mass
        slots are NaN (so consumers must use NaN-aware reductions).
    """
    layout = feature_layout()
    name_slots = features[..., layout["per_tool_name"]]  # (..., MAX_TOOLS)
    name_slots = np.asarray(name_slots, dtype=np.float64)
    total = np.nansum(name_slots, axis=-1, keepdims=True)
    # Where total > 0 we can normalise; everywhere else, NaN.
    safe_total = np.where(total > 0, total, np.nan)
    soft = name_slots / safe_total
    return soft.astype(np.float32, copy=False)


def hard_votes_from_soft(soft: np.ndarray) -> np.ndarray:
    """Argmax over the last axis. ``-1`` where the soft row is all-NaN.

    Returns shape ``soft.shape[:-1]`` of dtype int32.
    """
    arr = np.asarray(soft)
    masked = np.where(np.isnan(arr), -np.inf, arr)
    hard = masked.argmax(axis=-1).astype(np.int32, copy=False)
    valid = ~np.isnan(arr).all(axis=-1)
    hard = np.where(valid, hard, np.int32(-1))
    return hard


def hard_votes_from_features(features: np.ndarray) -> np.ndarray:
    """Convenience: soft → hard in one call."""
    return hard_votes_from_soft(soft_votes_from_features(features))


# ──────────────────────────────────────────────────────────────────────────
# Aggregation across a head set
# ──────────────────────────────────────────────────────────────────────────

def head_set_indices(
    head_set: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    num_layers: int,
    num_heads: int,
) -> np.ndarray:
    """Convert ``[(L1, H1), ...]`` to a 1-D index array into a flat ``L*H`` axis."""
    out = np.empty(len(head_set), dtype=np.int64)
    for i, (L, H) in enumerate(head_set):
        if not (0 <= L < num_layers and 0 <= H < num_heads):
            raise IndexError(f"head ({L}, {H}) outside grid ({num_layers}, {num_heads})")
        out[i] = L * num_heads + H
    return out


def majority_hard_vote(
    hard_votes: np.ndarray,
    *,
    head_axis: int,
) -> np.ndarray:
    """Modal hard-vote across ``head_axis``.

    Ignores positions where the hard vote is ``-1``. When all heads abstain
    the result is ``-1``. Ties are broken by smallest tool index.
    """
    # Move the head axis to the back for easier vectorised iteration.
    moved = np.moveaxis(hard_votes, head_axis, -1)
    out = np.full(moved.shape[:-1], -1, dtype=np.int32)
    flat = moved.reshape(-1, moved.shape[-1])
    for i, row in enumerate(flat):
        valid = row[row >= 0]
        if valid.size == 0:
            continue
        vals, counts = np.unique(valid, return_counts=True)
        out.flat[i] = int(vals[counts.argmax()])
    return out


def vote_distribution_across_heads(
    hard_votes: np.ndarray,
    *,
    n_tools: int,
    head_axis: int,
) -> np.ndarray:
    """Per-sample categorical distribution of hard votes across the head set.

    Returns shape ``hard_votes.shape`` minus the head axis, with an extra
    final axis of size ``n_tools`` containing fractions that sum to 1
    (or to 0 when every head abstained).
    """
    moved = np.moveaxis(hard_votes, head_axis, -1)
    out_shape = moved.shape[:-1] + (n_tools,)
    out = np.zeros(out_shape, dtype=np.float32)
    flat = moved.reshape(-1, moved.shape[-1])
    flat_out = out.reshape(-1, n_tools)
    for i, row in enumerate(flat):
        valid = row[row >= 0]
        if valid.size == 0:
            continue
        vals, counts = np.unique(valid, return_counts=True)
        flat_out[i, vals] = counts / float(valid.size)
    return out
