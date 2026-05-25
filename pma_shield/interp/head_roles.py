"""Classify selection heads into intent / tool-summary / intent-matching roles.

Given a per-head causal importance matrix and access to a sample attention
forward (via :func:`mcp_eval.interp.patching.capture_attention_pattern`), each
candidate head is assigned exactly one of three roles based on where its
attention mass lands:

* ``intent``           — concentrated on the user query span (last query token row).
* ``tool_summary``     — concentrated on tool-boundary / trailing tokens of *each*
                         tool, no preference for the chosen tool.
* ``intent_matching``  — concentrated on the chosen tool's span at the commit row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class SpanInfo:
    """Token spans used to score each head."""

    user_query: tuple[int, int]              # (start, end) inclusive
    tool_spans: tuple[tuple[int, int], ...]  # one (s, e) per candidate tool
    chosen_tool_index: int                   # index into tool_spans
    commit_pos: int


def score_head(attn: np.ndarray, span: SpanInfo) -> dict[str, float]:
    """Compute three role-score components from a (seq, seq) attention matrix.

    The scores all live in ``[0, 1]``.

    * ``intent``: mass at row ``commit_pos`` falling inside the user-query span,
      *normalised* by the mass falling inside any (tool ∪ user) span.
    * ``tool_summary``: mass at row ``commit_pos`` distributed roughly
      uniformly across all tool spans (1 − chosen tool share among tool mass).
    * ``intent_matching``: mass at row ``commit_pos`` concentrated on the
      chosen tool's span among tool mass.
    """
    row = attn[span.commit_pos]
    uq_mass = float(row[span.user_query[0]:span.user_query[1] + 1].sum())
    tool_masses = np.array([
        float(row[s:e + 1].sum()) for s, e in span.tool_spans
    ])
    tool_total = float(tool_masses.sum())
    total = uq_mass + tool_total + 1e-12

    intent = uq_mass / total
    chosen_share = float(tool_masses[span.chosen_tool_index] / (tool_total + 1e-12))
    intent_matching = (tool_total / total) * chosen_share
    tool_summary = (tool_total / total) * (1.0 - chosen_share)
    return {
        "intent": intent,
        "tool_summary": tool_summary,
        "intent_matching": intent_matching,
    }


def classify_heads(
    head_scores: Mapping[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], str]:
    """Assign each head the argmax role."""
    out: dict[tuple[int, int], str] = {}
    for head, scores in head_scores.items():
        out[head] = max(scores, key=scores.get)
    return out


def top_heads(
    importance: np.ndarray, k: int = 16,
) -> list[tuple[int, int]]:
    flat = importance.flatten()
    idx = np.argsort(flat)[::-1][:k]
    return [tuple(map(int, np.unravel_index(i, importance.shape))) for i in idx]


__all__ = ["SpanInfo", "classify_heads", "score_head", "top_heads"]
