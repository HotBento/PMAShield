"""
Stage 3 — identify tool-selection / user-intent / tool-output heads.

Two complementary approaches are provided:

Rule-based classifier (preferred, :func:`compute_head_type_map`)
----------------------------------------------------------------
Directly labels every head without clustering, using three interpretable rules:

* **Tool-selection head** (label=1): on the benign side, the head's combined
  attention on the selected tool's name+description span exceeds the combined
  attention on every other tool, for ≥ ``selection_threshold`` of valid pairs.
* **User-intent head** (label=2): mean ``attn_user_query`` over benign pairs
  exceeds ``user_intent_threshold``.
* **Tool-output head** (label=3): mean ``attn_self`` (attention on already-
  generated tokens) over benign pairs exceeds ``tool_output_threshold``.

Priority order when a head qualifies for multiple types: 1 > 2 > 3 > 0.

:func:`pick_selection_heads_from_type_map` extracts the tool-selection heads
into a :class:`SelectionHeads` object compatible with downstream stages.

Cluster-based picker (:func:`pick_selection_cluster`, kept for compatibility)
-----------------------------------------------------------------------------
A cluster is the tool-selection cluster iff EITHER (a) or (b) holds:

(a) Its centroid ranks in the **top-2** simultaneously on:
    - ``mean_tool_name``        (column 0 of :data:`SIGNATURE_COLUMNS`), and
    - ``mean_concentration``    (column 5).

(b) The cluster's **median** per-head argmax-match-rate against the model's
    selected tool exceeds ``argmax_match_threshold`` (default 0.5).

Sanity check
------------
Both approaches verify that the prior-known core heads
(:data:`config.KNOWN_CORE_HEADS`) appear in the identified set. Misses are
reported but do not abort by default — pass ``strict=True`` to fail.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from loguru import logger

from . import config
from .features import MAX_TOOLS, feature_layout

SIGNATURE_COLUMNS: tuple[str, ...] = (
    "mean_tool_name",
    "mean_tool_desc",
    "mean_user_query",
    "mean_format",
    "mean_entropy",
    "mean_concentration",
    "mean_selected_tool_attn",
    "mean_tool_param",
)
from .voting import hard_votes_from_features


# ──────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelectionHeads:
    """Result of Stage 3."""

    cluster_id: int
    heads: list[tuple[int, int]]
    argmax_match_rate: dict[tuple[int, int], float]
    passed_known_head_check: bool
    failed_known_heads: list[tuple[int, int]]
    rule_used: str
    cluster_summary: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Per-head match rate
# ──────────────────────────────────────────────────────────────────────────

def per_head_argmax_match_rate(
    captured: Any,
    *,
    side: str = "benign",
) -> np.ndarray:
    """For each head, fraction of samples whose argmax tool == model's
    selected tool. Shape ``(num_layers, num_heads)``.

    Samples with ``parse_ok=False`` or whose ``selected_tool`` is missing
    from ``tool_names`` are skipped. Denominators count only samples where
    the head produced a valid hard vote (i.e. some attention on a tool-name
    span).
    """
    if side not in {"benign", "malicious"}:
        raise ValueError(f"side must be 'benign' or 'malicious'; got {side!r}")
    feats = (
        np.asarray(captured.benign_features) if side == "benign"
        else np.asarray(captured.malicious_features)
    )
    n_pairs, num_layers, num_heads, _ = feats.shape
    hard = hard_votes_from_features(feats)  # (N, L, H)

    matches = np.zeros((num_layers, num_heads), dtype=np.int64)
    valids = np.zeros((num_layers, num_heads), dtype=np.int64)

    for entry in captured.meta:
        if entry.get("side") != side:
            continue
        pair_idx = int(entry["pair_idx"])
        if pair_idx >= n_pairs:
            continue
        if not entry.get("parse_ok"):
            continue
        sel = entry.get("selected_tool")
        tool_names = entry.get("tool_names") or []
        if sel is None or sel not in tool_names:
            continue
        target_idx = tool_names.index(sel)
        head_hard = hard[pair_idx]                  # (L, H)
        head_valid = head_hard >= 0
        match = (head_hard == target_idx) & head_valid
        matches += match.astype(np.int64)
        valids += head_valid.astype(np.int64)

    rate = np.divide(matches, np.maximum(valids, 1), dtype=np.float64)
    rate = np.where(valids > 0, rate, 0.0)
    return rate.astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────
# Rule-based head-type classifier
# ──────────────────────────────────────────────────────────────────────────

#: Integer label constants for :attr:`HeadTypeMap.labels`.
HEAD_TYPE_OTHER = 0
HEAD_TYPE_SELECTION = 1
HEAD_TYPE_USER_INTENT = 2
HEAD_TYPE_TOOL_OUTPUT = 3
HEAD_TYPE_NAMES: dict[int, str] = {
    HEAD_TYPE_OTHER: "other",
    HEAD_TYPE_SELECTION: "selection",
    HEAD_TYPE_USER_INTENT: "user_intent",
    HEAD_TYPE_TOOL_OUTPUT: "tool_output",
}


@dataclass(frozen=True)
class HeadTypeMap:
    """Rule-based head-type classification result.

    Attributes
    ----------
    selection_rate
        ``(num_layers, num_heads)`` — fraction of benign valid pairs where the
        head's top-attended tool (name+desc combined) equals the model's
        selected tool.
    user_intent_mean
        ``(num_layers, num_heads)`` — mean ``attn_user_query`` over benign pairs.
    tool_output_mean
        ``(num_layers, num_heads)`` — mean ``attn_self`` over benign pairs.
    labels
        ``(num_layers, num_heads)`` int8 — 0=other, 1=selection, 2=user_intent,
        3=tool_output.  When a head qualifies for multiple types the highest-
        priority type (selection > user_intent > tool_output > other) wins.
    thresholds
        The threshold values used during classification.
    """

    selection_rate: np.ndarray    # (L, H) float32
    user_intent_mean: np.ndarray  # (L, H) float32
    tool_output_mean: np.ndarray  # (L, H) float32
    labels: np.ndarray            # (L, H) int8
    thresholds: dict[str, float]

    def heads_of_type(self, type_label: int) -> list[tuple[int, int]]:
        """Return all (layer, head) pairs whose label equals ``type_label``."""
        ls, hs = np.where(self.labels == type_label)
        return list(zip(ls.tolist(), hs.tolist()))


def compute_head_type_map(
    captured: Any,
    *,
    selection_threshold: float = 0.5,
    user_intent_threshold: float = 0.25,
    tool_output_threshold: float = 0.25,
) -> HeadTypeMap:
    """Classify every head by type using benign-side features only.

    Parameters
    ----------
    captured
        Stage-1 :class:`mcp_eval.mcptox.capture.CapturedDataset`.
    selection_threshold
        Minimum fraction of valid pairs where the head's argmax
        (name+desc combined) must equal the selected tool.
    user_intent_threshold
        Minimum mean ``attn_user_query`` value for user-intent classification.
    tool_output_threshold
        Minimum mean ``attn_self`` value for tool-output classification.
    """
    layout = feature_layout()
    feats_b = np.asarray(captured.benign_features, dtype=np.float32)   # (N, L, H, D)
    N, num_layers, num_heads, _ = feats_b.shape

    # ── build selected_tool_index per pair (NaN when parse failed) ──────────
    sel_indices = np.full(N, np.nan, dtype=np.float32)
    tool_counts = np.zeros(N, dtype=np.int32)
    for entry in captured.meta:
        if entry.get("side") != "benign":
            continue
        if not entry.get("parse_ok"):
            continue
        sel = entry.get("selected_tool")
        tool_names = entry.get("tool_names") or []
        tool_count = len(tool_names)
        if sel is None or sel not in tool_names:
            continue
        p = int(entry["pair_idx"])
        if 0 <= p < N:
            tool_counts[p] = tool_count
            sel_indices[p] = float(tool_names.index(sel))

    # ── selection_rate ────────────────────────────────────────────────────
    # For each sample, head's combined name+desc+param on selected tool vs others
    name_slots  = feats_b[..., layout["per_tool_name"]]   # (N, L, H, MAX_TOOLS)
    desc_slots  = feats_b[..., layout["per_tool_desc"]]   # (N, L, H, MAX_TOOLS)
    param_slots = feats_b[..., layout["per_tool_param"]]  # (N, L, H, MAX_TOOLS)
    combined = name_slots + desc_slots + param_slots       # (N, L, H, MAX_TOOLS)

    correct = np.zeros((N, num_layers, num_heads), dtype=np.float32)
    valid = np.zeros((N,), dtype=bool)

    for i in range(N):
        idx = sel_indices[i]
        if not np.isfinite(idx) or idx < 0:
            continue
        tool_count = int(tool_counts[i])
        if tool_count <= 1:
            continue
        t = int(idx)
        valid[i] = True
        sample_combined = combined[i, :, :, :tool_count]
        attn_sel = sample_combined[:, :, t]         # (L, H)
        mask = np.arange(tool_count) != t
        other = sample_combined[:, :, mask]         # (L, H, tool_count-1)
        attn_max_other = np.nanmax(other, axis=-1)  # (L, H)
        correct[i] = (attn_sel > attn_max_other).astype(np.float32)

    n_valid = valid.sum()
    if n_valid == 0:
        selection_rate = np.zeros((num_layers, num_heads), dtype=np.float32)
    else:
        selection_rate = correct[valid].mean(axis=0).astype(np.float32)

    # ── user_intent_mean, tool_output_mean ───────────────────────────────
    user_intent_mean = np.nanmean(
        feats_b[..., layout["attn_user_query"]], axis=0
    ).astype(np.float32)
    tool_output_mean = np.nanmean(
        feats_b[..., layout["attn_self"]], axis=0
    ).astype(np.float32)

    # ── classify with priority: selection > user_intent > tool_output ────
    labels = np.zeros((num_layers, num_heads), dtype=np.int8)
    labels[tool_output_mean >= tool_output_threshold] = HEAD_TYPE_TOOL_OUTPUT
    labels[user_intent_mean >= user_intent_threshold] = HEAD_TYPE_USER_INTENT
    labels[selection_rate >= selection_threshold] = HEAD_TYPE_SELECTION

    n_sel = int((labels == HEAD_TYPE_SELECTION).sum())
    n_ui = int((labels == HEAD_TYPE_USER_INTENT).sum())
    n_to = int((labels == HEAD_TYPE_TOOL_OUTPUT).sum())
    logger.info(
        "HeadTypeMap: {}/{} heads — selection={}, user_intent={}, tool_output={}, other={}",
        num_layers * num_heads, num_layers * num_heads,
        n_sel, n_ui, n_to,
        num_layers * num_heads - n_sel - n_ui - n_to,
    )

    return HeadTypeMap(
        selection_rate=selection_rate,
        user_intent_mean=user_intent_mean,
        tool_output_mean=tool_output_mean,
        labels=labels,
        thresholds={
            "selection": float(selection_threshold),
            "user_intent": float(user_intent_threshold),
            "tool_output": float(tool_output_threshold),
        },
    )


def pick_selection_heads_from_type_map(
    type_map: HeadTypeMap,
    *,
    sanity_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
    strict: bool = False,
) -> SelectionHeads:
    """Extract tool-selection heads from a :class:`HeadTypeMap`.

    Returns a :class:`SelectionHeads` object with ``cluster_id=-1`` (no
    clustering was used) that is otherwise compatible with all downstream
    stages (disagreement, detection, etc.).
    """
    heads = type_map.heads_of_type(HEAD_TYPE_SELECTION)
    if not heads:
        raise RuntimeError(
            "No heads qualify as tool-selection heads at the current threshold "
            f"({type_map.thresholds['selection']}).  Consider lowering "
            "--selection-threshold."
        )

    known = [tuple(h) for h in sanity_heads]
    head_set = set(map(tuple, heads))
    failed = [h for h in known if h not in head_set]
    passed = len(failed) == 0
    if not passed:
        logger.warning(
            "Sanity check: {} of {} known core heads NOT in selection set "
            "(missing: {})",
            len(failed), len(known), failed,
        )
        if strict:
            raise RuntimeError(
                f"Strict sanity check failed: known heads {failed!r} not among "
                "rule-based selection heads."
            )

    rate_dict = {
        tuple(h): float(type_map.selection_rate[h[0], h[1]])
        for h in heads
    }

    logger.info(
        "Rule-based selection: {} heads, sanity passed={}",
        len(heads), passed,
    )
    return SelectionHeads(
        cluster_id=-1,
        heads=[tuple(h) for h in heads],
        argmax_match_rate=rate_dict,
        passed_known_head_check=passed,
        failed_known_heads=failed,
        rule_used="rule_name_desc_combined",
        cluster_summary={
            "method": "rule_based",
            "thresholds": type_map.thresholds,
        },
    )


def pick_top_k_heads(
    type_map: HeadTypeMap,
    k: int,
    *,
    sanity_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
    strict: bool = False,
) -> SelectionHeads:
    """Pick the top-K heads by ``selection_rate``, with no threshold.

    Unlike :func:`pick_selection_heads_from_type_map`, this never raises: it
    always returns exactly ``min(k, total_heads)`` heads ranked by
    ``selection_rate`` descending.  The only free parameter is *how many*
    heads to include (K), not *how good* a head must be.

    Parameters
    ----------
    type_map
        Stage-3 rule-based classification result (needs ``selection_rate``).
    k
        Number of heads to return.  If k > total heads it is clamped.
    sanity_heads
        Expected-present heads for the sanity check.
    strict
        Raise if any sanity head is missing from the top-K set.
    """
    sel_rate = type_map.selection_rate          # (L, H) float32
    num_layers, num_heads_per_layer = sel_rate.shape
    total = num_layers * num_heads_per_layer
    k_actual = min(k, total)
    if k_actual < k:
        logger.warning(
            "Requested K={} but model only has {} heads; clamped to {}",
            k, total, k_actual,
        )

    flat = sel_rate.flatten()                   # (L*H,)
    top_flat = np.argsort(-flat)[:k_actual]     # descending by selection_rate
    layers   = (top_flat // num_heads_per_layer).tolist()
    heads_i  = (top_flat %  num_heads_per_layer).tolist()
    heads    = list(zip(layers, heads_i))

    known    = [tuple(h) for h in sanity_heads]
    head_set = set(map(tuple, heads))
    failed   = [h for h in known if h not in head_set]
    passed   = not failed

    if failed:
        logger.warning(
            "Top-{} sanity: {} of {} known core heads missing: {}",
            k_actual, len(failed), len(known), failed,
        )
        if strict:
            raise RuntimeError(
                f"Strict top-K sanity check failed: {failed!r} not in top-{k_actual}."
            )

    rate_dict = {(l, h): float(sel_rate[l, h]) for l, h in heads}
    min_rate  = float(flat[top_flat[-1]])
    max_rate  = float(flat[top_flat[0]])

    logger.info(
        "Top-K selection (K={}): {} heads, rate=[{:.3f}, {:.3f}], sanity={}",
        k_actual, len(heads), min_rate, max_rate,
        "passed" if passed else "FAILED",
    )
    return SelectionHeads(
        cluster_id=-2,
        heads=heads,
        argmax_match_rate=rate_dict,
        passed_known_head_check=passed,
        failed_known_heads=failed,
        rule_used=f"top_k_{k_actual}",
        cluster_summary={
            "method": "top_k",
            "k": k_actual,
            "k_requested": k,
            "min_selection_rate": min_rate,
            "max_selection_rate": max_rate,
        },
    )


def search_k_by_detection(
    dataset: Any,
    type_map: HeadTypeMap,
    k_candidates: Sequence[int],
    *,
    seed: int = 0,
    sanity_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
) -> tuple["SelectionHeads", dict[int, float]]:
    """Find the K that maximises LOSO detection AUC.

    For each K in *k_candidates* this function:

    1. Builds a top-K :class:`SelectionHeads` from ``type_map.selection_rate``.
    2. Runs a lightweight detection evaluation (main LOSO only, no ablation
       controls) via :func:`mcp_eval.mcptox.detection.evaluate`.
    3. Returns the :class:`SelectionHeads` for the winning K together with a
       ``{k: loso_auc}`` log for all candidates.

    Notes
    -----
    Head selection and detection evaluation share the same dataset; LOSO
    mitigates but does not fully eliminate selection bias.  K is a single
    integer hyperparameter, so the bias is limited in practice.

    Parameters
    ----------
    dataset
        Stage-1 :class:`mcp_eval.mcptox.capture.CapturedDataset`.
    type_map
        Stage-3 result (only ``selection_rate`` is used).
    k_candidates
        Iterable of K values to try.  Duplicates are removed; values are
        sorted ascending before the search begins.
    seed
        Random seed passed to the LOSO logistic regression.
    sanity_heads
        Passed through to :func:`pick_top_k_heads`.
    """
    from . import detection as _det  # local import to avoid circular deps

    candidates = sorted(set(int(k) for k in k_candidates))
    logger.info(
        "Detection-driven K search: {} candidates {}",
        len(candidates), candidates,
    )

    k_to_auc: dict[int, float] = {}
    best_k:   int | None       = None
    best_auc: float            = float("-inf")

    for k in candidates:
        sel = pick_top_k_heads(type_map, k, sanity_heads=sanity_heads)
        try:
            # controls=() → skip all ablations, only run main LOSO (fast)
            report = _det.evaluate(dataset, sel, controls=(), seed=seed)
            auc = report.main_logistic.get("loso_auc_mean", float("nan"))
        except Exception as exc:
            logger.warning("K={} detection failed: {}", k, exc)
            auc = float("nan")

        k_to_auc[k] = auc
        is_best = auc == auc and auc > best_auc  # nan-safe comparison
        logger.info(
            "  K={:5d}: LOSO AUC = {:.4f}{}",
            k, auc, "  ← best" if is_best else "",
        )
        if is_best:
            best_auc = auc
            best_k   = k

    if best_k is None:
        best_k = candidates[0]
        logger.warning(
            "All K candidates failed; falling back to K={}. "
            "Check that the dataset contains malicious samples.",
            best_k,
        )
    else:
        logger.info(
            "Best K={} → LOSO AUC = {:.4f}  (searched {} candidates)",
            best_k, best_auc, len(candidates),
        )

    best_sel = pick_top_k_heads(type_map, best_k, sanity_heads=sanity_heads)
    return best_sel, k_to_auc


def save_head_type_map(type_map: HeadTypeMap, path: Path) -> None:
    """Atomic save of a :class:`HeadTypeMap` to ``.npz``."""
    import os as _os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            selection_rate=type_map.selection_rate,
            user_intent_mean=type_map.user_intent_mean,
            tool_output_mean=type_map.tool_output_mean,
            labels=type_map.labels,
        )
    _os.replace(tmp, path)
    # Side-car JSON for human inspection.
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    import json as _json
    tmp_m = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp_m.open("w", encoding="utf-8") as fh:
        _json.dump({"thresholds": type_map.thresholds}, fh, indent=2)
    _os.replace(tmp_m, meta_path)
    logger.info("Saved HeadTypeMap → {}", path)


def load_head_type_map(path: Path) -> HeadTypeMap:
    """Round-trip companion to :func:`save_head_type_map`."""
    import json as _json

    path = Path(path)
    with np.load(path) as npz:
        selection_rate = npz["selection_rate"]
        user_intent_mean = npz["user_intent_mean"]
        tool_output_mean = npz["tool_output_mean"]
        labels = npz["labels"]
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    thresholds: dict[str, float] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            thresholds = _json.load(fh).get("thresholds", {})
    return HeadTypeMap(
        selection_rate=selection_rate,
        user_intent_mean=user_intent_mean,
        tool_output_mean=tool_output_mean,
        labels=labels,
        thresholds=thresholds,
    )


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

def save(selection: SelectionHeads, path: Path) -> None:
    """JSON round-trip for :class:`SelectionHeads`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_id": selection.cluster_id,
        "heads": [list(h) for h in selection.heads],
        "argmax_match_rate": {
            f"{L}_{H}": v for (L, H), v in selection.argmax_match_rate.items()
        },
        "passed_known_head_check": selection.passed_known_head_check,
        "failed_known_heads": [list(h) for h in selection.failed_known_heads],
        "rule_used": selection.rule_used,
        "cluster_summary": selection.cluster_summary,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load(path: Path) -> SelectionHeads:
    """Companion to :func:`save`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    heads = [tuple(h) for h in payload["heads"]]
    rates: dict[tuple[int, int], float] = {}
    for k, v in payload.get("argmax_match_rate", {}).items():
        L, H = (int(p) for p in k.split("_"))
        rates[(L, H)] = float(v)
    return SelectionHeads(
        cluster_id=int(payload["cluster_id"]),
        heads=heads,
        argmax_match_rate=rates,
        passed_known_head_check=bool(payload["passed_known_head_check"]),
        failed_known_heads=[tuple(h) for h in payload.get("failed_known_heads", [])],
        rule_used=str(payload.get("rule_used", "")),
        cluster_summary=payload.get("cluster_summary", {}),
    )
