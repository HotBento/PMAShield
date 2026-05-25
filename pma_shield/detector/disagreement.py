"""
Stage 4 — sample-level disagreement metrics + paired statistical testing.

Per-sample metrics (computed over a head set ``H``, typically the
selection-cluster heads from Stage 3)
-------------------------------------
For each sample ``s`` with selected tool ``t_out(s)``:

1. **Argmax agreement rate**
   ``A(s) = max_t |{(L,H) ∈ H : argmax_t p_{L,H}(t) = t}| / |H_valid|``
   — higher = more consistent. ``H_valid`` excludes heads that abstained
   (no attention on any tool-name span).

2. **Vote entropy**
   ``E(s) = H(distribution of hard votes over tools across H_valid)``
   — higher = more disagreement (natural log).

3. **Mean pairwise Jensen–Shannon**
   ``D_JS(s) = mean over (h1, h2) ∈ binom(H_valid, 2) of JS(p_{h1} || p_{h2})``
   — higher = more disagreement.

4. **Output alignment**
   ``O(s) = |{(L,H) ∈ H_valid : argmax = t_out(s)}| / |H_valid|``
   — fraction of heads voting for what the model actually chose.

Paired test (across all pairs):
    ``ΔE = E(mal) − E(benign) > 0``,
    ``ΔD_JS > 0``,
    ``ΔA < 0``,
    ``ΔO < 0``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .voting import (
    hard_votes_from_features,
    hard_votes_from_soft,
    soft_votes_from_features,
)

# ──────────────────────────────────────────────────────────────────────────
# Single-sample metrics
# ──────────────────────────────────────────────────────────────────────────

def _entropy_of_distribution(probs: np.ndarray, *, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _vote_distribution(hard_per_head: np.ndarray, n_tools: int) -> np.ndarray:
    """Counts → probabilities over tools (length ``n_tools``).

    ``hard_per_head`` is 1-D with values in ``[0, n_tools)`` or ``-1``
    (abstain). Abstain entries are dropped from the denominator.
    """
    valid = hard_per_head[hard_per_head >= 0]
    if valid.size == 0:
        return np.zeros(n_tools, dtype=np.float64)
    counts = np.bincount(valid, minlength=n_tools).astype(np.float64)
    return counts / counts.sum()


def _argmax_agreement_rate(hard_per_head: np.ndarray, n_tools: int) -> float:
    """Fraction of heads voting for the modal tool. Abstain heads excluded
    from the denominator."""
    valid = hard_per_head[hard_per_head >= 0]
    if valid.size == 0:
        return float("nan")
    counts = np.bincount(valid, minlength=n_tools)
    return float(counts.max() / valid.size)


def _output_alignment(hard_per_head: np.ndarray, target_idx: int) -> float:
    valid = hard_per_head[hard_per_head >= 0]
    if valid.size == 0 or target_idx < 0:
        return float("nan")
    return float((valid == target_idx).sum() / valid.size)


def _mean_pairwise_js(soft_per_head: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean pairwise Jensen–Shannon over a head set's soft votes.

    ``soft_per_head`` has shape ``(N_heads, n_tools)``. Heads whose row is
    all-NaN or sums to 0 are dropped before computing pairs.
    """
    arr = np.asarray(soft_per_head, dtype=np.float64)
    # Drop invalid rows.
    valid_mask = ~np.isnan(arr).all(axis=1)
    arr = arr[valid_mask]
    if arr.shape[0] < 2:
        return float("nan")
    # Replace NaN slots with 0 (treat padded tool slots as zero probability).
    arr = np.where(np.isnan(arr), 0.0, arr)
    # Renormalise rows so they sum to 1; abstain rows would have sum 0 and
    # were already filtered above.
    sums = arr.sum(axis=1, keepdims=True)
    sums = np.where(sums <= 0, 1.0, sums)
    arr = arr / sums

    # Vectorised mixture / KL: build all unordered pairs.
    n = arr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    p = arr[iu]                     # (n_pairs, n_tools)
    q = arr[ju]
    m = 0.5 * (p + q)
    # KL(p || m) — masked log to avoid p log 0 when p > 0 and m == 0
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    m_safe = np.clip(m, eps, 1.0)
    kl_pm = (p * (np.log(p_safe) - np.log(m_safe))).sum(axis=1)
    kl_qm = (q * (np.log(q_safe) - np.log(m_safe))).sum(axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    return float(js.mean())


# ──────────────────────────────────────────────────────────────────────────
# Per-sample tabulator
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _PerSampleMetric:
    A: float = float("nan")
    E: float = float("nan")
    D_JS: float = float("nan")
    O: float = float("nan")
    n_valid_heads: int = 0


def _compute_one_side(
    feats_pair_side: np.ndarray,            # (L, H, FEAT_DIM) for one (pair, side)
    head_set: list[tuple[int, int]],
    n_tools: int,
    target_idx: int,
) -> _PerSampleMetric:
    """Compute the four metrics for one sample, restricted to ``head_set``."""
    if not head_set:
        return _PerSampleMetric()

    # Gather per-head soft / hard votes.
    soft_full = soft_votes_from_features(feats_pair_side)        # (L, H, MAX_TOOLS)
    hard_full = hard_votes_from_soft(soft_full)                  # (L, H)

    layers = np.array([h[0] for h in head_set])
    heads = np.array([h[1] for h in head_set])
    soft_set = soft_full[layers, heads, :n_tools]                # (|H_set|, n_tools)
    hard_set = hard_full[layers, heads]                          # (|H_set|,)

    n_valid = int((hard_set >= 0).sum())
    if n_valid == 0:
        return _PerSampleMetric()

    A = _argmax_agreement_rate(hard_set, n_tools=n_tools)
    distribution = _vote_distribution(hard_set, n_tools=n_tools)
    E = _entropy_of_distribution(distribution)
    D_JS = _mean_pairwise_js(soft_set)
    O = _output_alignment(hard_set, target_idx=target_idx)
    return _PerSampleMetric(A=A, E=E, D_JS=D_JS, O=O, n_valid_heads=n_valid)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def compute(
    captured: Any,
    head_set: Sequence[tuple[int, int]],
    *,
    n_jobs: int = 64,
) -> pd.DataFrame:
    """Compute the four metrics on benign and malicious sides.

    Parameters
    ----------
    captured
        Stage-1 :class:`mcp_eval.mcptox.capture.CapturedDataset`.
    head_set
        Selection-cluster heads (e.g. :attr:`SelectionHeads.heads`).

    Returns
    -------
    pandas.DataFrame
        One row per pair (1348 rows on the full corpus) with columns:
        ``sample_id``, ``mcp_server``, ``risk_category``, ``attack_paradigm``,
        ``A_benign``, ``A_mal``, ``dA``, ``E_benign``, ``E_mal``, ``dE``,
        ``D_JS_benign``, ``D_JS_mal``, ``dD_JS``, ``O_benign``, ``O_mal``,
        ``dO``, ``n_valid_heads_benign``, ``n_valid_heads_mal``,
        ``parse_ok_benign``, ``parse_ok_mal``.

    Parameters (parallelism)
    ------------------------
    n_jobs
        Number of worker threads for pair-wise computation. ``1`` (default)
        keeps serial behaviour. Values ``>1`` enable thread parallelism.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1; got {n_jobs}")

    head_set_list = [tuple(h) for h in head_set]
    feats_b = np.asarray(captured.benign_features)
    feats_m = np.asarray(captured.malicious_features)
    n_pairs = feats_b.shape[0]

    # Build a per-pair lookup of the meta records, indexed by (pair_idx, side).
    meta_by_pair_side: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in captured.meta:
        meta_by_pair_side[(int(entry["pair_idx"]), entry["side"])] = entry

    def _compute_pair_row(p: int) -> dict[str, Any] | None:
        eb = meta_by_pair_side.get((p, "benign"))
        em = meta_by_pair_side.get((p, "malicious"))
        if eb is None or em is None:
            logger.warning("Missing meta for pair {}; skipping", p)
            return None

        n_tools_b = len(eb.get("tool_names") or [])
        n_tools_m = len(em.get("tool_names") or [])
        target_b = (
            eb["tool_names"].index(eb["selected_tool"])
            if eb.get("selected_tool") in (eb.get("tool_names") or [])
            else -1
        )
        target_m = (
            em["tool_names"].index(em["selected_tool"])
            if em.get("selected_tool") in (em.get("tool_names") or [])
            else -1
        )

        m_b = _compute_one_side(
            feats_b[p], head_set_list, n_tools=n_tools_b, target_idx=target_b
        )
        m_m = _compute_one_side(
            feats_m[p], head_set_list, n_tools=n_tools_m, target_idx=target_m
        )

        return {
            "pair_idx": p,
            "sample_id": eb.get("sample_id"),
            "mcp_server": eb.get("mcp_server"),
            "risk_category": eb.get("risk_category"),
            "attack_paradigm": eb.get("attack_paradigm"),
            "parse_ok_benign": bool(eb.get("parse_ok")),
            "parse_ok_mal": bool(em.get("parse_ok")),
            "n_valid_heads_benign": m_b.n_valid_heads,
            "n_valid_heads_mal": m_m.n_valid_heads,
            "A_benign": m_b.A, "A_mal": m_m.A, "dA": m_m.A - m_b.A,
            "E_benign": m_b.E, "E_mal": m_m.E, "dE": m_m.E - m_b.E,
            "D_JS_benign": m_b.D_JS, "D_JS_mal": m_m.D_JS, "dD_JS": m_m.D_JS - m_b.D_JS,
            "O_benign": m_b.O, "O_mal": m_m.O, "dO": m_m.O - m_b.O,
        }

    rows: list[dict[str, Any]] = []
    if n_jobs == 1 or n_pairs <= 1:
        for p in range(n_pairs):
            row = _compute_pair_row(p)
            if row is not None:
                rows.append(row)
    else:
        max_workers = min(n_jobs, n_pairs)
        logger.info("Computing disagreement with thread parallelism: n_jobs={}", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for row in ex.map(_compute_pair_row, range(n_pairs)):
                if row is not None:
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ for paired delta arrays (treats inputs independently).

    Vectorised but O(|a| × |b|); fine up to ~10k samples. Returns NaN when
    either input is empty after dropping NaN.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def paired_test(
    df: pd.DataFrame,
    metric: str,
    *,
    direction: str,
    drop_failed_parse: bool = True,
) -> dict[str, Any]:
    """One-sided Wilcoxon signed-rank test on a Δmetric.

    Parameters
    ----------
    df
        Output of :func:`compute`.
    metric
        Column name for the delta (one of ``"dA"``, ``"dE"``, ``"dD_JS"``,
        ``"dO"``).
    direction
        ``"greater"`` or ``"less"`` — the side of the alternative hypothesis.
    drop_failed_parse
        If True (default), only count pairs where both sides parsed
        successfully.
    """
    from scipy.stats import wilcoxon

    if direction not in {"greater", "less"}:
        raise ValueError(f"direction must be 'greater' or 'less'; got {direction!r}")

    sub = df.copy()
    if drop_failed_parse:
        sub = sub[sub["parse_ok_benign"] & sub["parse_ok_mal"]]
    deltas = sub[metric].to_numpy(dtype=np.float64)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size < 5:
        return {
            "metric": metric, "direction": direction,
            "statistic": float("nan"), "p_value": float("nan"),
            "median_delta": float("nan"), "cliffs_delta": float("nan"),
            "n": int(deltas.size),
        }

    # zero_method='wilcox' (default) drops zero deltas; use 'pratt' to keep
    # them in the rank (safer when many tied deltas occur in noisy data).
    stat, p = wilcoxon(deltas, alternative=direction, zero_method="pratt")
    delta_pos = deltas[deltas > 0]
    delta_neg = deltas[deltas < 0]
    cd = _cliffs_delta(delta_pos, -delta_neg) if (delta_pos.size and delta_neg.size) else float("nan")
    return {
        "metric": metric,
        "direction": direction,
        "statistic": float(stat),
        "p_value": float(p),
        "median_delta": float(np.median(deltas)),
        "cliffs_delta": cd,
        "n": int(deltas.size),
    }


def per_category_tests(
    df: pd.DataFrame,
    *,
    metrics: Sequence[tuple[str, str]] = (
        ("dA", "less"), ("dE", "greater"),
        ("dD_JS", "greater"), ("dO", "less"),
    ),
    by: str = "risk_category",
) -> pd.DataFrame:
    """Run paired Wilcoxon for each (metric, direction, category) cell.

    Returns a long-form DataFrame with Bonferroni-corrected p-values across
    the (n_metrics × n_categories) tests.
    """
    rows: list[dict[str, Any]] = []
    cats = sorted(df[by].dropna().unique())
    n_tests = max(len(metrics) * len(cats), 1)
    for metric, direction in metrics:
        for cat in cats:
            sub = df[df[by] == cat]
            res = paired_test(sub, metric, direction=direction)
            res["category"] = cat
            res["p_value_bonferroni"] = min(res["p_value"] * n_tests, 1.0) if not np.isnan(res["p_value"]) else float("nan")
            rows.append(res)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

def compute_per_head_tool_attn_df(
    captured: Any,
    pairs: Sequence[Any],
    head_set: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    """Build a long-form DataFrame for the three-condition strip plot.

    For each (pair, head) combination, extracts the combined name+desc
    attention on three tools:

    * ``"benign_correct"``  — benign side, attention on the benign-selected tool
    * ``"mal_correct"``     — malicious side, attention on the same correct tool
    * ``"mal_poisoned"``    — malicious side, attention on the poisoned tool

    Parameters
    ----------
    captured
        Stage-1 :class:`~mcp_eval.mcptox.capture.CapturedDataset`.
    pairs
        Sequence of :class:`~mcp_eval.mcptox.data.MCPToxPair` objects (used
        for ``poisoned_tool_name``).
    head_set
        List of ``(layer, head)`` tuples (e.g. from
        :class:`~mcp_eval.mcptox.selection.SelectionHeads`).

    Returns
    -------
    pd.DataFrame
        Columns: ``head_id`` (str, e.g. ``"L24H29"``),
        ``condition`` (str, one of the three above),
        ``attn_value`` (float32).
    """
    from .features import MAX_TOOLS, feature_layout

    if not head_set:
        return pd.DataFrame(columns=["head_id", "condition", "attn_value"])

    layout = feature_layout()
    name_slice: slice = layout["per_tool_name"]   # 0:MAX_TOOLS
    desc_slice: slice = layout["per_tool_desc"]   # MAX_TOOLS:2*MAX_TOOLS

    b_feats = np.asarray(captured.benign_features)    # (N, L, H, FEAT_DIM)
    m_feats = np.asarray(captured.malicious_features) # (N, L, H, FEAT_DIM)
    n_pairs = b_feats.shape[0]

    # Build pair-indexed meta look-ups.
    benign_meta: dict[int, dict] = {}
    mal_meta: dict[int, dict] = {}
    for entry in captured.meta:
        p = int(entry["pair_idx"])
        if entry.get("side") == "benign":
            benign_meta[p] = entry
        elif entry.get("side") == "malicious":
            mal_meta[p] = entry

    # Build pair → poisoned tool index on the malicious side.
    pair_map: dict[int, Any] = {}
    for pair in pairs:
        pidx = int(pair.pair_idx) if hasattr(pair, "pair_idx") else -1
        if 0 <= pidx < n_pairs:
            pair_map[pidx] = pair

    rows: list[dict] = []

    layers = [h[0] for h in head_set]
    heads_idx = [h[1] for h in head_set]
    head_ids = [f"L{l}H{h}" for l, h in head_set]

    for p in range(n_pairs):
        b_entry = benign_meta.get(p, {})
        m_entry = mal_meta.get(p, {})

        if not b_entry.get("parse_ok") or not m_entry.get("parse_ok"):
            continue

        b_tool_names = b_entry.get("tool_names") or []
        m_tool_names = m_entry.get("tool_names") or []
        correct_tool = b_entry.get("selected_tool")

        if correct_tool is None or correct_tool not in b_tool_names:
            continue
        b_correct_idx = b_tool_names.index(correct_tool)

        # correct tool on malicious side
        m_correct_idx: int | None = None
        if correct_tool in m_tool_names:
            m_correct_idx = m_tool_names.index(correct_tool)

        # poisoned tool on malicious side
        m_poison_idx: int | None = None
        pair_obj = pair_map.get(p)
        if pair_obj is not None:
            pname = getattr(pair_obj, "poisoned_tool_name", None)
            if pname and pname in m_tool_names:
                m_poison_idx = m_tool_names.index(pname)

        for i, (L, H) in enumerate(zip(layers, heads_idx)):
            hid = head_ids[i]
            b_row = b_feats[p, L, H]  # (FEAT_DIM,)
            m_row = m_feats[p, L, H]

            # benign_correct
            val_bc = (
                float(b_row[name_slice][b_correct_idx])
                + float(b_row[desc_slice][b_correct_idx])
            )
            rows.append({"head_id": hid, "condition": "benign_correct", "attn_value": val_bc})

            # mal_correct
            if m_correct_idx is not None:
                val_mc = (
                    float(m_row[name_slice][m_correct_idx])
                    + float(m_row[desc_slice][m_correct_idx])
                )
                rows.append({"head_id": hid, "condition": "mal_correct", "attn_value": val_mc})

            # mal_poisoned
            if m_poison_idx is not None:
                val_mp = (
                    float(m_row[name_slice][m_poison_idx])
                    + float(m_row[desc_slice][m_poison_idx])
                )
                rows.append({"head_id": hid, "condition": "mal_poisoned", "attn_value": val_mp})

    return pd.DataFrame(rows, columns=["head_id", "condition", "attn_value"])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write (``.tmp → rename``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_summary_md(
    df: pd.DataFrame,
    *,
    overall_tests: Iterable[dict[str, Any]],
    per_cat_df: pd.DataFrame,
    out: Path,
) -> None:
    """Render a Markdown report with the headline numbers."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 4 — Disagreement metrics summary",
        "",
        f"Total pairs: **{len(df)}**.  Pairs where both sides parsed: "
        f"**{int((df['parse_ok_benign'] & df['parse_ok_mal']).sum())}**.",
        "",
        "## Overall paired tests",
        "",
        "| metric | direction | n | median Δ | Wilcoxon p | Cliff's δ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in overall_tests:
        lines.append(
            f"| {r['metric']} | {r['direction']} | {r['n']} | "
            f"{r['median_delta']:.4f} | {r['p_value']:.3g} | "
            f"{r['cliffs_delta']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-category breakdown (Bonferroni-corrected)")
    lines.append("")
    lines.append(
        "| metric | category | n | median Δ | p (Bonf) |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in per_cat_df.iterrows():
        lines.append(
            f"| {row['metric']} | {row['category']} | {int(row['n'])} | "
            f"{row['median_delta']:.4f} | "
            f"{row['p_value_bonferroni']:.3g} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote disagreement summary → {}", out)
