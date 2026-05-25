"""
Stage 5 — turn disagreement metrics into a no-training attack detector.

Setup
-----
* Task: binary classification — is the sample malicious?
* Features: the four disagreement metrics from :mod:`disagreement`
  (``A, E, D_JS, O``), optionally restricted to the selection cluster, or to
  an alternate head set (ablation).
* Splits: leave-one-server-out (LOSO) over the 45 MCP servers.

Classifiers
-----------
1. Single-threshold ROC per metric (no train/test split — pure ranking).
2. Logistic regression on the four metrics jointly, evaluated under LOSO.

Ablation controls
-----------------
A. **random_heads**     — replace selection heads with a random sample of
   ``|H|`` heads from outside the selection cluster.
B. **top6**             — restrict ``H`` to :data:`config.KNOWN_CORE_HEADS`.
C. **query_attn**       — single-feature classifier on
   ``Δ(mean attn_user_query)`` over the selection heads.
D. **logit_baseline**   — *NOT YET WIRED*: requires the provider to expose
   the selection-step logits (``output.scores[attn_step]``). Currently raises
   ``NotImplementedError`` — implement once Stage 1 saves
   ``last_attn_step_logits`` per side.

Public API
----------
* :func:`evaluate` — top-level entry point.
* :class:`DetectionReport` — container for all numbers + curves.
* :func:`write_report` — render Markdown + ROC PNG.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from . import config
from .disagreement import compute as compute_disagreement
from .features import feature_layout
from .selection import SelectionHeads


# ──────────────────────────────────────────────────────────────────────────
# Reshaping helpers
# ──────────────────────────────────────────────────────────────────────────

def _df_to_xy(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a per-pair (benign, malicious) DataFrame to X / y / groups
    suitable for sklearn (one row per *side* — 2 rows per pair)."""
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for side, label in (("benign", 0), ("mal", 1)):
            row: dict[str, Any] = {"y": label, "group": r["mcp_server"]}
            for col in feature_cols:
                row[col] = r.get(f"{col.split('_per_side')[0]}_{side}", float("nan"))
            rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=feature_cols).reset_index(drop=True)
    X = out[list(feature_cols)].to_numpy(dtype=np.float64)
    y = out["y"].to_numpy(dtype=np.int64)
    groups = out["group"].to_numpy()
    return X, y, groups


def _delta_xy_per_pair(df: pd.DataFrame, delta_cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use *Δmetrics* as features and a constant per-pair label of 1 vs. a
    matched zero-delta vector for "benign-only" — this would create an
    artificial paired classifier. Not used at present; included for the
    optional Δ-classifier ablation if we ever want it.
    """
    raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# Single-threshold ROC
# ──────────────────────────────────────────────────────────────────────────

def _per_metric_auc(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each disagreement metric, compute single-threshold ROC AUC.

    Treats the union of ``benign`` and ``malicious`` per-side rows as the
    point cloud and uses each metric's value (with sign chosen so that
    higher = more malicious-looking) as the score.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    per_metric: dict[str, dict[str, Any]] = {}
    # For each metric, decide sign: A (agreement) and O (output alignment)
    # are *lower* when malicious — flip for AUC.
    metric_specs = (
        ("A", -1),
        ("E", +1),
        ("D_JS", +1),
        ("O", -1),
    )
    for metric, sign in metric_specs:
        ben_col, mal_col = f"{metric}_benign", f"{metric}_mal"
        sub = df.dropna(subset=[ben_col, mal_col])
        scores = np.concatenate([sub[ben_col].to_numpy(), sub[mal_col].to_numpy()]) * sign
        labels = np.concatenate([np.zeros(len(sub)), np.ones(len(sub))])
        if len(sub) < 5:
            per_metric[metric] = {"auc": float("nan"), "fpr": np.array([]), "tpr": np.array([])}
            continue
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        # Find the threshold that maximises Youden's J.
        if fpr.size > 1:
            j = tpr - fpr
            best = int(j.argmax())
            best_threshold_score = float(scores[scores.argsort()][np.searchsorted(np.sort(scores), scores[scores.argsort()][int(round(best * len(scores) / (fpr.size - 1)))])]) if fpr.size else float("nan")
        else:
            best_threshold_score = float("nan")
        per_metric[metric] = {
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "score_sign": sign,
            "best_threshold_score": best_threshold_score,
        }
    return per_metric


# ──────────────────────────────────────────────────────────────────────────
# LOSO logistic regression
# ──────────────────────────────────────────────────────────────────────────

def _best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Sweep candidate thresholds and return ``(best_f1, best_threshold)``.

    Candidate set is the unique observed scores plus 0.5.  Degenerate
    thresholds (all-positive or all-negative predictions) are skipped.
    Returns ``(nan, nan)`` if no valid threshold exists.
    """
    from sklearn.metrics import f1_score
    if len(scores) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    candidates = np.unique(np.concatenate([scores, [0.5]]))
    best_f1, best_thr = -1.0, 0.5
    for thr in candidates:
        preds = (scores >= thr).astype(np.int64)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    if best_f1 < 0:
        return float("nan"), float("nan")
    return best_f1, best_thr


def _loso_logistic_auc(
    df: pd.DataFrame,
    feature_cols_per_side: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Leave-one-server-out logistic regression AUC and best-F1.

    ``feature_cols_per_side`` lists the per-metric column **bases**
    (e.g. ``("A", "E", "D_JS", "O")``) — both ``<metric>_benign`` and
    ``<metric>_mal`` rows are emitted per pair.

    Per-fold F1 protocol (nested threshold selection)
    ------------------------------------------------
    Within each LOSO fold:
      1. Fit a logistic regression on the N-1 training servers.
      2. Predict probabilities on the **training set** and sweep thresholds
         to maximise F1 on training labels.  The chosen threshold is a 1-D
         operating-point parameter selected without using the held-out
         server's labels.
      3. Apply that threshold to the held-out server's predictions to obtain
         the fold's F1 score.
    The aggregate ``f1_best`` is the mean fold F1.  We also keep
    ``f1_best_pooled`` (single global threshold over pooled out-of-fold
    predictions) for diagnostic comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score

    # Build per-side rows.
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for side, label in (("benign", 0), ("mal", 1)):
            entry: dict[str, Any] = {"y": label, "group": r["mcp_server"]}
            for col in feature_cols_per_side:
                entry[col] = r.get(f"{col}_{side}", float("nan"))
            rows.append(entry)
    sub = pd.DataFrame(rows).dropna(subset=list(feature_cols_per_side))
    nan_return = {
        "loso_auc_mean": float("nan"),
        "loso_auc_per_fold": {},
        "f1_best": float("nan"),
        "f1_best_threshold": float("nan"),
        "f1_per_fold": {},
        "threshold_per_fold": {},
        "f1_best_pooled": float("nan"),
        "f1_best_pooled_threshold": float("nan"),
        "n": len(sub),
    }
    if len(sub) < 20 or len(sub["group"].unique()) < 2:
        return nan_return

    X = sub[list(feature_cols_per_side)].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.int64)
    g = sub["group"].to_numpy()

    aucs: dict[str, float] = {}
    f1_per_fold: dict[str, float] = {}
    thr_per_fold: dict[str, float] = {}
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []

    for held in sorted(set(g)):
        train_mask = g != held
        test_mask = ~train_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue  # cannot compute AUC with a single class in the fold
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed,
        )
        clf.fit(X[train_mask], y[train_mask])

        # AUROC on the held-out fold.
        test_proba = clf.predict_proba(X[test_mask])[:, 1]
        aucs[str(held)] = float(roc_auc_score(y[test_mask], test_proba))

        # Threshold selection: maximise F1 on the training set, then apply to test.
        train_proba = clf.predict_proba(X[train_mask])[:, 1]
        _, fold_thr = _best_f1_threshold(y[train_mask], train_proba)
        if not np.isnan(fold_thr):
            test_pred = (test_proba >= fold_thr).astype(np.int64)
            fold_f1 = float(f1_score(y[test_mask], test_pred, zero_division=0))
            f1_per_fold[str(held)] = fold_f1
            thr_per_fold[str(held)] = fold_thr

        pooled_scores.extend(test_proba.tolist())
        pooled_labels.extend(y[test_mask].tolist())

    if not aucs:
        return nan_return

    # Aggregate fold-level F1.
    if f1_per_fold:
        f1_best = float(np.mean(list(f1_per_fold.values())))
        f1_best_thr = float(np.mean(list(thr_per_fold.values())))
    else:
        f1_best = float("nan")
        f1_best_thr = float("nan")

    # Diagnostic: pooled-threshold F1 (selects a single global threshold over
    # all out-of-fold predictions — biased upward because the threshold sees
    # test labels, kept for comparison only).
    pooled_scores_arr = np.asarray(pooled_scores, dtype=np.float64)
    pooled_labels_arr = np.asarray(pooled_labels, dtype=np.int64)
    f1_pooled, f1_pooled_thr = _best_f1_threshold(pooled_labels_arr, pooled_scores_arr)

    # Pooled out-of-fold ROC curve for downstream plotting.
    from sklearn.metrics import roc_curve
    pooled_fpr: list[float] = []
    pooled_tpr: list[float] = []
    if len(pooled_scores_arr) > 0 and len(np.unique(pooled_labels_arr)) == 2:
        fpr, tpr, _ = roc_curve(pooled_labels_arr, pooled_scores_arr)
        pooled_fpr = fpr.tolist()
        pooled_tpr = tpr.tolist()

    return {
        "loso_auc_mean": float(np.mean(list(aucs.values()))),
        "loso_auc_std": float(np.std(list(aucs.values()), ddof=0)),
        "loso_auc_per_fold": aucs,
        "f1_best": f1_best,
        "f1_best_threshold": f1_best_thr,
        "f1_per_fold": f1_per_fold,
        "threshold_per_fold": thr_per_fold,
        "f1_best_pooled": f1_pooled,
        "f1_best_pooled_threshold": f1_pooled_thr,
        "roc_fpr": pooled_fpr,
        "roc_tpr": pooled_tpr,
        "n": int(len(sub)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Ablation feature builders
# ──────────────────────────────────────────────────────────────────────────

def _random_non_selection_heads(
    selection: SelectionHeads,
    *,
    num_layers: int,
    num_heads: int,
    seed: int,
) -> list[tuple[int, int]]:
    sel_set = set(map(tuple, selection.heads))
    pool = [(L, H) for L in range(num_layers) for H in range(num_heads) if (L, H) not in sel_set]
    rng = random.Random(seed)
    n = min(len(selection.heads), len(pool))
    return rng.sample(pool, n)


def _query_attn_delta_per_pair(
    captured: Any,
    head_set: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    """Compute Δ(mean ``attn_user_query``) over a head set, per pair.

    Returns a DataFrame with one row per pair, columns:
    ``pair_idx``, ``mcp_server``, ``risk_category``, ``query_attn_benign``,
    ``query_attn_mal``, ``dquery_attn``.
    """
    layout = feature_layout()
    feats_b = np.asarray(captured.benign_features)
    feats_m = np.asarray(captured.malicious_features)
    layers = np.array([h[0] for h in head_set])
    heads = np.array([h[1] for h in head_set])
    if len(layers) == 0:
        raise ValueError("empty head_set passed to _query_attn_delta_per_pair")
    qcol = layout["attn_user_query"]
    qb = np.nanmean(feats_b[:, layers, heads, qcol], axis=1)
    qm = np.nanmean(feats_m[:, layers, heads, qcol], axis=1)

    meta_by_pair: dict[int, dict[str, Any]] = {}
    for entry in captured.meta:
        if entry["side"] == "benign":
            meta_by_pair[int(entry["pair_idx"])] = entry

    rows = []
    for p in range(feats_b.shape[0]):
        meta = meta_by_pair.get(p, {})
        rows.append({
            "pair_idx": p,
            "mcp_server": meta.get("mcp_server"),
            "risk_category": meta.get("risk_category"),
            "query_attn_benign": float(qb[p]) if not np.isnan(qb[p]) else float("nan"),
            "query_attn_mal": float(qm[p]) if not np.isnan(qm[p]) else float("nan"),
            "dquery_attn": float(qm[p] - qb[p]) if not np.isnan(qm[p] - qb[p]) else float("nan"),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Report container + driver
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionReport:
    """All numbers + curves emitted by :func:`evaluate`."""

    main_per_metric_auc: dict[str, float]
    main_per_metric_roc: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    main_logistic: dict[str, Any] = field(default_factory=dict)
    per_category_logistic: dict[str, dict[str, Any]] = field(default_factory=dict)
    ablations: dict[str, dict[str, Any]] = field(default_factory=dict)
    head_set_sizes: dict[str, int] = field(default_factory=dict)
    seed: int = 0


def evaluate(
    captured: Any,
    selection: SelectionHeads,
    *,
    controls: Sequence[str] = ("random_heads", "top6", "query_attn"),
    seed: int = 0,
    known_core_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
) -> DetectionReport:
    """End-to-end evaluator. Returns a :class:`DetectionReport`."""
    num_layers = int(captured.manifest["num_layers"])
    num_heads = int(captured.manifest["num_heads"])

    # Main: disagreement metrics on the selection cluster.
    df_main = compute_disagreement(captured, head_set=selection.heads)
    df_main = df_main[df_main["parse_ok_benign"] & df_main["parse_ok_mal"]].reset_index(drop=True)

    per_metric = _per_metric_auc(df_main)
    logistic = _loso_logistic_auc(df_main, ("A", "E", "D_JS", "O"), seed=seed)

    # Per-category AUC (joint logistic on the four metrics).
    per_cat: dict[str, dict[str, Any]] = {}
    for cat, cat_df in df_main.groupby("risk_category"):
        if len(cat_df) < 10:
            continue
        per_cat[str(cat)] = _loso_logistic_auc(cat_df, ("A", "E", "D_JS", "O"), seed=seed)

    # Ablations.
    ablations: dict[str, dict[str, Any]] = {}
    head_set_sizes = {"main": len(selection.heads)}

    if "random_heads" in controls:
        rh = _random_non_selection_heads(
            selection, num_layers=num_layers, num_heads=num_heads, seed=seed
        )
        head_set_sizes["random_heads"] = len(rh)
        df_rh = compute_disagreement(captured, head_set=rh)
        df_rh = df_rh[df_rh["parse_ok_benign"] & df_rh["parse_ok_mal"]].reset_index(drop=True)
        ablations["random_heads"] = {
            **_loso_logistic_auc(df_rh, ("A", "E", "D_JS", "O"), seed=seed),
            "per_metric_auc": {k: v["auc"] for k, v in _per_metric_auc(df_rh).items()},
        }

    if "top6" in controls:
        top6 = list(map(tuple, known_core_heads))
        head_set_sizes["top6"] = len(top6)
        if not top6:
            # No registered core heads for this model; skip the control
            # entirely rather than crash on an empty head_set.
            ablations["top6"] = {"loso_auc_mean": float("nan"), "per_metric_auc": {}}
        else:
            df_t6 = compute_disagreement(captured, head_set=top6)
            df_t6 = df_t6[df_t6["parse_ok_benign"] & df_t6["parse_ok_mal"]].reset_index(drop=True)
            ablations["top6"] = {
                **_loso_logistic_auc(df_t6, ("A", "E", "D_JS", "O"), seed=seed),
                "per_metric_auc": {k: v["auc"] for k, v in _per_metric_auc(df_t6).items()},
            }

    if "query_attn" in controls:
        df_q = _query_attn_delta_per_pair(captured, selection.heads)
        # Single-feature AUC: rank pairs by mal vs benign.
        from sklearn.metrics import roc_auc_score
        rows = []
        for _, r in df_q.dropna(subset=["query_attn_benign", "query_attn_mal"]).iterrows():
            rows.append({"y": 0, "group": r["mcp_server"], "feat": r["query_attn_benign"]})
            rows.append({"y": 1, "group": r["mcp_server"], "feat": r["query_attn_mal"]})
        sub = pd.DataFrame(rows)
        ablations["query_attn"] = {
            "per_feature_auc": float(roc_auc_score(sub["y"], sub["feat"]))
            if len(sub) > 4 and len(sub["y"].unique()) == 2
            else float("nan"),
        }

    if "logit_baseline" in controls:
        logger.warning(
            "Control 'logit_baseline' not yet implemented — requires the "
            "HuggingFaceProvider to cache `output.scores[attn_step]` per "
            "sample. Skipping."
        )
        ablations["logit_baseline"] = {"note": "not_implemented"}

    return DetectionReport(
        main_per_metric_auc={k: v["auc"] for k, v in per_metric.items()},
        main_per_metric_roc={
            k: {"fpr": v["fpr"], "tpr": v["tpr"]} for k, v in per_metric.items()
        },
        main_logistic=logistic,
        per_category_logistic=per_cat,
        ablations=ablations,
        head_set_sizes=head_set_sizes,
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

def write_report(report: DetectionReport, out: Path) -> None:
    """Render a Markdown summary alongside a JSON dump for programmatic use."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # JSON (numpy arrays → lists).
    payload: dict[str, Any] = {
        "main_per_metric_auc": dict(report.main_per_metric_auc),
        "main_logistic": _to_jsonable(report.main_logistic),
        "per_category_logistic": {
            k: _to_jsonable(v) for k, v in report.per_category_logistic.items()
        },
        "ablations": {k: _to_jsonable(v) for k, v in report.ablations.items()},
        "head_set_sizes": dict(report.head_set_sizes),
        "seed": report.seed,
    }
    json_path = out.with_suffix(".json")
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)

    # Markdown.
    lines = ["# Stage 5 — Attack-detection report", ""]
    lines.append("## Per-metric ROC AUC (single-threshold)")
    lines.append("")
    lines.append("| metric | AUC |")
    lines.append("| --- | --- |")
    for k, v in report.main_per_metric_auc.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append("## Joint logistic (LOSO) on selection cluster")
    lines.append(
        f"- Mean AUC: **{report.main_logistic.get('loso_auc_mean', float('nan')):.4f}**"
    )
    lines.append(
        f"- F1 (nested per-fold threshold): "
        f"**{report.main_logistic.get('f1_best', float('nan')):.4f}**  "
        f"(avg threshold = {report.main_logistic.get('f1_best_threshold', float('nan')):.3f})"
    )
    lines.append(
        f"- F1 (pooled threshold, diagnostic): "
        f"{report.main_logistic.get('f1_best_pooled', float('nan')):.4f}"
    )
    lines.append(f"- N (per-side rows): {report.main_logistic.get('n', '?')}")
    lines.append(f"- Folds: {len(report.main_logistic.get('loso_auc_per_fold', {}))}")
    lines.append("")
    if report.per_category_logistic:
        lines.append("## Per-risk-category LOSO AUC")
        lines.append("")
        lines.append("| category | n | AUC mean | AUC std |")
        lines.append("| --- | --- | --- | --- |")
        for cat, stats in sorted(report.per_category_logistic.items()):
            lines.append(
                f"| {cat} | {stats.get('n', '?')} | "
                f"{stats.get('loso_auc_mean', float('nan')):.4f} | "
                f"{stats.get('loso_auc_std', float('nan')):.4f} |"
            )
        lines.append("")
    lines.append("## Ablations")
    lines.append("")
    for k, stats in report.ablations.items():
        lines.append(f"### {k}")
        lines.append(f"- Head-set size: {report.head_set_sizes.get(k, '?')}")
        if "loso_auc_mean" in stats:
            lines.append(f"- Joint logistic AUC mean: {stats['loso_auc_mean']:.4f}")
        if "per_feature_auc" in stats:
            lines.append(f"- Per-feature AUC: {stats['per_feature_auc']:.4f}")
        if "per_metric_auc" in stats:
            for m, v in stats["per_metric_auc"].items():
                lines.append(f"- {m} AUC: {v:.4f}")
        if "note" in stats:
            lines.append(f"- Note: {stats['note']}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote detection report → {} (json side-car: {})", out, json_path)


def _to_jsonable(obj: Any) -> Any:
    """Coerce numpy and other awkward types into JSON-friendly forms."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
