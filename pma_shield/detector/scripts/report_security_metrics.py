"""
Security-metric post-processing CLI (rebuttal, reviewer 3Bi4 W2).

3Bi4 asked for TPR at low FPR, PR-AUC, calibration, and worst-case
per-server performance, in addition to plain AUROC. This script derives all
four from artifacts ``detection.py`` / ``baselines.py`` already produce —
**no new inference or re-training is needed**, it is pure post-processing
over the pooled out-of-fold ``(score, label)`` pairs and per-fold AUROC
dicts already saved to ``detection.json`` / ``baselines.json``.

Inputs
------
Any number of "curve sources", each pointing at a JSON produced by:
  - ``run_detection.py``  → ``detection.json`` (has ``main_logistic`` plus
    each entry in ``ablations`` with the same shape: ``pooled_scores``,
    ``pooled_labels``, ``loso_auc_per_fold``);
  - ``run_baselines.py``  → ``baselines.json`` (flat: one entry per baseline
    name, same field names as ``LOSOResult`` — ``pooled_scores``,
    ``pooled_labels``, ``auc_per_fold``).

Usage
-----
    python -m pma_shield.detector.scripts.report_security_metrics \\
        --detection results/mcptox/Qwen_Qwen3-8B/detection.json:main_logistic:PMAShield \\
        --baselines results/mcptox/Qwen_Qwen3-8B/baselines/baselines.json \\
        --out results/mcptox/Qwen_Qwen3-8B/security_metrics.md \\
        --fpr-targets 0.01 0.05 0.10 \\
        --bootstrap 2000

Each ``--detection`` argument is ``path:key:display_name`` where ``key`` is
either ``main_logistic`` or an ablation name (e.g. ``random_heads``); repeat
the flag to pull multiple curves out of the same or different
``detection.json`` files (e.g. one per model). ``--baselines`` pulls in
*every* entry of a ``baselines.json`` automatically (display name = the
JSON key, e.g. ``logit_margin``).

Output
------
A single Markdown table (plus a JSON side-car) with, per curve:
  - AUROC (mean of already-saved per-fold values — unchanged from before)
  - worst-fold AUROC (min over ``*_per_fold``, directly answers "worst-case
    per-server performance")
  - PR-AUC (average precision) computed from the pooled (score, label) pairs
  - TPR at each requested FPR operating point (linear interpolation along
    the empirical ROC curve)
  - Brier score (calibration — mean squared error between predicted
    probability and the 0/1 label; only meaningful for the supervised
    detectors, since ``residual_ood``'s "score" is an unbounded distance,
    not a probability — flagged as N/A for those)
  - Bootstrap 95% CI on AUROC and PR-AUC, resampling **pairs** (not
    individual rows) with replacement, since each pair contributes one
    benign + one malicious row that share a server label — resampling rows
    independently would understate variance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from loguru import logger


class Curve(NamedTuple):
    name: str
    scores: np.ndarray
    labels: np.ndarray
    per_fold_auc: dict[str, float]


# ──────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────

def _load_detection_curve(spec: str) -> Curve:
    """Parse ``path:key:display_name`` and pull one curve out of detection.json."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--detection spec must be 'path:key:display_name', got: {spec}")
    path_str, key, display_name = parts
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    block = payload["main_logistic"] if key == "main_logistic" else payload["ablations"][key]
    scores = np.asarray(block.get("pooled_scores", []), dtype=np.float64)
    labels = np.asarray(block.get("pooled_labels", []), dtype=np.int64)
    per_fold = block.get("loso_auc_per_fold", {})
    if scores.size == 0:
        logger.warning(
            "{}: no pooled_scores found for key={} — was this detection.json "
            "produced before the pooled_scores/pooled_labels field was added? "
            "Re-run run_detection.py.",
            path_str, key,
        )
    return Curve(name=display_name, scores=scores, labels=labels, per_fold_auc=per_fold)


def _load_baseline_curves(path: Path) -> list[Curve]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    curves = []
    for name, block in payload.items():
        scores = np.asarray(block.get("pooled_scores", []), dtype=np.float64)
        labels = np.asarray(block.get("pooled_labels", []), dtype=np.int64)
        per_fold = block.get("auc_per_fold", {})
        if scores.size == 0:
            logger.warning(
                "{}: no pooled_scores found for baseline={} — re-run "
                "run_baselines.py to regenerate with pooled scores.",
                path, name,
            )
        curves.append(Curve(name=name, scores=scores, labels=labels, per_fold_auc=per_fold))
    return curves


# ──────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────

def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared calibration error. Assumes ``probs`` are in [0, 1]
    (true for the supervised LR-based detectors; N/A for unbounded OOD
    distance scores — caller should skip this metric for those)."""
    if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def bootstrap_ci_by_pair(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float]:
    """95% CI via pair-level bootstrap resampling.

    Each pair contributes exactly one benign (label 0) and one malicious
    (label 1) row adjacent in the pooled arrays (see how detection.py /
    baselines.py build ``pooled_scores``/``pooled_labels`` — benign then
    malicious per pair, in pair order). Resampling *pairs* rather than
    individual rows keeps the benign/malicious count balanced in every
    bootstrap replicate and respects that both rows come from the same
    underlying sample, avoiding an overly optimistic (too-narrow) CI.
    """
    n_rows = len(labels)
    if n_rows < 4 or n_rows % 2 != 0:
        logger.warning(
            "bootstrap_ci_by_pair: expected an even number of rows "
            "(benign+malicious per pair), got {}; CI may be unreliable.",
            n_rows,
        )
    n_pairs = n_rows // 2
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pair_idx = rng.integers(0, n_pairs, size=n_pairs)
        row_idx = np.concatenate([[2 * i, 2 * i + 1] for i in pair_idx])
        v = metric_fn(labels[row_idx], scores[row_idx])
        if v == v:  # not NaN
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    lo = float(np.percentile(values, (1 - ci) / 2 * 100))
    hi = float(np.percentile(values, (1 + ci) / 2 * 100))
    return lo, hi


def summarize_curve(
    curve: Curve, *, fpr_targets: tuple[float, ...], n_boot: int
) -> dict[str, Any]:
    labels, scores = curve.labels, curve.scores
    if labels.size == 0:
        return {"name": curve.name, "error": "no pooled scores/labels available"}

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    worst_fold = min(curve.per_fold_auc.values()) if curve.per_fold_auc else float("nan")
    ap = pr_auc(labels, scores)
    tprs = {f"tpr@fpr={f}": tpr_at_fpr(labels, scores, f) for f in fpr_targets}

    is_probability = float(np.nanmin(scores)) >= -1e-6 and float(np.nanmax(scores)) <= 1 + 1e-6
    brier = brier_score(labels, scores) if is_probability else float("nan")

    auc_ci = bootstrap_ci_by_pair(
        labels, scores,
        metric_fn=lambda y, s: roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan"),
        n_boot=n_boot,
    )
    ap_ci = bootstrap_ci_by_pair(labels, scores, metric_fn=pr_auc, n_boot=n_boot)

    return {
        "name": curve.name,
        "n_rows": int(labels.size),
        "n_folds": len(curve.per_fold_auc),
        "auroc": auc,
        "auroc_ci95": auc_ci,
        "worst_fold_auroc": worst_fold,
        "pr_auc": ap,
        "pr_auc_ci95": ap_ci,
        "brier": brier,
        **tprs,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--detection", action="append", default=[],
        metavar="path:key:display_name",
        help="Pull one curve out of a detection.json (repeatable). key is "
        "'main_logistic' or an ablation name.",
    )
    p.add_argument(
        "--baselines", action="append", default=[],
        type=Path, metavar="path",
        help="Pull every curve out of a baselines.json (repeatable).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .md path (JSON side-car alongside).")
    p.add_argument(
        "--fpr-targets", type=float, nargs="+", default=[0.01, 0.05, 0.10],
        help="FPR operating points to report TPR at (default: 0.01 0.05 0.10).",
    )
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates (default: 2000).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fpr_targets = tuple(args.fpr_targets)

    curves: list[Curve] = []
    for spec in args.detection:
        curves.append(_load_detection_curve(spec))
    for path in args.baselines:
        curves.extend(_load_baseline_curves(path))

    if not curves:
        logger.error("No --detection or --baselines sources given; nothing to report.")
        return 1

    summaries = [summarize_curve(c, fpr_targets=fpr_targets, n_boot=args.bootstrap) for c in curves]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    fpr_cols = [f"tpr@fpr={f}" for f in fpr_targets]
    lines = ["# Security-metric report (TPR@low-FPR / PR-AUC / calibration / worst-case)", ""]
    header = ["Detector", "AUROC", "AUROC 95% CI", "Worst fold", "PR-AUC", "PR-AUC 95% CI", "Brier"] + fpr_cols
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for s in summaries:
        if "error" in s:
            lines.append(f"| {s['name']} | — | — | — | — | — | — | " + " | ".join("—" for _ in fpr_cols) + " | (" + s["error"] + ")")
            continue
        ci_lo, ci_hi = s["auroc_ci95"]
        ap_lo, ap_hi = s["pr_auc_ci95"]
        brier_str = f"{s['brier']:.3f}" if s["brier"] == s["brier"] else "N/A"
        row = [
            s["name"],
            f"{s['auroc']:.3f}",
            f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            f"{s['worst_fold_auroc']:.3f}",
            f"{s['pr_auc']:.3f}",
            f"[{ap_lo:.3f}, {ap_hi:.3f}]",
            brier_str,
        ] + [f"{s[c]:.3f}" for c in fpr_cols]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        f"Bootstrap: {args.bootstrap} replicates, pair-level resampling "
        f"(see `bootstrap_ci_by_pair` docstring). FPR targets: {list(fpr_targets)}."
    )

    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote {} (+ {})", args.out, json_path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
