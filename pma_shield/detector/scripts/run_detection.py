"""
Stage 5 CLI — leave-one-server-out attack-detection evaluation.

Usage
-----
    # Full run with all ablation controls.
    python -m mcp_eval.mcptox.scripts.run_detection \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox

    # Subset of controls (space-separated).
    python -m mcp_eval.mcptox.scripts.run_detection \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox \\
        --controls random_heads top6

    # Also write ROC-curve figures.
    python -m mcp_eval.mcptox.scripts.run_detection \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox \\
        --make-figures

Exit codes
----------
* 0 — main LOSO AUC > 0.5 (detector is better than chance).
* 1 — input missing, or main AUC ≤ 0.5.

Interpretation note (from the plan)
------------------------------------
If main AUC > control D (logit_baseline) by ≥ 0.05 the paper can claim
"attention disagreement carries signal beyond the model's own confidence."
The logit_baseline control requires Stage-1 to save per-step logits; until
then it is reported as ``not_implemented`` and excluded from the comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from pma_shield.detector import capture, config, detection, selection, viz

_ALL_CONTROLS = ("random_heads", "top6", "query_attn", "logit_baseline")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MCPTox Stage 5: LOSO attack detection + ablation controls."
    )
    p.add_argument(
        "--capture",
        type=Path,
        default=config.ATTN_CACHE_DIR,
        help="Stage-1 output directory (default: %(default)s).",
    )
    p.add_argument(
        "--selection",
        type=Path,
        default=config.CLUSTER_DIR / "selection_heads.json",
        help="Stage-3 selection_heads.json (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.DETECTION_DIR,
        help="Output directory for detection_report.md + .json (default: %(default)s).",
    )
    p.add_argument(
        "--controls",
        nargs="+",
        choices=_ALL_CONTROLS,
        default=list(_ALL_CONTROLS),
        metavar="CONTROL",
        help=(
            f"Ablation controls to run. Choices: {_ALL_CONTROLS}. "
            "Default: all four."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for LOSO logistic regression and random-heads ablation (default: 0).",
    )
    p.add_argument(
        "--make-figures",
        action="store_true",
        help="Write ROC-curve figures.",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory for figures (default: <out>/figures).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Load artifacts ──────────────────────────────────────────────────────
    if not (args.capture / "manifest.json").exists():
        logger.error("Stage-1 manifest not found: {}", args.capture / "manifest.json")
        logger.error(
            "Run Stage 1 first:  python -m mcp_eval.mcptox.scripts.run_capture"
        )
        return 1
    if not args.selection.exists():
        logger.error("selection_heads.json not found: {}", args.selection)
        logger.error(
            "Run Stage 2-3 first:  python -m mcp_eval.mcptox.scripts.run_cluster"
        )
        return 1

    logger.info("Loading Stage-1 dataset from {}", args.capture)
    dataset = capture.load(args.capture)
    logger.info("Loading selection heads from {}", args.selection)
    sel = selection.load(args.selection)
    logger.info(
        "Using {} selection heads (cluster {})", len(sel.heads), sel.cluster_id
    )

    # ── Stage 5 — evaluate ─────────────────────────────────────────────────
    # Resolve the diagnostic ("top6") head set from the captured dataset's
    # model_id so we never apply Qwen3-8B coordinates to a different model.
    captured_model_id = str(dataset.manifest.get("model_id", config.DEFAULT_MODEL_ID))
    known_core_heads = config.get_known_core_heads(captured_model_id)
    if "top6" in args.controls and not known_core_heads:
        logger.warning(
            "No KNOWN_CORE_HEADS registered for {} — 'top6' control will use an "
            "empty head set. Add the discovered heads to "
            "config.KNOWN_CORE_HEADS_BY_MODEL after running head analysis.",
            captured_model_id,
        )
    logger.info("Running detection evaluation (controls={}) …", args.controls)
    report = detection.evaluate(
        dataset,
        sel,
        controls=args.controls,
        seed=args.seed,
        known_core_heads=known_core_heads,
    )

    # ── Write report ────────────────────────────────────────────────────────
    report_path = args.out / "detection_report.md"
    detection.write_report(report, report_path)

    # ── Print one-line summary ──────────────────────────────────────────────
    main_auc = report.main_logistic.get("loso_auc_mean", float("nan"))
    print()
    print("=" * 60)
    print("  Stage 5 — Detection summary")
    print("=" * 60)
    print(f"  Head-set size (main): {report.head_set_sizes.get('main', '?')}")
    print()
    print("  Per-metric ROC AUC (single threshold):")
    for metric, auc in report.main_per_metric_auc.items():
        print(f"    {metric:<6} AUC = {auc:.4f}")
    print()
    print(f"  Joint logistic LOSO AUC:  {main_auc:.4f}")
    n_folds = len(report.main_logistic.get("loso_auc_per_fold", {}))
    print(f"  LOSO folds:               {n_folds}")
    print()
    if report.ablations:
        print("  Ablation controls:")
        for ctrl, stats in sorted(report.ablations.items()):
            if "loso_auc_mean" in stats:
                print(
                    f"    {ctrl:<18} joint LOSO AUC = {stats['loso_auc_mean']:.4f}"
                    + (
                        f"  Δ vs main = {stats['loso_auc_mean'] - main_auc:+.4f}"
                        if not (main_auc != main_auc)
                        else ""
                    )
                )
            elif "per_feature_auc" in stats:
                print(
                    f"    {ctrl:<18} per-feature AUC = {stats['per_feature_auc']:.4f}"
                )
            elif "note" in stats:
                print(f"    {ctrl:<18} {stats['note']}")
    print()
    print(f"  Report → {report_path}")
    print("=" * 60)

    # ── Optional figures ───────────────────────────────────────────────────
    if args.make_figures:
        fig_dir = args.fig_dir if args.fig_dir is not None else args.out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        viz.setup_style()
        _write_roc_curves(report, fig_dir)
        logger.info("Figures written → {}", fig_dir)

    # ── Exit code: success if main AUC > 0.5 ───────────────────────────────
    if main_auc != main_auc or main_auc <= 0.5:
        logger.warning("Main LOSO AUC ({:.4f}) ≤ 0.5 — detector is at chance.", main_auc)
        return 1
    return 0


def _write_roc_curves(report: detection.DetectionReport, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    try:
        ax = viz.roc_curves(
            report.main_per_metric_roc,
            report.main_per_metric_auc,
        )
        path = fig_dir / "roc_curves.png"
        ax.get_figure().savefig(path)
        plt.close(ax.get_figure())
        logger.info("Saved ROC curves → {}", path)
    except Exception as exc:
        logger.warning("Could not write ROC-curve figure: {}", exc)


if __name__ == "__main__":
    sys.exit(main())
