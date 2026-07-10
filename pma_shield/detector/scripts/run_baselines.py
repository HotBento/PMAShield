"""
Rebuttal-baselines CLI — logit-margin / activation-probe / residual-OOD /
attention-to-tool-mass, evaluated under the same LOSO protocol as PMAShield.

Usage
-----
    # attention-to-tool-mass only (works off an ordinary Stage-1 capture,
    # no re-inference needed):
    python -m pma_shield.detector.scripts.run_baselines \\
        --capture results/mcptox/Qwen_Qwen3-8B/attn_cache \\
        --selection results/mcptox/Qwen_Qwen3-8B/selection_heads.json \\
        --out results/mcptox/Qwen_Qwen3-8B/baselines \\
        --baselines attention_to_tool_mass

    # All four (requires a Stage-1 capture run with --capture-extras):
    python -m pma_shield.detector.scripts.run_baselines \\
        --capture results/mcptox/Qwen_Qwen3-8B/attn_cache_extras \\
        --selection results/mcptox/Qwen_Qwen3-8B/selection_heads.json \\
        --out results/mcptox/Qwen_Qwen3-8B/baselines \\
        --baselines all

Output
------
``<out>/baselines.json`` — one entry per requested baseline with
``auc_mean``, ``auc_std``, ``auc_per_fold``, ``f1_best``, ``n``, plus the
pooled ROC curve for plotting alongside Table 4 (main baseline comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from pma_shield.detector import baselines, capture, config, selection


_ALL_BASELINES = ("attention_to_tool_mass", "logit_margin", "activation_probe", "residual_ood")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuttal baselines: logit-margin / activation-probe / "
        "residual-OOD / attention-to-tool-mass, under LOSO."
    )
    p.add_argument(
        "--capture", type=Path, default=config.ATTN_CACHE_DIR,
        help="Stage-1 output directory (default: %(default)s). Must have been "
        "run with --capture-extras for logit_margin/activation_probe/residual_ood.",
    )
    p.add_argument(
        "--selection", type=Path, default=config.CLUSTER_DIR / "selection_heads.json",
        help="Stage-3 selection_heads.json — only needed for attention_to_tool_mass "
        "(default: %(default)s).",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for baselines.json.",
    )
    p.add_argument(
        "--baselines", nargs="+", default=["all"],
        choices=list(_ALL_BASELINES) + ["all"],
        help="Which baseline(s) to run (default: all).",
    )
    p.add_argument(
        "--hidden-layer", type=int, default=None,
        help="Which captured hidden-state layer to use for activation_probe/"
        "residual_ood, if more than one was saved (default: the only/first one).",
    )
    p.add_argument(
        "--ood-method", choices=["mahalanobis", "robust"], default="mahalanobis",
        help="Covariance estimator for residual_ood (default: mahalanobis / LedoitWolf).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    requested = list(_ALL_BASELINES) if "all" in args.baselines else args.baselines

    if not (args.capture / "manifest.json").exists():
        logger.error("Stage-1 manifest not found: {}", args.capture / "manifest.json")
        return 1
    logger.info("Loading Stage-1 dataset from {}", args.capture)
    dataset = capture.load(args.capture)

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    if "attention_to_tool_mass" in requested:
        if not args.selection.exists():
            logger.error("--selection required for attention_to_tool_mass: {}", args.selection)
        else:
            sel = selection.load(args.selection)
            logger.info("attention_to_tool_mass: using {} selection heads", len(sel.heads))
            _, res = baselines.evaluate_attention_to_tool_mass(dataset, sel.heads)
            results["attention_to_tool_mass"] = _result_to_dict(res)
            logger.info(
                "attention_to_tool_mass: AUROC={:.3f} (n={})", res.auc_mean, res.n
            )

    if "logit_margin" in requested:
        try:
            _, res = baselines.evaluate_logit_margin(dataset)
            results["logit_margin"] = _result_to_dict(res)
            logger.info("logit_margin: AUROC={:.3f} (n={})", res.auc_mean, res.n)
        except ValueError as exc:
            logger.error("Skipping logit_margin: {}", exc)

    if "activation_probe" in requested:
        try:
            res = baselines.evaluate_activation_probe(dataset, layer=args.hidden_layer)
            results["activation_probe"] = _result_to_dict(res)
            logger.info("activation_probe: AUROC={:.3f} (n={})", res.auc_mean, res.n)
        except ValueError as exc:
            logger.error("Skipping activation_probe: {}", exc)

    if "residual_ood" in requested:
        try:
            res = baselines.evaluate_residual_ood(
                dataset, layer=args.hidden_layer, method=args.ood_method
            )
            results["residual_ood"] = _result_to_dict(res)
            logger.info("residual_ood: AUROC={:.3f} (n={})", res.auc_mean, res.n)
        except ValueError as exc:
            logger.error("Skipping residual_ood: {}", exc)

    out_path = args.out / "baselines.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote {}", out_path)

    print()
    print(f"{'Baseline':<24} {'AUROC':>8} {'std':>8} {'F1':>8} {'n':>6}")
    print("-" * 58)
    for name, res in results.items():
        print(
            f"{name:<24} {res['auc_mean']:>8.3f} {res['auc_std']:>8.3f} "
            f"{res['f1_best']:>8.3f} {res['n']:>6}"
        )
    return 0


def _result_to_dict(res: "baselines.LOSOResult") -> dict[str, Any]:
    return {
        "auc_mean": res.auc_mean,
        "auc_std": res.auc_std,
        "auc_per_fold": res.auc_per_fold,
        "f1_best": res.f1_best,
        "n": res.n,
        "roc_fpr": res.roc_fpr,
        "roc_tpr": res.roc_tpr,
        "pooled_scores": res.pooled_scores,
        "pooled_labels": res.pooled_labels,
    }


if __name__ == "__main__":
    sys.exit(main())
