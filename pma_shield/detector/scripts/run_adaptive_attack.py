"""
Adaptive-attacker robustness CLI (rebuttal, reviewer texz W2).

Usage
-----
    python -m pma_shield.detector.scripts.run_adaptive_attack \\
        --capture results/mcptox/Qwen_Qwen3-8B/attn_cache \\
        --selection results/mcptox/Qwen_Qwen3-8B/selection_heads.json \\
        --out results/mcptox/Qwen_Qwen3-8B/adaptive

Output
------
``<out>/adaptive_sweep.json`` — AUROC as a function of the fraction of
selection heads the simulated attacker successfully hijacks (1.0 = full
hijack, reproduces RQ1; smaller fractions model an attacker who limits
their footprint to evade the disagreement signal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from pma_shield.detector import adaptive_attack, capture, config, disagreement, selection


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive-attacker robustness sweep (texz W2).")
    p.add_argument(
        "--capture", type=Path, default=config.ATTN_CACHE_DIR,
        help="Stage-1 output directory (default: %(default)s).",
    )
    p.add_argument(
        "--selection", type=Path, default=config.CLUSTER_DIR / "selection_heads.json",
        help="Stage-3 selection_heads.json (default: %(default)s).",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for adaptive_sweep.json.",
    )
    p.add_argument(
        "--fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.1],
        help="Fractions of the selection-head set the attacker hijacks "
        "(default: 1.0 0.75 0.5 0.25 0.1). 1.0 should reproduce RQ1's AUROC.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not (args.capture / "manifest.json").exists():
        logger.error("Stage-1 manifest not found: {}", args.capture / "manifest.json")
        return 1
    if not args.selection.exists():
        logger.error("selection_heads.json not found: {}", args.selection)
        return 1

    logger.info("Loading Stage-1 dataset from {}", args.capture)
    dataset = capture.load(args.capture)
    logger.info("Loading selection heads from {}", args.selection)
    sel = selection.load(args.selection)
    logger.info("Using {} selection heads (cluster {})", len(sel.heads), sel.cluster_id)

    logger.info("Computing real (full-hijack) disagreement metrics for training folds …")
    df_real = disagreement.compute(dataset, head_set=sel.heads)

    logger.info("Running adaptive-attacker sweep over fractions {}", args.fractions)
    results = adaptive_attack.run_adaptive_sweep(
        dataset, sel, df_real, fractions=args.fractions,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "adaptive_sweep.json"
    payload = [
        {
            "k": r.k,
            "fraction": r.fraction,
            "auc_mean": r.auc_mean,
            "auc_std": r.auc_std,
            "auc_per_fold": r.auc_per_fold,
            "n": r.n,
        }
        for r in results
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote {}", out_path)

    print()
    print(f"{'fraction':>10} {'k':>6} {'AUROC':>8} {'std':>8} {'n':>6}")
    print("-" * 44)
    for r in results:
        print(f"{r.fraction:>10.2f} {r.k:>6d} {r.auc_mean:>8.3f} {r.auc_std:>8.3f} {r.n:>6d}")

    if results and abs(results[0].fraction - 1.0) < 1e-9:
        logger.info(
            "Sanity check: fraction=1.0 AUROC={:.3f} should match RQ1's "
            "reported AUROC for this model/attack (full hijack = real data).",
            results[0].auc_mean,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
