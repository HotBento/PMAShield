"""Run the cross-attack transferability matrix on one model.

Prerequisites
-------------
For each attack listed in ``--attacks``, the following two paths must exist
(populated by the MCPTox capture + cluster pipeline):

    <base_dir>/<attack>/<model_safe>/attn_cache/      ← Stage-1 capture
    <base_dir>/<attack>/<model_safe>/selection_heads.json  ← Stage-3 selection

Usage
-----
    python -m mcp_eval.transferability.scripts.run_transfer \\
        --model Qwen/Qwen3-8B \\
        --attacks mcptox mpma injecagent \\
        --base-dir results \\
        --out results/transferability/Qwen_Qwen3-8B/transfer_report.md \\
        [--no-zscore]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from pma_shield.transferability import matrix as mat


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-attack transferability matrix runner."
    )
    p.add_argument(
        "--model", required=True, type=str,
        help="HuggingFace model id, e.g. Qwen/Qwen3-8B.",
    )
    p.add_argument(
        "--attacks", nargs="+", required=True,
        help="Attack short-names in the order they should appear in the matrix.",
    )
    p.add_argument(
        "--base-dir", type=Path, default=Path("results"),
        help="Root that contains per-attack subdirectories (default: results).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Output Markdown path. Defaults to "
            "<base-dir>/transferability/<model_safe>/transfer_report.md"
        ),
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional JSON sidecar with the raw numbers.",
    )
    p.add_argument(
        "--no-zscore", action="store_true",
        help="Disable per-attack benign-only z-score feature alignment.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    model_safe = args.model.replace("/", "_")
    base = args.base_dir

    bundles: dict[str, mat.AttackBundle] = {}
    for a in args.attacks:
        cap_dir = base / a / model_safe / "attn_cache"
        sel_path = base / a / model_safe / "selection_heads.json"
        if not (cap_dir / "manifest.json").exists():
            logger.error("Missing Stage-1 capture for attack {!r}: {}", a, cap_dir)
            return 1
        if not sel_path.exists():
            logger.error("Missing selection_heads.json for attack {!r}: {}", a, sel_path)
            return 1
        bundles[a] = mat.load_bundle(a, cap_dir, sel_path)
        logger.info(
            "Loaded bundle {!r}: capture from {}, {} selection heads",
            a, cap_dir, len(bundles[a].heads),
        )

    report = mat.run_transfer_matrix(
        bundles, align_zscore=not args.no_zscore, seed=args.seed,
    )

    out = args.out or (base / "transferability" / model_safe / "transfer_report.md")
    mat.write_report(report, out, model_id=args.model)

    json_out = args.json_out or out.with_suffix(".json")
    json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Raw numbers → {}", json_out)

    # Print a one-line summary for the operator.
    print("=" * 60)
    print(f"  Transfer report → {out}")
    print(f"  Diagonal AUC: {{ {', '.join(f'{a}={v:.3f}' for a, v in report.diagonal_auc.items())} }}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
