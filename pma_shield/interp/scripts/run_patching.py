"""Run layer-level + head-level activation patching for one model.

LOCAL / REMOTE: **REMOTE-ONLY** (loads HuggingFace model on GPU).

Example
-------

Smoke (4 pairs)::

    python -m mcp_eval.interp.scripts.run_patching \
        --model Qwen/Qwen3-8B --limit 4 \
        --out results/interp/Qwen_Qwen3-8B/patching/

Full run (default ≈ 30 contrastive pairs)::

    python -m mcp_eval.interp.scripts.run_patching \
        --model Qwen/Qwen3-8B \
        --out results/interp/Qwen_Qwen3-8B/patching/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pma_shield.interp import patching, probe_data
from pma_shield.interp.config import model_results_dir
from pma_shield.logger import logger, setup_file_logging


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace model id.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/interp/<safe>/patching/).")
    p.add_argument("--limit", type=int, default=None,
                   help="Use at most this many pairs (smoke test).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pairs-per-combo", type=int, default=5,
                   help="Pairs drawn per (cat_i, cat_j) combination.")
    p.add_argument("--skip-head", action="store_true",
                   help="Run layer-level patching only.")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = args.out or (model_results_dir(args.model) / "patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(out_dir, run_name="patching")

    from pma_shield.providers.llm import HuggingFaceProvider

    logger.info("loading model {}", args.model)
    provider = HuggingFaceProvider(
        model_id=args.model,
        torch_dtype=args.torch_dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    scenarios = probe_data.load_all()
    pairs = probe_data.make_pairs(
        scenarios, seed=args.seed, max_pairs_per_combo=args.max_pairs_per_combo
    )
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info("running patching over {} pairs", len(pairs))

    def _progress(i: int, total: int) -> None:
        logger.info("  pair {} / {}", i + 1, total)

    layer_result = patching.layer_patching(provider, pairs, progress=_progress)
    np.savez(out_dir / "layer_attn_mlp.npz", **layer_result)
    logger.info("wrote {}", out_dir / "layer_attn_mlp.npz")

    if not args.skip_head:
        head_result = patching.head_patching(provider, pairs, progress=_progress)
        np.savez(out_dir / "head_importance.npz", importance=head_result)
        logger.info("wrote {}", out_dir / "head_importance.npz")


if __name__ == "__main__":
    sys.exit(main())
