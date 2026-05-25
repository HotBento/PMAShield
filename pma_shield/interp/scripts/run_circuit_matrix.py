"""Compute the top-K head-pair circuit path-replacement matrix.

LOCAL / REMOTE: **REMOTE-ONLY** (GPU).

Reads ``head_importance.npz`` from a prior :mod:`run_patching` run, picks the
top-K heads, and computes the K×K path-replacement matrix used by
``fig:circuit-matrix`` (paper appendix).

Example::

    python -m mcp_eval.interp.scripts.run_circuit_matrix \
        --model Qwen/Qwen3-8B --top-k 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pma_shield.interp import head_roles, patching, probe_data
from pma_shield.interp.config import model_results_dir
from pma_shield.logger import logger, setup_file_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    out_dir = model_results_dir(args.model) / "patching"
    setup_file_logging(out_dir, run_name="circuit_matrix")

    imp_path = out_dir / "head_importance.npz"
    if not imp_path.exists():
        raise FileNotFoundError(
            f"{imp_path} not found — run run_patching.py first to discover heads."
        )
    importance = np.load(imp_path)["importance"]
    top = head_roles.top_heads(importance, k=args.top_k)
    logger.info("top heads: {}", top)

    from pma_shield.providers.llm import HuggingFaceProvider

    provider = HuggingFaceProvider(
        model_id=args.model,
        torch_dtype=args.torch_dtype,
        output_attentions=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    scenarios = probe_data.load_all()
    pairs = probe_data.make_pairs(scenarios)
    if args.limit:
        pairs = pairs[: args.limit]

    matrix = patching.head_pair_circuit_matrix(provider, pairs, top)
    head_labels = [f"L{l}H{h}" for l, h in top]
    np.savez(out_dir / "circuit_matrix.npz", matrix=matrix, head_labels=np.array(head_labels))
    logger.info("wrote {}", out_dir / "circuit_matrix.npz")


if __name__ == "__main__":
    main()
