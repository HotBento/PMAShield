"""Stage 1 CLI — run the model over MCPTox pairs and dump per-head features.

Usage::

    # Smoke run (first 100 pairs).
    python -m mcp_eval.mcptox.scripts.run_capture --limit 100

    # Full run with resume on by default.
    python -m mcp_eval.mcptox.scripts.run_capture

    # Batched run: 4 samples per generate call (≈2–3× faster on A100/H100).
    python -m mcp_eval.mcptox.scripts.run_capture --batch-size 4

    # Force a clean rerun (overwrites prior outputs).
    python -m mcp_eval.mcptox.scripts.run_capture --no-resume

Implementation notes
--------------------
* Single GPU is assumed. The provider is constructed inside ``capture.run``,
  so the model is only loaded once even when resuming.
* On Ctrl+C the process exits between chunk boundaries (no partial corruption)
  thanks to per-chunk atomic writes inside :func:`mcp_eval.mcptox.capture.run`.
* ``--batch-size`` must be even (one pair = benign + malicious = 2 samples).
  Odd values are rounded up automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pma_shield.logger import logger, setup_file_logging

from pma_shield.detector import capture, config, data


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MCPTox Stage-1 attention capture (Qwen3 / HuggingFaceProvider)."
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=config.MCPTOX_PAIRS_PATH,
        help="Pairs JSONL produced by Stage 0.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.ATTN_CACHE_DIR,
        help="Output directory for features.npy / meta.jsonl / topheads/.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL_ID,
        help="HuggingFace model id (default: %(default)s).",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optional override for layer count. Defaults to model-config auto-detect.",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Optional override for head count. Defaults to model-config auto-detect.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pairs (smoke runs).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh; overwrite existing outputs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of samples per model.generate() call in batched mode. "
            "Must be even (2 = one pair). Larger values trade memory for throughput. "
            "If batched capture hits CUDA OOM, the run automatically falls back "
            "to single-sample capture for that chunk."
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help=(
            "Maximum generation length for tool-call decode. "
            "Lower values reduce KV-cache and attention memory in long outputs. "
            "Default: 256."
        ),
    )
    p.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model with 8-bit quantization (saves ~75%% VRAM).",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model with 4-bit NF4 quantization (saves ~87%% VRAM).",
    )
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights. float16/bfloat16 reduce VRAM. Default: auto.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device mapping: 'auto', 'cuda', 'cpu', or device index. Default: auto.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Persist full run logs for post-hoc inspection.
    setup_file_logging(log_dir="log", run_name="mcptox_capture")

    if not args.pairs.exists():
        logger.error("Pairs JSONL not found: {}", args.pairs)
        logger.error("Run Stage 0 first:  python -m mcp_eval.mcptox.scripts.run_data")
        return 1

    pairs = data.load_pairs(args.pairs)
    
    # Build provider kwargs for memory optimization
    provider_kwargs = {
        "device": args.device,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
    }
    
    capture.run(
        pairs,
        out_dir=args.out,
        model_id=args.model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        limit=args.limit,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        provider_kwargs=provider_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
