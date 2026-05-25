"""Stage 0 CLI — clone-aware MCPTox loader → ``data/mcptox/pairs.jsonl``.

Usage::

    # Print high-level schema stats and exit (no JSONL produced).
    python -m mcp_eval.mcptox.scripts.run_data --inspect-schema

    # Build the full pairs.jsonl (1348 cases on the current MCPTox snapshot).
    python -m mcp_eval.mcptox.scripts.run_data

    # List all server names with pair counts and exit.
    python -m mcp_eval.mcptox.scripts.run_data --list-servers

    # Exclude specific servers (rebuild JSONL without them).
    python -m mcp_eval.mcptox.scripts.run_data --exclude Email Slack

    # Keep only specific servers.
    python -m mcp_eval.mcptox.scripts.run_data --include FileSystem GitHub

    # Smoke run — first 20 cases only.
    python -m mcp_eval.mcptox.scripts.run_data --limit 20 --out /tmp/pairs.jsonl

Exit codes
----------
* 0 — success.
* 1 — schema inspection or pair-build error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from loguru import logger

from pma_shield.detector import config, data


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MCPTox Stage-0 loader / pair builder.")
    p.add_argument(
        "--response-file",
        type=Path,
        default=config.MCPTOX_RESPONSE_FILE,
        help="Path to MCPTox response_all.json (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.MCPTOX_PAIRS_PATH,
        help="Output JSONL path (default: %(default)s).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only emit the first N pairs (smoke runs).",
    )
    p.add_argument(
        "--inspect-schema",
        action="store_true",
        help="Print schema stats and exit; do not build pairs.",
    )
    p.add_argument(
        "--on-error",
        choices=("raise", "skip"),
        default="raise",
        help="Behaviour when a single case fails parsing.",
    )
    p.add_argument(
        "--list-servers",
        action="store_true",
        help="Print all server names with pair counts and exit (no JSONL produced).",
    )
    filter_group = p.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--exclude",
        nargs="+",
        metavar="SERVER",
        default=None,
        help="Server name(s) to drop from the output. Case-sensitive.",
    )
    filter_group.add_argument(
        "--include",
        nargs="+",
        metavar="SERVER",
        default=None,
        help="Server name(s) to keep (all others are dropped). Case-sensitive.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.response_file.exists():
        logger.error("MCPTox response file not found: {}", args.response_file)
        logger.error(
            "Clone the benchmark first:  "
            "git clone https://github.com/zhiqiangwang4/MCPTox-Benchmark.git data/mcptox"
        )
        return 1

    if args.inspect_schema:
        info = data.inspect_schema(args.response_file)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    pairs_iter = data.build_pairs(args.response_file, on_error=args.on_error)
    if args.limit is not None:
        pairs_iter = islice(pairs_iter, args.limit)

    pairs = list(pairs_iter)
    if not pairs:
        logger.error("No pairs were emitted — check raw data.")
        return 1

    # --list-servers: print counts then exit (no JSONL produced)
    if args.list_servers:
        counts = data.list_servers(pairs)
        print(f"\n{'Server':<30} {'Pairs':>6}")
        print("-" * 38)
        for server, n in counts.items():
            print(f"{server:<30} {n:>6}")
        print("-" * 38)
        print(f"{'Total':<30} {sum(counts.values()):>6}")
        return 0

    # Server-name filtering
    try:
        pairs = data.filter_pairs(pairs, exclude=args.exclude, include=args.include)
    except ValueError as exc:
        logger.error("{}", exc)
        return 1

    if not pairs:
        logger.error("No pairs remain after filtering — check --exclude / --include.")
        return 1

    data.write_pairs_jsonl(pairs, args.out)

    # Print a small summary.
    shape_counts = Counter(p.attack_shape for p in pairs)
    risk_counts = Counter(p.benign.risk_category for p in pairs)
    server_counts = Counter(p.benign.mcp_server for p in pairs)
    print()
    print(f"Wrote {len(pairs)} pairs → {args.out}")
    print(f"Attack shapes: {dict(shape_counts)}")
    print(f"Servers ({len(server_counts)}): top-10 {server_counts.most_common(10)}")
    print(f"Risk categories ({len(risk_counts)}): {dict(risk_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
