"""
Stage 3: Discover tool-selection heads via rule-based classification.

Reads Stage-1 captured features and applies the rule-based head-type
classifier (§4 Offline Phase).  Each head is assigned a type label:
  1 = selection   — argmax attention matches the model's selected tool
                    in ≥ τ_sel fraction of benign pairs
  2 = user_intent — mean attention on the user-query span ≥ τ_ui
  3 = tool_output — mean self-attention ≥ τ_to
  0 = other

A grid search over τ_sel sweeps {0.1, 0.15, …, 0.5} and picks the value
that maximises the number of selection-type heads (used in Appendix D).

LOCAL / REMOTE: REMOTE-ONLY (requires Stage-1 features on disk).

EXAMPLE (single model, smoke):
  python -m pma_shield.detector.scripts.run_selection \\
      --capture results/mcptox/Qwen_Qwen3-8B/attn_cache \\
      --out     results/mcptox/Qwen_Qwen3-8B \\
      --tau-sel 0.50 --tau-ui 0.25 --tau-to 0.25

EXAMPLE (grid search, writes tau_grid_results.json):
  python -m pma_shield.detector.scripts.run_selection \\
      --capture results/mcptox/Qwen_Qwen3-8B/attn_cache \\
      --out     results/mcptox/Qwen_Qwen3-8B \\
      --grid-search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from pma_shield.detector import capture as cap_mod
from pma_shield.detector import config
from pma_shield.detector import selection as sel_mod


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3: discover tool-selection heads."
    )
    p.add_argument(
        "--capture",
        type=Path,
        default=config.ATTN_CACHE_DIR,
        help="Stage-1 output directory (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.CLUSTER_DIR,
        help="Output directory for selection_heads.json (default: %(default)s).",
    )
    p.add_argument(
        "--tau-sel",
        type=float,
        default=0.50,
        help="Selection-rate threshold τ_sel (default: %(default)s).",
    )
    p.add_argument(
        "--tau-ui",
        type=float,
        default=0.25,
        help="User-intent attention threshold τ_ui (default: %(default)s).",
    )
    p.add_argument(
        "--tau-to",
        type=float,
        default=0.25,
        help="Tool-output self-attention threshold τ_to (default: %(default)s).",
    )
    p.add_argument(
        "--grid-search",
        action="store_true",
        help="Sweep τ_sel in {0.10, 0.15, …, 0.50} and write tau_grid_results.json.",
    )
    return p


def _run_single(
    captured: object,
    tau_sel: float,
    tau_ui: float,
    tau_to: float,
) -> sel_mod.HeadTypeMap:
    return sel_mod.compute_head_type_map(
        captured,
        selection_threshold=tau_sel,
        user_intent_threshold=tau_ui,
        tool_output_threshold=tau_to,
    )


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Stage-1 capture from {}", args.capture)
    captured = cap_mod.load_captured_dataset(Path(args.capture))

    if args.grid_search:
        tau_values = np.arange(0.10, 0.55, 0.05).round(2).tolist()
        results: list[dict] = []
        for tau in tau_values:
            hm = _run_single(captured, tau, args.tau_ui, args.tau_to)
            sel_heads = hm.heads_of_type(sel_mod.HEAD_TYPE_SELECTION)
            results.append({
                "tau_sel": round(tau, 2),
                "tau_ui": args.tau_ui,
                "tau_to": args.tau_to,
                "n_selection": len(sel_heads),
                "n_user_intent": len(hm.heads_of_type(sel_mod.HEAD_TYPE_USER_INTENT)),
                "n_tool_output": len(hm.heads_of_type(sel_mod.HEAD_TYPE_TOOL_OUTPUT)),
            })
            logger.info("τ_sel={:.2f}: {} selection heads", tau, len(sel_heads))
        grid_path = out_dir / "tau_grid_results.json"
        grid_path.write_text(json.dumps(results, indent=2))
        logger.info("Grid results written to {}", grid_path)
        # Use τ that maximises selection count
        best = max(results, key=lambda r: r["n_selection"])
        logger.info("Best τ_sel={} with {} selection heads", best["tau_sel"], best["n_selection"])
        args.tau_sel = best["tau_sel"]

    type_map = _run_single(captured, args.tau_sel, args.tau_ui, args.tau_to)
    sel_heads = type_map.heads_of_type(sel_mod.HEAD_TYPE_SELECTION)
    logger.info(
        "τ_sel={}, τ_ui={}, τ_to={} → {} selection heads",
        args.tau_sel, args.tau_ui, args.tau_to, len(sel_heads),
    )

    out_path = out_dir / "head_type_map.npz"
    sel_mod.save_head_type_map(type_map, out_path)
    logger.info("Head type map saved to {}", out_path)

    sel_path = out_dir / "selection_heads.json"
    selection_out = sel_mod.pick_selection_heads_from_type_map(
        type_map,
        tau_sel=args.tau_sel,
        tau_ui=args.tau_ui,
        tau_to=args.tau_to,
    )
    sel_mod.save(selection_out, sel_path)
    logger.info("Selection heads ({}) saved to {}", len(selection_out.heads), sel_path)


if __name__ == "__main__":
    main()
