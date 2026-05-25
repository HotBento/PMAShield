"""Classify top-K selection heads into intent / tool-summary / intent-matching.

LOCAL / REMOTE: **REMOTE-ONLY** (GPU forward pass to capture attention).

Reads ``head_importance.npz``, picks top heads, captures their attention
matrices on a few probe scenarios, scores each role component, and writes a
JSON file consumed by :mod:`mcp_eval.interp.figures.table_head_roles` and
:mod:`mcp_eval.interp.figures.fig_head_patterns`.

Example::

    python -m mcp_eval.interp.scripts.run_head_roles --model Qwen/Qwen3-8B --top-k 24
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pma_shield.interp import head_roles, patching, probe_data
from pma_shield.interp.config import model_results_dir
from pma_shield.logger import logger, setup_file_logging


def _find_span(tokens: list[int], needle_text: str, tokenizer) -> tuple[int, int] | None:
    ids = tokenizer.encode(needle_text, add_special_tokens=False)
    if not ids:
        return None
    for i in range(len(tokens) - len(ids) + 1):
        if tokens[i:i + len(ids)] == ids:
            return i, i + len(ids) - 1
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--top-k", type=int, default=24)
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--save-patterns",
        action="store_true",
        help="Also save one representative attention matrix per role to "
             "patterns/<role>_lLhH.npz for the fig:head-patterns figure.",
    )
    args = p.parse_args()

    res_dir = model_results_dir(args.model)
    out_dir = res_dir
    setup_file_logging(res_dir, run_name="head_roles")

    importance = np.load(res_dir / "patching" / "head_importance.npz")["importance"]
    top = head_roles.top_heads(importance, k=args.top_k)

    from pma_shield.providers.llm import HuggingFaceProvider

    provider = HuggingFaceProvider(
        model_id=args.model,
        torch_dtype=args.torch_dtype,
        output_attentions=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    scenarios = probe_data.load_all()
    sample_pool = [s for cat in scenarios.values() for s in cat[:3]]

    head_to_scores: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {"intent": 0.0, "tool_summary": 0.0, "intent_matching": 0.0}
    )
    repr_scenario = None   # first scenario with valid spans, used for pattern panels
    repr_sinfo = None

    for scenario in sample_pool:
        input_ids, commit, _ = patching._build_forced_inputs(provider, scenario)  # type: ignore[attr-defined]
        input_ids = input_ids.to(next(provider.model.parameters()).device)
        tokens = input_ids[0].tolist()
        uq_span = _find_span(tokens, scenario.query, provider.tokenizer)
        tool_spans = []
        chosen_idx = 0
        for i, t in enumerate(scenario.tools):
            span = _find_span(tokens, t["name"], provider.tokenizer)
            if span is None:
                continue
            tool_spans.append(span)
            if t["name"] == scenario.intended_tool:
                chosen_idx = len(tool_spans) - 1
        if uq_span is None or not tool_spans:
            continue
        sinfo = head_roles.SpanInfo(
            user_query=uq_span,
            tool_spans=tuple(tool_spans),
            chosen_tool_index=chosen_idx,
            commit_pos=commit,
        )
        if repr_scenario is None:
            repr_scenario, repr_sinfo = scenario, sinfo
        for head in top:
            attn = patching.capture_attention_pattern(provider, scenario, head)
            scores = head_roles.score_head(attn, sinfo)
            for k, v in scores.items():
                head_to_scores[head][k] += v

    # average and classify
    for head in head_to_scores:
        for k in head_to_scores[head]:
            head_to_scores[head][k] /= max(1, len(sample_pool))
    classification = head_roles.classify_heads(head_to_scores)

    out_path = out_dir / "head_roles.json"
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "top_k": args.top_k,
                "heads": [
                    {
                        "layer": l,
                        "head": h,
                        "role": classification[(l, h)],
                        "scores": head_to_scores[(l, h)],
                    }
                    for l, h in top
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote {}", out_path)

    if args.save_patterns:
        _save_patterns(provider, args, top, head_to_scores, classification,
                       repr_scenario, repr_sinfo, out_dir)


def _save_patterns(provider, args, top, head_to_scores, classification,
                   repr_scenario, repr_sinfo, out_dir) -> None:
    """Save attention matrices for **all** top-K heads.

    Produces ``patterns/<role>_lLhH.npz`` with keys ``attn``, ``head_label``,
    ``query_span`` and ``tool_spans`` for every head in *top*.  The full set is
    consumed by ``make_figures._do_all_head_patterns`` to build the appendix
    grid figure; ``make_figures._do_head_patterns`` still picks one file per
    role for the main-text three-panel figure.
    """
    if repr_scenario is None or repr_sinfo is None:
        logger.warning("no scenario with valid spans; skipping pattern capture")
        return
    patterns_dir = out_dir / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    for (l, h) in top:
        role = classification[(l, h)]
        attn = patching.capture_attention_pattern(provider, repr_scenario, (l, h))
        np.savez(
            patterns_dir / f"{role}_l{l}h{h}.npz",
            attn=attn,
            head_label=f"L{l}H{h}",
            query_span=np.array(repr_sinfo.user_query),
            tool_spans=np.array(repr_sinfo.tool_spans),
            chosen_tool_idx=np.array(repr_sinfo.chosen_tool_index),
        )
        logger.info("saved pattern {} -> L{}H{}", role, l, h)


if __name__ == "__main__":
    main()
