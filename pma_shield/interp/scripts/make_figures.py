"""Assemble §3 + appendix figures from on-disk experiment artefacts.

LOCAL / REMOTE: **LOCAL** — pure I/O + matplotlib, no model loading. Run this
on your workstation after rsync-ing ``results/interp/`` and
``results/{mcptox,mpma,injecagent}/`` from the GPU server.

Output: ``figures/interp/*.pdf`` (paths consumed by the LaTeX source).

Expected inputs (per model)::

    results/interp/<SAFE>/patching/layer_attn_mlp.npz   {attn, mlp}      [primary only]
    results/interp/<SAFE>/patching/head_importance.npz  {importance}     [all]
    results/interp/<SAFE>/patching/circuit_matrix.npz   {matrix, head_labels}  [primary only]
    results/interp/<SAFE>/head_roles.json                                [primary only]
    results/interp/<SAFE>/patterns/{role}.npz           {attn, head_label,
                                                         query_span, tool_spans}  [primary only]
    results/{mcptox,mpma,injecagent}/<SAFE>/disagreement/points.npz  {r_bar, H}

Missing inputs are logged as warnings; the script still emits whatever can be
made. Use ``--make-mock`` to re-run the mock pipeline in addition (useful when
preparing a slides deck).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pma_shield.interp.config import (
    ALL_MODELS,
    INTERP_FIGURES_DIR,
    MODEL_DISPLAY,
    PRIMARY_MODEL,
    model_results_dir,
    model_safe,
)
from pma_shield.interp.figures.fig_attn_vs_mlp import plot_attn_vs_mlp
from pma_shield.interp.figures.fig_circuit_matrix import plot_circuit_matrix
from pma_shield.interp.figures.fig_disagreement import (
    plot_disagreement_multi_model,
    plot_disagreement_scatter,
)
from pma_shield.interp.figures.fig_head_heatmap import (
    plot_head_heatmap,
    plot_multi_model_heatmaps,
)
from pma_shield.interp.figures.fig_all_head_patterns import HeadPatternEntry, plot_all_head_patterns
from pma_shield.interp.figures.fig_head_patterns import PatternPanel, plot_head_patterns
from pma_shield.interp.figures.table_head_roles import render_table
from pma_shield.logger import logger


def _load_npz(path: Path):
    if not path.exists():
        logger.warning("missing {}", path)
        return None
    return dict(np.load(path, allow_pickle=True))


def _disagreement_points(attack: str, model_id: str) -> np.ndarray | None:
    base = Path("results") / attack / model_safe(model_id) / "disagreement" / "points.npz"
    data = _load_npz(base)
    if data is None:
        return None
    return np.column_stack([data["r_bar"], data["H"]])


def _benign_points(model_id: str) -> np.ndarray | None:
    # benign points are produced alongside any single attack run — pull from mcptox by default.
    base = Path("results/mcptox") / model_safe(model_id) / "disagreement" / "benign.npz"
    data = _load_npz(base)
    if data is None:
        return None
    return np.column_stack([data["r_bar"], data["H"]])


def _do_attn_vs_mlp(out_dir: Path) -> None:
    npz = _load_npz(model_results_dir(PRIMARY_MODEL) / "patching" / "layer_attn_mlp.npz")
    if npz is None:
        return
    plot_attn_vs_mlp(npz["attn"], npz["mlp"], out_dir=out_dir)


def _do_head_heatmap_primary(out_dir: Path) -> None:
    npz = _load_npz(model_results_dir(PRIMARY_MODEL) / "patching" / "head_importance.npz")
    if npz is None:
        return
    plot_head_heatmap(npz["importance"], out_dir=out_dir)


def _do_head_heatmap_multi(out_dir: Path) -> None:
    matrices: dict[str, np.ndarray] = {}
    for mid in ALL_MODELS:
        npz = _load_npz(model_results_dir(mid) / "patching" / "head_importance.npz")
        if npz is None:
            continue
        matrices[MODEL_DISPLAY[mid]] = npz["importance"]
    if matrices:
        plot_multi_model_heatmaps(matrices, out_dir=out_dir)


def _do_head_patterns(out_dir: Path) -> None:
    patterns_dir = model_results_dir(PRIMARY_MODEL) / "patterns"
    if not patterns_dir.exists():
        logger.warning("missing {}; skipping head-patterns figure", patterns_dir)
        return
    panels: list[PatternPanel] = []
    # tool_summary not found in top-24; only intent and intent_matching are shown.
    for role in ("intent", "intent_matching"):
        for path in sorted(patterns_dir.glob(f"{role}_*.npz")):
            data = dict(np.load(path, allow_pickle=True))
            attn = np.asarray(data["attn"])
            seq_len = attn.shape[0]

            # chosen_tool_idx: only meaningful for intent_matching heads.
            # Use saved value when available; otherwise infer from the tool span
            # with the highest commit-step attention sum.
            if role != "intent_matching":
                chosen_tool_idx = 0
            elif "chosen_tool_idx" in data:
                chosen_tool_idx = int(data["chosen_tool_idx"])
            elif "tool_spans" in data and len(data["tool_spans"]) > 0:
                commit = attn[-1, :]
                sums = [float(commit[s:e + 1].sum()) for s, e in data["tool_spans"]]
                chosen_tool_idx = int(np.argmax(sums))
            else:
                chosen_tool_idx = -1

            # Extend query_span end to seq_len-2 (the last token before the
            # commit token) so the annotation covers the full user turn.
            if "query_span" in data:
                qs = int(data["query_span"][0])
                query_span: tuple[int, int] | None = (qs, seq_len - 2)
            else:
                query_span = None

            panels.append(
                PatternPanel(
                    role=role,
                    head_label=str(data["head_label"]),
                    attn=attn,
                    highlight_query_span=query_span,
                    highlight_tool_spans=tuple(map(tuple, data["tool_spans"])) if "tool_spans" in data else (),
                    chosen_tool_idx=chosen_tool_idx,
                )
            )
            break
    if panels:
        plot_head_patterns(panels, out_dir=out_dir)


def _do_all_head_patterns(out_dir: Path) -> None:
    """Appendix grid: attention-pattern heatmap for every top-K head."""
    patterns_dir = model_results_dir(PRIMARY_MODEL) / "patterns"
    if not patterns_dir.exists():
        logger.warning("missing {}; skipping all-head-patterns grid", patterns_dir)
        return
    entries: list[HeadPatternEntry] = []
    for path in sorted(patterns_dir.glob("*.npz")):
        data = dict(np.load(path, allow_pickle=True))
        label = str(data["head_label"])   # e.g. "L29H11"
        role_tag = path.stem.rsplit("_", 1)[0]  # e.g. "intent_matching" from "intent_matching_l29h11"
        try:
            # Parse layer/head from file name suffix (lLhH)
            suffix = path.stem.rsplit("_l", 1)[1]  # e.g. "29h11"
            layer_s, head_s = suffix.split("h")
            layer, head = int(layer_s), int(head_s)
        except (IndexError, ValueError):
            logger.warning("cannot parse layer/head from {}", path.name)
            continue
        entries.append(
            HeadPatternEntry(
                layer=layer,
                head=head,
                role=role_tag,
                attn=np.asarray(data["attn"]),
            )
        )
    if entries:
        plot_all_head_patterns(entries, out_dir=out_dir)
    else:
        logger.warning("no pattern files found in {}", patterns_dir)


def _do_head_roles_table(out_dir: Path) -> None:
    path = model_results_dir(PRIMARY_MODEL) / "head_roles.json"
    if not path.exists():
        logger.warning("missing {}", path)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    role_to_heads: dict[str, list[tuple[int, int]]] = {"intent": [], "tool_summary": [], "intent_matching": []}
    for entry in data["heads"]:
        role = entry["role"]
        role_to_heads[role].append((entry["layer"], entry["head"]))
    role_to_heads = {k: v[:3] for k, v in role_to_heads.items()}
    render_table(role_to_heads, out_path=out_dir / "tab_head_roles.tex")


def _do_disagreement_main(out_dir: Path) -> None:
    benign = _benign_points(PRIMARY_MODEL)
    mcptox = _disagreement_points("mcptox", PRIMARY_MODEL)
    if benign is None or mcptox is None:
        return
    plot_disagreement_scatter(
        {"benign": benign, "MCPTox": mcptox},
        out_dir=out_dir,
    )


def _do_disagreement_multi(out_dir: Path) -> None:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for mid in ALL_MODELS:
        sub: dict[str, np.ndarray] = {}
        for attack_label in (("mcptox", "MCPTox"), ("mpma", "MPMA"), ("injecagent", "InjecAgent")):
            attack, label = attack_label
            arr = _disagreement_points(attack, mid)
            if arr is not None:
                sub[label] = arr
        b = _benign_points(mid)
        if b is not None:
            sub = {"benign": b, **sub}
        if sub:
            grouped[MODEL_DISPLAY[mid]] = sub
    if grouped:
        plot_disagreement_multi_model(grouped, out_dir=out_dir)


def _layer_of(label: str) -> int:
    """Parse the layer index from a head label like ``L32H5``."""
    return int(label[1:].split("H", 1)[0])


def _do_circuit_matrix(out_dir: Path) -> None:
    npz = _load_npz(model_results_dir(PRIMARY_MODEL) / "patching" / "circuit_matrix.npz")
    if npz is None:
        return
    labels = [str(x) for x in npz["head_labels"]]
    matrix = np.asarray(npz["matrix"])
    # Sort heads low -> high layer so the upper triangle is the causal
    # (upstream low-layer -> downstream high-layer) direction.
    order = sorted(range(len(labels)), key=lambda i: (_layer_of(labels[i]), labels[i]))
    labels = [labels[i] for i in order]
    matrix = matrix[np.ix_(order, order)]
    plot_circuit_matrix(matrix, labels, out_dir=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=INTERP_FIGURES_DIR)
    p.add_argument("--make-mock", action="store_true",
                   help="Also regenerate the mock preview figures.")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("writing figures to {}", out_dir.resolve())

    _do_attn_vs_mlp(out_dir)
    _do_head_heatmap_primary(out_dir)
    _do_head_patterns(out_dir)
    _do_all_head_patterns(out_dir)
    _do_head_roles_table(out_dir)
    _do_disagreement_main(out_dir)
    _do_head_heatmap_multi(out_dir)
    _do_disagreement_multi(out_dir)
    _do_circuit_matrix(out_dir)

    if args.make_mock:
        from pma_shield.interp.scripts import make_mock_figures
        make_mock_figures.main()


if __name__ == "__main__":
    main()
