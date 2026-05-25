"""Emit ``tab:head-roles`` (Finding 3) as a LaTeX booktabs fragment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


ROLE_DISPLAY: dict[str, str] = {
    "intent": "Intent heads",
    "tool_summary": "Tool summarisation heads",
    "intent_matching": "Intent-matching heads",
}

ROLE_SIGNATURE: dict[str, str] = {
    "intent": "Attention from user query tokens to sequence end",
    "tool_summary": "Concentrate on trailing tokens of individual tools",
    "intent_matching": "Route chosen tool's signal to the selection position",
}


def render_table(
    role_to_heads: dict[str, Sequence[tuple[int, int]]],
    *,
    out_path: Path,
) -> Path:
    """Write ``tab:head-roles`` LaTeX fragment to *out_path*.

    ``role_to_heads`` keys are role names; values are lists of ``(layer, head)``
    tuples — typically the top-N representatives of each role.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{p{2cm}p{3.2cm}p{2cm}}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Role} & \textbf{Signature} & \textbf{Example heads} \\")
    lines.append(r"  \midrule")
    for role in ("intent", "tool_summary", "intent_matching"):
        heads = role_to_heads.get(role, [])
        head_str = ", ".join(f"L{l}H{h}" for l, h in heads) or "--"
        display = ROLE_DISPLAY[role]
        signature = ROLE_SIGNATURE[role]
        lines.append(f"  {display} & {signature} & {head_str} \\\\")
        lines.append(r"  \addlinespace")
    lines = lines[:-1]  # drop trailing addlinespace
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
