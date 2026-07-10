"""Emit a cross-model circuit-concentration comparison table (rebuttal, Nyqg W3).

Renders attention share / Gini / top-6 share / heads-for-half side by side
for every model with available Stage-1 patching artefacts, so the "same
qualitative circuit structure generalises across models" claim has a
tabular cross-model reference alongside the existing multi-model heatmap
figure (``fig_head_heatmap.plot_multi_model_heatmaps``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pma_shield.interp.concentration_stats import ConcentrationSummary


def render_concentration_table(
    summaries: Sequence[ConcentrationSummary],
    *,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"  \toprule")
    lines.append(
        r"  \textbf{Model} & \textbf{Attn.\ share} & \textbf{Gini} & "
        r"\textbf{Top-6 share} & \textbf{Heads for 50\%} & \textbf{$N_{\text{heads}}$} \\"
    )
    lines.append(r"  \midrule")
    for s in summaries:
        attn_s = f"{s.attn_share:.2f}" if s.attn_share == s.attn_share else "--"
        gini_s = f"{s.gini:.2f}" if s.gini == s.gini else "--"
        top6_s = f"{s.top6_share:.1%}".replace("%", r"\%") if s.top6_share == s.top6_share else "--"
        half_s = str(s.heads_for_half) if s.heads_for_half >= 0 else "--"
        n_s = str(s.n_heads_total) if s.n_heads_total > 0 else "--"
        lines.append(f"  {s.model_id} & {attn_s} & {gini_s} & {top6_s} & {half_s} & {n_s} \\\\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
