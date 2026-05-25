"""
Stage 4 CLI — compute disagreement metrics + paired Wilcoxon tests.

Usage
-----
    # Full run with default paths.
    python -m mcp_eval.mcptox.scripts.run_disagreement \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox

    # Also render strip-plot figures.
    python -m mcp_eval.mcptox.scripts.run_disagreement \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox \\
        --make-figures

    # Use a different significance level for exit code.
    python -m mcp_eval.mcptox.scripts.run_disagreement \\
        --capture results/mcptox/attn_cache \\
        --selection results/mcptox/selection_heads.json \\
        --out results/mcptox \\
        --alpha 0.05

Exit codes
----------
* 0 — all four primary Wilcoxon tests reject H0 at the requested α.
* 1 — input missing, or at least one primary test does NOT reject H0.

The exit-code semantics make it easy to detect regressions in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from pma_shield.detector import capture, config, disagreement, selection, viz


# Four primary tests: (metric, direction, human description)
_PRIMARY_TESTS: list[tuple[str, str, str]] = [
    ("dA",    "less",    "A drops  (more disagreement) under attack"),
    ("dE",    "greater", "E rises  (higher entropy)    under attack"),
    ("dD_JS", "greater", "D_JS rises (higher JS div)   under attack"),
    ("dO",    "less",    "O drops  (less output align) under attack"),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MCPTox Stage 4: disagreement metrics + Wilcoxon tests."
    )
    p.add_argument(
        "--capture",
        type=Path,
        default=config.ATTN_CACHE_DIR,
        help="Stage-1 output directory (default: %(default)s).",
    )
    p.add_argument(
        "--selection",
        type=Path,
        default=config.CLUSTER_DIR / "selection_heads.json",
        help="Stage-3 selection_heads.json (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.DISAGREEMENT_DIR,
        help="Output directory for disagreement.csv + disagreement_stats.md (default: %(default)s).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for exit-code check (default: 0.01).",
    )
    p.add_argument(
        "--by",
        default="risk_category",
        help="Column used for per-category breakdown (default: risk_category).",
    )
    p.add_argument(
        "--make-figures",
        action="store_true",
        help="Write Δmetric strip-plot figures.",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory for figures (default: <out>/figures).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Load artifacts ──────────────────────────────────────────────────────
    if not (args.capture / "manifest.json").exists():
        logger.error("Stage-1 manifest not found: {}", args.capture / "manifest.json")
        logger.error(
            "Run Stage 1 first:  python -m mcp_eval.mcptox.scripts.run_capture"
        )
        return 1
    if not args.selection.exists():
        logger.error("selection_heads.json not found: {}", args.selection)
        logger.error(
            "Run Stage 2-3 first:  python -m mcp_eval.mcptox.scripts.run_cluster"
        )
        return 1

    logger.info("Loading Stage-1 dataset from {}", args.capture)
    dataset = capture.load(args.capture)
    logger.info("Loading selection heads from {}", args.selection)
    sel = selection.load(args.selection)
    logger.info(
        "Using {} selection heads (cluster {})", len(sel.heads), sel.cluster_id
    )

    # ── Stage 4 — compute disagreement ─────────────────────────────────────
    logger.info("Computing per-pair disagreement metrics …")
    df = disagreement.compute(dataset, head_set=sel.heads)
    logger.info(
        "Computed {} rows ({} both-parsed pairs)",
        len(df),
        int((df["parse_ok_benign"] & df["parse_ok_mal"]).sum()),
    )

    csv_path = args.out / "disagreement.csv"
    disagreement.save_csv(df, csv_path)

    # ── Primary Wilcoxon tests ──────────────────────────────────────────────
    all_pass = True
    overall_results: list[dict] = []
    print()
    print(f"{'Metric':<8} {'Dir':<8} {'n':>6} {'median Δ':>10} {'p-value':>10} {'Cliff δ':>8}  hypothesis")
    print("-" * 75)
    for metric, direction, desc in _PRIMARY_TESTS:
        res = disagreement.paired_test(df, metric, direction=direction)
        overall_results.append(res)
        p = res["p_value"]
        reject = (not (p != p)) and p < args.alpha  # NaN-safe
        flag = "✓" if reject else "✗"
        print(
            f"{metric:<8} {direction:<8} {res['n']:>6} "
            f"{res['median_delta']:>10.4f} {p:>10.3g} "
            f"{res['cliffs_delta']:>8.3f}  {flag} {desc}"
        )
        if not reject:
            all_pass = False
    print()

    # ── Per-category breakdown ──────────────────────────────────────────────
    logger.info("Running per-category tests (by={}) …", args.by)
    per_cat_df = disagreement.per_category_tests(df, by=args.by)

    # ── Write Markdown summary ──────────────────────────────────────────────
    md_path = args.out / "disagreement_stats.md"
    disagreement.write_summary_md(
        df,
        overall_tests=overall_results,
        per_cat_df=per_cat_df,
        out=md_path,
    )

    print(f"CSV   → {csv_path}")
    print(f"Stats → {md_path}")
    if all_pass:
        print(f"\nAll 4 tests significant at α={args.alpha} — H confirmed.")
    else:
        print(f"\nWARNING: one or more tests did not reach α={args.alpha}.")

    # ── Optional figures ───────────────────────────────────────────────────
    if args.make_figures:
        fig_dir = args.fig_dir if args.fig_dir is not None else args.out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        viz.setup_style()
        _write_strip_plots(df, fig_dir)
        logger.info("Figures written → {}", fig_dir)

    return 0 if all_pass else 1


def _write_strip_plots(df, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for delta_col, _dir, _desc in _PRIMARY_TESTS:
        if delta_col not in df.columns:
            continue
        try:
            ax = viz.disagreement_paired_strip(df, metric=delta_col)
            path = fig_dir / f"strip_{delta_col}.png"
            ax.get_figure().savefig(path)
            plt.close(ax.get_figure())
            logger.info("Saved strip plot → {}", path)
        except Exception as exc:
            logger.warning("Could not write strip plot for {}: {}", delta_col, exc)


if __name__ == "__main__":
    sys.exit(main())
