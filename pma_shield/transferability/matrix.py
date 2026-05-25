"""Cross-attack transferability matrix.

Given Stage-1 captures + Stage-3 selection-head files for ``N`` attacks on
the same model, this module computes:

* **Diagonal (baseline)** — train + eval on each attack's own disagreement
  features under its own selection heads.  AUC measured via the same
  leave-one-server-out logistic regression that ``mcptox.detection`` uses.
* **4a — detector transfer** — train LR on attack ``X``'s features (under
  ``X``'s heads); evaluate AUC on attack ``Y``'s features (under ``Y``'s
  heads).  Only the LR coefficients move.
* **4b — head-set transfer** — recompute attack ``Y``'s disagreement
  features under attack ``X``'s heads, then train + eval LR on that
  re-featured dataset.
* **4c — joint detector** — pool features from all attacks, evaluate
  under leave-one-attack-out.

The single public entry point is :func:`run_transfer_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger


METRIC_COLS: tuple[str, ...] = ("A", "E", "D_JS", "O")


# ─────────────────────────────────────────────────────────────────────────────
# Capture / selection loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttackBundle:
    """One attack's loaded artefacts."""
    name:        str                # e.g. "mcptox" / "mpma" / "injecagent"
    captured:    Any                # mcptox.capture.CapturedDataset
    selection:   Any                # mcptox.selection.SelectionHeads
    heads:       list[tuple[int, int]] = field(default_factory=list)


def load_bundle(name: str, capture_dir: Path, selection_path: Path) -> AttackBundle:
    """Load one attack's :class:`AttackBundle` from disk.

    Parameters
    ----------
    name
        Short identifier — labels the rows of the output matrix.
    capture_dir
        Stage-1 output directory containing ``features.npy``, ``meta.jsonl``,
        ``manifest.json``.
    selection_path
        Path to the attack's ``selection_heads.json``.
    """
    from pma_shield.detector import capture as mcap, selection as msel
    captured = mcap.load(capture_dir)
    sel = msel.load(selection_path)
    return AttackBundle(
        name=name, captured=captured, selection=sel,
        heads=[tuple(h) for h in sel.heads],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disagreement → long-form rows
# ─────────────────────────────────────────────────────────────────────────────

def _disagreement_df(bundle: AttackBundle, heads: Sequence[tuple[int, int]]) -> pd.DataFrame:
    """Run ``mcptox.disagreement.compute`` on ``bundle.captured`` under
    the requested head set.  Filters out pairs with parse failures so the
    AUC computation has a non-degenerate label."""
    from pma_shield.detector import disagreement as mdis
    df = mdis.compute(bundle.captured, head_set=list(heads))
    df = df[df["parse_ok_benign"] & df["parse_ok_mal"]].reset_index(drop=True)
    return df


def _melt_long(df: pd.DataFrame, attack_label: str | None = None) -> pd.DataFrame:
    """Wide pair-level df → long sample-level df (benign and malicious
    on separate rows).  Output columns:
    ``y, group, attack, A, E, D_JS, O, sample_id``.
    """
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for side, label in (("benign", 0), ("mal", 1)):
            entry: dict[str, Any] = {
                "y": label,
                "group": r.get("mcp_server", "Unknown"),
                "attack": attack_label,
                "sample_id": r.get("sample_id"),
            }
            for col in METRIC_COLS:
                entry[col] = r.get(f"{col}_{side}", float("nan"))
            rows.append(entry)
    out = pd.DataFrame(rows).dropna(subset=list(METRIC_COLS))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature alignment (per-attack z-score on benign rows)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZScoreStats:
    """Mean + std of each feature, fitted on benign-only rows of one attack."""
    mean: np.ndarray   # shape (len(METRIC_COLS),)
    std:  np.ndarray


def fit_zscore(long_df: pd.DataFrame) -> ZScoreStats:
    """Fit z-score on the benign rows only — keeps the malicious-side shift
    signal intact while normalising the *distributional* scale of the four
    metrics.  This is what makes cross-attack LR meaningful when one attack
    has 2-tool pairs and another has 5+ (different entropy dynamic range)."""
    benign = long_df[long_df["y"] == 0]
    if len(benign) < 5:
        return ZScoreStats(mean=np.zeros(len(METRIC_COLS)), std=np.ones(len(METRIC_COLS)))
    arr = benign[list(METRIC_COLS)].to_numpy(dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return ZScoreStats(mean=mean, std=std)


def apply_zscore(long_df: pd.DataFrame, stats: ZScoreStats) -> pd.DataFrame:
    """Z-score the four metric columns *in place* on a copy."""
    out = long_df.copy()
    arr = out[list(METRIC_COLS)].to_numpy(dtype=np.float64)
    arr = (arr - stats.mean[None, :]) / stats.std[None, :]
    for i, c in enumerate(METRIC_COLS):
        out[c] = arr[:, i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Within-attack baseline (LOSO logistic; mirrors detection._loso_logistic_auc)
# ─────────────────────────────────────────────────────────────────────────────

def _loso_logistic(long_df: pd.DataFrame, *, seed: int = 0) -> float:
    """Mean leave-one-server-out AUC on the long-form df.  Returns NaN
    when there are too few groups or rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if len(long_df) < 20 or long_df["group"].nunique() < 2:
        return float("nan")
    X = long_df[list(METRIC_COLS)].to_numpy(dtype=np.float64)
    y = long_df["y"].to_numpy(dtype=np.int64)
    g = long_df["group"].to_numpy()

    aucs: list[float] = []
    for held in sorted(set(g)):
        train_mask = g != held
        test_mask = ~train_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X[train_mask], y[train_mask])
        proba = clf.predict_proba(X[test_mask])[:, 1]
        aucs.append(float(roc_auc_score(y[test_mask], proba)))
    return float(np.mean(aucs)) if aucs else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attack detector transfer (4a + 4b)
# ─────────────────────────────────────────────────────────────────────────────

def _train_eval_lr(
    train_long: pd.DataFrame,
    test_long:  pd.DataFrame,
    *,
    seed: int = 0,
) -> float:
    """Fit LR on ``train_long``; return AUC on ``test_long``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if len(train_long) < 10 or len(test_long) < 4:
        return float("nan")
    if train_long["y"].nunique() < 2 or test_long["y"].nunique() < 2:
        return float("nan")
    X_tr = train_long[list(METRIC_COLS)].to_numpy(dtype=np.float64)
    y_tr = train_long["y"].to_numpy(dtype=np.int64)
    X_te = test_long[list(METRIC_COLS)].to_numpy(dtype=np.float64)
    y_te = test_long["y"].to_numpy(dtype=np.int64)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ─────────────────────────────────────────────────────────────────────────────
# Joint detector (4c) — leave-one-attack-out
# ─────────────────────────────────────────────────────────────────────────────

def _leave_one_attack_out(pooled_long: pd.DataFrame, *, seed: int = 0) -> dict[str, float]:
    """LOAO AUC: train on all attacks except ``A``, evaluate on ``A``."""
    out: dict[str, float] = {}
    attacks = sorted(pooled_long["attack"].dropna().unique())
    for held in attacks:
        train = pooled_long[pooled_long["attack"] != held]
        test  = pooled_long[pooled_long["attack"] == held]
        out[held] = _train_eval_lr(train, test, seed=seed)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Head-set Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> float:
    """Jaccard similarity of two head sets, treating each (L, H) as atomic."""
    sa, sb = set(map(tuple, a)), set(map(tuple, b))
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransferReport:
    """Output of :func:`run_transfer_matrix`."""
    attacks:           list[str]
    diagonal_auc:      dict[str, float]                    # per-attack baseline
    transfer_4a:       dict[str, dict[str, float]]          # train_attack → test_attack → AUC
    transfer_4b:       dict[str, dict[str, float]]
    joint_loao_auc:    dict[str, float]                    # held-out attack → AUC
    head_jaccard:      dict[str, dict[str, float]]          # head-set similarity
    n_pairs:           dict[str, int]                       # per-attack pair count after parse filter

    def to_dict(self) -> dict[str, Any]:
        return {
            "attacks":        self.attacks,
            "diagonal_auc":   self.diagonal_auc,
            "transfer_4a":    self.transfer_4a,
            "transfer_4b":    self.transfer_4b,
            "joint_loao_auc": self.joint_loao_auc,
            "head_jaccard":   self.head_jaccard,
            "n_pairs":        self.n_pairs,
        }


def run_transfer_matrix(
    bundles: Mapping[str, AttackBundle],
    *,
    align_zscore: bool = True,
    seed: int = 0,
) -> TransferReport:
    """Run the full transferability evaluation.

    Parameters
    ----------
    bundles
        Ordered mapping ``{attack_name → AttackBundle}``.
    align_zscore
        Whether to fit per-attack benign-only z-score statistics and
        normalise features before any cross-attack LR.  Required when the
        attacks differ in pair-tool-count (e.g. InjecAgent has 2-tool
        pairs while MCPTox has 5+).
    seed
        Reproducibility seed for LR + folds.
    """
    attacks = list(bundles.keys())
    # 1. Compute long-form features under each attack's own heads (used for
    #    mode 4a and the diagonal).
    long_own: dict[str, pd.DataFrame] = {}
    zstats:  dict[str, ZScoreStats] = {}
    n_pairs: dict[str, int] = {}
    for a in attacks:
        b = bundles[a]
        df_wide = _disagreement_df(b, b.heads)
        long = _melt_long(df_wide, attack_label=a)
        n_pairs[a] = int(len(df_wide))
        zstats[a] = fit_zscore(long)
        long_own[a] = apply_zscore(long, zstats[a]) if align_zscore else long
        logger.info(
            "[{}] {} pairs, {} long rows after parse filter, {} groups",
            a, n_pairs[a], len(long_own[a]),
            long_own[a]["group"].nunique(),
        )

    # 2. Diagonal AUC (LOSO within each attack, under own heads).
    diagonal: dict[str, float] = {
        a: _loso_logistic(long_own[a], seed=seed) for a in attacks
    }
    logger.info("Diagonal LOSO AUC: {}", {a: f"{v:.4f}" for a, v in diagonal.items()})

    # 3. 4a — detector transfer (own heads on both sides; cross-attack LR).
    transfer_4a: dict[str, dict[str, float]] = {}
    for a_train in attacks:
        transfer_4a[a_train] = {}
        for a_test in attacks:
            if a_train == a_test:
                # Use the diagonal LOSO AUC for the cell (more honest than
                # in-sample LR which would be ≈1.0).
                transfer_4a[a_train][a_test] = diagonal[a_test]
            else:
                transfer_4a[a_train][a_test] = _train_eval_lr(
                    long_own[a_train], long_own[a_test], seed=seed,
                )

    # 4. 4b — head-set transfer.  For each (train_attack, test_attack) pair,
    #    recompute the test attack's features under the *train attack's* heads.
    transfer_4b: dict[str, dict[str, float]] = {}
    for a_train in attacks:
        transfer_4b[a_train] = {}
        train_heads = bundles[a_train].heads
        for a_test in attacks:
            test_bundle = bundles[a_test]
            # Skip empty head sets — would produce NaN features anyway.
            if not train_heads:
                transfer_4b[a_train][a_test] = float("nan")
                continue
            df_wide = _disagreement_df(test_bundle, train_heads)
            long = _melt_long(df_wide, attack_label=a_test)
            if align_zscore:
                # Fit a fresh z-score on this re-featured df's benign rows.
                zs = fit_zscore(long)
                long = apply_zscore(long, zs)
            if a_train == a_test:
                # In-attack diagonal: LOSO within attack (honest).
                transfer_4b[a_train][a_test] = _loso_logistic(long, seed=seed)
            else:
                transfer_4b[a_train][a_test] = _train_eval_lr(
                    long_own[a_train], long, seed=seed,
                )

    # 5. 4c — joint detector, leave-one-attack-out.
    pooled = pd.concat([long_own[a] for a in attacks], ignore_index=True)
    joint = _leave_one_attack_out(pooled, seed=seed)
    logger.info("LOAO AUC: {}", {a: f"{v:.4f}" for a, v in joint.items()})

    # 6. Head-set Jaccard.
    jacc: dict[str, dict[str, float]] = {}
    for a in attacks:
        jacc[a] = {b: _jaccard(bundles[a].heads, bundles[b].heads) for b in attacks}

    return TransferReport(
        attacks=attacks,
        diagonal_auc=diagonal,
        transfer_4a=transfer_4a,
        transfer_4b=transfer_4b,
        joint_loao_auc=joint,
        head_jaccard=jacc,
        n_pairs=n_pairs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown writer
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_matrix(table: dict[str, dict[str, float]], attacks: Sequence[str]) -> str:
    """Render an N×N AUC matrix as a Markdown table."""
    head = "| train \\ test |" + "".join(f" {a} |" for a in attacks)
    sep  = "|" + "---|" * (len(attacks) + 1)
    rows = [head, sep]
    for a_train in attacks:
        cells = [f"{table[a_train].get(a_test, float('nan')):.3f}" for a_test in attacks]
        rows.append(f"| **{a_train}** |" + "".join(f" {c} |" for c in cells))
    return "\n".join(rows)


def write_report(report: TransferReport, out_path: Path, *, model_id: str = "?") -> Path:
    """Write a human-readable Markdown summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = report.attacks
    parts: list[str] = []
    parts.append(f"# Cross-attack transferability — model `{model_id}`\n")
    parts.append("## Setup\n")
    parts.append("| attack | n_pairs (parse_ok) | |H_select| |")
    parts.append("|---|---|---|")
    # We don't have the head counts in the report directly; show them as
    # extracted from the diagonal cells.
    for a in A:
        n = report.n_pairs.get(a, 0)
        parts.append(f"| {a} | {n} | — |")
    parts.append("")
    parts.append("## Diagonal — within-attack LOSO AUC (own heads)\n")
    for a in A:
        parts.append(f"- **{a}**: {report.diagonal_auc[a]:.4f}")
    parts.append("")
    parts.append("## 4a — detector transfer (own heads, cross-attack LR)\n")
    parts.append(_fmt_matrix(report.transfer_4a, A))
    parts.append("")
    parts.append("## 4b — head-set transfer (train-attack heads applied to test data)\n")
    parts.append(_fmt_matrix(report.transfer_4b, A))
    parts.append("")
    parts.append("## 4c — joint detector, leave-one-attack-out\n")
    for a, v in report.joint_loao_auc.items():
        parts.append(f"- held out **{a}**: AUC = {v:.4f}")
    parts.append("")
    parts.append("## Head-set Jaccard similarity\n")
    parts.append(_fmt_matrix(report.head_jaccard, A))
    parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Transfer report → {}", out_path)
    return out_path
