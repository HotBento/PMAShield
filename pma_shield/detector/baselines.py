"""
Rebuttal baselines — white-box / gray-box alternatives to head-disagreement.

Reviewer 3Bi4 (ACL 2026 submission 7367) asked for stronger alternatives to
Random / Perplexity / LLM-Guard, specifically naming: "logit-margin
detectors, activation probes, residual-stream OOD detectors, or simple
attention-to-tool-mass baselines." This module implements exactly those
four, reusing the Stage-1/3/4/5 infrastructure in this package so that all
four baselines are evaluated under the identical Leave-One-Server-Out (LOSO)
protocol as PMAShield itself (see ``detection.py``).

Data requirements
------------------
* **attention_to_tool_mass**: needs only an ordinary Stage-1 capture
  (``features.npy`` / ``meta.jsonl``) — no new inference. Uses the
  ``concentration`` scalar feature (per-head "how much of this head's
  tool-name attention lands on a single tool") already computed by
  ``features.extract_all_heads``, averaged over a head set, WITHOUT the
  cross-head disagreement structure PMAShield relies on. This isolates
  whether disagreement (vs. raw single-head focus) is doing the work.
* **logit_margin**, **activation_probe**, **residual_ood**: need a Stage-1
  capture run with ``capture_extras=True`` (see ``capture.run``), which
  additionally persists commit-step logits (``meta.jsonl`` fields
  ``commit_logit_top1`` / ``commit_logit_top2``) and commit-step hidden
  states (``topheads/pair_<NNNN>.npz`` key ``commit_hidden_*``,
  see ``CapturedDataset.commit_hidden``).

Design note
-----------
``activation_probe`` and ``residual_ood`` both read the *same* hidden-state
vectors but answer different questions:
  - activation_probe: supervised — can a linear classifier trained on
    benign/attacked hidden states (LOSO) separate them?
  - residual_ood: unsupervised per fold — fit only on the training folds'
    *benign* hidden states, score the held-out server (both benign and
    malicious) by distance from that benign distribution. This is the
    "true" OOD framing: it never sees an attacked example during fitting,
    unlike every other detector in this paper (including PMAShield itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .features import feature_layout


# ──────────────────────────────────────────────────────────────────────────
# Generic LOSO logistic-regression harness (X/y/groups form)
#
# detection.py's `_loso_logistic_auc` takes a wide per-metric DataFrame and
# is specialised to the 4 named disagreement columns. The baselines here
# need to feed either a single scalar column (logit-margin,
# attention-to-tool-mass) or a full hidden-state vector (activation probe),
# so we work directly with (X, y, groups) instead of re-deriving column
# names, and keep the fold protocol (fit LR per held-out server, threshold
# tuned on training folds only) identical to detection.py for comparability.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class LOSOResult:
    auc_mean: float
    auc_std: float
    auc_per_fold: dict[str, float]
    f1_best: float
    n: int
    roc_fpr: list[float]
    roc_tpr: list[float]
    # Raw pooled out-of-fold (score, label) pairs, for exact downstream
    # TPR@low-FPR / PR-AUC / bootstrap-CI computation — see
    # scripts/report_security_metrics.py. Empty unless the caller populated
    # them (all functions in this module do).
    pooled_scores: list[float] = None  # type: ignore[assignment]
    pooled_labels: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.pooled_scores is None:
            self.pooled_scores = []
        if self.pooled_labels is None:
            self.pooled_labels = []


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    order = np.argsort(scores)
    best_f1, best_thr = -1.0, float("nan")
    candidates = np.unique(scores)
    if candidates.size == 0:
        return float("nan"), float("nan")
    for thr in candidates:
        pred = (scores >= thr).astype(np.int64)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def loso_logistic_from_Xy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 0,
    standardize: bool = True,
) -> LOSOResult:
    """LOSO logistic regression AUC/F1 on an arbitrary feature matrix.

    Same fold protocol as ``detection._loso_logistic_auc``: per held-out
    server, fit on the rest, tune the F1 threshold on the training fold
    only, evaluate AUROC + thresholded F1 on the held-out fold.

    Parameters
    ----------
    X
        ``(N, D)`` feature matrix (D=1 for scalar baselines, D=d_model for
        the activation probe).
    y
        ``(N,)`` binary labels (1 = malicious).
    groups
        ``(N,)`` server ids for LOSO grouping.
    standardize
        Z-score each feature using the training fold's mean/std (recommended
        for the activation probe, harmless for scalar baselines).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y, groups = X[mask], y[mask], groups[mask]

    nan_result = LOSOResult(
        auc_mean=float("nan"), auc_std=float("nan"), auc_per_fold={},
        f1_best=float("nan"), n=int(mask.sum()), roc_fpr=[], roc_tpr=[],
    )
    if len(y) < 20 or len(set(groups.tolist())) < 2:
        return nan_result

    aucs: dict[str, float] = {}
    f1s: list[float] = []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []

    for held in sorted(set(groups.tolist())):
        train_mask = groups != held
        test_mask = ~train_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        if standardize:
            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
            X_train = (X_train - mu) / sigma
            X_test = (X_test - mu) / sigma

        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X_train, y[train_mask])

        test_proba = clf.predict_proba(X_test)[:, 1]
        aucs[str(held)] = float(roc_auc_score(y[test_mask], test_proba))

        train_proba = clf.predict_proba(X_train)[:, 1]
        _, thr = _best_f1_threshold(y[train_mask], train_proba)
        if not np.isnan(thr):
            pred = (test_proba >= thr).astype(np.int64)
            f1s.append(float(f1_score(y[test_mask], pred, zero_division=0)))

        pooled_scores.extend(test_proba.tolist())
        pooled_labels.extend(y[test_mask].tolist())

    if not aucs:
        return nan_result

    pooled_scores_arr = np.asarray(pooled_scores)
    pooled_labels_arr = np.asarray(pooled_labels)
    fpr: list[float] = []
    tpr: list[float] = []
    if len(np.unique(pooled_labels_arr)) == 2:
        fpr_arr, tpr_arr, _ = roc_curve(pooled_labels_arr, pooled_scores_arr)
        fpr, tpr = fpr_arr.tolist(), tpr_arr.tolist()

    return LOSOResult(
        auc_mean=float(np.mean(list(aucs.values()))),
        auc_std=float(np.std(list(aucs.values()), ddof=0)),
        auc_per_fold=aucs,
        f1_best=float(np.mean(f1s)) if f1s else float("nan"),
        n=int(len(y)),
        roc_fpr=fpr,
        roc_tpr=tpr,
        pooled_scores=pooled_scores_arr.tolist(),
        pooled_labels=pooled_labels_arr.tolist(),
    )


# ──────────────────────────────────────────────────────────────────────────
# Baseline 1 — attention-to-tool-mass (no disagreement structure)
# ──────────────────────────────────────────────────────────────────────────

def attention_to_tool_mass_df(captured: Any, head_set: Sequence[tuple[int, int]]) -> pd.DataFrame:
    """Per-pair mean single-head tool-attention concentration.

    Uses the ``concentration`` scalar (max tool-name attention / total
    tool-name attention for that head, Eq. in features.py) averaged over
    ``head_set``, computed independently per side — deliberately *not*
    comparing heads against each other, to isolate whether disagreement
    (rather than raw attentional focus) is what PMAShield's gain comes from.

    Requires only a standard Stage-1 capture (no ``capture_extras``).
    """
    layout = feature_layout()
    feats_b = np.asarray(captured.benign_features)
    feats_m = np.asarray(captured.malicious_features)
    layers = np.array([h[0] for h in head_set])
    heads = np.array([h[1] for h in head_set])
    if len(layers) == 0:
        raise ValueError("empty head_set passed to attention_to_tool_mass_df")

    ccol = layout["concentration"]
    mass_b = np.nanmean(feats_b[:, layers, heads, ccol], axis=1)
    mass_m = np.nanmean(feats_m[:, layers, heads, ccol], axis=1)

    meta_by_pair: dict[int, dict[str, Any]] = {
        int(e["pair_idx"]): e for e in captured.meta if e["side"] == "benign"
    }
    rows = []
    for p in range(feats_b.shape[0]):
        meta = meta_by_pair.get(p, {})
        rows.append({
            "pair_idx": p,
            "mcp_server": meta.get("mcp_server"),
            "risk_category": meta.get("risk_category"),
            "mass_benign": float(mass_b[p]) if not np.isnan(mass_b[p]) else float("nan"),
            "mass_mal": float(mass_m[p]) if not np.isnan(mass_m[p]) else float("nan"),
            "dmass": float(mass_m[p] - mass_b[p]),
        })
    return pd.DataFrame(rows)


def evaluate_attention_to_tool_mass(
    captured: Any, head_set: Sequence[tuple[int, int]], *, seed: int = 0
) -> tuple[pd.DataFrame, LOSOResult]:
    df = attention_to_tool_mass_df(captured, head_set)
    X, y, groups = _wide_df_to_Xy(df, ["mass"])
    return df, loso_logistic_from_Xy(X, y, groups, seed=seed, standardize=False)


# ──────────────────────────────────────────────────────────────────────────
# Baseline 2 — logit-margin detector
# ──────────────────────────────────────────────────────────────────────────

def logit_margin_df(captured: Any) -> pd.DataFrame:
    """Per-pair commit-step top1-vs-runner-up vocab logit margin.

    Requires a capture run with ``capture_extras=True`` (populates
    ``commit_logit_top1`` / ``commit_logit_top2`` in ``meta.jsonl``).
    A small margin means the model was less "sure" about which token (tool)
    to commit to; PMAs are hypothesised to *reduce* this margin on the
    poisoned side even when the model still emits the malicious tool,
    because the runner-up logit (often the correct tool) stays competitive.
    """
    meta_by_pair_side: dict[tuple[int, str], dict[str, Any]] = {
        (int(e["pair_idx"]), e["side"]): e for e in captured.meta
    }
    n_pairs = int(captured.manifest["n_pairs"])
    rows = []
    for p in range(n_pairs):
        eb = meta_by_pair_side.get((p, "benign"))
        em = meta_by_pair_side.get((p, "malicious"))
        if eb is None or em is None:
            continue
        mb = eb.get("commit_logit_top1", float("nan")) - eb.get("commit_logit_top2", float("nan"))
        mm = em.get("commit_logit_top1", float("nan")) - em.get("commit_logit_top2", float("nan"))
        rows.append({
            "pair_idx": p,
            "mcp_server": eb.get("mcp_server"),
            "risk_category": eb.get("risk_category"),
            "margin_benign": mb,
            "margin_mal": mm,
            "dmargin": mm - mb,
        })
    return pd.DataFrame(rows)


def evaluate_logit_margin(captured: Any, *, seed: int = 0) -> tuple[pd.DataFrame, LOSOResult]:
    if not captured.manifest.get("capture_extras"):
        raise ValueError(
            "logit_margin baseline requires a capture run with capture_extras=True "
            "(re-run Stage 1 with --capture-extras)."
        )
    df = logit_margin_df(captured)
    X, y, groups = _wide_df_to_Xy(df, ["margin"])
    return df, loso_logistic_from_Xy(X, y, groups, seed=seed, standardize=True)


# ──────────────────────────────────────────────────────────────────────────
# Baselines 3 & 4 — activation probe / residual-stream OOD
# (share the same commit-step hidden-state vectors)
# ──────────────────────────────────────────────────────────────────────────

def _load_hidden_Xy(captured: Any, *, layer: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack commit-step hidden vectors across all pairs into (X, y, groups).

    One row per (pair, side): y=0 for benign, y=1 for malicious.
    ``layer`` selects which saved hidden-state layer to use when more than
    one was captured (default: the first / only one, typically the final
    transformer layer — see ``capture.run(hidden_layers=...)``).
    """
    if not captured.manifest.get("capture_extras"):
        raise ValueError(
            "activation_probe/residual_ood require a capture run with "
            "capture_extras=True (re-run Stage 1 with --capture-extras)."
        )
    hidden_layers = captured.manifest.get("hidden_layers") or []
    layer_idx = 0 if layer is None else hidden_layers.index(layer)

    meta_by_pair: dict[int, dict[str, Any]] = {
        int(e["pair_idx"]): e for e in captured.meta if e["side"] == "benign"
    }
    n_pairs = int(captured.manifest["n_pairs"])

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    g_rows: list[str] = []
    for p in range(n_pairs):
        ch = captured.commit_hidden(p)
        if ch is None:
            continue
        server = meta_by_pair.get(p, {}).get("mcp_server")
        X_rows.append(ch["benign"][layer_idx].astype(np.float64))
        y_rows.append(0)
        g_rows.append(server)
        X_rows.append(ch["malicious"][layer_idx].astype(np.float64))
        y_rows.append(1)
        g_rows.append(server)

    if not X_rows:
        raise ValueError("No commit_hidden entries found; was capture_extras actually enabled?")
    return np.stack(X_rows, axis=0), np.array(y_rows), np.array(g_rows)


def evaluate_activation_probe(captured: Any, *, layer: int | None = None, seed: int = 0) -> LOSOResult:
    """Supervised linear probe on commit-step hidden states, under LOSO.

    Directly controls for "does access to internal *activations* in
    general (rather than our specific attention-disagreement construction)
    already explain the detection gain?" — same protocol as PMAShield's own
    LR classifier, different feature space.
    """
    X, y, groups = _load_hidden_Xy(captured, layer=layer)
    return loso_logistic_from_Xy(X, y, groups, seed=seed, standardize=True)


def evaluate_residual_ood(
    captured: Any, *, layer: int | None = None, method: str = "mahalanobis"
) -> LOSOResult:
    """Unsupervised residual-stream OOD detector, under LOSO.

    Unlike every other detector evaluated in this paper (PMAShield included),
    this one is fit *without ever seeing a malicious example*: for each
    held-out server, we fit a Gaussian (or robust covariance) to the
    training folds' **benign-only** hidden states, then score every held-out
    example (benign + malicious) by its Mahalanobis distance from that
    benign distribution. Larger distance = more anomalous = more
    attack-like. This is the most direct test of whether the malicious
    hidden state is simply "out of distribution" relative to benign
    activity, without any attack supervision at all.
    """
    from sklearn.covariance import LedoitWolf, MinCovDet
    from sklearn.metrics import roc_auc_score, roc_curve

    X, y, groups = _load_hidden_Xy(captured, layer=layer)
    aucs: dict[str, float] = {}
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []

    for held in sorted(set(groups.tolist())):
        train_mask = (groups != held) & (y == 0)  # benign-only, other servers
        test_mask = groups == held
        if train_mask.sum() < 10 or test_mask.sum() < 2:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue

        X_train = X[train_mask]
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        X_train_n = (X_train - mu) / sigma
        X_test_n = (X[test_mask] - mu) / sigma

        try:
            if method == "mahalanobis":
                cov = LedoitWolf().fit(X_train_n)
            elif method == "robust":
                cov = MinCovDet(support_fraction=0.9).fit(X_train_n)
            else:
                raise ValueError(f"unknown method: {method}")
            dist = cov.mahalanobis(X_test_n)
        except Exception as exc:
            logger.warning("residual_ood: covariance fit failed for fold {}: {}", held, exc)
            continue

        aucs[str(held)] = float(roc_auc_score(y[test_mask], dist))
        pooled_scores.extend(dist.tolist())
        pooled_labels.extend(y[test_mask].tolist())

    if not aucs:
        return LOSOResult(float("nan"), float("nan"), {}, float("nan"), int(len(y)), [], [])

    pooled_scores_arr = np.asarray(pooled_scores)
    pooled_labels_arr = np.asarray(pooled_labels)
    fpr: list[float] = []
    tpr: list[float] = []
    if len(np.unique(pooled_labels_arr)) == 2:
        fpr_arr, tpr_arr, _ = roc_curve(pooled_labels_arr, pooled_scores_arr)
        fpr, tpr = fpr_arr.tolist(), tpr_arr.tolist()

    return LOSOResult(
        auc_mean=float(np.mean(list(aucs.values()))),
        auc_std=float(np.std(list(aucs.values()), ddof=0)),
        auc_per_fold=aucs,
        f1_best=float("nan"),  # unsupervised score has no natural F1 threshold
        n=int(len(y)),
        roc_fpr=fpr,
        roc_tpr=tpr,
        pooled_scores=pooled_scores_arr.tolist(),
        pooled_labels=pooled_labels_arr.tolist(),
    )


# ──────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────

def _wide_df_to_Xy(df: pd.DataFrame, col_bases: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same per-side reshape convention as detection.py: one row per (pair, side)."""
    rows = []
    for _, r in df.iterrows():
        for side, label in (("benign", 0), ("mal", 1)):
            row: dict[str, Any] = {"y": label, "group": r["mcp_server"]}
            for base in col_bases:
                row[base] = r.get(f"{base}_{side}", float("nan"))
            rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=list(col_bases))
    X = out[list(col_bases)].to_numpy(dtype=np.float64)
    y = out["y"].to_numpy(dtype=np.int64)
    groups = out["group"].to_numpy()
    return X, y, groups
