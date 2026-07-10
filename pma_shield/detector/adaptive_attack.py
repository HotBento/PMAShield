"""
Adaptive-attacker robustness simulation (rebuttal, reviewer texz W2).

texz asked whether PMAShield considers an attacker who knows the discovered
selection-head set and deliberately limits how many of those heads their
injection actually influences, in order to suppress the inter-head
disagreement signal the detector relies on.

We cannot re-run MCPTox/MPMA-DPMA with a "partial-hijack" injection (no such
attack variant exists in either benchmark), so we simulate the effect
directly on already-captured features: for a given malicious pair and a
target hijack count ``k <= |S|``, we build a *hybrid* feature array that uses
the malicious-side attention for the ``k`` heads the attacker is assumed to
target and the benign-side attention for the remaining ``|S| - k`` heads
(as if the attacker never touched them). We then recompute the four
disagreement metrics on this hybrid, exactly as ``disagreement.compute``
does for the real benign/malicious pair, giving a synthetic
"malicious-but-only-k-heads-hijacked" sample.

Attacker capability model
--------------------------
We assume a *worst-case* adaptive attacker: they know each selection head's
offline selection rate ``r(h)`` (Eq. 3 in the paper, stored in
``SelectionHeads.argmax_match_rate``) and prioritise hijacking the heads
PMAShield relies on most heavily first. This is the strongest attacker the
threat model in the Limitations section describes, so a robustness curve
against it is a conservative (worst-case) bound.

Evaluation protocol — frozen detector
--------------------------------------
The detector is trained ONCE per LOSO fold on the *real* benign/malicious
disagreement metrics (identical to RQ1 — ``detection._loso_logistic_auc``),
then evaluated on the held-out server's benign rows (real) paired with the
synthetic partial-hijack malicious rows for the same server. This mirrors a
deployed detector that has not seen the adaptive attacker's weaker attacks
during training/calibration — the realistic scenario the reviewer describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .disagreement import compute_one_side


DISAGREEMENT_METRICS: tuple[str, ...] = ("A", "E", "D_JS", "O")


# ──────────────────────────────────────────────────────────────────────────
# Synthetic partial-hijack feature construction
# ──────────────────────────────────────────────────────────────────────────

def rank_heads_by_importance(
    selection_heads: Any, head_set: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Order ``head_set`` by descending offline selection rate r(h).

    Models a worst-case attacker who knows exactly which selection heads
    PMAShield relies on most and hijacks those first.
    """
    rates = selection_heads.argmax_match_rate
    return sorted(head_set, key=lambda h: rates.get(tuple(h), 0.0), reverse=True)


def _hybrid_features(
    feats_benign_pair: np.ndarray,
    feats_mal_pair: np.ndarray,
    hijacked_heads: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Feature array = malicious values at ``hijacked_heads``, benign elsewhere.

    Only the ``hijacked_heads`` slices are overwritten; every other
    (layer, head) — including heads outside the selection set entirely,
    which ``compute_one_side`` ignores anyway — keeps its benign value.
    """
    hybrid = feats_benign_pair.copy()
    if hijacked_heads:
        layers = [h[0] for h in hijacked_heads]
        heads = [h[1] for h in hijacked_heads]
        hybrid[layers, heads, :] = feats_mal_pair[layers, heads, :]
    return hybrid


def partial_hijack_df(
    captured: Any,
    head_set: Sequence[tuple[int, int]],
    ranked_heads: Sequence[tuple[int, int]],
    k: int,
) -> pd.DataFrame:
    """Disagreement metrics for the "attacker hijacks only the top-k heads" scenario.

    Same row schema as ``disagreement.compute`` (one row per pair, with
    ``A_mal`` / ``E_mal`` / ``D_JS_mal`` / ``O_mal`` now computed from the
    synthetic hybrid rather than the real malicious features). ``_benign``
    columns are unchanged (real benign features) so this DataFrame can be
    directly compared against ``disagreement.compute``'s output for k=|S|.
    """
    hijacked = list(ranked_heads[:k])
    feats_b = np.asarray(captured.benign_features)
    feats_m = np.asarray(captured.malicious_features)
    n_pairs = feats_b.shape[0]

    meta_by_pair_side: dict[tuple[int, str], dict[str, Any]] = {
        (int(e["pair_idx"]), e["side"]): e for e in captured.meta
    }

    rows = []
    for p in range(n_pairs):
        eb = meta_by_pair_side.get((p, "benign"))
        em = meta_by_pair_side.get((p, "malicious"))
        if eb is None or em is None:
            continue

        n_tools_b = len(eb.get("tool_names") or [])
        n_tools_m = len(em.get("tool_names") or [])
        target_b = (
            eb["tool_names"].index(eb["selected_tool"])
            if eb.get("selected_tool") in (eb.get("tool_names") or []) else -1
        )
        target_m = (
            em["tool_names"].index(em["selected_tool"])
            if em.get("selected_tool") in (em.get("tool_names") or []) else -1
        )

        m_b = compute_one_side(feats_b[p], list(head_set), n_tools=n_tools_b, target_idx=target_b)
        hybrid = _hybrid_features(feats_b[p], feats_m[p], hijacked)
        m_hy = compute_one_side(hybrid, list(head_set), n_tools=n_tools_m, target_idx=target_m)

        rows.append({
            "pair_idx": p,
            "mcp_server": eb.get("mcp_server"),
            "risk_category": eb.get("risk_category"),
            "k": k,
            "n_selection_heads": len(head_set),
            "A_benign": m_b.A, "A_mal": m_hy.A,
            "E_benign": m_b.E, "E_mal": m_hy.E,
            "D_JS_benign": m_b.D_JS, "D_JS_mal": m_hy.D_JS,
            "O_benign": m_b.O, "O_mal": m_hy.O,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Frozen-detector LOSO evaluation against the synthetic attacker
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveResult:
    k: int
    fraction: float
    auc_mean: float
    auc_std: float
    auc_per_fold: dict[str, float]
    n: int


def evaluate_adaptive_loso(
    df_real: pd.DataFrame,
    df_partial: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = DISAGREEMENT_METRICS,
    seed: int = 0,
) -> AdaptiveResult:
    """Train on real attacks per LOSO fold; test on the synthetic partial-hijack.

    ``df_real`` is ``disagreement.compute``'s output (the real, full-hijack
    benign/malicious pairs — used only for training). ``df_partial`` is
    ``partial_hijack_df``'s output for a single ``k`` (used only for the
    held-out server's malicious side at test time; its ``_benign`` columns
    are identical to ``df_real`` and not otherwise used).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    real_by_pair = df_real.set_index("pair_idx")
    partial_by_pair = df_partial.set_index("pair_idx")
    k = int(df_partial["k"].iloc[0]) if len(df_partial) else 0
    n_sel = int(df_partial["n_selection_heads"].iloc[0]) if len(df_partial) else 0

    def _rows(df: pd.DataFrame, side_values: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
        out = []
        for pair_idx, r in df.iterrows():
            for side, label in (("benign", 0), ("mal", 1)):
                entry: dict[str, Any] = {"y": label, "group": r["mcp_server"], "pair_idx": pair_idx}
                for col in feature_cols:
                    entry[col] = r.get(f"{col}_{side}", float("nan"))
                out.append(entry)
        return out

    train_pool = pd.DataFrame(_rows(real_by_pair, {}))
    # Test pool: benign rows from df_real, malicious rows from df_partial,
    # restricted to pairs present in both.
    common_idx = real_by_pair.index.intersection(partial_by_pair.index)
    test_benign = pd.DataFrame([
        {"y": 0, "group": real_by_pair.loc[p, "mcp_server"], "pair_idx": p,
         **{col: real_by_pair.loc[p, f"{col}_benign"] for col in feature_cols}}
        for p in common_idx
    ])
    test_mal = pd.DataFrame([
        {"y": 1, "group": partial_by_pair.loc[p, "mcp_server"], "pair_idx": p,
         **{col: partial_by_pair.loc[p, f"{col}_mal"] for col in feature_cols}}
        for p in common_idx
    ])
    test_pool = pd.concat([test_benign, test_mal], ignore_index=True)

    train_pool = train_pool.dropna(subset=list(feature_cols))
    test_pool = test_pool.dropna(subset=list(feature_cols))

    aucs: dict[str, float] = {}
    for held in sorted(set(train_pool["group"]) | set(test_pool["group"])):
        train_fold = train_pool[train_pool["group"] != held]
        test_fold = test_pool[test_pool["group"] == held]
        if len(train_fold) < 5 or len(test_fold) < 2:
            continue
        if test_fold["y"].nunique() < 2:
            continue

        X_train = train_fold[list(feature_cols)].to_numpy(dtype=np.float64)
        y_train = train_fold["y"].to_numpy(dtype=np.int64)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X_train, y_train)

        X_test = test_fold[list(feature_cols)].to_numpy(dtype=np.float64)
        y_test = test_fold["y"].to_numpy(dtype=np.int64)
        proba = clf.predict_proba(X_test)[:, 1]
        aucs[str(held)] = float(roc_auc_score(y_test, proba))

    if not aucs:
        return AdaptiveResult(k=k, fraction=(k / n_sel if n_sel else float("nan")),
                               auc_mean=float("nan"), auc_std=float("nan"),
                               auc_per_fold={}, n=0)

    return AdaptiveResult(
        k=k,
        fraction=(k / n_sel if n_sel else float("nan")),
        auc_mean=float(np.mean(list(aucs.values()))),
        auc_std=float(np.std(list(aucs.values()), ddof=0)),
        auc_per_fold=aucs,
        n=int(len(test_pool)),
    )


def run_adaptive_sweep(
    captured: Any,
    selection_heads: Any,
    df_real: pd.DataFrame,
    *,
    fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.1),
    feature_cols: Sequence[str] = DISAGREEMENT_METRICS,
    seed: int = 0,
) -> list[AdaptiveResult]:
    """Sweep attacker capability (fraction of selection heads hijacked).

    Returns one :class:`AdaptiveResult` per requested fraction, in the same
    order, sorted descending (fraction=1.0 should reproduce RQ1's AUROC as
    a sanity check — it hijacks every selection head, same as the real data).
    """
    head_set = list(selection_heads.heads)
    ranked = rank_heads_by_importance(selection_heads, head_set)
    n_sel = len(head_set)

    results = []
    for frac in fractions:
        k = max(1, round(frac * n_sel)) if n_sel else 0
        logger.info("Adaptive sweep: fraction={:.2f} -> k={}/{} heads hijacked", frac, k, n_sel)
        df_partial = partial_hijack_df(captured, head_set, ranked, k)
        res = evaluate_adaptive_loso(df_real, df_partial, feature_cols=feature_cols, seed=seed)
        results.append(res)
        logger.info(
            "  fraction={:.2f} k={} -> AUROC={:.3f} (n={})",
            frac, k, res.auc_mean, res.n,
        )
    return results
