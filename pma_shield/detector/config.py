"""Central configuration: filesystem paths, model defaults, schema constants."""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Paths (resolved from this file's location so cwd doesn't matter)
# mcp_eval/mcptox/config.py → parents[2] = project root
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

MCPTOX_RAW_DIR: Path = DATA_DIR / "mcptox"
MCPTOX_RESPONSE_FILE: Path = MCPTOX_RAW_DIR / "response_all.json"
MCPTOX_PAIRS_PATH: Path = MCPTOX_RAW_DIR / "pairs.jsonl"

ATTN_CACHE_DIR: Path = RESULTS_DIR / "mcptox" / "attn_cache"
CLUSTER_DIR: Path = RESULTS_DIR / "mcptox"
DISAGREEMENT_DIR: Path = RESULTS_DIR / "mcptox"
DETECTION_DIR: Path = RESULTS_DIR / "mcptox"

# ──────────────────────────────────────────────────────────────────────────
# Model defaults — match the prior whitebox experiments
# ──────────────────────────────────────────────────────────────────────────
# Legacy constants kept for backwards compatibility (Qwen3-8B specific).
# Architecture parameters for other models are in mcp_eval.model_registry.
DEFAULT_MODEL_ID: str = "Qwen/Qwen3-8B"

NUM_LAYERS: int = 40
NUM_HEADS: int = 32

# Used as sanity-check anchors in :mod:`mcp_eval.mcptox.selection` and as the
# diagnostic-row set in :mod:`mcp_eval.mcptox.capture`.
KNOWN_CORE_HEADS: tuple[tuple[int, int], ...] = (
    (24, 29),
    (32, 11),
    (24, 14),
    (30, 7),
    (23, 26),
    (29, 11),
)

#: Per-model known core heads.  Used by capture.py to log diagnostic rows for
#: each model.  Entries for new models are filled after running head analysis.
KNOWN_CORE_HEADS_BY_MODEL: dict[str, tuple[tuple[int, int], ...]] = {
    "Qwen/Qwen3-8B": KNOWN_CORE_HEADS,
    # Entries below are stubs; populate after running whitebox head analysis
    # on each model (head coordinates are architecture-specific).
    "Qwen/Qwen3-4B": (),
    "microsoft/Phi-4-mini-instruct": (),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (),
}


def get_known_core_heads(
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[tuple[int, int], ...]:
    """Return known core heads for *model_id*.

    Returns an empty tuple for unregistered models. Callers (e.g. capture's
    ``_stack_topheads_rows``) must tolerate an empty head set — the Qwen3-8B
    coordinates ``(L32, H29, …)`` are out-of-bounds for smaller models.
    """
    return KNOWN_CORE_HEADS_BY_MODEL.get(model_id, ())

# KMeans K candidates for Stage 2 (silhouette sweep when no explicit K is passed).
CLUSTER_K_GRID: tuple[int, ...] = (3, 4, 5, 6, 8)

# Populated lazily by data.load_pairs at first call so analysis code can group
# without re-scanning the raw files.
MCPTOX_RISK_CATEGORIES: tuple[str, ...] | None = None
