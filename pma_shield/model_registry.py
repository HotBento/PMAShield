"""Shared model architecture registry for backdoor and MCPTox experiments.

Maps HuggingFace model IDs to architecture parameters (num_layers, num_heads,
head_dim, num_kv_heads).  When a model is not in the registry, parameters are
derived dynamically from the HuggingFace PretrainedConfig object.

Usage::

    from pma_shield.model_registry import get_arch, list_models

    arch = get_arch("meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(arch.num_layers, arch.num_heads, arch.head_dim)

    # After loading a model whose ID is not registered:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    arch = get_arch(model_id, hf_config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelArchParams:
    """Architecture parameters for a transformer LLM."""
    model_id:     str
    num_layers:   int   # number of transformer blocks (= num_hidden_layers)
    num_heads:    int   # query attention heads
    head_dim:     int   # per-head dimension
    num_kv_heads: int   # key/value heads (= num_heads for MHA, < num_heads for GQA)
    family:       str   # "qwen", "gemma", "phi", "llama", "mistral", "unknown"


# ---------------------------------------------------------------------------
# Registry: verified from HuggingFace model configs (or best-effort estimates
# pending server-side AutoConfig verification).  Estimates are flagged with
# a comment; exact values are read at runtime via get_arch(model_id, cfg).
# ---------------------------------------------------------------------------
_KNOWN_MODELS: dict[str, ModelArchParams] = {
    # ── Qwen3 (already in use) ─────────────────────────────────────────────
    "Qwen/Qwen3-8B": ModelArchParams(
        "Qwen/Qwen3-8B", num_layers=40, num_heads=32, head_dim=128, num_kv_heads=8, family="qwen"
    ),
    # ── Qwen3-4B (same series, smaller) ────────────────────────────────────
    "Qwen/Qwen3-4B": ModelArchParams(
        "Qwen/Qwen3-4B", num_layers=36, num_heads=32, head_dim=128, num_kv_heads=8, family="qwen"
    ),
    # ── Gemma 3 4B (March 2025) ────────────────────────────────────────────
    # Largest in-range size of the standard Gemma 3 family (sizes are
    # 1B / 4B / 12B / 27B — no 9B exists).  Alternating sliding-window
    # (5/6 layers, window=1024) + full-attention layers.  Estimates below
    # are overwritten by AutoConfig at load time.
    "google/gemma-3-4b-it": ModelArchParams(
        "google/gemma-3-4b-it", num_layers=34, num_heads=8, head_dim=256, num_kv_heads=4, family="gemma"
    ),
    # ── Gemma 4 E4B (latest Gemma generation) ──────────────────────────────
    # Effective 4B variant.  Alternating sliding-window (window≈512) and
    # full-attention layers — selection-head analysis only meaningful on
    # full-attention layers.  Values below are estimates; AutoConfig at load
    # time overrides them.
    "google/gemma-4-E4B-it": ModelArchParams(
        "google/gemma-4-E4B-it", num_layers=42, num_heads=8, head_dim=320, num_kv_heads=2, family="gemma"
    ),
    # ── Phi-4-mini (February 2025, latest Phi-4 in 4B range) ───────────────
    "microsoft/Phi-4-mini-instruct": ModelArchParams(
        "microsoft/Phi-4-mini-instruct", num_layers=32, num_heads=32, head_dim=96, num_kv_heads=8, family="phi"
    ),
    # ── LLaMA 3.1 8B (July 2024, latest Meta LLaMA 8B) ─────────────────────
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ModelArchParams(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", num_layers=32, num_heads=32, head_dim=128, num_kv_heads=8, family="llama"
    ),
}


def list_models() -> list[str]:
    """Return all registered model IDs."""
    return list(_KNOWN_MODELS.keys())


def _detect_family(model_id: str) -> str:
    """Infer model family from model ID string."""
    mid = model_id.lower()
    if "gemma" in mid:
        return "gemma"
    if "phi" in mid:
        return "phi"
    if "llama" in mid:
        return "llama"
    if "qwen" in mid:
        return "qwen"
    if "mistral" in mid or "mixtral" in mid:
        return "mistral"
    if "deepseek" in mid:
        return "deepseek"
    return "unknown"


def _arch_from_hf_config(model_id: str, cfg: Any) -> ModelArchParams:
    """Derive ModelArchParams from a HuggingFace PretrainedConfig object."""
    num_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    # head_dim: use explicit attribute if present, else infer from hidden_size
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    family = _detect_family(model_id)
    return ModelArchParams(
        model_id=model_id,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv,
        family=family,
    )


def get_arch(model_id: str, hf_config: Any = None) -> ModelArchParams:
    """Return architecture parameters for *model_id*.

    Parameters
    ----------
    model_id
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-8B"``).
    hf_config
        Optional HuggingFace ``PretrainedConfig`` object.  When provided and
        the model is not in the registry, parameters are read from this object.
        When the model IS in the registry, this is ignored.

    Raises
    ------
    KeyError
        If the model is not registered and *hf_config* is not provided.
    """
    if model_id in _KNOWN_MODELS:
        return _KNOWN_MODELS[model_id]
    if hf_config is not None:
        return _arch_from_hf_config(model_id, hf_config)
    raise KeyError(
        f"Model {model_id!r} is not in the registry and no hf_config was provided. "
        f"Registered models: {list(_KNOWN_MODELS.keys())}. "
        "Either add the model to mcp_eval/model_registry.py or pass "
        "hf_config=AutoConfig.from_pretrained(model_id)."
    )
