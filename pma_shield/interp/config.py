"""Paths and constants for mcp_eval.interp.

All paths resolve relative to the project root so cwd does not matter.
"""

from __future__ import annotations

from pathlib import Path

from pma_shield.model_registry import ModelArchParams, get_arch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

PROBE_DIR: Path = DATA_DIR / "probe_scenarios"
INTERP_RESULTS_DIR: Path = RESULTS_DIR / "interp"
INTERP_FIGURES_DIR: Path = FIGURES_DIR / "interp"
MOCK_FIGURES_DIR: Path = FIGURES_DIR / "interp_mock"

PRIMARY_MODEL: str = "Qwen/Qwen3-8B"

ALL_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-3-4b-it",
    "google/gemma-4-E4B-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
)

MODEL_DISPLAY: dict[str, str] = {
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    "google/gemma-3-4b-it": "Gemma-3-4B",
    "google/gemma-4-E4B-it": "Gemma-4-E4B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}


def model_safe(model_id: str) -> str:
    """Filesystem-safe version of a HuggingFace model id (matches scripts/multi_model)."""
    return model_id.replace("/", "_")


def model_results_dir(model_id: str) -> Path:
    return INTERP_RESULTS_DIR / model_safe(model_id)


def model_arch(model_id: str) -> ModelArchParams:
    return get_arch(model_id)


# ACL single-column ≈ 3.15in, double-column ≈ 6.5in.
FIG_SINGLE_W: float = 3.3
FIG_DOUBLE_W: float = 6.7
FIG_HEIGHT_DEFAULT: float = 2.4
