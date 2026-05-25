"""Loader for ``data/probe_scenarios/*.yaml``.

The YAML schema is documented in :file:`data/probe_scenarios/README.md`.
Each scenario carries a user query plus a small candidate tool set; exactly
one tool is marked as the ``intended_tool``. Cross-category scenario pairs
form the contrastive ``(s_A, s_B)`` pairs consumed by
:mod:`mcp_eval.interp.patching`.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from pma_shield.interp.config import PROBE_DIR

# YAML is an optional dep elsewhere in the repo but used heavily by mcptox/.
# Import here so any caller of probe_data fails fast if the env is broken.
import yaml  # type: ignore


@dataclass(frozen=True)
class ProbeScenario:
    """A single contrastive prompt scenario."""

    id: str
    category: str
    query: str
    intended_tool: str
    tools: tuple[dict[str, Any], ...]

    def tool_names(self) -> tuple[str, ...]:
        return tuple(t["name"] for t in self.tools)

    def function_calling_tools(self) -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling JSON format."""
        return [{"type": "function", "function": dict(t)} for t in self.tools]


@dataclass(frozen=True)
class ProbePair:
    """A contrastive pair ``(s_A, s_B)`` with distinct intended tools."""

    a: ProbeScenario
    b: ProbeScenario

    @property
    def id(self) -> str:
        return f"{self.a.id}__vs__{self.b.id}"


CATEGORIES: tuple[str, ...] = ("weather", "paper_search", "stock", "web_search")


def load_category(category: str, root: Path = PROBE_DIR) -> list[ProbeScenario]:
    """Parse one YAML file into a list of :class:`ProbeScenario`."""
    path = root / f"{category}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"probe scenario file not found: {path}")
    with path.open() as fh:
        doc = yaml.safe_load(fh)
    if doc.get("category") != category:
        raise ValueError(
            f"category mismatch: {path} declares {doc.get('category')!r} but "
            f"was loaded as {category!r}"
        )
    out: list[ProbeScenario] = []
    for entry in doc.get("scenarios", []):
        tools = tuple(entry["tools"])
        intended = entry["intended_tool"]
        if intended not in {t["name"] for t in tools}:
            raise ValueError(
                f"scenario {entry['id']!r}: intended_tool {intended!r} not "
                f"present in tools {[t['name'] for t in tools]}"
            )
        out.append(
            ProbeScenario(
                id=entry["id"],
                category=category,
                query=entry["query"],
                intended_tool=intended,
                tools=tools,
            )
        )
    return out


def load_all(categories: Iterable[str] = CATEGORIES, root: Path = PROBE_DIR) -> dict[str, list[ProbeScenario]]:
    return {c: load_category(c, root) for c in categories}


def make_pairs(
    scenarios_by_category: dict[str, list[ProbeScenario]],
    *,
    seed: int = 0,
    max_pairs_per_combo: int | None = 5,
) -> list[ProbePair]:
    """Build contrastive pairs across distinct categories.

    For every unordered pair of categories we draw up to ``max_pairs_per_combo``
    scenario pairs. The randomness uses ``seed`` so the pair set is stable
    across reruns — important for resumability.
    """
    rng = random.Random(seed)
    pairs: list[ProbePair] = []
    cats = sorted(scenarios_by_category.keys())
    for ca, cb in itertools.combinations(cats, 2):
        sa = scenarios_by_category[ca]
        sb = scenarios_by_category[cb]
        candidates = [(a, b) for a in sa for b in sb]
        rng.shuffle(candidates)
        if max_pairs_per_combo is not None:
            candidates = candidates[:max_pairs_per_combo]
        pairs.extend(ProbePair(a, b) for a, b in candidates)
    return pairs


__all__ = [
    "CATEGORIES",
    "ProbePair",
    "ProbeScenario",
    "load_all",
    "load_category",
    "make_pairs",
]
