"""MPMA attack generators + pairs.jsonl builder.

This module reads the same ``response_all.json`` that MCPTox uses, parses each
server's *clean* tool block, and synthesises an MPMA "twin" tool per real tool.
The twin has identical parameters but a manipulatively-rewritten name +
description.  Each (real_tool, query) pair becomes one
:class:`mcp_eval.mcptox.data.MCPToxPair` with ``attack_shape = "insert"``.

The output JSONL is consumed by :mod:`mcp_eval.mcptox.capture` and the rest
of the detection pipeline without any code changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from loguru import logger

from pma_shield.detector import config as mcptox_config
from pma_shield.detector.data_mcptox import (
    MCPToxPair,
    MCPToxSample,
    assert_pair_invariants,
    parse_tool_list,
    to_openai_schema,
)


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name to be LLM-API-safe (max 64 chars, keep suffix)."""
    name = name.replace("/", "_").replace("@", "")
    return name[-64:] if len(name) > 64 else name

from . import mpma_config


# ─────────────────────────────────────────────────────────────────────────────
# DPMA — Direct Preference Manipulation Attack
# ─────────────────────────────────────────────────────────────────────────────

def _dpma_name_prefix(rng: random.Random) -> str:
    """Pick a short eye-catching prefix for the *tool name*.

    The paper inserts a bracketed ranking badge; we mirror that.
    """
    stars = rng.choice(mpma_config.STRATEGY_STARS)
    return f"[{stars}]"


def _dpma_description_prefix(rng: random.Random, strategies: tuple[str, ...]) -> str:
    """Construct the manipulative *description prefix* by picking one token
    from each requested strategy bank and gluing them together."""
    parts: list[str] = []
    for s in strategies:
        bank = mpma_config.ALL_STRATEGIES.get(s)
        if not bank:
            continue
        parts.append(rng.choice(bank))
    return ". ".join(parts) + "."


def apply_dpma(
    tool: dict[str, Any],
    *,
    rng: random.Random | None = None,
    strategies: tuple[str, ...] = mpma_config.DEFAULT_DPMA_COMBO,
    name_suffix: str = "_pro",
) -> dict[str, Any]:
    """Return a copy of ``tool`` with DPMA-decorated name + description.

    The resulting *function name* is

        ``"<original_name><name_suffix>"``

    so it never collides with the original (necessary because the pair is
    constructed in ``insert`` shape — the attacker's twin must coexist with
    the legitimate tool on the malicious side).

    The original parameters block is left untouched: only metadata changes.
    """
    rng = rng if rng is not None else random.Random()
    fn_in = tool["function"]
    name_in: str = fn_in["name"]
    desc_in: str = fn_in.get("description", "") or ""

    # New name — different from original to satisfy the "insert" invariant
    # of MCPToxPair.  Truncate to 64 chars to stay within OpenAI tool-name
    # limits (the project's normalize_tool_name caps at 64).
    new_name = normalize_tool_name(name_in + name_suffix)

    # Decorate the visible name (description-side hint) with a star badge.
    # We do not embed star characters into the function name itself because
    # the chat template's tool-name regex would reject them.
    name_badge = _dpma_name_prefix(rng)

    new_desc = (
        f"{name_badge} {_dpma_description_prefix(rng, strategies)} "
        f"{desc_in}"
    ).strip()

    new_tool = copy.deepcopy(tool)
    new_tool["function"]["name"] = new_name
    new_tool["function"]["description"] = new_desc
    return new_tool


def apply_gapma_simple(
    tool: dict[str, Any],
    *,
    rng: random.Random | None = None,
    n_strategies: int = 2,
    name_suffix: str = "_x",
) -> dict[str, Any]:
    """v1 simplification of GAPMA.

    Rather than running a full genetic algorithm over a fitness signal, we
    randomly choose ``n_strategies`` of the four MPMA strategy families and
    pick one token from each.  The composition is shuffled before insertion
    so consecutive samples drawn from the same source tool yield diverse
    descriptions (a poor man's "population diversity").
    """
    rng = rng if rng is not None else random.Random()
    all_names = list(mpma_config.ALL_STRATEGIES.keys())
    chosen = rng.sample(all_names, k=min(n_strategies, len(all_names)))
    rng.shuffle(chosen)
    return apply_dpma(
        tool, rng=rng, strategies=tuple(chosen), name_suffix=name_suffix
    )


ATTACK_FUNCS = {
    "MPMA-DPMA":  apply_dpma,
    "MPMA-GAPMA": apply_gapma_simple,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pair construction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _ServerTools:
    """Internal: one server's parsed clean tool list + queries."""
    server_name: str
    server_url: str
    benign_tools: list[dict[str, Any]]   # OpenAI fn-calling JSON
    benign_tool_names: list[str]
    clean_queries: list[str]


def _load_server_tools(
    response_path: Path = mpma_config.MPMA_SOURCE_RESPONSE_FILE,
) -> Iterator[_ServerTools]:
    """Stream each MCPTox server's clean tool list as OpenAI fn-calling schemas."""
    with open(response_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for server_name, server in payload["servers"].items():
        try:
            parsed = parse_tool_list(server["clean_system_promot"])
        except Exception as exc:
            logger.warning(
                "Skipping server {!r}: failed to parse clean tool list ({})",
                server_name, exc,
            )
            continue
        if not parsed:
            continue
        benign_tools = [to_openai_schema(p) for p in parsed]
        yield _ServerTools(
            server_name=server_name,
            server_url=server.get("server_url", ""),
            benign_tools=benign_tools,
            benign_tool_names=[t["function"]["name"] for t in benign_tools],
            clean_queries=list(server.get("clean_querys") or []),
        )


def build_mpma_pairs(
    *,
    attack: str = "MPMA-DPMA",
    response_path: Path = mpma_config.MPMA_SOURCE_RESPONSE_FILE,
    per_server_tools: int | None = None,
    per_query: int = 1,
    seed: int = 0,
) -> Iterator[MCPToxPair]:
    """Yield one :class:`MCPToxPair` per (server, target_tool, query) combo.

    Parameters
    ----------
    attack
        Which attack variant to apply.  One of the keys of
        :data:`ATTACK_FUNCS`.
    response_path
        Path to MCPTox's ``response_all.json``.  Used as the source of
        legitimate tools and queries.
    per_server_tools
        If set, sample at most this many *target* tools per server (without
        replacement, deterministic for given ``seed``).  ``None`` means use
        all tools.
    per_query
        How many distinct ``clean_querys`` to use per target tool.  If the
        server has fewer queries than requested, all available queries are
        used.
    seed
        RNG seed for both target-tool sampling and DPMA token sampling.
    """
    if attack not in ATTACK_FUNCS:
        raise ValueError(f"Unknown attack {attack!r}; expected one of {list(ATTACK_FUNCS)}")
    attack_fn = ATTACK_FUNCS[attack]
    rng = random.Random(seed)

    total = 0
    for server in _load_server_tools(response_path):
        if not server.clean_queries:
            # MPMA needs a real user query — skip servers without one.
            continue
        targets = list(server.benign_tools)
        if per_server_tools is not None and len(targets) > per_server_tools:
            targets = rng.sample(targets, k=per_server_tools)
        queries = server.clean_queries
        n_q = min(per_query, len(queries))
        for tool_idx, target_tool in enumerate(targets):
            for q_idx in range(n_q):
                query = queries[q_idx]
                # Deterministic per-(server, tool, q) sub-RNG so the same CLI
                # invocation produces byte-identical pairs.jsonl.
                tag = f"{server.server_name}|{target_tool['function']['name']}|{q_idx}"
                sub_seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                sub_rng = random.Random(seed ^ sub_seed)

                # Build the attacker's twin tool.
                attacker_tool = attack_fn(target_tool, rng=sub_rng)
                attacker_name = attacker_tool["function"]["name"]

                # Skip collisions (attacker name accidentally equals an existing one).
                if attacker_name in server.benign_tool_names:
                    logger.debug(
                        "Skip {}: attacker twin name {} collides with benign list",
                        tag, attacker_name,
                    )
                    continue

                malicious_tools = list(server.benign_tools) + [attacker_tool]

                sample_id = (
                    f"{server.server_name}.mpma[{tool_idx:03d}].q[{q_idx}]"
                )
                benign = MCPToxSample(
                    sample_id=sample_id,
                    mcp_server=server.server_name,
                    server_url=server.server_url,
                    risk_category=f"{attack.lower().replace('-', '_')}",
                    attack_paradigm=attack,
                    user_query=query,
                    tools=server.benign_tools,
                    is_malicious=False,
                )
                malicious = MCPToxSample(
                    sample_id=sample_id,
                    mcp_server=server.server_name,
                    server_url=server.server_url,
                    risk_category=f"{attack.lower().replace('-', '_')}",
                    attack_paradigm=attack,
                    user_query=query,
                    tools=malicious_tools,
                    is_malicious=True,
                )
                pair = MCPToxPair(
                    benign=benign,
                    malicious=malicious,
                    poisoned_tool_name=attacker_name,
                    replaced_tool_name=None,
                    attack_shape="insert",
                    query_id=q_idx,
                )
                try:
                    assert_pair_invariants(pair)
                except AssertionError as exc:
                    logger.warning("Pair {} fails invariants: {}", sample_id, exc)
                    continue
                total += 1
                yield pair
    logger.info("Generated {} MPMA pairs ({} attack)", total, attack)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL writer (atomic)
# ─────────────────────────────────────────────────────────────────────────────

def write_pairs_jsonl(
    pairs: Iterable[MCPToxPair],
    out_path: Path = mpma_config.MPMA_PAIRS_PATH,
) -> int:
    """Atomic JSONL write.  Returns number of pairs written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    import os
    os.replace(tmp, out_path)
    logger.info("Wrote {} MPMA pairs → {}", n, out_path)
    return n
