"""
Stage 0 — MCPTox raw-data loader, pairing, and schema normalization.

Schema (confirmed by inspection of the cloned repo on 2026-04-25)
-----------------------------------------------------------------
Top-level file: ``response_all.json``.

::

    {
      "data_length": 1348,
      "attack_scopes":  [...11 risk categories...],
      "label_scopes":   [...],
      "call_behaviors": ["Template-1", "Template-2", "Template-3", "Other"],
      "save_dir":       "data_local.json",
      "servers": {
          "FileSystem": {
              "server_id":           int,
              "server_name":         str,
              "server_url":          str,
              "tool_names":          list[str],   # legitimate MCP tool names
              "clean_system_promot": str,         # benign system prompt (full
                                                  # tool block + chat instructions)
              "clean_querys":        list[str],   # benign queries
              "malicious_instance":  list[dict]   # one dict per attack case
          },
          ...44 more servers...
      }
    }

Each ``malicious_instance`` entry::

    {
      "security_risk_description": str,
      "wrong_data": int,
      "datas": [
          {
              "id":             int,
              "query":          str,        # benign user query
              "system":         str,        # full malicious system prompt
              "response":       dict,       # reference model outputs (ignored)
              "label":          dict,       # per-model classification (ignored)
              "online_result":  dict,       # reference (ignored)
              "poisoned_tool":  str,        # poisoned tool block (whitespace-cleaned;
                                            # NOT byte-equal to what is inside
                                            # ``system``, so do not rely on it)
          }
      ],
      "metadata": {"paradigm": str, "security risk": str},
      "poisoned_tool": str,                  # also unreliable (whitespace-cleaned)
    }

Total cases: 1348. ``len(datas) == 1`` for every malicious_instance.

Attack-shape invariant verified across all 1348 cases
-----------------------------------------------------
Parsing the system prompt with the regex ``r'^Tool: (\\S+)'`` yields the set
of tool names visible to the model. For every case::

    set(parse(mal_system)) − set(parse(clean_system))   has exactly 1 element

That extra name is the poisoned tool. Two attack shapes coexist:

* **insert** — ``set(legit) ⊆ set(mal)``; the poisoned tool is added.
* **replace** — ``set(legit) ⊄ set(mal)``; one legit tool is overwritten with
  the poisoned tool. (The "missing" legit tool's name is recorded in
  :attr:`MCPToxPair.replaced_tool_name`.)

In both shapes, the benign and malicious tool lists differ in exactly the
tool object whose name equals ``poisoned_tool_name`` (and, for replace, also
the legit tool that the poisoned one impersonated). Tool **ordering** of the
legit tools is preserved.

Output schema (this module's job)
---------------------------------
Each MCPTox case → one ``MCPToxPair`` written as a JSONL row:

::

    {
      "sample_id":          "FileSystem.mi[0].d[0]",
      "mcp_server":         "FileSystem",
      "server_url":         "https://github.com/...",
      "risk_category":      "infrastructure_damage",
      "attack_paradigm":    "Template-2",
      "attack_shape":       "insert",
      "user_query":         "Read the content of ...",
      "tool_list_benign":   [<OpenAI fn-calling JSON>, ...],
      "tool_list_malicious":[<OpenAI fn-calling JSON>, ...],
      "poisoned_tool_name": "qubit",
      "replaced_tool_name": null,            # str for "replace", null for "insert"
      "query_id":           3,
    }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from loguru import logger

from . import config


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name to be LLM-API-safe (max 64 chars, keep suffix)."""
    name = name.replace("/", "_").replace("@", "")
    return name[-64:] if len(name) > 64 else name

# ──────────────────────────────────────────────────────────────────────────
# Plain-text tool-block parser
# ──────────────────────────────────────────────────────────────────────────

# Match a tool name on its own line — names are single-token in MCPTox.
_TOOL_HEADER_RE = re.compile(r"^Tool: (\S+)\s*$", re.MULTILINE)
# Capture an "Arguments:" line followed by zero or more "- ..." bullets.
_ARG_LINE_RE = re.compile(r"^- (.+?)$", re.MULTILINE)
# Detect "(required)" suffix in argument descriptions.
_REQUIRED_SUFFIX_RE = re.compile(r"\s*\(required\)\s*$")
# Split on "name: description" inside an argument bullet.
_ARG_NAME_DESC_RE = re.compile(r"^([^:]+?):\s*(.*)$")


@dataclass(frozen=True)
class _ParsedTool:
    """Internal: result of parsing one ``Tool: ...`` block."""

    name: str
    description: str
    arguments: list[tuple[str, str, bool]]  # (arg_name, arg_description, required)


def _parse_tool_block(block_text: str) -> _ParsedTool:
    """Parse a single tool block.

    Expected layout::

        Tool: <name>
        Description: <free text, may span multiple lines>
        Arguments:
        - arg1: <description> (required)
        - arg2: <description>
        - No arguments        # placeholder for parameterless tools

    Trailing blank lines are tolerated.
    """
    lines = block_text.strip("\n").split("\n")

    # Line 1: "Tool: <name>"
    m = _TOOL_HEADER_RE.match(lines[0])
    if not m:
        raise ValueError(f"Tool block does not start with 'Tool: <name>': {block_text[:120]!r}")
    name = m.group(1).strip()

    # Find "Description:" / "Arguments:" anchors.
    desc_start: int | None = None
    args_start: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("Description:"):
            desc_start = i
        elif line.startswith("Arguments:"):
            args_start = i
            break

    if desc_start is None or args_start is None:
        raise ValueError(
            f"Tool block missing Description/Arguments section (name={name!r}): "
            f"{block_text[:200]!r}"
        )

    desc_text = lines[desc_start][len("Description:"):].lstrip()
    if args_start > desc_start + 1:
        # Multi-line description — concatenate continuation lines.
        desc_text = "\n".join([desc_text, *lines[desc_start + 1 : args_start]]).rstrip()

    args: list[tuple[str, str, bool]] = []
    for arg_line in lines[args_start + 1 :]:
        arg_line = arg_line.rstrip()
        if not arg_line.startswith("- "):
            # Trailing blank lines or unrelated content — stop parsing args.
            if arg_line.strip() == "":
                continue
            break
        body = arg_line[2:]
        if body.strip().lower() == "no arguments":
            break

        required = bool(_REQUIRED_SUFFIX_RE.search(body))
        body_no_required = _REQUIRED_SUFFIX_RE.sub("", body).strip()
        nm = _ARG_NAME_DESC_RE.match(body_no_required)
        if nm:
            arg_name = nm.group(1).strip()
            arg_desc_raw = nm.group(2).strip()
            arg_desc = "" if arg_desc_raw.lower() == "no description" else arg_desc_raw
        else:
            arg_name = body_no_required.strip()
            arg_desc = ""
        args.append((arg_name, arg_desc, required))

    return _ParsedTool(name=name, description=desc_text, arguments=args)


def parse_tool_list(system_prompt: str) -> list[_ParsedTool]:
    """Extract every ``Tool: ...`` block from a system prompt, in order.

    Robust to leading boilerplate ("You are a helpful assistant ...") and the
    trailing "IMPORTANT" chat-formatting instructions that follow the last
    tool. Returns tools in the order they appear in the prompt.
    """
    headers = list(_TOOL_HEADER_RE.finditer(system_prompt))
    if not headers:
        raise ValueError("No 'Tool: ...' headers found in system prompt.")

    tools: list[_ParsedTool] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(system_prompt)
        block = system_prompt[start:end]
        # Strip the trailing chat-instructions footer that follows the last
        # tool: it is separated by a blank line from the last "Arguments:"
        # bullet block.
        if i == len(headers) - 1:
            # Cut at the first occurrence of "\n\n\n" — that separates the
            # final tool block from the footer in MCPTox prompts.
            if "\n\n\n" in block:
                block = block[: block.index("\n\n\n")]
        tools.append(_parse_tool_block(block))
    return tools


# ──────────────────────────────────────────────────────────────────────────
# OpenAI fn-calling JSON conversion
# ──────────────────────────────────────────────────────────────────────────

def to_openai_schema(parsed: _ParsedTool) -> dict[str, Any]:
    """Convert a parsed plain-text tool to OpenAI fn-calling JSON.

    All argument types default to ``"string"`` (MCPTox does not record real
    JSON-schema types). The ``required`` array is populated from per-argument
    flags.
    """
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for arg_name, arg_desc, is_required in parsed.arguments:
        prop: dict[str, str] = {"type": "string"}
        if arg_desc:
            prop["description"] = arg_desc
        properties[arg_name] = prop
        if is_required:
            required.append(arg_name)

    return {
        "type": "function",
        "function": {
            "name": normalize_tool_name(parsed.name),
            "description": parsed.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# Sample / pair dataclasses
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MCPToxSample:
    """One side of a paired evaluation (benign or malicious)."""

    sample_id: str
    mcp_server: str
    server_url: str
    risk_category: str
    attack_paradigm: str
    user_query: str
    tools: list[dict[str, Any]]
    is_malicious: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MCPToxPair:
    """A benign/malicious pair sharing the same query."""

    benign: MCPToxSample
    malicious: MCPToxSample
    poisoned_tool_name: str
    replaced_tool_name: str | None  # set for "replace"-shape attacks
    attack_shape: str               # "insert" or "replace"
    query_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.benign.sample_id,
            "mcp_server": self.benign.mcp_server,
            "server_url": self.benign.server_url,
            "risk_category": self.benign.risk_category,
            "attack_paradigm": self.benign.attack_paradigm,
            "attack_shape": self.attack_shape,
            "user_query": self.benign.user_query,
            "tool_list_benign": self.benign.tools,
            "tool_list_malicious": self.malicious.tools,
            "poisoned_tool_name": self.poisoned_tool_name,
            "replaced_tool_name": self.replaced_tool_name,
            "query_id": self.query_id,
        }


# ──────────────────────────────────────────────────────────────────────────
# Pair construction
# ──────────────────────────────────────────────────────────────────────────

def _normalize_risk(category: str) -> str:
    """``"Privacy Leakage"`` → ``"privacy_leakage"`` for safe groupby."""
    return category.strip().lower().replace(" ", "_")


def assert_pair_invariants(pair: MCPToxPair) -> None:
    """Defensive sanity check used both during build and as a Stage-0 utility.

    Asserts:
      - same ``user_query``
      - same ``mcp_server`` / ``risk_category`` / ``attack_paradigm``
      - benign side does NOT contain the poisoned tool name
      - malicious side contains the poisoned tool name exactly once
      - benign tool ordering is a sub-sequence of malicious tool ordering
        (after removing the poisoned tool from malicious, and re-inserting the
        replaced tool for replace-shape attacks)
    """
    b, m = pair.benign, pair.malicious
    assert b.user_query == m.user_query, "user_query mismatch in pair"
    assert b.mcp_server == m.mcp_server
    assert b.risk_category == m.risk_category
    assert b.attack_paradigm == m.attack_paradigm
    assert not b.is_malicious and m.is_malicious

    benign_names = [t["function"]["name"] for t in b.tools]
    mal_names = [t["function"]["name"] for t in m.tools]

    poisoned = normalize_tool_name(pair.poisoned_tool_name)
    assert poisoned not in benign_names, f"poisoned name leaked into benign: {poisoned}"
    assert mal_names.count(poisoned) == 1, (
        f"poisoned name should appear exactly once in malicious; got "
        f"{mal_names.count(poisoned)}"
    )

    if pair.attack_shape == "insert":
        # Malicious is benign with one extra tool inserted.
        mal_minus_poisoned = [n for n in mal_names if n != poisoned]
        assert mal_minus_poisoned == benign_names, (
            "insert-shape attack should leave the legit-tool sequence unchanged"
        )
    else:
        # Replace-shape: one legit tool is missing from malicious; reinsert it
        # in the same position.
        replaced = normalize_tool_name(pair.replaced_tool_name or "")
        assert replaced and replaced in benign_names, (
            f"replace-shape attack but replaced tool not found in benign: {replaced}"
        )
        # Build the "what malicious would look like with the poisoned tool
        # swapped back" version, then compare.
        rebuilt = [replaced if n == poisoned else n for n in mal_names]
        assert rebuilt == benign_names, (
            "replace-shape: substituting poisoned→replaced should restore benign order"
        )


def _iter_raw_cases(response_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per MCPTox attack case (1348 total)."""
    logger.info("Loading raw MCPTox response file: {}", response_path)
    with response_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    for server_name, sv in payload["servers"].items():
        clean_sys = sv["clean_system_promot"]
        server_url = sv.get("server_url", "")
        for mi_idx, mi in enumerate(sv["malicious_instance"]):
            metadata = mi.get("metadata", {})
            for d_idx, dd in enumerate(mi["datas"]):
                yield {
                    "server_name": server_name,
                    "server_url": server_url,
                    "mi_idx": mi_idx,
                    "d_idx": d_idx,
                    "query": dd["query"],
                    "query_id": dd.get("id", -1),
                    "clean_system_prompt": clean_sys,
                    "malicious_system_prompt": dd["system"],
                    "risk_category": metadata.get("security risk", "Unknown"),
                    "attack_paradigm": metadata.get("paradigm", "Unknown"),
                }


def _build_one_pair(case: dict[str, Any]) -> MCPToxPair:
    legit_parsed = parse_tool_list(case["clean_system_prompt"])
    mal_parsed = parse_tool_list(case["malicious_system_prompt"])

    legit_names = [t.name for t in legit_parsed]
    mal_names = [t.name for t in mal_parsed]
    legit_set = set(legit_names)
    mal_set = set(mal_names)

    extras = mal_set - legit_set
    if len(extras) != 1:
        raise ValueError(
            f"Case {case['server_name']}.mi[{case['mi_idx']}].d[{case['d_idx']}]: "
            f"expected exactly 1 poisoned tool name, got {extras!r}"
        )
    poisoned_name = next(iter(extras))

    missing = legit_set - mal_set
    if len(missing) == 0:
        attack_shape = "insert"
        replaced_name: str | None = None
    elif len(missing) == 1:
        attack_shape = "replace"
        replaced_name = next(iter(missing))
    else:
        raise ValueError(
            f"Case {case['server_name']}.mi[{case['mi_idx']}].d[{case['d_idx']}]: "
            f"more than one legit tool missing in malicious side: {missing!r}"
        )

    sample_id = f"{case['server_name']}.mi[{case['mi_idx']}].d[{case['d_idx']}]"
    risk = _normalize_risk(case["risk_category"])
    paradigm = case["attack_paradigm"]
    server = case["server_name"]
    server_url = case["server_url"]
    query = case["query"]

    benign_tools = [to_openai_schema(t) for t in legit_parsed]
    malicious_tools = [to_openai_schema(t) for t in mal_parsed]

    benign = MCPToxSample(
        sample_id=sample_id,
        mcp_server=server,
        server_url=server_url,
        risk_category=risk,
        attack_paradigm=paradigm,
        user_query=query,
        tools=benign_tools,
        is_malicious=False,
    )
    malicious = MCPToxSample(
        sample_id=sample_id,
        mcp_server=server,
        server_url=server_url,
        risk_category=risk,
        attack_paradigm=paradigm,
        user_query=query,
        tools=malicious_tools,
        is_malicious=True,
    )
    pair = MCPToxPair(
        benign=benign,
        malicious=malicious,
        poisoned_tool_name=poisoned_name,
        replaced_tool_name=replaced_name,
        attack_shape=attack_shape,
        query_id=case["query_id"],
    )
    assert_pair_invariants(pair)
    return pair


def build_pairs(
    response_path: Path = config.MCPTOX_RESPONSE_FILE,
    *,
    on_error: str = "raise",
) -> Iterator[MCPToxPair]:
    """Iterate every MCPTox case and yield :class:`MCPToxPair`.

    ``on_error``:
      - ``"raise"`` (default): re-raise parser/invariant failures.
      - ``"skip"``: log and continue. Useful for first-pass smoke runs.
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    n_emitted = 0
    n_failed = 0
    for case in _iter_raw_cases(response_path):
        try:
            pair = _build_one_pair(case)
        except Exception as exc:
            n_failed += 1
            if on_error == "raise":
                raise
            logger.warning(
                "Skipped case {}.mi[{}].d[{}]: {}",
                case["server_name"],
                case["mi_idx"],
                case["d_idx"],
                exc,
            )
            continue
        n_emitted += 1
        yield pair
    logger.info("Built {} pairs ({} skipped due to errors)", n_emitted, n_failed)


# ──────────────────────────────────────────────────────────────────────────
# JSONL persistence
# ──────────────────────────────────────────────────────────────────────────

def write_pairs_jsonl(
    pairs: Iterable[MCPToxPair], out_path: Path = config.MCPTOX_PAIRS_PATH
) -> int:
    """Atomically write pairs to JSONL (``.tmp → rename``). Returns the count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    n = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp_path, out_path)
    logger.info("Wrote {} pairs → {}", n, out_path)
    return n


def load_pairs(path: Path = config.MCPTOX_PAIRS_PATH) -> list[MCPToxPair]:
    """Load a JSONL produced by :func:`write_pairs_jsonl` back into memory.

    The returned list is the canonical input to Stage 1 capture.
    """
    path = Path(path)
    pairs: list[MCPToxPair] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            benign = MCPToxSample(
                sample_id=obj["sample_id"],
                mcp_server=obj["mcp_server"],
                server_url=obj["server_url"],
                risk_category=obj["risk_category"],
                attack_paradigm=obj["attack_paradigm"],
                user_query=obj["user_query"],
                tools=obj["tool_list_benign"],
                is_malicious=False,
            )
            malicious = MCPToxSample(
                sample_id=obj["sample_id"],
                mcp_server=obj["mcp_server"],
                server_url=obj["server_url"],
                risk_category=obj["risk_category"],
                attack_paradigm=obj["attack_paradigm"],
                user_query=obj["user_query"],
                tools=obj["tool_list_malicious"],
                is_malicious=True,
            )
            pairs.append(
                MCPToxPair(
                    benign=benign,
                    malicious=malicious,
                    poisoned_tool_name=obj["poisoned_tool_name"],
                    replaced_tool_name=obj.get("replaced_tool_name"),
                    attack_shape=obj["attack_shape"],
                    query_id=obj["query_id"],
                )
            )
    logger.info("Loaded {} pairs from {}", len(pairs), path)
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# Server-name filtering
# ──────────────────────────────────────────────────────────────────────────

def filter_pairs(
    pairs: list[MCPToxPair],
    *,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
) -> list[MCPToxPair]:
    """Return a subset of *pairs* filtered by ``mcp_server`` name.

    Parameters
    ----------
    pairs
        List produced by :func:`load_pairs` or :func:`build_pairs`.
    exclude
        Server names to **drop**. Case-sensitive. Mutually exclusive with
        *include*.
    include
        Server names to **keep** (everything else is dropped). Case-sensitive.
        Mutually exclusive with *exclude*.

    Returns
    -------
    Filtered list preserving the original order.

    Raises
    ------
    ValueError
        If both *exclude* and *include* are given, or if a requested server
        name does not appear in *pairs* (typo guard).

    Examples
    --------
    ::

        # Drop two noisy servers before capture
        pairs = filter_pairs(pairs, exclude=["Email", "Slack"])

        # Keep only the servers you care about
        pairs = filter_pairs(pairs, include=["FileSystem", "GitHub"])
    """
    if exclude is not None and include is not None:
        raise ValueError("Specify at most one of 'exclude' or 'include', not both.")

    available = {p.benign.mcp_server for p in pairs}

    if exclude is not None:
        unknown = set(exclude) - available
        if unknown:
            raise ValueError(
                f"filter_pairs(exclude=...): unknown server names: {sorted(unknown)}\n"
                f"Available: {sorted(available)}"
            )
        excluded_set = set(exclude)
        filtered = [p for p in pairs if p.benign.mcp_server not in excluded_set]
        logger.info(
            "filter_pairs: excluded {} server(s) {} — {} → {} pairs",
            len(excluded_set), sorted(excluded_set), len(pairs), len(filtered),
        )
        return filtered

    if include is not None:
        unknown = set(include) - available
        if unknown:
            raise ValueError(
                f"filter_pairs(include=...): unknown server names: {sorted(unknown)}\n"
                f"Available: {sorted(available)}"
            )
        included_set = set(include)
        filtered = [p for p in pairs if p.benign.mcp_server in included_set]
        logger.info(
            "filter_pairs: kept {} server(s) {} — {} → {} pairs",
            len(included_set), sorted(included_set), len(pairs), len(filtered),
        )
        return filtered

    return list(pairs)


def list_servers(pairs: list[MCPToxPair]) -> dict[str, int]:
    """Return a ``{server_name: pair_count}`` mapping, sorted by name."""
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.benign.mcp_server] = counts.get(p.benign.mcp_server, 0) + 1
    return dict(sorted(counts.items()))


# ──────────────────────────────────────────────────────────────────────────
# Convenience: schema inspection (used by scripts/run_data.py --inspect-schema)
# ──────────────────────────────────────────────────────────────────────────

def inspect_schema(response_path: Path = config.MCPTOX_RESPONSE_FILE) -> dict[str, Any]:
    """Return high-level statistics for a quick sanity check at first run."""
    with response_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    servers = payload["servers"]

    n_cases = sum(
        len(mi["datas"]) for sv in servers.values() for mi in sv["malicious_instance"]
    )
    risk_counts: dict[str, int] = {}
    paradigm_counts: dict[str, int] = {}
    for sv in servers.values():
        for mi in sv["malicious_instance"]:
            for _ in mi["datas"]:
                meta = mi.get("metadata", {})
                risk_counts[meta.get("security risk", "Unknown")] = (
                    risk_counts.get(meta.get("security risk", "Unknown"), 0) + 1
                )
                paradigm_counts[meta.get("paradigm", "Unknown")] = (
                    paradigm_counts.get(meta.get("paradigm", "Unknown"), 0) + 1
                )

    return {
        "data_length": payload.get("data_length"),
        "computed_case_count": n_cases,
        "n_servers": len(servers),
        "servers": sorted(servers.keys()),
        "risk_categories": dict(sorted(risk_counts.items())),
        "paradigms": dict(sorted(paradigm_counts.items())),
    }
