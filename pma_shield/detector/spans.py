"""
Token-span identification utilities (used by Stage 1 capture).

Why this is a dedicated module
------------------------------
Span finding is the single most error-prone part of the pipeline. Tokenisers
silently merge whitespace, descriptions can contain the literal tool name,
``<|im_start|>`` markers must be respected, etc. Isolating it keeps the
capture loop small and lets us unit-test the tricky cases.

Public API
----------
* ``Span``                — a half-open ``[start, end)`` token range.
* ``PromptSpans``         — full per-sample span manifest.
* ``find_tool_name_spans``
* ``find_tool_desc_spans``
* ``find_tool_param_spans``
* ``find_tool_struct_spans``
* ``find_user_query_span``
* ``find_all_spans``      — top-level convenience.
* ``to_region_masks``     — convert a ``PromptSpans`` into a dict of bool masks
                            of length ``input_len``, ready for use by
                            :mod:`mcp_eval.mcptox.features`.

Span conventions (Track C)
--------------------------
All returned spans are extended to include surrounding JSON syntax so adjacent
spans are contiguous:

* **name** span  — covers ``"name": "<value>",\n``
* **desc** span  — covers ``"description": "<value>",\n``
* **param** span — covers ``"parameters": {...},\n``
* **struct** span — covers the full ``{...},\n`` for each tool object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────

Span = tuple[int, int]


class _TokenizerLike(Protocol):
    """Minimal tokenizer interface needed by this module."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class PromptSpans:
    """All spans located inside the model's tokenised prompt.

    Coordinates are half-open ``[start, end)`` token indices into the prompt
    (length ``input_len``). A missing span is represented by ``None`` (rather
    than a sentinel value) so callers must handle missing data explicitly.

    Track-C additions
    -----------------
    * ``tool_param``  — ``"parameters": {...},\\n`` block per tool.
    * ``tool_struct`` — full ``{...},\\n`` wrapper per tool object.
    Both default to empty dict for backwards compatibility.
    """

    tool_name:   dict[str, Span]
    tool_desc:   dict[str, Span]
    user_query:  Span | None
    input_len:   int
    tool_order:  list[str] = field(default_factory=list)
    tool_param:  dict[str, Span] = field(default_factory=dict)
    tool_struct: dict[str, Span] = field(default_factory=dict)

    def n_tools(self) -> int:
        return len(self.tool_order)

    def all_tool_name_positions(self) -> set[int]:
        out: set[int] = set()
        for s, e in self.tool_name.values():
            out.update(range(s, e))
        return out


# ──────────────────────────────────────────────────────────────────────────
# Sliding-window subsequence search
# ──────────────────────────────────────────────────────────────────────────

def _find_subsequence(
    haystack: Sequence[int], needle: Sequence[int], start: int = 0
) -> int | None:
    """Return the smallest index ``i ≥ start`` such that
    ``haystack[i : i + len(needle)] == needle``, or ``None``.
    """
    n, m = len(haystack), len(needle)
    if m == 0 or m > n - start:
        return None
    needle_t = tuple(needle)
    haystack_t = tuple(haystack)
    for i in range(start, n - m + 1):
        if haystack_t[i : i + m] == needle_t:
            return i
    return None


def _extend_span_trailing(
    input_ids: Sequence[int],
    end: int,
    tokenizer: _TokenizerLike,
    max_extra: int = 4,
) -> int:
    """Extend span end rightward to absorb trailing comma/newline/whitespace.

    Looks ahead up to ``max_extra`` tokens and includes any whose decoded text
    consists entirely of characters in ``{' ', '\\n', '\\t', ','}``.
    """
    n = len(input_ids)
    extra = 0
    while end < n and extra < max_extra:
        try:
            tok = tokenizer.decode([input_ids[end]])
        except Exception:
            break
        if tok and all(c in ' \n\t,' for c in tok):
            end += 1
            extra += 1
        else:
            break
    return end


def _extend_span_with_label(
    input_ids: Sequence[int],
    start: int,
    label_text: str,
    tokenizer: _TokenizerLike,
    window: int = 20,
) -> tuple[int, int]:
    """Extend span start leftward to include the JSON label.

    Searches ``input_ids[max(0, start-window):start]`` for the token sequence
    that encodes ``label_text`` (e.g. ``'"description": "'``).  Returns a
    ``(new_start, label_end)`` tuple.  ``new_start`` is the position of the
    first token of the matched label; ``label_end`` is the position immediately
    after it (i.e. the first value token).  When no label is found, returns
    ``(start, start)``.

    Callers should verify ``label_end == value_start`` to ensure the matched
    label is directly adjacent to the value tokens (no gap), preventing
    false-positive matches where the label belongs to a different tool.
    """
    # Build a list of candidate token sequences for the label.  BPE
    # tokenisers encode ``"name": "`` differently depending on the surrounding
    # JSON context (e.g. preceded by ``{`` vs. standing alone), so we try
    # both the standalone form and the ``{``-prefixed form.
    # A truncated variant (without the trailing ' "' token) is also tried for
    # the case where the closing quote has merged with the first character of
    # the tool name (e.g. ' "' + '@' → ' "@' in MCP @-style names).  When the
    # truncated variant is used, we require that the token immediately after
    # the matched label equals ``input_ids[start]`` — this ensures we have
    # found the label that directly precedes the value we located, preventing
    # false-positive matches on other tools' name labels.
    candidate_label_ids: list[list[int]] = []
    # truncated variants stored separately so we can apply the extra check
    truncated_label_ids: list[list[int]] = []

    def _add(seq: list[int], trunc: bool = False) -> None:
        if seq and seq not in candidate_label_ids and seq not in truncated_label_ids:
            (truncated_label_ids if trunc else candidate_label_ids).append(seq)

    standalone = tokenizer.encode(label_text, add_special_tokens=False)
    _add(standalone)

    open_brace_ids = tokenizer.encode("{", add_special_tokens=False)
    ctx_ids = tokenizer.encode("{" + label_text, add_special_tokens=False)
    ctx_label: list[int] = []
    if open_brace_ids and len(ctx_ids) > len(open_brace_ids):
        ctx_label = ctx_ids[len(open_brace_ids):]
        _add(ctx_label)

    # Truncated variants (drop trailing ' "' token): used when the closing
    # quote has BPE-merged with the first character of the value.
    if len(standalone) > 1:
        _add(standalone[:-1], trunc=True)
    if len(ctx_label) > 1:
        _add(ctx_label[:-1], trunc=True)

    best_start = start
    best_end = start  # label_end == start means "not found"
    for label_ids in candidate_label_ids:
        n_label = len(label_ids)
        for i in range(start - n_label, max(-1, start - window - 1), -1):
            if i < 0:
                break
            if list(input_ids[i : i + n_label]) == label_ids:
                if i < best_start:
                    best_start = i
                    best_end = i + n_label
                break

    # Only try truncated variants if full variants didn't improve the result,
    # and only accept a truncated match when its immediately-following token
    # equals input_ids[start] (confirming this label precedes our target value).
    if best_start == start:
        for label_ids in truncated_label_ids:
            n_label = len(label_ids)
            for i in range(start - n_label, max(-1, start - window - 1), -1):
                if i < 0:
                    break
                if list(input_ids[i : i + n_label]) == label_ids:
                    # Verify the label is directly followed by the value token
                    # we're looking for (i.e. no extra tokens in between).
                    if i + n_label < len(input_ids) and input_ids[i + n_label] == input_ids[start]:
                        if i < best_start:
                            best_start = i
                            best_end = i + n_label
                    break

    return best_start, best_end


def _candidate_token_sequences(
    text: str,
    tokenizer: _TokenizerLike,
    *,
    json_context: bool,
    return_full_context: bool = False,
) -> list[list[int]]:
    """Return tokenisations of ``text`` to try in priority order.

    Long text (descriptions, queries) is robust to context, so a single
    standalone encoding suffices. Short tool names can tokenise differently
    depending on their JSON neighbours, so we additionally try the in-context
    form ``'"name": "<text>"'`` and slice out the inner tokens.

    When ``return_full_context=True`` (only relevant when ``json_context=True``)
    the full ``'"name": "<text>"'`` token sequence is returned instead of the
    inner slice, enabling callers to locate the entire label+value in one
    search.  Strategies 2 and 3 are suppressed in that case because they do
    not include the JSON label.
    """
    sequences: list[list[int]] = []

    # Strategy 1 — JSON context (only useful for short tokens / names).
    if json_context:
        prefix = '"name": "'
        suffix = '"'
        ctx_ids = tokenizer.encode(prefix + text + suffix, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        prefix_ok = (
            len(ctx_ids) > len(prefix_ids) + len(suffix_ids)
            and ctx_ids[: len(prefix_ids)] == prefix_ids
            and ctx_ids[-len(suffix_ids) :] == suffix_ids
        )
        suffix_ok = (
            len(suffix_ids) > 0
            and ctx_ids[-len(suffix_ids) :] == suffix_ids
        )
        if prefix_ok:
            if return_full_context:
                sequences.append(list(ctx_ids))
            else:
                inner = ctx_ids[len(prefix_ids) : len(ctx_ids) - len(suffix_ids)]
                if inner:
                    sequences.append(inner)
        elif suffix_ok and len(ctx_ids) > len(suffix_ids):
            # BPE merged the prefix tokens with the first char of ``text``
            # (e.g. `"@` becomes a single token).  Fall back to using the
            # full ``ctx_ids`` as the search key for Strategy A, or the
            # prefix-merged inner (everything except trailing suffix) for B.
            merged_inner = ctx_ids[: len(ctx_ids) - len(suffix_ids)]
            if return_full_context:
                sequences.append(list(ctx_ids))
            else:
                if merged_inner:
                    sequences.append(merged_inner)

    if not return_full_context:
        # Strategy 2 — standalone encoding (always tried as fallback).
        standalone = tokenizer.encode(text, add_special_tokens=False)
        if standalone and (not sequences or standalone != sequences[0]):
            sequences.append(standalone)

        # Strategy 3 — leading-space variant (BPE often re-uses these tokens).
        if not text.startswith(" "):
            leading = tokenizer.encode(" " + text, add_special_tokens=False)
            if leading and leading not in sequences:
                sequences.append(leading)

        # Strategy 4 — space-quote prefix variant.
        # BPE tokenisers (Qwen, etc.) often merge the surrounding JSON quote
        # with the first character of the tool name value (e.g. ``"@smithery``
        # becomes token `` "@``).  Encoding `` "`` + text reproduces this
        # merged form and makes the subsequence searchable.
        space_quote = ' "' + text
        sq_ids = tokenizer.encode(space_quote, add_special_tokens=False)
        if sq_ids and sq_ids not in sequences:
            sequences.append(sq_ids)

    return sequences


# ──────────────────────────────────────────────────────────────────────────
# Span locators
# ──────────────────────────────────────────────────────────────────────────

def _detect_and_search_gemma_tool_spans(
    input_ids: Sequence[int],
    tool_names: Sequence[str],
    tokenizer: _TokenizerLike,
) -> dict[str, Span] | None:
    """Try to locate tool names in Gemma-4 custom tool format.

    Gemma-4 uses format like: ``<|tool>declaration:TOOLNAME{...}<tool|>``
    instead of standard JSON ``"name": "TOOLNAME"``.

    Returns None if this doesn't appear to be a Gemma model or no matches found.
    """
    # Check if this looks like Gemma by looking for special Gemma tokens
    try:
        gemma_tool_token = tokenizer.encode("<|tool>", add_special_tokens=False)
        if not gemma_tool_token:
            return None
    except Exception:
        return None

    spans: dict[str, Span] = {}
    found_any = False

    for name in tool_names:
        if not name:
            continue

        # Construct the Gemma tool declaration pattern:
        # Search for "declaration:{name}{"
        pattern_text = f"declaration:{name}{{"
        try:
            pattern_ids = tokenizer.encode(pattern_text, add_special_tokens=False)
        except Exception:
            continue

        idx = _find_subsequence(input_ids, pattern_ids)
        if idx is None:
            continue

        # Found it! Now extend the span backwards to include <|tool> marker
        start = idx
        while start > 0:
            # Look for <|tool> backwards
            tool_marker_ids = tokenizer.encode("<|tool>", add_special_tokens=False)
            if tool_marker_ids and list(input_ids[start : start + len(tool_marker_ids)]) == tool_marker_ids:
                start = start
                break
            start -= 1

        # Extend forward to include closing <tool|> marker
        end = idx + len(pattern_ids)
        closing_ids = tokenizer.encode("<tool|>", add_special_tokens=False)
        if closing_ids:
            search_end = min(end + 200, len(input_ids))  # reasonable search window
            for pos in range(end, search_end):
                if (
                    pos + len(closing_ids) <= len(input_ids)
                    and list(input_ids[pos : pos + len(closing_ids)]) == closing_ids
                ):
                    end = pos + len(closing_ids)
                    break

        spans[name] = (start, end)
        found_any = True

    return spans if found_any else None


def find_tool_name_spans(
    input_ids: Sequence[int],
    tool_names: Sequence[str],
    tokenizer: _TokenizerLike,
) -> dict[str, Span]:
    """Locate the JSON-definition occurrence of every tool name in ``input_ids``.

    Tries multiple strategies:
    1. Gemma-4 custom format: ``<|tool>declaration:NAME{...}<tool|>``
    2. Standard JSON format: ``"name": "NAME"``

    The returned span covers the full tool definition so that name, description
    and parameter spans are contiguous.
    """
    spans: dict[str, Span] = {}
    misses: list[str] = []

    # Strategy 0: Try Gemma-4 custom format first
    gemma_spans = _detect_and_search_gemma_tool_spans(input_ids, tool_names, tokenizer)
    if gemma_spans:
        return gemma_spans

    for name in tool_names:
        if not name:
            misses.append(name)
            continue

        found = False

        # Strategy A: search for full "name": "value" sequence
        for seq in _candidate_token_sequences(
            name, tokenizer, json_context=True, return_full_context=True
        ):
            idx = _find_subsequence(input_ids, seq)
            if idx is not None:
                end = _extend_span_trailing(input_ids, idx + len(seq), tokenizer)
                spans[name] = (idx, end)
                found = True
                break

        if found:
            continue

        # Strategy B fallback: inner tokens + extend label backwards.
        # Important: we REQUIRE that _extend_span_with_label actually found the
        # '"name": "' prefix (new_start < idx).  Without this check a tool name
        # that appears verbatim inside another tool's description (e.g. via a
        # backtick reference like `realtime_options`) would be misidentified.
        # When the label is absent we skip that occurrence and continue searching.
        for seq in _candidate_token_sequences(
            name, tokenizer, json_context=True, return_full_context=False
        ):
            search_start = 0
            while True:
                idx = _find_subsequence(input_ids, seq, start=search_start)
                if idx is None:
                    break
                new_start, label_end = _extend_span_with_label(
                    input_ids, idx, '"name": "', tokenizer
                )
                if new_start >= idx or label_end != idx:
                    # Label not found here, or it is not directly adjacent to
                    # the value tokens (gap means the name appeared inside
                    # another tool's description, not in its own "name" field).
                    search_start = idx + 1
                    continue
                end = idx + len(seq)
                # include closing quote if immediately following
                quote_ids = tokenizer.encode('"', add_special_tokens=False)
                if (
                    quote_ids
                    and end + len(quote_ids) <= len(input_ids)
                    and list(input_ids[end : end + len(quote_ids)]) == quote_ids
                ):
                    end += len(quote_ids)
                end = _extend_span_trailing(input_ids, end, tokenizer)
                spans[name] = (new_start, end)
                found = True
                break
            if found:
                break

        if not found:
            misses.append(name)

    if misses:
        raise ValueError(
            f"Failed to locate token spans for tool name(s): {misses!r}. "
            "This usually means the chat template wasn't applied or the "
            "tokenizer differs from the one used at inference."
        )

    # Sanity: tool-name spans should be pairwise disjoint.
    sorted_spans = sorted(spans.values())
    for (a, b), (c, d) in zip(sorted_spans, sorted_spans[1:]):
        if c < b:
            logger.warning(
                "Overlapping tool-name spans detected: ({}, {}) and ({}, {}). "
                "First occurrences may have grabbed the wrong tool.",
                a, b, c, d,
            )
    return spans


def find_tool_desc_spans(
    input_ids: Sequence[int],
    tools: Sequence[Mapping[str, Any]],
    tokenizer: _TokenizerLike,
    *,
    name_spans: Mapping[str, Span] | None = None,
    description_prefix_chars: int = 80,
) -> dict[str, Span]:
    """Locate each tool's description span.

    The returned span is extended to cover ``"description": "<value>",\\n``
    for contiguity with the name and parameter spans.

    Strategy: encode the **full** description text and locate its token
    sequence in the prompt. Falls back to the leading
    ``description_prefix_chars`` characters when the full encoding doesn't
    match.  Search is anchored immediately after the corresponding tool-name
    span so we pin to the right tool even when descriptions share words.
    """
    spans: dict[str, Span] = {}
    name_spans = dict(name_spans or {})
    name_start_by_tool = {n: s for n, (s, _e) in name_spans.items()}
    sorted_by_pos = sorted(name_start_by_tool.items(), key=lambda kv: kv[1])
    next_start: dict[str, int] = {}
    for i, (n, _) in enumerate(sorted_by_pos):
        next_start[n] = (
            sorted_by_pos[i + 1][1] if i + 1 < len(sorted_by_pos) else len(input_ids)
        )

    for tool in tools:
        fn = tool.get("function", {})
        tool_name = fn.get("name", "")
        description = (fn.get("description") or "").strip()
        if not tool_name or not description:
            continue
        anchor = name_spans.get(tool_name, (0, 0))[1]
        right_bound = next_start.get(tool_name, len(input_ids))

        raw_span: Span | None = None

        # Try the full description first.
        for seq in _candidate_token_sequences(description, tokenizer, json_context=False):
            idx = _find_subsequence(input_ids, seq, start=anchor)
            if idx is None:
                idx = _find_subsequence(input_ids, seq)
            if idx is not None and idx + len(seq) <= right_bound:
                raw_span = (idx, idx + len(seq))
                break

        if raw_span is None:
            # Fallback: prefix-only span.
            prefix = description[:description_prefix_chars]
            for seq in _candidate_token_sequences(prefix, tokenizer, json_context=False):
                idx = _find_subsequence(input_ids, seq, start=anchor)
                if idx is None:
                    idx = _find_subsequence(input_ids, seq)
                if idx is not None and idx + len(seq) <= right_bound:
                    raw_span = (idx, idx + len(seq))
                    break

        if raw_span is None:
            # Label-first fallback: search for the "description": " label token
            # sequence directly.  This handles descriptions that start with
            # tokens that co-tokenize with the preceding '"' (e.g. "<IMPORTANT>"),
            # making the content-based search fail.
            desc_label = '"description": "'
            desc_label_ids = tokenizer.encode(desc_label, add_special_tokens=False)
            if desc_label_ids:
                lbl_idx = _find_subsequence(input_ids, desc_label_ids, start=anchor)
                if lbl_idx is not None and lbl_idx < right_bound:
                    # Value starts immediately after the label.
                    val_start = lbl_idx + len(desc_label_ids)
                    # Find end: look for "parameters": label or right_bound.
                    param_label_ids = tokenizer.encode('"parameters": ', add_special_tokens=False)
                    val_end = right_bound
                    if param_label_ids:
                        p_idx = _find_subsequence(input_ids, param_label_ids, start=val_start)
                        if p_idx is not None and p_idx < right_bound:
                            val_end = p_idx
                    raw_span = (val_start, val_end)
                    # Trim trailing quote/comma/newline tokens from end.
                    while val_end > val_start:
                        try:
                            tok = tokenizer.decode([input_ids[val_end - 1]])
                        except Exception:
                            break
                        if tok and all(c in ' \n\t,"' for c in tok):
                            val_end -= 1
                        else:
                            break
                    raw_span = (val_start, val_end)

        if raw_span is None:
            logger.debug("No description span located for tool {!r}", tool_name)
            continue

        # Extend to cover "description": "..." label and trailing ,\n
        s, e = raw_span
        new_start, _label_end = _extend_span_with_label(
            input_ids, s, '"description": "', tokenizer
        )
        # Include closing quote if immediately following the raw text
        quote_ids = tokenizer.encode('"', add_special_tokens=False)
        if (
            quote_ids
            and e + len(quote_ids) <= len(input_ids)
            and list(input_ids[e : e + len(quote_ids)]) == quote_ids
        ):
            e += len(quote_ids)
        e = _extend_span_trailing(input_ids, e, tokenizer)
        spans[tool_name] = (new_start, e)

    return spans


def find_tool_param_spans(
    input_ids: Sequence[int],
    tools: Sequence[Mapping[str, Any]],
    tokenizer: _TokenizerLike,
    *,
    desc_spans: Mapping[str, Span] | None = None,
    name_spans: Mapping[str, Span] | None = None,
) -> dict[str, Span]:
    """Locate each tool's ``"parameters": {...}`` block.

    The span covers from the ``"parameters":`` label token to the matching
    closing ``}`` (inclusive), plus any trailing comma/newline tokens.
    """
    spans: dict[str, Span] = {}

    param_label = '"parameters": '
    param_label_ids = tokenizer.encode(param_label, add_special_tokens=False)
    if not param_label_ids:
        return spans

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue

        # Search anchor: prefer desc span end, fall back to name span end, then 0.
        anchor = 0
        if desc_spans and name in desc_spans:
            anchor = desc_spans[name][1]
        elif name_spans and name in name_spans:
            anchor = name_spans[name][1]

        idx = _find_subsequence(input_ids, param_label_ids, start=anchor)
        if idx is None:
            logger.debug("No 'parameters': span for tool {!r}", name)
            continue

        label_end = idx + len(param_label_ids)

        # Find the opening brace {
        open_pos: int | None = None
        for p in range(label_end, min(label_end + 6, len(input_ids))):
            tok_str = tokenizer.decode([input_ids[p]])
            if '{' in tok_str:
                open_pos = p
                break

        if open_pos is None:
            logger.debug("No opening brace after 'parameters': for {!r}", name)
            continue

        # Count brackets to find the matching }
        depth = 0
        close_pos: int | None = None
        for p in range(open_pos, len(input_ids)):
            tok_str = tokenizer.decode([input_ids[p]])
            depth += tok_str.count('{')
            depth -= tok_str.count('}')
            if depth <= 0:
                close_pos = p
                break

        if close_pos is None:
            logger.debug("Unmatched brace in 'parameters' for {!r}", name)
            continue

        span_end = close_pos + 1
        span_end = _extend_span_trailing(input_ids, span_end, tokenizer)
        spans[name] = (idx, span_end)

    return spans


def find_tool_struct_spans(
    input_ids: Sequence[int],
    tool_order: Sequence[str],
    name_spans: Mapping[str, Span],
    param_spans: Mapping[str, Span],
    desc_spans: Mapping[str, Span],
    tokenizer: _TokenizerLike,
) -> dict[str, Span]:
    """Locate the complete ``{...},\\n`` wrapper for each tool object.

    The span extends from the opening ``{`` (before the name key) to the
    closing ``}`` of the whole tool object plus any trailing comma/newline,
    so that adjacent tool struct spans are contiguous.
    """
    spans: dict[str, Span] = {}

    for name in tool_order:
        if name not in name_spans:
            continue
        ns, _ne = name_spans[name]

        # Find opening { before name span (up to 6 tokens back)
        open_pos: int | None = None
        for p in range(ns - 1, max(-1, ns - 7), -1):
            tok_str = tokenizer.decode([input_ids[p]])
            if '{' in tok_str:
                open_pos = p
                break

        if open_pos is None:
            continue

        # Count brackets from open_pos to find matching }
        depth = 0
        close_pos: int | None = None
        for p in range(open_pos, len(input_ids)):
            tok_str = tokenizer.decode([input_ids[p]])
            depth += tok_str.count('{')
            depth -= tok_str.count('}')
            if depth <= 0:
                close_pos = p
                break

        if close_pos is None:
            continue

        span_end = close_pos + 1
        span_end = _extend_span_trailing(input_ids, span_end, tokenizer)
        spans[name] = (open_pos, span_end)

    return spans


def find_user_query_span(
    input_ids: Sequence[int],
    user_query: str,
    tokenizer: _TokenizerLike,
    *,
    min_prefix_chars: int = 30,
) -> Span | None:
    """Locate the user query inside the prompt.

    Uses the leading ``min_prefix_chars`` characters of the query as the
    anchor (short queries are encoded whole). Returns the full query span if
    located, otherwise ``None`` and a debug log line.
    """
    if not user_query:
        return None
    prefix = user_query[: max(min_prefix_chars, 1)]
    candidates = _candidate_token_sequences(prefix, tokenizer, json_context=False)

    for seq in candidates:
        idx = _find_subsequence(input_ids, seq)
        if idx is None:
            continue
        # The query likely extends beyond the prefix; encode the full query
        # to estimate its end position.
        full_seq = tokenizer.encode(user_query, add_special_tokens=False)
        if full_seq:
            end_candidate = _find_subsequence(input_ids, full_seq)
            if end_candidate is not None:
                return (end_candidate, end_candidate + len(full_seq))
        return (idx, idx + len(seq))

    logger.debug("Failed to locate user-query span (preview: {!r})", user_query[:80])
    return None


def find_all_spans(
    input_ids: Sequence[int],
    tools: Sequence[Mapping[str, Any]],
    user_query: str,
    tokenizer: _TokenizerLike,
) -> PromptSpans:
    """Top-level convenience that returns a fully-populated :class:`PromptSpans`."""
    tool_order = [t.get("function", {}).get("name", "") for t in tools]
    name_spans = find_tool_name_spans(input_ids, tool_order, tokenizer)
    desc_spans = find_tool_desc_spans(
        input_ids, tools, tokenizer, name_spans=name_spans
    )
    param_spans = find_tool_param_spans(
        input_ids, tools, tokenizer, desc_spans=desc_spans, name_spans=name_spans
    )
    struct_spans = find_tool_struct_spans(
        input_ids, tool_order, name_spans, param_spans, desc_spans, tokenizer
    )
    query_span = find_user_query_span(input_ids, user_query, tokenizer)
    return PromptSpans(
        tool_name=name_spans,
        tool_desc=desc_spans,
        user_query=query_span,
        input_len=len(input_ids),
        tool_order=tool_order,
        tool_param=param_spans,
        tool_struct=struct_spans,
    )


# ──────────────────────────────────────────────────────────────────────────
# Mask construction
# ──────────────────────────────────────────────────────────────────────────

def to_region_masks(spans: PromptSpans) -> dict[str, np.ndarray]:
    """Convert spans to ``{region_name → bool mask of length input_len}``.

    Regions:
      * ``tool_name_total``   — union of all per-tool name spans
      * ``tool_desc_total``   — union of all per-tool description spans
      * ``tool_param_total``  — union of all per-tool parameter spans
      * ``tool_struct_total`` — union of all per-tool struct spans (superset)
      * ``user_query``
      * ``other``             — everything in ``[0, input_len)`` not covered above

    Priority for disjoint ``other`` mask: name > desc > param > query.
    The ``tool_struct_total`` mask is the union of name+desc+param; it is
    provided for inspection but does not affect ``other``.
    """
    L = spans.input_len
    name_mask = np.zeros(L, dtype=bool)
    for s, e in spans.tool_name.values():
        name_mask[s:e] = True

    desc_mask = np.zeros(L, dtype=bool)
    for s, e in spans.tool_desc.values():
        desc_mask[s:e] = True

    param_mask = np.zeros(L, dtype=bool)
    for s, e in spans.tool_param.values():
        param_mask[s:e] = True

    struct_mask = np.zeros(L, dtype=bool)
    for s, e in spans.tool_struct.values():
        struct_mask[s:e] = True

    query_mask = np.zeros(L, dtype=bool)
    if spans.user_query is not None:
        s, e = spans.user_query
        query_mask[s:e] = True

    # Resolve overlaps: name > desc > param > query
    desc_mask &= ~name_mask
    param_mask &= ~(name_mask | desc_mask)
    query_mask &= ~(name_mask | desc_mask | param_mask)
    other_mask = ~(name_mask | desc_mask | param_mask | query_mask)

    return {
        "tool_name_total": name_mask,
        "tool_desc_total": desc_mask,
        "tool_param_total": param_mask,
        "tool_struct_total": struct_mask,
        "user_query": query_mask,
        "other": other_mask,
    }
