"""Forced tool-call output prefix construction (parity with the notebooks).

Why this exists
---------------

Tool selection is only legible at the autoregressive step where the model is
about to emit the **tool name**. The early notebooks
(`tool_selection_patching.ipynb`, `head_heatmap_patching.ipynb`,
`attn_head_viewer.ipynb`) achieve this by appending a *forced output prefix* to
the chat-template text so that the model is teacher-forced into committing to a
tool call. For Qwen3 the prefix is::

    <think>\n</think>\n\n<tool_call>\n{"name": "

After this prefix the very next token is the first token of the chosen tool's
name, so:

* the **commit position** is simply the last token of the forced input, and
* the **target token** is ``first_token('{"name": "' + tool_name)``.

Without the prefix (the bug fixed here) the forward pass measures logits at the
generic generation-prompt position, which does *not* reflect the tool-selection
decision and produces results that disagree with the notebooks.

This module derives the correct prefix per model family from the provider's
``_TOOL_CALL_TOKENS`` registry so the same code path works for Qwen, Phi,
Llama, Gemma-3 and Gemma-4.
"""

from __future__ import annotations

from dataclasses import dataclass

JSON_NAME_PREFIX = '{"name": "'


@dataclass(frozen=True)
class ForcedPrefix:
    """Resolved forced-output prefix for one model."""

    suffix: str          # appended to the chat-template text
    name_context: str    # the literal text immediately preceding the tool name
                         # (used to compute the name's first token in-context)


def resolve_forced_prefix(provider) -> ForcedPrefix:
    """Return the forced output prefix for *provider*'s loaded model."""
    tokens = provider.get_tool_call_tokens() or {}
    model_id = getattr(provider, "model_id", "").lower()

    # Qwen3 is a reasoning model: force an empty <think> block so the model
    # immediately emits the tool call (non-thinking mode), matching the notebooks.
    think = "<think>\n</think>\n\n" if "qwen" in model_id else ""

    gemma4 = tokens.get("gemma4_tool_call_prefix", "")
    if gemma4:
        # Gemma-4: "<|tool_call>call:TOOLNAME{...}" — the name follows ``call:``
        # directly (no JSON wrapper).
        return ForcedPrefix(suffix=think + gemma4, name_context=gemma4)

    call_start = tokens.get("call_start", "")
    if call_start:
        suffix = think + call_start + "\n" + JSON_NAME_PREFIX
    else:
        # Llama-3.1 custom tools / Gemma-2/3 emit a bare JSON object.
        suffix = think + JSON_NAME_PREFIX
    return ForcedPrefix(suffix=suffix, name_context=JSON_NAME_PREFIX)


def render_forced_text(provider, query: str, tools: list[dict]) -> tuple[str, ForcedPrefix]:
    """Render chat template + forced prefix into a single decodable string."""
    rendered = provider.tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    fp = resolve_forced_prefix(provider)
    # Guard against templates that already opened a <think> block at the tail.
    suffix = fp.suffix
    if suffix.startswith("<think>") and rendered.rstrip().endswith("<think>"):
        suffix = suffix[len("<think>"):]
    return rendered + suffix, fp


def build_forced_input_ids(provider, query: str, tools: list[dict]):
    """Tokenise ``chat_template(query, tools) + forced_prefix`` → (input_ids, ForcedPrefix).

    ``input_ids`` is a ``(1, seq)`` LongTensor whose **last** position is the
    commit position. ``add_special_tokens=False`` because the rendered text
    already contains the template's special-token strings.
    """
    text, fp = render_forced_text(provider, query, tools)
    input_ids = provider.tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    return input_ids, fp


def target_token_id(provider, tool_name: str, name_context: str) -> int:
    """First token id of *tool_name* **in context** of *name_context*.

    Tokenisation is context-sensitive, so the first token of ``"weather"``
    encoded alone can differ from its first token after ``'{"name": "'``.
    Computing it in-context matches the notebooks exactly.
    """
    prefix_ids = provider.tokenizer.encode(name_context, add_special_tokens=False)
    full_ids = provider.tokenizer.encode(name_context + tool_name, add_special_tokens=False)
    name_ids = full_ids[len(prefix_ids):]
    if not name_ids:
        raise ValueError(
            f"could not isolate name tokens for {tool_name!r} in context "
            f"{name_context!r} (prefix={prefix_ids}, full={full_ids})"
        )
    return int(name_ids[0])


__all__ = [
    "ForcedPrefix",
    "JSON_NAME_PREFIX",
    "build_forced_input_ids",
    "render_forced_text",
    "resolve_forced_prefix",
    "target_token_id",
]
