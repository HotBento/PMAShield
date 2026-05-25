"""
Per-head compact feature extraction from a single attention capture.

Goal
----
Storing the raw attention rows for thousands of samples is infeasible
(``N_samples × 40 layers × 32 heads × ~3000 tokens × float32`` is many GB).
This module reduces each attention row to a fixed-length feature vector that
keeps everything Stage 2/4 actually need:

* per-tool attention mass on each tool's name span (for soft / hard vote)
* per-tool attention mass on each tool's description span (for cluster signature)
* attention mass on user query / system / "self" (= the model's own already-generated tokens)
* entropy of the full row (specialisation indicator)
* argmax-region label (which kind of position carries the head's peak attention)
* concentration = max-tool-name-attn / total-tool-name-attn (single-tool focus)
* n_tools (sanity / used by downstream code to mask padding)

Per-tool slots are padded to :data:`MAX_TOOLS` with NaN so the resulting tensor
is fixed-shape ``(N_pairs, 2, num_layers, num_heads, FEAT_DIM)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .spans import PromptSpans

# ──────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────

#: Hard cap on the number of tools per sample. The MCPTox snapshot used in
#: this project has at most 21 tools per sample (FileSystem and Github tied),
#: so 24 leaves head-room.
MAX_TOOLS: int = 24

#: Numeric ids for ``argmax_region`` — the type of token that carries the
#: head's maximum attention at the selection step.
REGION_NAME = 0
REGION_DESC = 1
REGION_QUERY = 2
REGION_SELF = 3
REGION_OTHER = 4
REGION_PARAM = 5
REGION_NAMES = ("tool_name", "tool_desc", "user_query", "self", "other", "tool_param")

#: Number of scalar (non-per-tool) features.
N_SCALAR = 7  # query, self, other, entropy, argmax_region, concentration, n_tools

#: Total feature dimension per (sample, side, layer, head).
#: Layout: [0:MAX_TOOLS]=per_tool_name, [MAX_TOOLS:2*MAX_TOOLS]=per_tool_desc,
#:         [2*MAX_TOOLS:3*MAX_TOOLS]=per_tool_param, then 7 scalars.
FEAT_DIM: int = 3 * MAX_TOOLS + N_SCALAR


def feature_layout() -> dict[str, slice | int]:
    """Return slices/indices into the packed feature vector.

    Use as ``feats[..., feature_layout()['per_tool_name']]`` etc.
    """
    return {
        "per_tool_name":  slice(0, MAX_TOOLS),
        "per_tool_desc":  slice(MAX_TOOLS, 2 * MAX_TOOLS),
        "per_tool_param": slice(2 * MAX_TOOLS, 3 * MAX_TOOLS),
        "attn_user_query": 3 * MAX_TOOLS + 0,
        "attn_self":       3 * MAX_TOOLS + 1,
        "attn_other":      3 * MAX_TOOLS + 2,
        "attn_entropy":    3 * MAX_TOOLS + 3,
        "argmax_region":   3 * MAX_TOOLS + 4,
        "concentration":   3 * MAX_TOOLS + 5,
        "n_tools":         3 * MAX_TOOLS + 6,
    }


# ──────────────────────────────────────────────────────────────────────────
# Attention-row extraction
# ──────────────────────────────────────────────────────────────────────────

def attention_row_per_head(layer_attn: Any) -> np.ndarray:
    """Return per-head attention vector at the selection query position.

    Accepts either a ``torch.Tensor`` or ``np.ndarray`` of shape

            * ``(num_heads, key_len)``           — already row-only
      * ``(num_heads, 1, key_len)``        — generation step (KV cached)
      * ``(num_heads, q_len, key_len)``    — full prefill: take the last query

    Returns a ``np.ndarray`` of shape ``(num_heads, key_len)`` in float32.
    """
    # `torch.bfloat16` cannot always be converted directly by NumPy.
    # Convert via float32 first when needed.
    if hasattr(layer_attn, "detach") and hasattr(layer_attn, "cpu"):
        tensor = layer_attn.detach().cpu()
        if str(getattr(tensor, "dtype", "")).endswith("bfloat16"):
            tensor = tensor.float()
        arr = np.asarray(tensor.numpy(), dtype=np.float64)
    else:
        arr = np.asarray(layer_attn, dtype=np.float64)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(
            f"layer_attn must be 2-D or 3-D; got shape {arr.shape}"
        )
    if arr.shape[1] == 1:
        return arr[:, 0, :].astype(np.float32, copy=False)
    return arr[:, -1, :].astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────
# Mask construction over the full key axis
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _SampleMasks:
    """Boolean masks of length ``key_len`` for one sample."""

    per_tool_name:  np.ndarray  # (N_tools, key_len)
    per_tool_desc:  np.ndarray  # (N_tools, key_len)
    per_tool_param: np.ndarray  # (N_tools, key_len)  — NEW
    user_query:  np.ndarray     # (key_len,)
    self_region: np.ndarray     # (key_len,)
    other_region: np.ndarray    # (key_len,)
    tool_order: list[str]


def build_sample_masks(
    spans: PromptSpans,
    *,
    key_len: int,
) -> _SampleMasks:
    """Build masks of length ``key_len`` from a :class:`PromptSpans`.

    The "self" region covers indices ``[input_len, key_len)`` — the tokens
    the model has emitted up to (and including) the selection step
    (``<tool_call>\\n{"name": "``). The "other" region is everything left
    inside ``[0, input_len)`` after subtracting tool / query coverage.

    Priority for disjoint masks: name > desc > param > query.
    """
    L = key_len
    n_tools = spans.n_tools()
    name_per_tool  = np.zeros((n_tools, L), dtype=bool)
    desc_per_tool  = np.zeros((n_tools, L), dtype=bool)
    param_per_tool = np.zeros((n_tools, L), dtype=bool)
    for i, t in enumerate(spans.tool_order):
        if t in spans.tool_name:
            s, e = spans.tool_name[t]
            name_per_tool[i, s:e] = True
        if t in spans.tool_desc:
            s, e = spans.tool_desc[t]
            desc_per_tool[i, s:e] = True
        if t in spans.tool_param:
            s, e = spans.tool_param[t]
            param_per_tool[i, s:e] = True

    name_total  = name_per_tool.any(axis=0)
    desc_total  = desc_per_tool.any(axis=0)  & ~name_total
    param_total = param_per_tool.any(axis=0) & ~name_total & ~desc_total

    query_mask = np.zeros(L, dtype=bool)
    if spans.user_query is not None:
        s, e = spans.user_query
        query_mask[s:e] = True
    query_mask &= ~(name_total | desc_total | param_total)

    self_mask = np.zeros(L, dtype=bool)
    self_mask[spans.input_len :] = True

    other_mask = np.ones(L, dtype=bool)
    other_mask &= ~name_total
    other_mask &= ~desc_total
    other_mask &= ~param_total
    other_mask &= ~query_mask
    other_mask &= ~self_mask

    # Re-disjointise per-tool masks with the same priority order.
    desc_per_tool  = desc_per_tool  & ~name_total
    param_per_tool = param_per_tool & ~name_total & ~desc_total

    return _SampleMasks(
        per_tool_name=name_per_tool,
        per_tool_desc=desc_per_tool,
        per_tool_param=param_per_tool,
        user_query=query_mask,
        self_region=self_mask,
        other_region=other_mask,
        tool_order=list(spans.tool_order),
    )


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────

def extract_head_features(
    attn_row: np.ndarray,
    masks: _SampleMasks,
) -> np.ndarray:
    """Compute the packed feature vector for ONE head.

    ``attn_row`` is shape ``(key_len,)``. Returns shape ``(FEAT_DIM,)`` in
    float32 with NaN padding for unused per-tool slots.
    """
    out = np.full(FEAT_DIM, np.nan, dtype=np.float32)
    layout = feature_layout()

    n_tools = len(masks.tool_order)
    if n_tools > MAX_TOOLS:
        raise ValueError(
            f"Sample has {n_tools} tools — exceeds MAX_TOOLS={MAX_TOOLS}. "
            "Increase MAX_TOOLS in features.py."
        )

    # Per-tool sums.
    name_sums  = (masks.per_tool_name  * attn_row[None, :]).sum(axis=1)  # (n_tools,)
    desc_sums  = (masks.per_tool_desc  * attn_row[None, :]).sum(axis=1)
    param_sums = (masks.per_tool_param * attn_row[None, :]).sum(axis=1)
    out[: n_tools] = name_sums
    out[MAX_TOOLS : MAX_TOOLS + n_tools] = desc_sums
    out[2 * MAX_TOOLS : 2 * MAX_TOOLS + n_tools] = param_sums

    # Region totals.
    out[layout["attn_user_query"]] = attn_row[masks.user_query].sum()
    out[layout["attn_self"]] = attn_row[masks.self_region].sum()
    out[layout["attn_other"]] = attn_row[masks.other_region].sum()

    # Entropy over the full attention row (natural log).
    p = np.clip(attn_row, 1e-12, 1.0)
    out[layout["attn_entropy"]] = float(-(p * np.log(p)).sum())

    # Argmax-region: which region contains the position with max attention.
    argmax_idx = int(attn_row.argmax())
    name_total  = masks.per_tool_name.any(axis=0)
    desc_total  = masks.per_tool_desc.any(axis=0)
    param_total = masks.per_tool_param.any(axis=0)
    if name_total[argmax_idx]:
        region = REGION_NAME
    elif desc_total[argmax_idx]:
        region = REGION_DESC
    elif param_total[argmax_idx]:
        region = REGION_PARAM
    elif masks.user_query[argmax_idx]:
        region = REGION_QUERY
    elif masks.self_region[argmax_idx]:
        region = REGION_SELF
    else:
        region = REGION_OTHER
    out[layout["argmax_region"]] = float(region)

    # Concentration: max(p_t) / sum(p_t) over tool-name attention.
    name_total_mass = float(name_sums.sum())
    if name_total_mass > 0.0:
        out[layout["concentration"]] = float(name_sums.max() / name_total_mass)
    else:
        out[layout["concentration"]] = np.nan

    out[layout["n_tools"]] = float(n_tools)
    return out


def extract_all_heads(
    last_attentions: Sequence[Any],
    spans: PromptSpans,
) -> np.ndarray:
    """Aggregate over all (layer, head) for one sample.

    ``last_attentions`` is the per-layer tuple from
    :class:`mcp_eval.providers.llm.HuggingFaceProvider.last_attentions`.
    Returns an array of shape ``(num_layers, num_heads, FEAT_DIM)``.
    """
    if len(last_attentions) == 0:
        raise ValueError("last_attentions is empty")

    row_shapes = [attention_row_per_head(x).shape for x in last_attentions]
    num_heads = row_shapes[0][0]
    if any(s[0] != num_heads for s in row_shapes):
        raise ValueError(
            f"Inconsistent num_heads across layers: {row_shapes}"
        )
    key_len = max(s[1] for s in row_shapes)
    num_layers = len(last_attentions)

    masks = build_sample_masks(spans, key_len=key_len)
    feats = np.full((num_layers, num_heads, FEAT_DIM), np.nan, dtype=np.float32)

    for layer_idx, layer_attn in enumerate(last_attentions):
        rows = attention_row_per_head(layer_attn)
        if rows.shape[0] != num_heads:
            raise ValueError(
                f"Inconsistent attention shapes: layer {layer_idx} has shape "
                f"{rows.shape}, expected {(num_heads, key_len)}"
            )
        if rows.shape[1] < key_len:
            # Sliding-window layers expose a shorter key axis. Right-align so
            # indices near sequence end remain aligned across layers.
            pad = np.zeros((num_heads, key_len - rows.shape[1]), dtype=np.float32)
            rows = np.concatenate([pad, rows], axis=1)
        elif rows.shape[1] > key_len:
            rows = rows[:, -key_len:]
        for h in range(num_heads):
            feats[layer_idx, h] = extract_head_features(rows[h], masks)

    return feats
