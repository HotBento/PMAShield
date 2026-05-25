"""Activation patching for tool selection (paper §3.1–§3.3).

LOCAL / REMOTE: **REMOTE-ONLY** — loads a HuggingFace causal LM and runs forwards
on a GPU. Local development uses :mod:`mcp_eval.interp.scripts.make_mock_figures`
to iterate on plot style without ever touching this module.

Method
------

For a contrastive scenario pair ``(s_A, s_B)`` (e.g. weather vs paper search),
we measure how much causal information about *tool identity* is carried by a
component (a whole transformer layer's attention/MLP output, or a single
attention head):

1. Build each input as ``chat_template(query, tools) + forced_prefix`` where
   the forced prefix teacher-forces the model into committing to a tool call
   (for Qwen3: ``<think>\n</think>\n\n<tool_call>\n{"name": "``). The **commit
   position** is then simply the last token of the forced input — the step at
   which the next token is the tool name. See :mod:`mcp_eval.interp.prompting`.
2. Run ``s_A`` and cache the activation of the target component at the commit
   position.
3. Run ``s_B``; install a forward hook that **overwrites** the component
   output at the commit position with the cached value from ``s_A``.
4. Read out the logit assigned to the first token of ``s_A.intended_tool``,
   computed *in JSON context* (``first_tok('{"name": "' + name)``).

This forced-prefix construction is essential — without it the forward measures
logits at the generic generation-prompt position, which does not reflect the
tool-selection decision and disagrees with the reference notebooks.

The metric reported is::

    Δlog p = log p(t_A | s_B, patched) - log p(t_A | s_B, clean)

averaged over both patching directions for each pair. Aggregating across all
contrastive pairs gives the per-component importance reported in
``fig:attn-vs-mlp`` (layer-level) and ``fig:head-heatmap`` (head-level).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pma_shield.interp import prompting
from pma_shield.interp.probe_data import ProbePair, ProbeScenario
from pma_shield.logger import logger


# Lazy torch import — we never want :mod:`mcp_eval.interp.patching` to import
# torch when run from the mock figure path.
def _torch():
    import torch  # type: ignore
    return torch


@dataclass
class CommitState:
    """Cached state for one scenario forward (commit-position slices only).

    Only the commit-position vectors are retained — that is the single token
    whose component activations we patch, matching the reference notebooks
    (`o[0, -1, :]` / `inp[0][0, -1, :]`).
    """

    input_ids: "object"        # torch.LongTensor (1, seq)
    commit_pos: int            # absolute token index of the commit position
    target_token_id: int       # first token of intended_tool name (JSON context)
    attn_at_commit: list       # per-layer (hidden,) attention module output
    mlp_at_commit: list        # per-layer (hidden,) MLP module output
    oproj_in_at_commit: list   # per-layer (n_heads*head_dim,) o_proj input
    clean_logits: "object"     # (1, vocab) logits at commit_pos
    head_dim: int              # per-head slice width in the o_proj input


def _build_forced_inputs(provider, scenario: ProbeScenario):
    """Render *scenario* with the forced tool-call prefix.

    Returns ``(input_ids, commit_pos, target_token_id)`` where ``commit_pos`` is
    the last token index and ``target_token_id`` is the first token of the
    intended tool name *in JSON context*.
    """
    input_ids, fp = prompting.build_forced_input_ids(
        provider, scenario.query, scenario.function_calling_tools()
    )
    commit_pos = int(input_ids.shape[1]) - 1
    target = prompting.target_token_id(provider, scenario.intended_tool, fp.name_context)
    return input_ids, commit_pos, target


# Backwards-compatible alias used by run_head_roles.py span finding.
def _build_input_ids(provider, scenario: ProbeScenario):
    input_ids, _, _ = _build_forced_inputs(provider, scenario)
    return input_ids


def _commit_position(provider, scenario: ProbeScenario) -> int:
    _, commit_pos, _ = _build_forced_inputs(provider, scenario)
    return commit_pos


def _attn_module(block):
    return getattr(block, "self_attn", None) or getattr(block, "attention", None)


def _mlp_module(block):
    return getattr(block, "mlp", None) or getattr(block, "feed_forward", None)


def _oproj_module(block):
    attn = _attn_module(block)
    for name in ("o_proj", "dense", "out_proj", "wo"):
        mod = getattr(attn, name, None)
        if mod is not None:
            return mod
    raise RuntimeError("could not locate the attention output projection module")


def _head_dim(provider) -> int:
    cfg = provider.model.config
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "num_heads")
    return int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))


def _capture_clean_state(provider, scenario: ProbeScenario) -> CommitState:
    """Single forward recording commit-position attn / MLP / o_proj-input slices."""
    torch = _torch()
    layers = provider._get_transformer_layers()
    n_layers = len(layers)
    head_dim = _head_dim(provider)

    input_ids, commit_pos, target = _build_forced_inputs(provider, scenario)
    input_ids = input_ids.to(next(provider.model.parameters()).device)

    attn_at: list = [None] * n_layers
    mlp_at: list = [None] * n_layers
    oproj_at: list = [None] * n_layers
    handles = []

    def _make_attn_hook(idx: int):
        def hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            attn_at[idx] = tensor[0, commit_pos, :].detach().clone()
            return output
        return hook

    def _make_mlp_hook(idx: int):
        def hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            mlp_at[idx] = tensor[0, commit_pos, :].detach().clone()
            return output
        return hook

    def _make_oproj_pre_hook(idx: int):
        def hook(_mod, inp):
            tensor = inp[0]               # (1, seq, n_heads*head_dim)
            oproj_at[idx] = tensor[0, commit_pos, :].detach().clone()
            return None
        return hook

    for idx, block in enumerate(layers):
        attn_mod = _attn_module(block)
        if attn_mod is not None:
            handles.append(attn_mod.register_forward_hook(_make_attn_hook(idx)))
        mlp_mod = _mlp_module(block)
        if mlp_mod is not None:
            handles.append(mlp_mod.register_forward_hook(_make_mlp_hook(idx)))
        handles.append(_oproj_module(block).register_forward_pre_hook(_make_oproj_pre_hook(idx)))

    with torch.no_grad():
        out = provider.model(input_ids=input_ids, use_cache=False)
        clean_logits_at_commit = out.logits[:, commit_pos, :].detach().clone()

    for h in handles:
        h.remove()

    return CommitState(
        input_ids=input_ids,
        commit_pos=commit_pos,
        target_token_id=target,
        attn_at_commit=attn_at,
        mlp_at_commit=mlp_at,
        oproj_in_at_commit=oproj_at,
        clean_logits=clean_logits_at_commit,
        head_dim=head_dim,
    )


def _logodds(logits_row, pos_id: int, neg_id: int) -> float:
    """Contrastive log-odds ``log p(pos) - log p(neg)`` at one position."""
    torch = _torch()
    logp = torch.log_softmax(logits_row.float(), dim=-1)
    return float((logp[0, pos_id] - logp[0, neg_id]).cpu().item())


def _patched_logodds(
    provider,
    target_state: CommitState,
    source_state: CommitState,
    *,
    kind: str,
    layer_idx: int,
    head_idx: int | None = None,
) -> float:
    """Patch one component (source→target) and return the contrastive log-odds.

    The log-odds is ``log p(t_src) - log p(t_tgt)`` at the target's commit
    position, i.e. how much the patched component pushes the prediction toward
    the *source scenario's* tool.
    """
    torch = _torch()
    block = provider._get_transformer_layers()[layer_idx]
    commit_t = target_state.commit_pos
    handle = None

    if kind == "attn":
        src_act = source_state.attn_at_commit[layer_idx]

        def _hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            tensor[0, commit_t, :] = src_act.to(tensor.device, tensor.dtype)
            if isinstance(output, tuple):
                return (tensor,) + output[1:]
            return tensor

        handle = _attn_module(block).register_forward_hook(_hook)
    elif kind == "mlp":
        src_act = source_state.mlp_at_commit[layer_idx]

        def _hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            tensor[0, commit_t, :] = src_act.to(tensor.device, tensor.dtype)
            if isinstance(output, tuple):
                return (tensor,) + output[1:]
            return tensor

        handle = _mlp_module(block).register_forward_hook(_hook)
    elif kind == "head":
        if head_idx is None:
            raise ValueError("kind='head' requires head_idx")
        hd = target_state.head_dim
        lo, hi = head_idx * hd, (head_idx + 1) * hd
        src_slice = source_state.oproj_in_at_commit[layer_idx][lo:hi]

        def _pre_hook(_mod, inp):
            tensor = inp[0].clone()
            tensor[0, commit_t, lo:hi] = src_slice.to(tensor.device, tensor.dtype)
            return (tensor,) + inp[1:]

        handle = _oproj_module(block).register_forward_pre_hook(_pre_hook)
    else:
        raise ValueError(f"unknown patch kind: {kind!r}")

    try:
        with torch.no_grad():
            out = provider.model(input_ids=target_state.input_ids, use_cache=False)
            logits = out.logits[:, commit_t, :]
            return _logodds(logits, source_state.target_token_id, target_state.target_token_id)
    finally:
        if handle is not None:
            handle.remove()


def _clean_logodds(target_state: CommitState, source_state: CommitState) -> float:
    return _logodds(
        target_state.clean_logits,
        source_state.target_token_id,
        target_state.target_token_id,
    )


def layer_patching(
    provider,
    pairs: Sequence[ProbePair],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Per-layer attention vs MLP patching across *pairs*.

    Returns a dict with keys ``"attn"`` and ``"mlp"``, each a 1-D array of
    length ``n_layers`` containing ``|Δlog-odds|`` averaged across pairs and
    both patching directions, where log-odds is contrastive between the pair's
    two tools.
    """
    n_layers = len(provider._get_transformer_layers())
    attn_sum = np.zeros(n_layers, dtype=np.float64)
    mlp_sum = np.zeros(n_layers, dtype=np.float64)
    count = 0

    for i, pair in enumerate(pairs):
        if progress is not None:
            progress(i, len(pairs))
        state_a = _capture_clean_state(provider, pair.a)
        state_b = _capture_clean_state(provider, pair.b)

        # Both patching directions: (target=B, source=A) and (target=A, source=B).
        for tgt, src in [(state_b, state_a), (state_a, state_b)]:
            base = _clean_logodds(tgt, src)
            for ly in range(n_layers):
                if src.attn_at_commit[ly] is not None and tgt.attn_at_commit[ly] is not None:
                    val = _patched_logodds(provider, tgt, src, kind="attn", layer_idx=ly)
                    attn_sum[ly] += abs(val - base)
                if src.mlp_at_commit[ly] is not None and tgt.mlp_at_commit[ly] is not None:
                    val = _patched_logodds(provider, tgt, src, kind="mlp", layer_idx=ly)
                    mlp_sum[ly] += abs(val - base)
            count += 1

    if count == 0:
        raise RuntimeError("no patching directions produced data")
    return {"attn": attn_sum / count, "mlp": mlp_sum / count}


def head_patching(
    provider,
    pairs: Sequence[ProbePair],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Per-head ``o_proj``-input patching → ``(n_layers, n_heads)`` importance.

    Each head's ``[h*head_dim:(h+1)*head_dim]`` slice of the o_proj input at the
    commit position is replaced from the contrastive scenario; importance is the
    absolute contrastive log-odds shift, averaged over pairs and directions.
    """
    n_layers = len(provider._get_transformer_layers())
    arch_n_heads = getattr(provider.model.config, "num_attention_heads", None) or \
        provider.model.config.num_heads
    importance = np.zeros((n_layers, arch_n_heads), dtype=np.float64)
    count = 0

    for i, pair in enumerate(pairs):
        if progress is not None:
            progress(i, len(pairs))
        state_a = _capture_clean_state(provider, pair.a)
        state_b = _capture_clean_state(provider, pair.b)
        for tgt, src in [(state_b, state_a), (state_a, state_b)]:
            base = _clean_logodds(tgt, src)
            for ly in range(n_layers):
                if src.oproj_in_at_commit[ly] is None or tgt.oproj_in_at_commit[ly] is None:
                    continue
                for hd in range(arch_n_heads):
                    val = _patched_logodds(
                        provider, tgt, src, kind="head", layer_idx=ly, head_idx=hd
                    )
                    importance[ly, hd] += abs(val - base)
            count += 1

    if count == 0:
        raise RuntimeError("no patching directions produced data")
    return importance / count


def _attn_last_row(provider, input_ids, commit_pos: int, layers: Sequence[int],
                   *, patch=None):
    """Capture the commit-row attention of every head in *layers*.

    Returns ``{layer: tensor(n_heads, seq)}`` — the attention from the commit
    position over all key positions, for each head in that layer.

    ``patch`` optionally installs a single ``o_proj``-input head patch
    ``(layer_idx, head_idx, src_vector, head_dim)`` before the forward, so the
    captured attention reflects the downstream effect of replacing that head
    (path patching). Uses the eager attention implementation so attention
    weights are materialised.
    """
    torch = _torch()
    handle = None
    if patch is not None:
        p_layer, p_head, src_vec, head_dim = patch
        lo, hi = p_head * head_dim, (p_head + 1) * head_dim

        def _pre_hook(_mod, inp):
            tensor = inp[0].clone()
            tensor[0, commit_pos, lo:hi] = src_vec.to(tensor.device, tensor.dtype)
            return (tensor,) + inp[1:]

        block = provider._get_transformer_layers()[p_layer]
        handle = _oproj_module(block).register_forward_pre_hook(_pre_hook)

    cfg = provider.model.config
    orig_impl = getattr(cfg, "_attn_implementation", None)
    try:
        cfg._attn_implementation = "eager"
        with torch.no_grad():
            out = provider.model(input_ids=input_ids, output_attentions=True, use_cache=False)
        result: dict[int, "object"] = {}
        for L in layers:
            # out.attentions[L]: (batch, n_heads, q, k)
            result[L] = out.attentions[L][0, :, commit_pos, :].detach().float().cpu()
        return result
    finally:
        if orig_impl is not None:
            cfg._attn_implementation = orig_impl
        if handle is not None:
            handle.remove()


def _oproj_in_all_layers(provider, input_ids, commit_pos: int) -> dict[int, "object"]:
    """Capture the ``o_proj`` input at *commit_pos* for every layer."""
    torch = _torch()
    layers = provider._get_transformer_layers()
    out: dict[int, "object"] = {}
    handles = []

    def _make(idx):
        def hook(_mod, inp):
            out[idx] = inp[0][0, commit_pos, :].detach().clone()
            return None
        return hook

    for idx, block in enumerate(layers):
        handles.append(_oproj_module(block).register_forward_pre_hook(_make(idx)))
    with torch.no_grad():
        provider.model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()
    return out


def head_pair_circuit_matrix(
    provider,
    pairs: Sequence[ProbePair],
    head_set: Sequence[tuple[int, int]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Directed path-patching circuit matrix among heads in *head_set*.

    Mirrors ``notebooks/head_circuit_analysis.ipynb``. For an ordered pair
    ``(A, B)`` we replace upstream head ``A``'s ``o_proj``-input slice with its
    value from the contrastive scenario and measure how much downstream head
    ``B``'s commit-row attention pattern changes:

    .. math::
        M[A, B] = \\operatorname{mean}\\bigl|\\,
            \\alpha_B^{\\text{patched}} - \\alpha_B^{\\text{baseline}}\\,\\bigr|

    averaged over both contrastive directions and all pairs. The matrix is
    \\emph{asymmetric} (A influences B, not vice versa); the diagonal is NaN.
    Cells where ``layer(A) >= layer(B)`` are a causal control (an upstream head
    cannot influence an earlier layer through the residual stream).
    """
    n = len(head_set)
    head_dim = _head_dim(provider)
    layers_of_interest = sorted({L for L, _ in head_set})
    acc = np.zeros((n, n), dtype=np.float64)
    cnt = np.zeros((n, n), dtype=np.float64)
    device = next(provider.model.parameters()).device

    for k, pair in enumerate(pairs):
        if progress is not None:
            progress(k, len(pairs))
        for tgt_scn, src_scn in [(pair.a, pair.b), (pair.b, pair.a)]:
            tgt_ids, tgt_commit, _ = _build_forced_inputs(provider, tgt_scn)
            src_ids, src_commit, _ = _build_forced_inputs(provider, src_scn)
            tgt_ids = tgt_ids.to(device)
            src_ids = src_ids.to(device)

            # source o_proj inputs (the patch sources) and target baseline attention
            src_pre = _oproj_in_all_layers(provider, src_ids, src_commit)
            base_attn = _attn_last_row(provider, tgt_ids, tgt_commit, layers_of_interest)

            for i, (La, Ha) in enumerate(head_set):
                obs = _attn_last_row(
                    provider, tgt_ids, tgt_commit, layers_of_interest,
                    patch=(La, Ha, src_pre[La][Ha * head_dim:(Ha + 1) * head_dim], head_dim),
                )
                for j, (Lb, Hb) in enumerate(head_set):
                    if i == j:
                        continue
                    base_row = base_attn[Lb][Hb]
                    obs_row = obs[Lb][Hb]
                    m = min(base_row.shape[-1], obs_row.shape[-1])
                    diff = (obs_row[:m] - base_row[:m]).abs().mean().item()
                    acc[i, j] += diff
                    cnt[i, j] += 1.0

    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    np.fill_diagonal(out, np.nan)
    return out


def capture_attention_pattern(
    provider,
    scenario: ProbeScenario,
    head: tuple[int, int],
) -> np.ndarray:
    """Return the post-softmax attention matrix for one ``(layer, head)``.

    Uses ``output_attentions=True`` on a single forward of *scenario* built with
    the forced tool-call prefix (so the commit row reflects tool selection). The
    returned array has shape ``(seq_len, seq_len)``.
    """
    torch = _torch()
    input_ids, _, _ = _build_forced_inputs(provider, scenario)
    input_ids = input_ids.to(next(provider.model.parameters()).device)
    with torch.no_grad():
        out = provider.model(input_ids=input_ids, output_attentions=True, use_cache=False)
    layer, head_idx = head
    attn = out.attentions[layer][0, head_idx].detach().float().cpu().numpy()
    return attn


__all__ = [
    "CommitState",
    "capture_attention_pattern",
    "head_pair_circuit_matrix",
    "head_patching",
    "layer_patching",
]
