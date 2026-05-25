"""
Stage 1 — orchestrator: drive the model over a list of :class:`MCPToxPair`
samples and accumulate per-head features to disk.

What this file does
-------------------
* Load a ``HuggingFaceProvider`` with ``output_attentions=True``.
* For each pair (``benign``, ``malicious``):
    1. ``provider.select_tools(tools, user_query)`` — populates the provider's
       white-box caches (``_last_input_ids``, ``last_attentions``,
       ``last_attn_gen_prefix``).
    2. ``spans.find_all_spans(...)`` — locate tool / desc / query spans.
    3. ``features.extract_all_heads(...)`` — compute
       ``(num_layers, num_heads, FEAT_DIM)`` features.
    4. Save the full attention row for the six known core heads as
       diagnostics.
    5. Record the model's actually-selected tool name + parse status.
* Persist incrementally so a long run is resumable.

Memory note
-----------
``capture_batch()`` is kept for throughput, but can be memory-heavy on long
generations because ``generate(output_attentions=True)`` retains attentions for
all decode steps. To keep the original behaviour while improving robustness,
``run()`` will fall back to single-sample ``capture_one()`` for the current
chunk if a CUDA OOM is raised in batched mode.

Output layout::

    out_dir/
      features.npy           # memmap, shape (N_pairs, 2, L, H, FEAT_DIM)
      meta.jsonl             # one JSON record per (pair, side)
      topheads/pair_<NNNN>.npz   # one file per pair, keys 'benign'/'malicious'
      checkpoint.json        # {"last_completed_pair_idx": int, ...}
      manifest.json          # one-shot config (model id, n_pairs, dims, ...)

The caller is expected to set ``output_attentions=True`` only — hidden states
are not needed for this stage.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from loguru import logger

from . import config
from .data import MCPToxPair, MCPToxSample, load_pairs
from .features import FEAT_DIM, MAX_TOOLS, extract_all_heads
from .spans import PromptSpans, find_all_spans


def _cuda_mem_snapshot(provider: Any) -> tuple[int, int, int] | None:
    """Return CUDA memory stats (allocated, reserved, max_allocated) in bytes."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    model = getattr(provider, "model", None)
    if model is None:
        return None

    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        device = torch.device("cuda")

    if device.type != "cuda":
        return None

    idx = device.index if device.index is not None else torch.cuda.current_device()
    return (
        int(torch.cuda.memory_allocated(idx)),
        int(torch.cuda.memory_reserved(idx)),
        int(torch.cuda.max_memory_allocated(idx)),
    )


def _fmt_mib(n_bytes: int) -> str:
    return f"{(n_bytes / (1024 ** 2)):.1f} MiB"


def _log_mem_stats(prefix: str, stats: tuple[int, int, int] | None) -> None:
    if stats is None:
        return
    alloc_b, reserv_b, peak_b = stats
    logger.info(
        "{} | cuda_mem allocated={} reserved={} peak_alloc={} ",
        prefix,
        _fmt_mib(alloc_b),
        _fmt_mib(reserv_b),
        _fmt_mib(peak_b),
    )

# ──────────────────────────────────────────────────────────────────────────
# Per-sample record
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CaptureRecord:
    """One side of one pair's capture result.

    The bulk arrays (``features``, ``topheads_full``) are large and not
    serialised inside the dataclass — they are written to disk separately.
    This object exists primarily for in-memory work in tests / notebooks.
    """

    sample_id: str
    is_malicious: bool
    mcp_server: str
    risk_category: str
    attack_paradigm: str
    tool_names: list[str]
    selected_tool: str | None
    parse_ok: bool
    input_len: int
    key_len: int
    tool_name_spans:  dict[str, tuple[int, int]] = field(default_factory=dict)
    tool_desc_spans:  dict[str, tuple[int, int]] = field(default_factory=dict)
    tool_param_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    user_query_span: tuple[int, int] | None = None
    elapsed_sec: float = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Provider-handling
# ──────────────────────────────────────────────────────────────────────────

def _build_provider(model_id: str, **kwargs: Any) -> Any:
    """Lazily import to keep this module importable without torch/transformers.
    
    Parameters
    ----------
    model_id
        HuggingFace model ID.
    **kwargs
        Passed to HuggingFaceProvider.__init__. Can include:
        - device: device mapping (default: "auto")
        - load_in_8bit: 8-bit quantization (default: False)
        - load_in_4bit: 4-bit quantization (default: False)
        - torch_dtype: torch dtype (default: "auto")
    """
    from pma_shield.providers.llm import HuggingFaceProvider

    provider = HuggingFaceProvider(
        model_id=model_id,
        output_attentions=True,
        output_hidden_states=False,
        **kwargs,
    )
    return provider


def _flatten_input_ids(input_ids: Any) -> list[int]:
    """Provider's ``_last_input_ids`` is a torch.Tensor of shape (1, L); flatten."""
    if hasattr(input_ids, "tolist"):
        flat = input_ids.tolist()
    else:
        flat = list(input_ids)
    while isinstance(flat, list) and flat and isinstance(flat[0], list):
        flat = flat[0]
    return [int(x) for x in flat]


def _stack_topheads_rows(
    last_attentions: Sequence[Any],
    head_set: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Extract full attention rows for the requested ``(layer, head)`` pairs.

    Returns shape ``(len(head_set), key_len)`` in float32.  Returns a
    ``(0, key_len)`` array when ``head_set`` is empty (no diagnostic heads
    known for this model yet).  Silently drops any (layer, head) pair that
    is out-of-bounds for the loaded model and logs a warning.
    """
    from .features import attention_row_per_head

    if len(last_attentions) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    num_layers = len(last_attentions)
    row_shapes = [attention_row_per_head(x).shape for x in last_attentions]
    num_heads_model = row_shapes[0][0]
    key_len = max(s[1] for s in row_shapes)

    rows = []
    skipped: list[tuple[int, int]] = []
    for layer_idx, head_idx in head_set:
        if layer_idx < 0 or layer_idx >= num_layers or head_idx < 0 or head_idx >= num_heads_model:
            skipped.append((layer_idx, head_idx))
            continue
        layer_attn = last_attentions[layer_idx]
        per_head = attention_row_per_head(layer_attn)
        row = per_head[head_idx].astype(np.float32, copy=False)
        if row.shape[0] < key_len:
            pad = np.zeros((key_len - row.shape[0],), dtype=np.float32)
            row = np.concatenate([pad, row], axis=0)
        elif row.shape[0] > key_len:
            row = row[-key_len:]
        rows.append(row)
    if skipped:
        logger.warning(
            "_stack_topheads_rows: skipped {} out-of-bounds head(s) {} "
            "(model has {} layers × {} heads)",
            len(skipped), skipped, num_layers, num_heads_model,
        )
    if not rows:
        return np.zeros((0, key_len), dtype=np.float32)
    return np.stack(rows, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Per-side capture
# ──────────────────────────────────────────────────────────────────────────

def capture_one(
    provider: Any,
    sample: MCPToxSample,
    *,
    known_core_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
) -> tuple[CaptureRecord, np.ndarray, np.ndarray]:
    """Run inference on a single sample and compute features.

    Returns
    -------
    (record, features, topheads_full)
        ``features``       — shape ``(num_layers, num_heads, FEAT_DIM)``
        ``topheads_full``  — shape ``(len(known_core_heads), key_len)``
    """
    t0 = time.time()
    mem_before = _cuda_mem_snapshot(provider)
    _log_mem_stats(f"capture_one start sample={sample.sample_id}", mem_before)

    result = provider.select_tools(sample.tools, sample.user_query)

    last_attentions = provider.last_attentions
    if last_attentions is None:
        raise RuntimeError(
            "provider.last_attentions is None — did you pass output_attentions=True?"
        )

    input_ids = _flatten_input_ids(provider._last_input_ids)
    spans: PromptSpans = find_all_spans(
        input_ids, sample.tools, sample.user_query, provider.tokenizer
    )

    features = extract_all_heads(last_attentions, spans)

    # Determine key_len from the first layer's attention.
    from .features import attention_row_per_head
    rows0 = attention_row_per_head(last_attentions[0])
    key_len = int(rows0.shape[1])

    topheads_full = _stack_topheads_rows(last_attentions, known_core_heads)

    selected = (result.selected_tools or [None])[0]
    record = CaptureRecord(
        sample_id=sample.sample_id,
        is_malicious=sample.is_malicious,
        mcp_server=sample.mcp_server,
        risk_category=sample.risk_category,
        attack_paradigm=sample.attack_paradigm,
        tool_names=list(spans.tool_order),
        selected_tool=selected,
        parse_ok=selected is not None,
        input_len=spans.input_len,
        key_len=key_len,
        tool_name_spans={k: list(v) for k, v in spans.tool_name.items()},   # type: ignore[misc]
        tool_desc_spans={k: list(v) for k, v in spans.tool_desc.items()},   # type: ignore[misc]
        tool_param_spans={k: list(v) for k, v in spans.tool_param.items()}, # type: ignore[misc]
        user_query_span=tuple(spans.user_query) if spans.user_query else None,
        elapsed_sec=time.time() - t0,
    )

    mem_after = _cuda_mem_snapshot(provider)
    _log_mem_stats(f"capture_one end sample={sample.sample_id}", mem_after)
    if mem_before is not None and mem_after is not None:
        d_alloc = mem_after[0] - mem_before[0]
        d_reserv = mem_after[1] - mem_before[1]
        logger.info(
            "capture_one delta sample={} | alloc_delta={} reserved_delta={}",
            sample.sample_id,
            _fmt_mib(d_alloc),
            _fmt_mib(d_reserv),
        )

    return record, features, topheads_full


# ──────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────

def _find_selection_step(
    new_token_ids: list[int],
    provider: Any,
    n_steps: int,
) -> int:
    """Return the generation step index for the attention capture.

    Mirrors :meth:`HuggingFaceProvider._select_tools_stepwise`: search for
    ``<call_start>\\n{"name": "`` in the generated token stream and return
    the step immediately after.  For models with no dedicated call_start
    token (Llama 3.1 custom tools, standard Gemma instruct), the bare
    ``'\\n{"name": "'`` (and ``'{"name": "'``) prefix is used.  Falls back
    to step 0 if no prefix is found.
    """
    call_start_str = provider.get_tool_call_tokens().get("call_start", "")
    try:
        jp_ids_nl    = provider.tokenizer.encode('\n{"name": "', add_special_tokens=False)
        jp_ids_plain = provider.tokenizer.encode('{"name": "',   add_special_tokens=False)
    except Exception:
        return 0

    candidates: list[list[int]] = []
    if call_start_str:
        try:
            cs_ids = provider.tokenizer.encode(call_start_str, add_special_tokens=False)
            candidates.append(cs_ids + jp_ids_nl)
            candidates.append(cs_ids + jp_ids_plain)
        except Exception:
            pass
    candidates.append(jp_ids_nl)
    candidates.append(jp_ids_plain)

    seen = set()
    for full_prefix in candidates:
        if not full_prefix:
            continue
        key = tuple(full_prefix)
        if key in seen:
            continue
        seen.add(key)
        full_len = len(full_prefix)
        for idx in range(len(new_token_ids) - full_len + 1):
            if new_token_ids[idx: idx + full_len] == full_prefix:
                candidate = idx + full_len
                if candidate < n_steps:
                    return candidate
    return 0


def capture_batch(
    provider: Any,
    samples: Sequence[MCPToxSample],
    *,
    known_core_heads: Sequence[tuple[int, int]] = config.KNOWN_CORE_HEADS,
) -> list[tuple[CaptureRecord, np.ndarray, np.ndarray] | None]:
    """Run batched inference on multiple samples in a single ``model.generate``.

    Samples are left-padded to the length of the longest prompt.  Per-sample
    selection steps are detected independently after generation so that
    different samples in the batch can emit ``<tool_call>`` at different steps.

    Parameters
    ----------
    provider
        ``HuggingFaceProvider`` built with ``output_attentions=True``.
    samples
        Arbitrary list of :class:`MCPToxSample` objects (may mix benign /
        malicious from different pairs).
    known_core_heads
        (layer, head) pairs whose full attention rows are saved as diagnostics.

    Returns
    -------
    list of ``(CaptureRecord, features, topheads_full)`` tuples, one per input
    sample, in the same order.  A slot is ``None`` if that sample's capture
    raised an exception.
    """
    import torch

    t0 = time.time()
    B = len(samples)
    sample_ids = [s.sample_id for s in samples]
    mem_before = _cuda_mem_snapshot(provider)
    _log_mem_stats(
        f"capture_batch start size={B} first={sample_ids[0]} last={sample_ids[-1]}",
        mem_before,
    )

    # ── 1. Tokenise each sample independently ─────────────────────────────
    per_ids: list[torch.Tensor] = []
    for s in samples:
        raw = provider._build_input_ids(s.tools, s.user_query)
        norm = provider._normalize_model_inputs(raw)
        ids = norm["input_ids"]
        if ids.dim() == 2:
            ids = ids[0]
        per_ids.append(ids)

    prompt_lens = [ids.shape[0] for ids in per_ids]
    max_len = max(prompt_lens)
    pad_id = provider.tokenizer.pad_token_id or provider.tokenizer.eos_token_id

    # ── 2. Left-pad into a (B, max_len) batch ─────────────────────────────
    padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask  = torch.zeros((B, max_len), dtype=torch.long)
    for i, (ids, plen) in enumerate(zip(per_ids, prompt_lens)):
        padded_ids[i, max_len - plen:] = ids
        attn_mask[i,  max_len - plen:] = 1

    device = next(provider.model.parameters()).device
    padded_ids = padded_ids.to(device)
    attn_mask  = attn_mask.to(device)

    # ── 3. Batched generate ────────────────────────────────────────────────
    with torch.no_grad():
        output = provider.model.generate(
            input_ids=padded_ids,
            attention_mask=attn_mask,
            max_new_tokens=provider.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_attentions=True,
        )

    n_steps = len(output.attentions)

    # ── 4. Per-sample extraction ───────────────────────────────────────────
    results: list[tuple[CaptureRecord, np.ndarray, np.ndarray] | None] = []

    for i, (sample, plen) in enumerate(zip(samples, prompt_lens)):
        try:
            # Generated token ids for this sample
            new_ids = output.sequences[i, max_len:].tolist()

            # Find the selection step for this sample
            attn_step = _find_selection_step(new_ids, provider, n_steps)

            # Unpadded slice offset: padding lives in [0, max_len - plen)
            unpad_start = max_len - plen

            # Extract per-layer attention for this sample at attn_step.
            # Shape at each step: (B, heads, 1, max_len + step).
            # Slice to remove padding → (heads, 1, plen + step).
            last_attentions = tuple(
                output.attentions[attn_step][layer_idx][i, :, :, unpad_start:]
                .detach().cpu()
                for layer_idx in range(len(output.attentions[attn_step]))
            )

            # Unpadded prompt token ids (for span detection)
            input_ids_unpadded: list[int] = padded_ids[i, unpad_start:].tolist()

            spans: PromptSpans = find_all_spans(
                input_ids_unpadded, sample.tools, sample.user_query,
                provider.tokenizer,
            )
            features = extract_all_heads(last_attentions, spans)

            from .features import attention_row_per_head
            rows0 = attention_row_per_head(last_attentions[0])
            key_len = int(rows0.shape[1])

            topheads_full = _stack_topheads_rows(last_attentions, known_core_heads)

            # Parse tool selection
            output_text = provider.tokenizer.decode(
                new_ids, skip_special_tokens=False
            )
            tool_names = [t["function"]["name"] for t in sample.tools]
            selected_list = provider._parse_tool_calls(output_text, tool_names)
            selected = (selected_list or [None])[0]

            record = CaptureRecord(
                sample_id=sample.sample_id,
                is_malicious=sample.is_malicious,
                mcp_server=sample.mcp_server,
                risk_category=sample.risk_category,
                attack_paradigm=sample.attack_paradigm,
                tool_names=list(spans.tool_order),
                selected_tool=selected,
                parse_ok=selected is not None,
                input_len=spans.input_len,
                key_len=key_len,
                tool_name_spans={k: list(v) for k, v in spans.tool_name.items()},
                tool_desc_spans={k: list(v) for k, v in spans.tool_desc.items()},
                tool_param_spans={k: list(v) for k, v in spans.tool_param.items()},
                user_query_span=(
                    tuple(spans.user_query) if spans.user_query else None
                ),
                elapsed_sec=(time.time() - t0) / B,
            )
            results.append((record, features, topheads_full))

        except Exception as exc:
            logger.warning(
                "capture_batch: sample {} ({}) failed: {}",
                sample.sample_id, "mal" if sample.is_malicious else "ben", exc,
            )
            results.append(None)

    # Free the large generate output eagerly
    del output

    mem_after = _cuda_mem_snapshot(provider)
    _log_mem_stats(
        f"capture_batch end size={B} first={sample_ids[0]} last={sample_ids[-1]}",
        mem_after,
    )
    if mem_before is not None and mem_after is not None:
        d_alloc = mem_after[0] - mem_before[0]
        d_reserv = mem_after[1] - mem_before[1]
        logger.info(
            "capture_batch delta size={} | alloc_delta={} reserved_delta={}",
            B,
            _fmt_mib(d_alloc),
            _fmt_mib(d_reserv),
        )

    return results


# ──────────────────────────────────────────────────────────────────────────
# Disk plumbing
# ──────────────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _allocate_features_memmap(
    path: Path, *, n_pairs: int, num_layers: int, num_heads: int
) -> np.memmap:
    arr = np.memmap(
        path,
        dtype="float32",
        mode="w+",
        shape=(n_pairs, 2, num_layers, num_heads, FEAT_DIM),
    )
    arr[:] = np.nan
    arr.flush()
    return arr


def _open_features_memmap(
    path: Path, *, n_pairs: int, num_layers: int, num_heads: int
) -> np.memmap:
    return np.memmap(
        path,
        dtype="float32",
        mode="r+",
        shape=(n_pairs, 2, num_layers, num_heads, FEAT_DIM),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


# ──────────────────────────────────────────────────────────────────────────
# Top-level run
# ──────────────────────────────────────────────────────────────────────────

def run(
    pairs: Sequence[MCPToxPair],
    *,
    out_dir: Path = config.ATTN_CACHE_DIR,
    model_id: str = config.DEFAULT_MODEL_ID,
    num_layers: int | None = None,
    num_heads: int | None = None,
    known_core_heads: Sequence[tuple[int, int]] | None = None,
    limit: int | None = None,
    resume: bool = True,
    provider: Any | None = None,
    cuda_empty_cache_every: int = 8,
    batch_size: int = 1,
    provider_kwargs: dict[str, Any] | None = None,
) -> None:
    """Batch driver for Stage 1.

    Parameters
    ----------
    pairs
        Output of :func:`mcp_eval.mcptox.data.load_pairs`.
    out_dir
        Directory under which all artifacts are written. Created if missing.
    model_id, num_layers, num_heads
        Model identification and expected dimensions. If ``num_layers`` /
        ``num_heads`` are omitted, they are inferred from
        ``provider.model.config``. If provided, they are validated against
        the model config when available.
    known_core_heads
        Heads whose full attention rows are saved per sample for diagnostics.
    limit
        If set, only process the first ``limit`` pairs. Used by the
        100-pair smoke test.
    resume
        If True (default), pick up from the saved checkpoint. Set to False
        to force a clean run (overwrites prior outputs).
    provider
        Optional pre-built provider (used in tests). When None, this function
        instantiates one.
    cuda_empty_cache_every
        Call ``torch.cuda.empty_cache()`` after every N *pairs*. Mitigates the
        memory growth caused by ``output_attentions=True`` retaining all
        per-step attention tensors.
    batch_size
        Number of samples (not pairs) to feed into a single ``model.generate``
        call. Must be even so that whole pairs stay together; if odd, it is
        rounded up to the next even number. Larger values give higher
        throughput at the cost of more GPU memory.
    provider_kwargs
        Additional keyword arguments for HuggingFaceProvider (e.g.,
        load_in_8bit, load_in_4bit, torch_dtype, device). Default: {}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topheads_dir = out_dir / "topheads"
    topheads_dir.mkdir(exist_ok=True)
    features_path = out_dir / "features.npy"
    meta_path = out_dir / "meta.jsonl"
    manifest_path = out_dir / "manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"

    if limit is not None:
        pairs = list(pairs)[:limit]
    n_pairs = len(pairs)
    if n_pairs == 0:
        logger.warning("No pairs to process; exiting early.")
        return

    # Resolve diagnostic heads from the per-model registry; unknown models
    # get an empty set (Qwen3-8B coordinates are out-of-bounds for smaller
    # models, so we never fall back to them silently).
    if known_core_heads is None:
        known_core_heads = config.get_known_core_heads(model_id)
        if not known_core_heads:
            logger.warning(
                "No KNOWN_CORE_HEADS registered for {}; topheads diagnostic "
                "rows will be empty. Run MCPTox head analysis and add the "
                "discovered heads to config.KNOWN_CORE_HEADS_BY_MODEL.",
                model_id,
            )

    # Provider — lazy.
    own_provider = provider is None
    if own_provider:
        if provider_kwargs is None:
            provider_kwargs = {}
        provider = _build_provider(model_id, **provider_kwargs)

    # Infer dims from model config (preferred) when not explicitly provided.
    cfg = getattr(getattr(provider, "model", None), "config", None)
    cfg_layers = getattr(cfg, "num_hidden_layers", None)
    cfg_heads = getattr(cfg, "num_attention_heads", None)

    # Some multimodal/chat configs (e.g. Gemma-3) keep text-transformer
    # dimensions under cfg.text_config rather than top-level fields.
    text_cfg = getattr(cfg, "text_config", None)
    if cfg_layers is None and text_cfg is not None:
        cfg_layers = getattr(text_cfg, "num_hidden_layers", None)
    if cfg_heads is None and text_cfg is not None:
        cfg_heads = getattr(text_cfg, "num_attention_heads", None)

    # Final fallback: shared registry (covers known models even when config
    # layout differs across families).
    if cfg_layers is None or cfg_heads is None:
        try:
            from pma_shield.model_registry import get_arch

            arch = get_arch(model_id, hf_config=cfg)
            if cfg_layers is None:
                cfg_layers = int(arch.num_layers)
            if cfg_heads is None:
                cfg_heads = int(arch.num_heads)
        except Exception:
            pass

    if num_layers is None:
        if cfg_layers is None:
            raise RuntimeError(
                "Cannot infer num_layers from provider.model.config; "
                "please pass num_layers explicitly."
            )
        num_layers = int(cfg_layers)
    elif cfg_layers is not None and int(num_layers) != int(cfg_layers):
        raise ValueError(
            f"num_layers mismatch: CLI/config requested {num_layers}, "
            f"but model config reports {cfg_layers}."
        )

    if num_heads is None:
        if cfg_heads is None:
            raise RuntimeError(
                "Cannot infer num_heads from provider.model.config; "
                "please pass num_heads explicitly."
            )
        num_heads = int(cfg_heads)
    elif cfg_heads is not None and int(num_heads) != int(cfg_heads):
        raise ValueError(
            f"num_heads mismatch: CLI/config requested {num_heads}, "
            f"but model config reports {cfg_heads}."
        )

    # Manifest is rewritten every run (cheap; helps resume sanity-check).
    manifest = {
        "model_id": model_id,
        "n_pairs": n_pairs,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "feat_dim": FEAT_DIM,
        "max_tools": MAX_TOOLS,
        "known_core_heads": [list(h) for h in known_core_heads],
    }

    # Resume: detect existing memmap + checkpoint and reopen.
    ck = _load_checkpoint(checkpoint_path) if resume else {}
    start_idx = 0
    if ck and resume:
        if (
            ck.get("model_id") != model_id
            or ck.get("n_pairs") != n_pairs
            or ck.get("num_layers") != num_layers
            or ck.get("num_heads") != num_heads
        ):
            raise RuntimeError(
                "Checkpoint manifest mismatch — refuse to resume into a "
                "different layout. Pass resume=False to start fresh."
            )
        start_idx = int(ck.get("last_completed_pair_idx", -1)) + 1
        feats_mm = _open_features_memmap(
            features_path, n_pairs=n_pairs, num_layers=num_layers, num_heads=num_heads
        )
        logger.info("Resuming Stage 1 from pair index {}", start_idx)
    else:
        # Fresh run: allocate, clear meta.
        feats_mm = _allocate_features_memmap(
            features_path, n_pairs=n_pairs, num_layers=num_layers, num_heads=num_heads
        )
        if meta_path.exists():
            meta_path.unlink()
        for f in topheads_dir.glob("pair_*.npz"):
            f.unlink()

    _atomic_write_json(manifest_path, manifest)

    cuda = None
    if own_provider:
        try:
            import torch
            cuda = torch.cuda
        except ImportError:
            cuda = None

    # Normalise batch_size: must be >= 2 and even so whole pairs stay together.
    effective_batch = max(1, int(batch_size))
    if effective_batch > 1 and effective_batch % 2 != 0:
        effective_batch += 1
    pairs_per_chunk = effective_batch // 2 if effective_batch > 1 else 1

    try:
        i = start_idx
        while i < n_pairs:
            chunk_end = min(i + pairs_per_chunk, n_pairs)
            chunk = [pairs[j] for j in range(i, chunk_end)]

            if effective_batch == 1:
                pair = chunk[0]
                try:
                    rec_b, feats_b, topheads_b = capture_one(
                        provider, pair.benign, known_core_heads=known_core_heads
                    )
                    rec_m, feats_m, topheads_m = capture_one(
                        provider, pair.malicious, known_core_heads=known_core_heads
                    )
                except Exception as exc:
                    logger.exception(
                        "Capture failed for pair {} ({}): {}", i, pair.benign.sample_id, exc
                    )
                    i += 1
                    continue

                _write_chunk_results(
                    [(i, rec_b, feats_b, topheads_b, rec_m, feats_m, topheads_m)],
                    feats_mm, meta_path, topheads_dir,
                )
                elapsed = rec_b.elapsed_sec + rec_m.elapsed_sec

            else:
                samples = []
                for pair in chunk:
                    samples.append(pair.benign)
                    samples.append(pair.malicious)

                t0 = time.time()
                try:
                    batch_results = capture_batch(
                        provider, samples, known_core_heads=known_core_heads
                    )
                    elapsed = time.time() - t0

                    to_write = []
                    for k, pair in enumerate(chunk):
                        res_b = batch_results[2 * k]
                        res_m = batch_results[2 * k + 1]
                        if res_b is None or res_m is None:
                            logger.warning(
                                "Batch capture: pair {} ({}) partially failed; skipping.",
                                i + k, pair.benign.sample_id,
                            )
                            continue
                        rec_b, feats_b, topheads_b = res_b
                        rec_m, feats_m, topheads_m = res_m
                        to_write.append(
                            (i + k, rec_b, feats_b, topheads_b, rec_m, feats_m, topheads_m)
                        )

                    _write_chunk_results(to_write, feats_mm, meta_path, topheads_dir)

                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" not in msg and "cuda" not in msg:
                        raise

                    logger.warning(
                        "Batch capture OOM for pairs [{}..{}], fallback to single-sample mode for this chunk.",
                        i, chunk_end - 1,
                    )
                    if cuda is not None and cuda.is_available():
                        cuda.empty_cache()
                    gc.collect()

                    to_write = []
                    t1 = time.time()
                    for k, pair in enumerate(chunk):
                        try:
                            rec_b, feats_b, topheads_b = capture_one(
                                provider, pair.benign, known_core_heads=known_core_heads
                            )
                            rec_m, feats_m, topheads_m = capture_one(
                                provider, pair.malicious, known_core_heads=known_core_heads
                            )
                        except Exception as inner_exc:
                            logger.exception(
                                "Fallback single capture failed for pair {} ({}): {}",
                                i + k,
                                pair.benign.sample_id,
                                inner_exc,
                            )
                            continue
                        to_write.append(
                            (i + k, rec_b, feats_b, topheads_b, rec_m, feats_m, topheads_m)
                        )

                    _write_chunk_results(to_write, feats_mm, meta_path, topheads_dir)
                    elapsed = time.time() - t1

            last_pair_idx = chunk_end - 1
            ck = {
                "model_id": model_id,
                "n_pairs": n_pairs,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "last_completed_pair_idx": last_pair_idx,
                "elapsed_sec_last": elapsed,
            }
            _save_checkpoint(checkpoint_path, ck)

            if (
                cuda is not None
                and cuda.is_available()
                and (chunk_end) % cuda_empty_cache_every == 0
            ):
                cuda.empty_cache()
                gc.collect()

            if chunk_end % 50 == 0 or chunk_end >= n_pairs:
                logger.info(
                    "Captured pair {}/{} ({:.1f}s for last chunk of {} pairs)",
                    chunk_end, n_pairs, elapsed, len(chunk),
                )

            i = chunk_end

    finally:
        del feats_mm


def _write_chunk_results(
    items: list[tuple],
    feats_mm: np.memmap,
    meta_path: Path,
    topheads_dir: Path,
) -> None:
    """Write a list of per-pair capture results to disk.

    Each item in *items* is a 7-tuple:
    ``(pair_idx, rec_b, feats_b, topheads_b, rec_m, feats_m, topheads_m)``.
    """
    for pair_idx, rec_b, feats_b, topheads_b, rec_m, feats_m, topheads_m in items:
        feats_mm[pair_idx, 0] = feats_b
        feats_mm[pair_idx, 1] = feats_m
        feats_mm.flush()

        np.savez_compressed(
            topheads_dir / f"pair_{pair_idx:04d}.npz",
            benign=topheads_b,
            malicious=topheads_m,
        )

        for side_idx, rec in enumerate((rec_b, rec_m)):
            _append_jsonl(
                meta_path,
                {
                    "pair_idx": pair_idx,
                    "side": "benign" if side_idx == 0 else "malicious",
                    **_record_to_jsonable(rec),
                },
            )


def _record_to_jsonable(rec: CaptureRecord) -> dict[str, Any]:
    return {
        "sample_id": rec.sample_id,
        "is_malicious": rec.is_malicious,
        "mcp_server": rec.mcp_server,
        "risk_category": rec.risk_category,
        "attack_paradigm": rec.attack_paradigm,
        "tool_names": rec.tool_names,
        "selected_tool": rec.selected_tool,
        "parse_ok": rec.parse_ok,
        "input_len": rec.input_len,
        "key_len": rec.key_len,
        "tool_name_spans": rec.tool_name_spans,
        "tool_desc_spans": rec.tool_desc_spans,
        "tool_param_spans": rec.tool_param_spans,
        "user_query_span": list(rec.user_query_span) if rec.user_query_span else None,
        "elapsed_sec": rec.elapsed_sec,
    }


# ──────────────────────────────────────────────────────────────────────────
# Loading what `run` produced
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CapturedDataset:
    """In-memory view of the artifacts written by :func:`run`."""

    features: np.memmap                     # (N_pairs, 2, L, H, FEAT_DIM)
    meta: list[dict[str, Any]]              # one entry per (pair, side)
    manifest: dict[str, Any]
    topheads_dir: Path
    out_dir: Path

    def topheads(self, pair_idx: int) -> dict[str, np.ndarray]:
        """Lazy-load the per-pair top-heads diagnostics file."""
        with np.load(self.topheads_dir / f"pair_{pair_idx:04d}.npz") as data:
            return {"benign": data["benign"], "malicious": data["malicious"]}

    @property
    def benign_features(self) -> np.memmap:
        return self.features[:, 0]

    @property
    def malicious_features(self) -> np.memmap:
        return self.features[:, 1]


def load(out_dir: Path = config.ATTN_CACHE_DIR) -> CapturedDataset:
    """Reconstruct a :class:`CapturedDataset` from disk."""
    out_dir = Path(out_dir)
    with (out_dir / "manifest.json").open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    feats = _open_features_memmap(
        out_dir / "features.npy",
        n_pairs=manifest["n_pairs"],
        num_layers=manifest["num_layers"],
        num_heads=manifest["num_heads"],
    )
    meta: list[dict[str, Any]] = []
    with (out_dir / "meta.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            meta.append(json.loads(line))
    return CapturedDataset(
        features=feats,
        meta=meta,
        manifest=manifest,
        topheads_dir=out_dir / "topheads",
        out_dir=out_dir,
    )


# ──────────────────────────────────────────────────────────────────────────
# Re-export the JSONL pair loader for convenience
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    "CaptureRecord",
    "CapturedDataset",
    "capture_one",
    "run",
    "load",
    "load_pairs",  # forwarded from data.py
]
