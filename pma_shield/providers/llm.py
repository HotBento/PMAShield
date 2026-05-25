"""
LLM provider implementations.
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional
from pma_shield.core.base import LLMProvider, SelectionResult, RateLimitError
from pma_shield.logger import logger
import json
import os
import re


def _ensure_transformers_losskwargs_compat(transformers_module: Any) -> None:
    """Provide a minimal ``transformers.utils.LossKwargs`` shim when missing.

    Some trust-remote-code model repos (e.g. certain Phi checkpoints) still do
    ``from transformers.utils import LossKwargs``. Newer upstream transformers
    versions no longer export this symbol from ``transformers.utils``.
    The remote code only uses it as a typing mixin, so a TypedDict placeholder
    is sufficient for runtime compatibility.
    """
    utils_mod = getattr(transformers_module, "utils", None)
    if utils_mod is None or hasattr(utils_mod, "LossKwargs"):
        return

    from typing import TypedDict

    class _LossKwargsCompat(TypedDict, total=False):
        """Compatibility placeholder for remote model typing imports."""

    utils_mod.LossKwargs = _LossKwargsCompat
    logger.warning(
        "transformers.utils.LossKwargs is missing; installed a compatibility "
        "TypedDict shim for trust_remote_code models."
    )


def _ensure_transformers_phi_tied_weights_compat(transformers_module: Any) -> None:
    """Patch transformers tied-weight expansion for legacy remote Phi classes.

    Some Phi remote-code checkpoints define ``_tied_weights_keys`` as a list of
    parameter names (legacy style). Newer transformers expects a dict mapping
    target->source and calls ``.keys()``/``.values()`` on it, which raises an
    AttributeError during model ``post_init``.

    This patch rewrites list/tuple forms to an empty dict so that transformers
    does not attempt to fill the keys from a corrupt self-referential mapping.
    The actual weight tying (lm_head.weight = embed_tokens.weight) is handled
    by the model's ``tie_weights()`` call that transformers invokes after loading.
    """
    modeling_utils = getattr(transformers_module, "modeling_utils", None)
    if modeling_utils is None:
        return

    pre_trained_model = getattr(modeling_utils, "PreTrainedModel", None)
    if pre_trained_model is None:
        return

    method_name = "get_expanded_tied_weights_keys"
    original = getattr(pre_trained_model, method_name, None)
    if original is None:
        return
    if getattr(original, "_mcp_phi_tied_patch", False):
        return

    def _patched(self, all_submodels: bool = True):
        tied_mapping = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied_mapping, (list, tuple)):
            # Convert to empty dict so transformers skips the corrupt
            # self-referential mapping; tie_weights() will handle the actual
            # binding after the checkpoint is loaded.
            setattr(self, "_tied_weights_keys", {})
        return original(self, all_submodels=all_submodels)

    _patched._mcp_phi_tied_patch = True
    setattr(pre_trained_model, method_name, _patched)
    logger.warning(
        "Installed compatibility patch for legacy list-style _tied_weights_keys "
        "in trust_remote_code models."
    )


def _fix_phi_meta_rotary_embeddings(model: Any) -> None:
    """Re-initialize Phi3RotaryEmbedding non-persistent buffers left on meta device.

    When loading with ``device_map=<device>``, transformers 5.x uses meta device
    initialization.  Non-persistent buffers (``inv_freq``) and plain attributes
    derived from them (``original_inv_freq``) are never loaded from the
    checkpoint, so they remain on the meta device.  Calling ``.to(device)`` on a
    meta tensor raises ``NotImplementedError``.

    This function walks all ``Phi3RotaryEmbedding`` modules and re-computes
    ``inv_freq`` / ``original_inv_freq`` on the model's actual device.
    """
    import torch

    try:
        target_device = next(model.parameters()).device
    except StopIteration:
        return

    fixed = 0
    for module in model.modules():
        if type(module).__name__ != "Phi3RotaryEmbedding":
            continue
        if not hasattr(module, "rope_init_fn") or not hasattr(module, "config"):
            continue

        inv_freq = getattr(module, "inv_freq", None)
        original_inv_freq = getattr(module, "original_inv_freq", None)
        needs_fix = (
            (inv_freq is not None and inv_freq.device.type == "meta")
            or (original_inv_freq is not None and original_inv_freq.device.type == "meta")
        )
        if not needs_fix:
            continue

        try:
            new_inv_freq, new_scaling = module.rope_init_fn(module.config, target_device)
        except Exception:
            continue

        with torch.no_grad():
            module.register_buffer("inv_freq", new_inv_freq, persistent=False)
            module.original_inv_freq = new_inv_freq.clone()
            module.attention_scaling = new_scaling
        fixed += 1

    if fixed:
        logger.warning(
            f"Re-initialized {fixed} Phi3RotaryEmbedding module(s) whose "
            "non-persistent buffers were left on meta device after from_pretrained."
        )


class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace local model provider for white-box analysis.

    Supports any chat/instruction model that follows the transformers
    chat-template convention.  Tool calls are parsed from generated text
    using multi-strategy heuristics covering the formats used by
    Qwen2.5-Instruct, Llama-3.1-Instruct, Mistral-Instruct, and others.

    The logits of the last generated token are cached in ``self.last_logits``
    (a CPU tensor) after every call to ``_select_tools_impl``, making them
    available for downstream white-box interpretability analysis.
    """

    # ------------------------------------------------------------------ #
    # Per-family tool-call special tokens                                  #
    # ------------------------------------------------------------------ #
    # Keys follow a fixed schema:
    #   call_start / call_end       – delimit a single tool call block
    #   calls_start / calls_end     – delimit the outer multi-call wrapper (if any)
    #   response_start / response_end – delimit the tool result injected back
    #   responses_start / responses_end – outer wrapper for multiple results (if any)
    #   def_start / def_end         – delimit tool/function *definitions*
    #   sep                         – separator token within a call (e.g. DeepSeek)
    #   list_start / list_end       – delimit the available-tools list (Mistral)
    #
    # ``get_tool_call_tokens()`` looks up the right entry by matching the
    # lower-cased model_id against each key.
    _TOOL_CALL_TOKENS: Dict[str, Dict[str, str]] = {
        # Qwen2.5 / Qwen3 — XML tags registered as special tokens
        # Token IDs: <tool_call>=151657, </tool_call>=151658,
        #            <tool_response>=151665, </tool_response>=151666
        "qwen": {
            "call_start": "<tool_call>",
            "call_end": "</tool_call>",
            "response_start": "<tool_response>",
            "response_end": "</tool_response>",
        },
        # Llama 3.x — header-based format.  For *built-in* tools
        # (code_interpreter / brave_search / wolfram_alpha) Llama emits a
        # <|python_tag|> prefix; for *custom* JSON-schema tools passed via
        # apply_chat_template(tools=...), Llama 3.1-Instruct outputs the JSON
        # tool call directly with NO <|python_tag|>.  We therefore leave
        # call_start empty so the generic '\n{"name": "' fallback in
        # _select_tools_stepwise / _find_selection_step is used, and rely on
        # the parser's JSON-object scan to extract the tool name.
        "llama": {
            "call_start": "",
            "call_end": "<|eot_id|>",
            "response_start": "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "response_end": "<|eot_id|>",
            # Reference only — emitted only for the 3 Meta-shipped built-ins.
            "python_tag": "<|python_tag|>",
        },
        # Mistral / Mixtral — square-bracket control tokens
        "mistral": {
            "call_start": "[TOOL_CALLS]",
            "call_end": "</s>",
            "response_start": "[TOOL_RESULTS]",
            "response_end": "[/TOOL_RESULTS]",
            "list_start": "[AVAILABLE_TOOLS]",
            "list_end": "[/AVAILABLE_TOOLS]",
        },
        # Mixtral shares the same format as Mistral
        "mixtral": {
            "call_start": "[TOOL_CALLS]",
            "call_end": "</s>",
            "response_start": "[TOOL_RESULTS]",
            "response_end": "[/TOOL_RESULTS]",
            "list_start": "[AVAILABLE_TOOLS]",
            "list_end": "[/AVAILABLE_TOOLS]",
        },
        # Gemma 2 / 3 / 3n / 4 instruct — different formatting depending on version:
        #   - Gemma-2/3: JSON object (sometimes wrapped in ```tool_code)
        #   - Gemma-4 with tool declarations: <|tool_call>call:TOOLNAME{...}<tool_call|>
        # For selection-step triggering in white-box analysis, Gemma-4 uses a special
        # marker "gemma4_tool_call_prefix" to detect when the model commits to calling
        # a tool and is about to emit the tool name.
        "gemma": {
            "call_start": "",
            "call_end": "<end_of_turn>",
            "fn_call_start": "<start_function_call>",
            "fn_call_end": "<end_function_call>",
            "fn_response_start": "<start_function_response>",
            "fn_response_end": "<end_function_response>",
            # Gemma-4 specific: use this as a marker for selection-step detection
            "gemma4_tool_call_prefix": "<|tool_call>call:",
        },
        # Phi-4 / Phi-4-mini-instruct (microsoft/Phi-4-*)
        "phi": {
            "call_start": "<|tool_call|>",
            "call_end": "<|/tool_call|>",
            "response_start": "<|tool_response|>",
            "response_end": "<|end|>",
            "def_start": "<|tool|>",
            "def_end": "<|/tool|>",
        },
        # DeepSeek-V3 / DeepSeek-R1 — full-width Unicode separator tokens
        "deepseek": {
            "call_start": "<｜tool▁call▁begin｜>",
            "call_end": "<｜tool▁call▁end｜>",
            "calls_start": "<｜tool▁calls▁begin｜>",
            "calls_end": "<｜tool▁calls▁end｜>",
            "sep": "<｜tool▁sep｜>",
            "response_start": "<｜tool▁output▁begin｜>",
            "response_end": "<｜tool▁output▁end｜>",
            "responses_start": "<｜tool▁outputs▁begin｜>",
            "responses_end": "<｜tool▁outputs▁end｜>",
        },
    }

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 1024,
        torch_dtype: str = "auto",
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        rpm_limit: Optional[int] = None,
        max_retries: int = 3,
    ):
        """
        Args:
            model_id: HuggingFace model ID or local path.
            device: ``"auto"`` (device_map), ``"cuda"``, ``"cpu"``, etc.
            load_in_4bit: Load with 4-bit NF4 quantisation (requires bitsandbytes).
            load_in_8bit: Load with 8-bit LLM.int8 quantisation (requires bitsandbytes).
            max_new_tokens: Maximum tokens to generate per call.
            torch_dtype: ``"auto"``, ``"float16"``, ``"bfloat16"``, or ``"float32"``.
            output_attentions: If True, cache attention weights after each call.
                               Enables ``get_last_attentions()`` for Phase 1 analysis.
                               Only the selection-step last attention row is cached.
                               Note: requesting attentions still increases runtime/VRAM.
            output_hidden_states: If True, cache all layer hidden states after each call.
                                  Enables ``get_last_hidden_states()`` for Phase 3 analysis.
                                  Note: increases VRAM usage significantly.
            rpm_limit: Requests-per-minute cap (useful when testing rate limits).
            max_retries: Retries on RateLimitError.
        """
        super().__init__(rpm_limit=rpm_limit, max_retries=max_retries)
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Please install transformers and torch: uv add transformers torch"
            ) from exc

        import torch
        import transformers
        _ensure_transformers_losskwargs_compat(transformers)
        _ensure_transformers_phi_tied_weights_compat(transformers)

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._output_attentions = output_attentions
        self._output_hidden_states = output_hidden_states
        self.last_logits = None             # exposed for white-box analysis
        self.last_attentions = None         # per-layer attention rows (CPU), if enabled
        self.last_attentions_prefill = None # deprecated; retained for compatibility
        self.last_hidden_states = None      # per-layer hidden states (CPU), if enabled
        self._last_input_ids = None    # cached for saliency computation
        self._steering_hooks: list = []  # handles for active residual-stream steering hooks

        # Resolve torch dtype
        _DTYPE_MAP = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = _DTYPE_MAP.get(torch_dtype, "auto")

        # Decide whether to use remote code.
        # If the model_type is natively supported by the installed transformers
        # (i.e. it appears in the built-in AUTO mapping), prefer
        # trust_remote_code=False to avoid stale or incompatible remote code
        # bundled in the checkpoint (e.g. Phi-4-mini-instruct ships a
        # modeling_phi3.py that imports removed symbols like LossKwargs).
        _use_remote_code = True
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )
            _cfg = transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=False
            )
            if getattr(_cfg, "model_type", None) in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                _use_remote_code = False
                logger.info(
                    f"model_type='{_cfg.model_type}' is natively supported by "
                    "transformers; using trust_remote_code=False to avoid "
                    "stale remote code."
                )
        except Exception:
            pass  # config fetch failed; fall back to trust_remote_code=True

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=_use_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build model kwargs
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": _use_remote_code,
            "device_map": device,
        }
        if output_attentions:
            # Some models default to SDPA/Flash attention, which may not return
            # attention tensors even when output_attentions=True.
            model_kwargs["attn_implementation"] = "eager"
        if resolved_dtype != "auto":
            model_kwargs["torch_dtype"] = resolved_dtype

        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "Quantisation requires bitsandbytes: uv add bitsandbytes"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        _fix_phi_meta_rotary_embeddings(self.model)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt building                                                      #
    # ------------------------------------------------------------------ #

    def _build_input_ids(self, tools: List[Dict[str, Any]], prompt: str):
        """Tokenize prompt + tools via the model's chat template.

        Strategy:
        1. Use ``apply_chat_template(tools=tools)`` — supported by Qwen2.5,
           Llama-3.1+, Mistral, etc.
        2. Fall back to embedding the tools as plain JSON in a system message.
        """
        messages = [{"role": "user", "content": prompt}]

        # Strategy 1: native tool-calling chat template
        # Some tokenizers accept ``tools=...`` but silently ignore it. To
        # avoid downstream span-finding failures, verify rendered text really
        # contains tool names before accepting this strategy.
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            tool_names = [
                t.get("function", {}).get("name", "")
                for t in tools
                if isinstance(t, dict)
            ]
            has_tools = any(name and name in rendered for name in tool_names)
            if has_tools:
                return self.tokenizer(rendered, return_tensors="pt").input_ids
        except Exception:
            pass

        # Strategy 2: embed tool definitions as JSON in system message
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        system_content = (
            "You are a helpful assistant with access to the following tools.\n"
            "Call the most appropriate tool for the user's request.\n"
            'Respond ONLY with a JSON object in the format: '
            '{"name": "<tool_name>", "arguments": {...}}\n\n'
            f"Available tools:\n{tools_json}"
        )
        fallback_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                fallback_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return input_ids
        except Exception:
            pass

        # Strategy 3: plain tokenisation of rendered template text
        text = self.tokenizer.apply_chat_template(
            fallback_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids

    def _normalize_model_inputs(self, model_inputs):
        """Normalize tokenized inputs to a dict accepted by ``model.generate``.

        Depending on tokenizer/model, ``apply_chat_template`` may return either a
        tensor of input IDs or a ``BatchEncoding`` containing ``input_ids`` and
        optional fields such as ``attention_mask``.
        """
        if isinstance(model_inputs, dict):
            return model_inputs
        if hasattr(model_inputs, "keys") and hasattr(model_inputs, "__getitem__"):
            return dict(model_inputs)
        return {"input_ids": model_inputs}

    def _extract_input_ids(self, model_inputs):
        """Extract the input IDs tensor from normalized model inputs."""
        if isinstance(model_inputs, dict):
            return model_inputs.get("input_ids")
        return None

    # ------------------------------------------------------------------ #
    # Tool-call parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_tool_calls(self, text: str, tool_names: List[str]) -> List[str]:
        """Extract called tool names from raw model output.

        Strategies tried in order:
        1. Model-specific tokens from ``_TOOL_CALL_TOKENS`` (via
           ``get_tool_call_tokens()``), with three structural sub-cases:

           a. Symmetric XML-wrap (Qwen, Phi, Gemma, DeepSeek):
              ``<call_start>JSON<call_end>``
           b. Prefix-tag with lookahead (Llama):
              ``<|python_tag|>JSON`` terminated by the next special token
           c. Array-wrap (Mistral / Mixtral):
              ``[TOOL_CALLS] [{...}, ...]``

        2–4. Hardcoded fallbacks for unrecognised model IDs that still emit
             one of the three formats above.
        5.   Generic JSON object scan — any ``{...}`` containing a ``"name"``
             key that matches a known tool.
        6.   Bare tool-name substring match — last resort.
        """
        selected: List[str] = []

        def _extract_name(obj: dict) -> Optional[str]:
            name = obj.get("name") or obj.get("tool_name")
            if name and name in tool_names:
                return name
            # some models nest the name under "function"
            if isinstance(obj.get("function"), dict):
                n = obj["function"].get("name")
                if n and n in tool_names:
                    return n
            return None

        # ── Strategy 1: model-specific tokens ────────────────────────────
        tokens = self.get_tool_call_tokens()
        call_start = tokens.get("call_start", "")
        call_end = tokens.get("call_end", "")
        if call_start and call_end:
            if call_start == "[TOOL_CALLS]":
                # 1c — Mistral / Mixtral: [TOOL_CALLS] [{...}, ...]
                m = re.search(r"\[TOOL_CALLS\]\s*(\[[\s\S]*?\])", text)
                if m:
                    try:
                        for call in json.loads(m.group(1)):
                            name = _extract_name(call)
                            if name:
                                selected.append(name)
                    except (json.JSONDecodeError, ValueError):
                        pass
            elif call_end in ("<|eom_id|>", "<|eot_id|>"):
                # 1b — Llama: prefix tag, no symmetric closing token
                pat = (
                    re.escape(call_start)
                    + r"\s*([\s\S]*?)(?="
                    + re.escape(call_end)
                    + r"|$)"
                )
                for m in re.finditer(pat, text):
                    try:
                        name = _extract_name(json.loads(m.group(1).strip()))
                        if name:
                            selected.append(name)
                    except (json.JSONDecodeError, ValueError):
                        pass
            else:
                # 1a — symmetric wrap: Qwen, Phi, Gemma, DeepSeek
                pat = (
                    re.escape(call_start)
                    + r"\s*([\s\S]*?)\s*"
                    + re.escape(call_end)
                )
                for m in re.finditer(pat, text):
                    try:
                        name = _extract_name(json.loads(m.group(1)))
                        if name:
                            selected.append(name)
                    except (json.JSONDecodeError, ValueError):
                        pass
        if selected:
            return selected

        # ── Strategy 2: hardcoded fallback — <tool_call>...</tool_call> ──
        for m in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text):
            try:
                name = _extract_name(json.loads(m.group(1)))
                if name:
                    selected.append(name)
            except (json.JSONDecodeError, ValueError):
                pass
        if selected:
            return selected

        # ── Strategy 3: hardcoded fallback — <|python_tag|>{...} ─────────
        for m in re.finditer(r"<\|python_tag\|>\s*([\s\S]*?)(?=<\||$)", text):
            try:
                name = _extract_name(json.loads(m.group(1).strip()))
                if name:
                    selected.append(name)
            except (json.JSONDecodeError, ValueError):
                pass
        if selected:
            return selected

        # ── Strategy 4: hardcoded fallback — [TOOL_CALLS] [...] ──────────
        m = re.search(r"\[TOOL_CALLS\]\s*(\[[\s\S]*?\])", text)
        if m:
            try:
                for call in json.loads(m.group(1)):
                    name = _extract_name(call)
                    if name:
                        selected.append(name)
            except (json.JSONDecodeError, ValueError):
                pass
        if selected:
            return selected

        # ── Strategy 5: generic JSON objects containing "name" ────────────
        for m in re.finditer(r"\{[\s\S]*?\}", text):
            try:
                name = _extract_name(json.loads(m.group(0)))
                if name and name not in selected:
                    selected.append(name)
            except (json.JSONDecodeError, ValueError):
                pass
        if selected:
            return selected

        # ── Strategy 6: bare tool-name substring match (last resort) ─────
        for tname in tool_names:
            if tname in text:
                selected.append(tname)
                break  # only the first match to avoid false positives

        return selected

    # ------------------------------------------------------------------ #
    # Core inference                                                       #
    # ------------------------------------------------------------------ #

    def _select_tools_stepwise(
        self,
        model_inputs,
        prompt_tokens: int,
    ):
        """Greedy decode one token at a time and retain only needed tensors.

        This avoids ``generate(output_attentions=True)`` keeping attention
        tensors for every generation step. We also avoid requesting prefill
        attentions entirely and only retain the per-layer last attention row
        for the first generated token and the selection step.
        """
        import torch

        target_device = next(self.model.parameters()).device
        input_ids = self._extract_input_ids(model_inputs)
        if input_ids is None:
            raise ValueError("Tokenized inputs do not contain 'input_ids'.")

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=target_device)
        else:
            attention_mask = attention_mask.to(target_device)

        eos_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.append(int(self.tokenizer.eos_token_id))
        call_end = self.get_tool_call_tokens().get("call_end", "")
        if call_end:
            try:
                end_ids = self.tokenizer.encode(call_end, add_special_tokens=False)
                if len(end_ids) == 1:
                    eos_token_ids.append(int(end_ids[0]))
            except Exception:
                pass
        eos_token_ids = list(dict.fromkeys(eos_token_ids))
        eos_token_id_set = set(eos_token_ids)

        # Selection-step trigger: when the model has just emitted
        #   <call_start>\n{"name": "
        # we know it has committed to a tool call and the next generated
        # token is the first character of the tool name — that is the step
        # whose attention pattern we want to cache.
        #
        # For models with no dedicated call_start token (Llama 3.1 custom
        # tools, Gemma 2/3 instruct), call_start_str is empty
        # and we trigger purely on '{"name": "' so the same code path
        # works.  We try a couple of leading whitespace variants because
        # some chat templates emit '\n' while others emit ' ' or '"'.
        #
        # For Gemma-4, which uses <|tool_call>call:TOOLNAME{}, we use the
        # special "gemma4_tool_call_prefix" marker instead.
        call_start_str = self.get_tool_call_tokens().get("call_start", "")
        gemma4_prefix = self.get_tool_call_tokens().get("gemma4_tool_call_prefix", "")
        
        full_prefix_candidates: List[List[int]] = []
        try:
            if gemma4_prefix:
                # Gemma-4: match <|tool_call>call: as the selection-step trigger
                gemma4_ids = self.tokenizer.encode(gemma4_prefix, add_special_tokens=False)
                if gemma4_ids:
                    full_prefix_candidates.append(gemma4_ids)
            
            jp_ids_nl    = self.tokenizer.encode('\n{"name": "', add_special_tokens=False)
            jp_ids_plain = self.tokenizer.encode('{"name": "',   add_special_tokens=False)
            if call_start_str:
                cs_ids = self.tokenizer.encode(call_start_str, add_special_tokens=False)
                full_prefix_candidates.append(cs_ids + jp_ids_nl)
                full_prefix_candidates.append(cs_ids + jp_ids_plain)
            # Always try the bare JSON prefix as a fallback (works for Llama
            # custom tools, standard Gemma 2/3, and anything else that emits
            # a plain JSON object).
            full_prefix_candidates.append(jp_ids_nl)
            full_prefix_candidates.append(jp_ids_plain)
        except Exception:
            full_prefix_candidates = []
        # De-duplicate while preserving order
        seen = set()
        full_prefix_candidates = [
            ids for ids in full_prefix_candidates
            if ids and tuple(ids) not in seen and not seen.add(tuple(ids))
        ]
        # Legacy alias retained so the existing match code below still works
        # for the common case (only one candidate).
        full_prefix_ids = full_prefix_candidates[0] if full_prefix_candidates else []

        generated_ids: List[int] = []
        past_key_values = None
        current_input_ids = input_ids.to(target_device)
        capture_next_attn = False
        first_generated_attn = None
        first_step_hidden_states = None

        self.last_attentions = None
        self.last_attentions_prefill = None
        self.last_hidden_states = None
        self.last_logits = None

        for _step in range(self.max_new_tokens):
            want_attn = False
            if self._output_attentions and past_key_values is not None:
                want_attn = capture_next_attn or first_generated_attn is None

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=want_attn,
                    output_hidden_states=self._output_hidden_states,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token_cpu = next_token.detach().cpu()
            next_token_id = int(next_token_cpu[0, 0].item())
            generated_ids.append(next_token_id)
            self.last_logits = logits[0].detach().cpu()

            if self._output_attentions and hasattr(outputs, "attentions") and outputs.attentions:
                attn_cpu = tuple(a[0, :, -1, :].detach().cpu() for a in outputs.attentions)
                if first_generated_attn is None:
                    first_generated_attn = attn_cpu
                if capture_next_attn and self.last_attentions is None:
                    self.last_attentions = attn_cpu

            if self._output_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hs_cpu = tuple(h[0].detach().cpu() for h in outputs.hidden_states)
                if first_step_hidden_states is None:
                    first_step_hidden_states = hs_cpu
                if capture_next_attn and self.last_hidden_states is None:
                    self.last_hidden_states = hs_cpu

            past_key_values = outputs.past_key_values
            del outputs

            if full_prefix_candidates and not capture_next_attn:
                for _pref in full_prefix_candidates:
                    tail_len = len(_pref)
                    if len(generated_ids) >= tail_len and generated_ids[-tail_len:] == _pref:
                        capture_next_attn = True
                        break

            if next_token_id in eos_token_id_set:
                break

            current_input_ids = next_token_cpu.to(target_device)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )

        if self.last_attentions is None and first_generated_attn is not None:
            self.last_attentions = first_generated_attn
        if self.last_hidden_states is None and first_step_hidden_states is not None:
            self.last_hidden_states = first_step_hidden_states

        new_token_ids = torch.tensor(generated_ids, dtype=torch.long)
        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens
        return new_token_ids, completion_tokens, total_tokens

    def _select_tools_impl(
        self, tools: List[Dict[str, Any]], prompt: str
    ) -> SelectionResult:
        """Run local inference and parse the tool call from the output."""
        import torch

        model_inputs = self._normalize_model_inputs(self._build_input_ids(tools, prompt))

        # Move input to the model's device
        try:
            target_device = next(self.model.parameters()).device
            model_inputs = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }
        except StopIteration:
            pass

        input_ids = self._extract_input_ids(model_inputs)
        if input_ids is None:
            raise ValueError("Tokenized inputs do not contain 'input_ids'.")

        prompt_tokens = input_ids.shape[-1]

        # Cache input_ids for gradient saliency computation (Phase 2)
        self._last_input_ids = input_ids

        if self._output_attentions or self._output_hidden_states:
            new_token_ids, completion_tokens, total_tokens = self._select_tools_stepwise(
                model_inputs,
                prompt_tokens,
            )
        else:
            # Add model-specific tool-call end token as an additional EOS when possible.
            # This prevents very long runaway generations during tool selection.
            eos_token_ids = []
            if self.tokenizer.eos_token_id is not None:
                eos_token_ids.append(int(self.tokenizer.eos_token_id))
            call_end = self.get_tool_call_tokens().get("call_end", "")
            if call_end:
                try:
                    end_ids = self.tokenizer.encode(call_end, add_special_tokens=False)
                    if len(end_ids) == 1:
                        eos_token_ids.append(int(end_ids[0]))
                except Exception:
                    pass
            eos_token_ids = list(dict.fromkeys(eos_token_ids))

            with torch.no_grad():
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids if eos_token_ids else None,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            new_token_ids = output.sequences[0][prompt_tokens:].detach().cpu()
            if output.scores:
                self.last_logits = output.scores[-1][0].detach().cpu()
            del output
            completion_tokens = len(new_token_ids)
            total_tokens = prompt_tokens + completion_tokens

        del model_inputs

        generated_text = self.tokenizer.decode(
            new_token_ids, skip_special_tokens=False
        )

        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        selected = self._parse_tool_calls(generated_text, tool_names)

        if not selected:
            logger.warning(
                "HuggingFace model ({}) selected no tool. Raw output: {}",
                self.model_id,
                generated_text[-400:],  # show the last 400 chars which should contain the tool call if present
            )

        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        return SelectionResult(
            selected_tools=selected,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------ #
    # White-box access                                                     #
    # ------------------------------------------------------------------ #

    def get_tool_call_tokens(self) -> Dict[str, str]:
        """Return the tool-call special tokens for the loaded model family.

        Matches ``self.model_id`` (case-insensitive) against the keys of
        ``_TOOL_CALL_TOKENS`` and returns the corresponding token dict.
        Returns an empty dict for unrecognised model families.

        Common keys in the returned dict (not all families define all keys):
          ``call_start`` / ``call_end``         – single tool-call delimiters
          ``calls_start`` / ``calls_end``       – outer multi-call wrapper
          ``response_start`` / ``response_end`` – tool result delimiters
          ``responses_start`` / ``responses_end``– outer multi-result wrapper
          ``def_start`` / ``def_end``           – tool definition delimiters
          ``list_start`` / ``list_end``         – available-tools list markers
          ``sep``                               – intra-call separator token
        """
        model_lower = self.model_id.lower()
        for family, tokens in self._TOOL_CALL_TOKENS.items():
            if family in model_lower:
                return tokens
        return {}

    def get_last_logits(self):
        """Return logits of the last generated token (CPU tensor).

        Available after each ``select_tools`` call.  Returns ``None``
        before the first call.
        """
        return self.last_logits

    def get_last_attentions(self):
        """Return per-layer attention rows from the first generated tool token.

        Shape: tuple of ``(num_heads, key_len)`` tensors, one per layer.
        Only populated when ``output_attentions=True`` was passed to ``__init__``.
        """
        return self.last_attentions

    def get_last_hidden_states(self):
        """Return per-layer hidden states from the first generated token.

        Shape: tuple of ``(seq_len, d_model)`` tensors, one per layer
        (layer 0 = embedding output, layer L = final transformer layer).
        Only populated when ``output_hidden_states=True`` was passed to ``__init__``.
        """
        return self.last_hidden_states

    def compute_saliency(
        self,
        tools: List[Dict[str, Any]],
        prompt: str,
        target_token_id: Optional[int] = None,
    ):
        """Compute gradient-based input saliency for the tool selection decision.

        Uses a single forward pass (not ``generate``) and backpropagates the
        logit of *target_token_id* (defaults to the argmax of ``last_logits``)
        through the input embeddings.

        Args:
            tools: Tool list passed to the model (same format as ``select_tools``).
            prompt: User prompt string.
            target_token_id: Vocabulary index to differentiate. Defaults to the
                             ``argmax`` of the most recently cached ``last_logits``.
                             Pass ``None`` to auto-select.

        Returns:
            Dict with keys:
              ``"saliency"``   : 1-D CPU tensor, L2-norm of embedding gradient per token.
              ``"input_ids"``  : 1-D CPU tensor of input token IDs.
              ``"tokens"``     : list of decoded token strings.
              ``"target_token_id"``: the vocabulary index used.
        """
        import torch

        # ── Build input ids ──────────────────────────────────────────────
        model_inputs = self._normalize_model_inputs(self._build_input_ids(tools, prompt))
        try:
            target_device = next(self.model.parameters()).device
            model_inputs = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }
        except StopIteration:
            pass

        input_ids = self._extract_input_ids(model_inputs)
        if input_ids is None:
            raise ValueError("Tokenized inputs do not contain 'input_ids'.")

        # ── Get embeddings with gradient tracking ────────────────────────
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(input_ids)   # (1, seq_len, d_model)
        input_embeds = input_embeds.detach().requires_grad_(True)

        # ── Forward pass using inputs_embeds ─────────────────────────────
        with torch.enable_grad():
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=model_inputs.get("attention_mask"),
                use_cache=False,
            )
            # outputs.logits shape: (1, seq_len, vocab_size)
            # We care about the *next-token* logit — the last position
            next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

            if target_token_id is None:
                if self.last_logits is not None:
                    target_token_id = int(self.last_logits.argmax())
                else:
                    target_token_id = int(next_token_logits.argmax())

            target_logit = next_token_logits[target_token_id]
            target_logit.backward()

        # ── Saliency = L2-norm of gradient per token ─────────────────────
        # input_embeds.grad shape: (1, seq_len, d_model)
        grad = input_embeds.grad[0]           # (seq_len, d_model)
        saliency = grad.norm(dim=-1).cpu()    # (seq_len,)

        flat_ids = input_ids[0].cpu()
        tokens = [
            self.tokenizer.decode([tid]) for tid in flat_ids.tolist()
        ]

        return {
            "saliency": saliency,
            "input_ids": flat_ids,
            "tokens": tokens,
            "target_token_id": target_token_id,
        }

    # ------------------------------------------------------------------ #
    # Phase 5 – Residual-stream steering                                  #
    # ------------------------------------------------------------------ #

    def forward_all_hidden(
        self,
        tools: List[Dict[str, Any]],
        prompt: str,
    ) -> List:
        """Run a non-generation forward pass and return per-layer hidden states.

        Returns a list of CPU tensors of shape ``(d_model,)``, one per
        transformer layer (embedding output excluded, index 0 = first
        transformer block output, index -1 = last block output).

        Unlike ``get_last_hidden_states()``, this method does *not* call
        ``model.generate()``; it uses a plain ``model.forward()`` call, which
        is cheaper and does not require ``output_hidden_states=True`` to be
        set at construction time.
        """
        import torch

        model_inputs = self._normalize_model_inputs(self._build_input_ids(tools, prompt))
        try:
            target_device = next(self.model.parameters()).device
            model_inputs = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }
        except StopIteration:
            pass

        input_ids = self._extract_input_ids(model_inputs)
        if input_ids is None:
            raise ValueError("Tokenized inputs do not contain 'input_ids'.")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=model_inputs.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states: tuple of (batch, seq_len, d_model), one per layer
        # index 0 = embedding output; we skip it and return transformer-layer outputs
        return [hs[0, -1, :].detach().cpu() for hs in outputs.hidden_states[1:]]

    def _get_transformer_layers(self):
        """Return the list of transformer block modules for the loaded model.

        Tries common HuggingFace architecture naming conventions.
        Raises ``RuntimeError`` if no match is found.
        """
        m = self.model
        for attr_path in (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("model", "decoder", "layers"),
        ):
            obj = m
            try:
                for attr in attr_path:
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        raise RuntimeError(
            f"Cannot locate transformer layer list for model {self.model_id}. "
            "Supported attributes: model.layers, transformer.h, gpt_neox.layers."
        )

    def register_layer_steering_hook(
        self,
        layer_idx: int,
        direction,
        lam: float = 1.0,
    ):
        """Register a residual-stream steering hook on transformer layer *layer_idx*.

        The hook subtracts the component of the hidden state along *direction*
        at the **last input token position**, and fires **only on the first
        forward call** (tracked per-hook via a closure counter) so that
        KV-cache autoregressive decoding steps are unaffected.

        Args:
            layer_idx: Index into the transformer layer list (0 = first layer,
                       -1 = last layer).
            direction: 1-D tensor of shape ``(d_model,)``; will be normalised.
            lam: Steering strength (multiplier on the projected component).

        Returns:
            The ``torch.utils.hooks.RemovableHook`` handle (also stored in
            ``self._steering_hooks`` for bulk removal via
            ``remove_steering_hooks()``).
        """
        import torch

        layers = self._get_transformer_layers()
        layer = layers[layer_idx]
        d = direction.clone().float()
        d = d / (d.norm() + 1e-12)

        call_count = [0]

        def _hook(module, input, output):
            if call_count[0] > 0:
                # Skip KV-cache decoding steps (only steer the first call)
                call_count[0] += 1
                return output
            call_count[0] += 1

            h = output[0] if isinstance(output, tuple) else output
            d_dev = d.to(h.device, h.dtype)
            # Project out bias direction at the last token position
            proj = (h[:, -1, :] @ d_dev).unsqueeze(-1) * d_dev  # (batch, d_model)
            h[:, -1, :] = h[:, -1, :] - lam * proj
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        handle = layer.register_forward_hook(_hook)
        self._steering_hooks.append(handle)
        return handle

    def remove_steering_hooks(self) -> None:
        """Remove all active residual-stream steering hooks."""
        for handle in self._steering_hooks:
            handle.remove()
        self._steering_hooks.clear()

    def get_provider_name(self) -> str:
        return "huggingface"

    def get_model_name(self) -> str:
        return self.model_id
