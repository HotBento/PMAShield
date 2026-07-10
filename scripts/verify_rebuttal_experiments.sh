#!/usr/bin/env bash
# Rebuttal-experiment verification checklist (run on the GPU server FIRST,
# before scaling any of the new experiments to full size).
#
# None of the new code paths added for the ACL rebuttal (commit-step
# logits/hidden-state capture, the four baselines, the adaptive-attack
# simulation, the Qwen3-4B cross-model reproduction) have been exercised
# against a real model — only against synthetic in-memory data on a
# non-GPU machine. This script runs each of them on a small sample first
# and asserts basic sanity properties, so a structural bug shows up in
# minutes rather than after a multi-hour full run.
#
# Usage:
#   bash scripts/verify_rebuttal_experiments.sh --model Qwen/Qwen3-8B --gpu 0
#
# Exit code 0 iff every check passes. On failure, re-run the failing stage
# manually with the same flags to see the full log — this script trims
# output to keep the summary readable.

set -uo pipefail
# NOTE: deliberately no `-e` — this script's whole purpose is to run every
# verification step and report PASS/FAIL per step, so one failing stage
# must not abort the rest. Each step below checks its own exit code /
# output existence explicitly instead.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/common.sh

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SMOKE_LIMIT="${SMOKE_LIMIT:-2}"        # pairs for the capture_extras structural check
BASELINE_LIMIT="${BASELINE_LIMIT:-20}" # pairs for the tiny baseline dry run

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --gpu)   GPU="$2"; shift 2 ;;
        --smoke-limit) SMOKE_LIMIT="$2"; shift 2 ;;
        --baseline-limit) BASELINE_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_SAFE="$(model_safe "$MODEL")"
CAPTURE_DIR="results/mcptox/${MODEL_SAFE}/attn_cache"
EXTRAS_DIR="results/mcptox/${MODEL_SAFE}/attn_cache_extras_smoke"
SELECTION_JSON="results/mcptox/${MODEL_SAFE}/selection_heads.json"
VERIFY_OUT="results/mcptox/${MODEL_SAFE}/_rebuttal_verify"

ensure_dir "$VERIFY_OUT"

PASS=1
check() {
    local desc="$1"
    local cond="$2"
    if [[ "$cond" == "1" ]]; then
        echo "  [PASS] $desc"
    else
        echo "  [FAIL] $desc"
        PASS=0
    fi
}

echo "=========================================================="
echo " Step A — capture_extras structural check ($SMOKE_LIMIT pairs)"
echo "=========================================================="
echo "Prerequisite: data/mcptox/pairs.jsonl must exist (run_data.py)."
if [[ ! -f "data/mcptox/pairs.jsonl" ]]; then
    echo "  Generating pairs.jsonl first..."
    run_py -m pma_shield.detector.scripts.run_data --out "data/mcptox/"
fi

CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.detector.scripts.run_capture \
    --model "$MODEL" \
    --pairs "data/mcptox/pairs.jsonl" \
    --out "$EXTRAS_DIR" \
    --batch-size 1 \
    --capture-extras \
    --limit "$SMOKE_LIMIT" \
    --no-resume
CAPTURE_RC=$?
check "run_capture --capture-extras exits 0" "$([[ $CAPTURE_RC -eq 0 ]] && echo 1 || echo 0)"

run_py - "$EXTRAS_DIR" <<'PYEOF'
import sys, json
from pathlib import Path
out_dir = Path(sys.argv[1])

manifest = json.loads((out_dir / "manifest.json").read_text())
assert manifest.get("capture_extras") is True, "manifest.capture_extras is not True"
assert manifest.get("hidden_layers"), "manifest.hidden_layers is empty"
print("  manifest OK: hidden_layers =", manifest["hidden_layers"])

meta = [json.loads(l) for l in (out_dir / "meta.jsonl").read_text().splitlines()]
assert len(meta) > 0, "meta.jsonl is empty"
n_valid_logits = sum(
    1 for m in meta
    if m.get("commit_logit_top1") == m.get("commit_logit_top1")  # not NaN
    and m.get("commit_logit_top2") == m.get("commit_logit_top2")
)
print(f"  meta.jsonl: {n_valid_logits}/{len(meta)} records have non-NaN commit logits")
assert n_valid_logits > 0, "ALL commit_logit_top1/top2 are NaN — provider capture likely broken"

import numpy as np
pair0 = np.load(out_dir / "topheads" / "pair_0000.npz")
has_hidden = "commit_hidden_benign" in pair0 and "commit_hidden_malicious" in pair0
print("  pair_0000.npz keys:", list(pair0.keys()))
assert has_hidden, "commit_hidden_benign/malicious missing from pair_0000.npz"
hb = pair0["commit_hidden_benign"]
print(f"  commit_hidden_benign shape={hb.shape} dtype={hb.dtype}")
assert hb.ndim == 2 and hb.shape[0] == len(manifest["hidden_layers"]), "unexpected commit_hidden shape"
assert not np.isnan(hb).all(), "commit_hidden_benign is all-NaN"
print("STRUCTURAL_CHECK_OK")
PYEOF
STEP_A_RC=$?
check "capture_extras produces valid logits + hidden states" "$([[ $STEP_A_RC -eq 0 ]] && echo 1 || echo 0)"

echo
echo "=========================================================="
echo " Step B — attention_to_tool_mass baseline (existing capture, no extras needed)"
echo "=========================================================="
if [[ ! -d "$CAPTURE_DIR" ]]; then
    echo "  [SKIP] $CAPTURE_DIR not found — run scripts/phase_detection.sh first to"
    echo "         produce the ordinary (non-extras) Stage-1 capture + selection_heads.json."
    check "attention_to_tool_mass baseline runs" 0
else
    run_py -m pma_shield.detector.scripts.run_baselines \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$VERIFY_OUT/baselines_smoke" \
        --baselines attention_to_tool_mass
    if [[ -f "$VERIFY_OUT/baselines_smoke/baselines.json" ]]; then
        check "attention_to_tool_mass baseline runs" 1
    else
        check "attention_to_tool_mass baseline runs" 0
    fi
fi

echo
echo "=========================================================="
echo " Step C — logit_margin / activation_probe / residual_ood on the smoke-extras capture"
echo "=========================================================="
echo "  (n=$SMOKE_LIMIT pairs is too small for a real LOSO AUROC — this only checks"
echo "   that the code path runs without crashing and reports n > 0.)"
run_py -m pma_shield.detector.scripts.run_baselines \
    --capture "$EXTRAS_DIR" \
    --out "$VERIFY_OUT/baselines_extras_smoke" \
    --baselines logit_margin activation_probe residual_ood || true
if [[ -f "$VERIFY_OUT/baselines_extras_smoke/baselines.json" ]]; then
    check "logit_margin/activation_probe/residual_ood code paths run" 1
else
    check "logit_margin/activation_probe/residual_ood code paths run" 0
fi

echo
echo "=========================================================="
echo " Step D — adaptive-attack sweep sanity check (fraction=1.0 should match RQ1)"
echo "=========================================================="
if [[ ! -d "$CAPTURE_DIR" ]]; then
    echo "  [SKIP] $CAPTURE_DIR not found."
    check "adaptive sweep fraction=1.0 reproduces RQ1 AUROC" 0
else
    run_py -m pma_shield.detector.scripts.run_adaptive_attack \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$VERIFY_OUT/adaptive_smoke" \
        --fractions 1.0
    echo "  >>> Manually compare the printed AUROC above against Table 2/3's"
    echo "      reported MCPTox AUROC for $MODEL (e.g. 0.741 for Qwen3-8B)."
    echo "      They should match closely (small differences from RNG seed are OK;"
    echo "      a large gap means partial_hijack_df / evaluate_adaptive_loso has a bug."
    if [[ -f "$VERIFY_OUT/adaptive_smoke/adaptive_sweep.json" ]]; then
        check "adaptive sweep produced output" 1
    else
        check "adaptive sweep produced output" 0
    fi
fi

echo
echo "=========================================================="
echo " Step E — Qwen3-4B interp reproduction (Finding 1 & 2), tiny --limit"
echo "=========================================================="
CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.interp.scripts.run_patching \
    --model Qwen/Qwen3-4B --limit 4 \
    --out results/interp/Qwen_Qwen3-4B/patching/ || true
if [[ -f "results/interp/Qwen_Qwen3-4B/patching/layer_attn_mlp.npz" && \
      -f "results/interp/Qwen_Qwen3-4B/patching/head_importance.npz" ]]; then
    check "Qwen3-4B patching produces layer_attn_mlp.npz + head_importance.npz" 1
else
    check "Qwen3-4B patching produces layer_attn_mlp.npz + head_importance.npz" 0
fi

echo
echo "=========================================================="
echo " Step F — cross-model figures/table (local, no GPU needed once E has run)"
echo "=========================================================="
run_py -m pma_shield.interp.scripts.make_figures || true
if [[ -f "figures/interp/fig_attn_vs_mlp_multi_model.pdf" && \
      -f "figures/interp/tab_concentration.tex" ]]; then
    check "cross-model attn-share figure + concentration table generated" 1
else
    echo "  (expected if Qwen3-8B's own patching/ artefacts aren't present yet —"
    echo "   the multi-model chart needs >= 2 models with layer_attn_mlp.npz)"
    check "cross-model attn-share figure + concentration table generated" 0
fi

echo
echo "=========================================================="
if [[ "$PASS" == "1" ]]; then
    echo " ALL CHECKS PASSED — safe to scale up to the full experiment plan"
    echo " (see rebuttal_experiment_plan_v1.md §1)."
else
    echo " SOME CHECKS FAILED — do not launch full-scale runs yet."
    echo " Re-run the failing stage manually (commands above) to see the full"
    echo " traceback, fix, then re-run this script."
fi
echo "=========================================================="
[[ "$PASS" == "1" ]]
