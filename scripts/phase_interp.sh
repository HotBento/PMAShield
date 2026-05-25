#!/usr/bin/env bash
# Interpretability pipeline (paper §3): activation patching → head roles → circuit.
#
# LOCAL / REMOTE: REMOTE-ONLY (loads model on GPU).
#
# Single model:
#   bash scripts/phase_interp.sh --gpu 0 --model Qwen/Qwen3-8B
#
# All 6 models (sequential):
#   for m in Qwen/Qwen3-4B Qwen/Qwen3-8B microsoft/Phi-4-mini-instruct \
#            google/gemma-3-4b-it google/gemma-4-E4B-it \
#            meta-llama/Meta-Llama-3.1-8B-Instruct; do
#       bash scripts/phase_interp.sh --gpu 0 --model "$m"
#   done
#
# Stages:
#   patching        layer + head causal patching
#   head_roles      classify top-K heads into intent / intent-matching
#   circuit         top-K head-pair path-replacement matrix
#   all             run patching → head_roles → circuit (default)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/common.sh

STAGE="${STAGE:-all}"
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TOP_K="${TOP_K:-16}"
ROLE_TOP_K="${ROLE_TOP_K:-24}"
LIMIT="${LIMIT:-}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_PAIRS_PER_COMBO="${MAX_PAIRS_PER_COMBO:-5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)               STAGE="$2"; shift 2 ;;
        --gpu)                 GPU="$2"; shift 2 ;;
        --model)               MODEL="$2"; shift 2 ;;
        --top-k)               TOP_K="$2"; shift 2 ;;
        --role-top-k)          ROLE_TOP_K="$2"; shift 2 ;;
        --limit)               LIMIT="$2"; shift 2 ;;
        --torch-dtype)         TORCH_DTYPE="$2"; shift 2 ;;
        --load-in-8bit)        LOAD_IN_8BIT="1"; shift ;;
        --load-in-4bit)        LOAD_IN_4BIT="1"; shift ;;
        --max-pairs-per-combo) MAX_PAIRS_PER_COMBO="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_SAFE="$(model_safe "$MODEL")"
OUT_DIR="results/interp/${MODEL_SAFE}"
PATCH_DIR="${OUT_DIR}/patching"
ensure_dir "$PATCH_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

extra_args=()
[[ "$LOAD_IN_8BIT" == "1" ]] && extra_args+=(--load-in-8bit)
[[ "$LOAD_IN_4BIT" == "1" ]] && extra_args+=(--load-in-4bit)
[[ -n "$LIMIT" ]] && extra_args+=(--limit "$LIMIT")

run_patching() {
    echo "=== Interp Stage 1: patching ($MODEL) ==="
    run_py -m pma_shield.interp.scripts.run_patching \
        --model "$MODEL" \
        --out "$PATCH_DIR" \
        --max-pairs-per-combo "$MAX_PAIRS_PER_COMBO" \
        --torch-dtype "$TORCH_DTYPE" \
        "${extra_args[@]}"
}

run_roles() {
    echo "=== Interp Stage 2: head roles ($MODEL) ==="
    run_py -m pma_shield.interp.scripts.run_head_roles \
        --model "$MODEL" \
        --top-k "$ROLE_TOP_K" \
        --torch-dtype "$TORCH_DTYPE"
}

run_circuit() {
    echo "=== Interp Stage 3: circuit matrix ($MODEL) ==="
    run_py -m pma_shield.interp.scripts.run_circuit_matrix \
        --model "$MODEL" \
        --top-k "$TOP_K" \
        --torch-dtype "$TORCH_DTYPE"
}

case "$STAGE" in
    patching)   run_patching ;;
    head_roles) run_roles ;;
    circuit)    run_circuit ;;
    all)        run_patching; run_roles; run_circuit ;;
    *) echo "Unknown stage: $STAGE"; exit 1 ;;
esac

echo "Done: $MODEL → $OUT_DIR"
