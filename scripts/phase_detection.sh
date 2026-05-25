#!/usr/bin/env bash
# Detection pipeline (paper §4–§5): capture → selection → disagreement → detection.
#
# LOCAL / REMOTE: REMOTE-ONLY (loads model on GPU).
#
# Usage:
#   bash scripts/phase_detection.sh --model Qwen/Qwen3-8B --attack mcptox --gpu 0
#   bash scripts/phase_detection.sh --model Qwen/Qwen3-8B --attack mpma   --gpu 0
#
# Stages:
#   data          Stage 0: generate pairs.jsonl from raw dataset
#   capture       Stage 1: extract per-head attention features
#   selection     Stage 3: discover tool-selection heads (τ grid search)
#   disagreement  Stage 4: compute A / E / D_JS / O metrics
#   detection     Stage 5: LOSO classifier + ablation
#   transfer      RQ2: cross-attack transfer AUROC (requires both attacks captured)
#   all           run data → capture → selection → disagreement → detection (default)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/common.sh

STAGE="${STAGE:-all}"
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
ATTACK="${ATTACK:-mcptox}"   # mcptox | mpma
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)         STAGE="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --attack)        ATTACK="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --torch-dtype)   TORCH_DTYPE="$2"; shift 2 ;;
        --load-in-8bit)  LOAD_IN_8BIT="1"; shift ;;
        --load-in-4bit)  LOAD_IN_4BIT="1"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_SAFE="$(model_safe "$MODEL")"
CAPTURE_DIR="results/${ATTACK}/${MODEL_SAFE}/attn_cache"
OUT_DIR="results/${ATTACK}/${MODEL_SAFE}"
SELECTION_JSON="${OUT_DIR}/selection_heads.json"

ensure_dir "$CAPTURE_DIR"
ensure_dir "$OUT_DIR"

quantize_args=()
[[ "$LOAD_IN_8BIT" == "1" ]] && quantize_args+=(--load-in-8bit)
[[ "$LOAD_IN_4BIT" == "1" ]] && quantize_args+=(--load-in-4bit)

limit_args=()
[[ -n "$LIMIT" ]] && limit_args+=(--limit "$LIMIT")

run_data() {
    echo "=== Stage 0: generate pairs ($ATTACK) ==="
    run_py -m pma_shield.detector.scripts.run_data \
        --attack "$ATTACK" --out "data/${ATTACK}/"
}

run_capture() {
    echo "=== Stage 1: capture ($MODEL, $ATTACK) ==="
    CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.detector.scripts.run_capture \
        --model "$MODEL" \
        --pairs "data/${ATTACK}/pairs.jsonl" \
        --out "$CAPTURE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --torch-dtype "$TORCH_DTYPE" \
        "${quantize_args[@]}" \
        "${limit_args[@]}"
}

run_selection() {
    echo "=== Stage 3: selection head discovery ($MODEL) ==="
    run_py -m pma_shield.detector.scripts.run_selection \
        --capture "$CAPTURE_DIR" \
        --out "$OUT_DIR" \
        --grid-search
}

run_disagreement() {
    echo "=== Stage 4: disagreement metrics ($MODEL, $ATTACK) ==="
    run_py -m pma_shield.detector.scripts.run_disagreement \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$OUT_DIR"
}

run_detection() {
    echo "=== Stage 5: LOSO detection ($MODEL, $ATTACK) ==="
    run_py -m pma_shield.detector.scripts.run_detection \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$OUT_DIR"
}

if [[ "$STAGE" == "all" || "$STAGE" == "data" ]];         then run_data; fi
if [[ "$STAGE" == "all" || "$STAGE" == "capture" ]];      then run_capture; fi
if [[ "$STAGE" == "all" || "$STAGE" == "selection" ]];    then run_selection; fi
if [[ "$STAGE" == "all" || "$STAGE" == "disagreement" ]]; then run_disagreement; fi
if [[ "$STAGE" == "all" || "$STAGE" == "detection" ]];    then run_detection; fi

echo "Done: $MODEL / $ATTACK → $OUT_DIR"
