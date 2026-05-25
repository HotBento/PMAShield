#!/usr/bin/env bash
# Run the full detection pipeline for all six models on both attack families.
#
# LOCAL / REMOTE: REMOTE-ONLY.
#
# Usage:
#   bash scripts/run_all_models.sh --gpu 0
#   bash scripts/run_all_models.sh --gpu 0 --load-in-4bit  # for memory-limited GPUs

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)          GPU="$2"; shift 2 ;;
        --load-in-8bit) EXTRA_ARGS+=(--load-in-8bit); shift ;;
        --load-in-4bit) EXTRA_ARGS+=(--load-in-4bit); shift ;;
        --torch-dtype)  EXTRA_ARGS+=(--torch-dtype "$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODELS=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-4B"
    "microsoft/Phi-4-mini-instruct"
    "google/gemma-3-4b-it"
    "google/gemma-4-E4B-it"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

ATTACKS=(mcptox mpma)

for model in "${MODELS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        echo "=============================="
        echo "Model: $model  Attack: $attack"
        echo "=============================="
        bash scripts/phase_detection.sh \
            --gpu "$GPU" \
            --model "$model" \
            --attack "$attack" \
            "${EXTRA_ARGS[@]}"
    done
done

echo "All models and attacks complete."
