#!/usr/bin/env bash
# End-to-end runner for all ACL-rebuttal experiments (PMAShield, submission 7367).
#
# Runs, in order: pairs generation -> Stage-1 capture -> selection heads ->
# main detection (LOSO, now with pooled_scores/pooled_labels) -> extras
# capture (commit-step logits + hidden states) -> the four 3Bi4 baselines ->
# adaptive-attacker sweep (texz W2) -> security-metric report (TPR@FPR /
# PR-AUC / bootstrap CI, 3Bi4 W2) -> Qwen3-4B cross-model reproduction
# (Nyqg W3) -> cross-model figures/tables.
#
# RESUMABILITY: every stage first checks whether its expected output already
# exists (and, where relevant, contains the fields this rebuttal round added)
# before doing any GPU work. If a stage fails, fix the code and re-run this
# same script — completed stages are skipped automatically and the script
# picks up at the failed stage. Use --force to ignore all of that and
# re-run everything from scratch.
#
# Usage:
#   bash scripts/run_rebuttal_experiments.sh --gpu 0
#   bash scripts/run_rebuttal_experiments.sh --gpu 1 --model Qwen/Qwen3-8B \
#       --secondary-model Qwen/Qwen3-4B --extras-limit 350
#   bash scripts/run_rebuttal_experiments.sh --gpu 0 --force        # redo everything
#   bash scripts/run_rebuttal_experiments.sh --gpu 0 --only baselines  # just one stage
#
# See rebuttal_experiment_plan_v1.md for the full experiment-design writeup.

set -uo pipefail
# Deliberately no `-e`: each stage's success/failure is checked explicitly so
# we can print which stage failed and stop there, rather than an opaque abort.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/common.sh

# ── Defaults (override via flags) ──────────────────────────────────────────
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SECONDARY_MODEL="${SECONDARY_MODEL:-Qwen/Qwen3-4B}"   # Nyqg W3 cross-model repro
EXTRAS_LIMIT="${EXTRAS_LIMIT:-350}"                    # see plan doc §3 for rationale
CAPTURE_BATCH_SIZE="${CAPTURE_BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LOAD_IN_8BIT=""
LOAD_IN_4BIT=""
FORCE=0
ONLY=""
FPR_TARGETS="0.01 0.05 0.10"
BOOTSTRAP=2000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)              GPU="$2"; shift 2 ;;
        --model)             MODEL="$2"; shift 2 ;;
        --secondary-model)   SECONDARY_MODEL="$2"; shift 2 ;;
        --extras-limit)      EXTRAS_LIMIT="$2"; shift 2 ;;
        --batch-size)        CAPTURE_BATCH_SIZE="$2"; shift 2 ;;
        --max-new-tokens)    MAX_NEW_TOKENS="$2"; shift 2 ;;
        --torch-dtype)       TORCH_DTYPE="$2"; shift 2 ;;
        --load-in-8bit)      LOAD_IN_8BIT="--load-in-8bit"; shift ;;
        --load-in-4bit)      LOAD_IN_4BIT="--load-in-4bit"; shift ;;
        --force)             FORCE=1; shift ;;
        --only)              ONLY="$2"; shift 2 ;;
        --bootstrap)         BOOTSTRAP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_SAFE="$(model_safe "$MODEL")"
SECONDARY_SAFE="$(model_safe "$SECONDARY_MODEL")"

DATA_DIR="data/mcptox"
PAIRS_JSONL="$DATA_DIR/pairs.jsonl"

OUT_DIR="results/mcptox/${MODEL_SAFE}"
CAPTURE_DIR="$OUT_DIR/attn_cache"
EXTRAS_DIR="$OUT_DIR/attn_cache_extras"
SELECTION_JSON="$OUT_DIR/selection_heads.json"
DETECTION_JSON="$OUT_DIR/detection_report.json"
BASELINES_DIR="$OUT_DIR/baselines"
BASELINES_JSON="$BASELINES_DIR/baselines.json"
ADAPTIVE_DIR="$OUT_DIR/adaptive"
ADAPTIVE_JSON="$ADAPTIVE_DIR/adaptive_sweep.json"
SECURITY_MD="$OUT_DIR/security_metrics.md"

SECONDARY_PATCH_DIR="results/interp/${SECONDARY_SAFE}/patching"

ensure_dir "$OUT_DIR"

echo "=========================================================="
echo " PMAShield rebuttal experiments"
echo "   Primary model:   $MODEL  ($MODEL_SAFE)"
echo " Secondary model:   $SECONDARY_MODEL  ($SECONDARY_SAFE)"
echo "   GPU:              $GPU"
echo "   Extras limit:     $EXTRAS_LIMIT pairs"
echo "   Force re-run:      $([[ $FORCE -eq 1 ]] && echo yes || echo no)"
[[ -n "$ONLY" ]] && echo "   Only stage:        $ONLY"
echo "=========================================================="
echo

# ── Generic stage driver ────────────────────────────────────────────────────
# run_stage <name> <check_fn> <run_fn>
#   check_fn returns 0 (true) if the stage's output already exists and is
#   valid -> skipped unless --force. run_fn performs the actual work; on
#   non-zero exit the whole script stops immediately (fix + re-run resumes
#   here, since already-done stages will report [SKIP] again).
run_stage() {
    local name="$1" check_fn="$2" run_fn="$3"
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
        return 0
    fi
    echo "---------------------------------------------------------"
    echo "Stage: $name"
    echo "---------------------------------------------------------"
    if [[ "$FORCE" -ne 1 ]] && "$check_fn"; then
        echo "  [SKIP] output already present and valid — pass --force to redo."
        echo
        return 0
    fi
    if "$run_fn"; then
        echo "  [OK] $name"
        echo
    else
        echo
        echo "  [FAIL] Stage '$name' failed (see log above)."
        echo "         Fix the issue, then re-run this exact command —"
        echo "         completed stages will be skipped automatically and"
        echo "         this script will resume at '$name'."
        exit 1
    fi
}

# ── Stage 0: pairs.jsonl ─────────────────────────────────────────────────────
check_data() { [[ -f "$PAIRS_JSONL" ]]; }
do_data() {
    ensure_dir "$DATA_DIR"
    run_py -m pma_shield.detector.scripts.run_data --out "$DATA_DIR/"
}

# ── Stage 1: ordinary Stage-1 capture (attention only) ──────────────────────
check_capture() { [[ -f "$CAPTURE_DIR/manifest.json" ]]; }
do_capture() {
    ensure_dir "$CAPTURE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.detector.scripts.run_capture \
        --model "$MODEL" \
        --pairs "$PAIRS_JSONL" \
        --out "$CAPTURE_DIR" \
        --batch-size "$CAPTURE_BATCH_SIZE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --torch-dtype "$TORCH_DTYPE" \
        $LOAD_IN_8BIT $LOAD_IN_4BIT
}

# ── Stage 2: selection-head discovery ───────────────────────────────────────
check_selection() { [[ -f "$SELECTION_JSON" ]]; }
do_selection() {
    run_py -m pma_shield.detector.scripts.run_selection \
        --capture "$CAPTURE_DIR" \
        --out "$OUT_DIR" \
        --grid-search
}

# ── Stage 3: main detection (LOSO) — needs the new pooled_scores field ──────
check_detection() {
    [[ -f "$DETECTION_JSON" ]] && grep -q '"pooled_scores"' "$DETECTION_JSON" 2>/dev/null
}
do_detection() {
    run_py -m pma_shield.detector.scripts.run_detection \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$OUT_DIR"
}

# ── Stage 4: extras capture (commit-step logits + hidden states) ───────────
check_extras_capture() {
    [[ -f "$EXTRAS_DIR/manifest.json" ]] && grep -q '"capture_extras": *true' "$EXTRAS_DIR/manifest.json" 2>/dev/null
}
do_extras_capture() {
    ensure_dir "$EXTRAS_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.detector.scripts.run_capture \
        --model "$MODEL" \
        --pairs "$PAIRS_JSONL" \
        --out "$EXTRAS_DIR" \
        --batch-size 1 \
        --capture-extras \
        --limit "$EXTRAS_LIMIT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --torch-dtype "$TORCH_DTYPE" \
        $LOAD_IN_8BIT $LOAD_IN_4BIT
}

# ── Stage 5: four 3Bi4 baselines ────────────────────────────────────────────
check_baselines() {
    [[ -f "$BASELINES_JSON" ]] \
        && grep -q '"pooled_scores"' "$BASELINES_JSON" 2>/dev/null \
        && grep -q 'logit_margin' "$BASELINES_JSON" 2>/dev/null \
        && grep -q 'activation_probe' "$BASELINES_JSON" 2>/dev/null \
        && grep -q 'residual_ood' "$BASELINES_JSON" 2>/dev/null
}
do_baselines() {
    ensure_dir "$BASELINES_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.detector.scripts.run_baselines \
        --capture "$EXTRAS_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$BASELINES_DIR" \
        --baselines all
}

# ── Stage 6: adaptive-attacker sweep (texz W2) ──────────────────────────────
check_adaptive() { [[ -f "$ADAPTIVE_JSON" ]]; }
do_adaptive() {
    ensure_dir "$ADAPTIVE_DIR"
    run_py -m pma_shield.detector.scripts.run_adaptive_attack \
        --capture "$CAPTURE_DIR" \
        --selection "$SELECTION_JSON" \
        --out "$ADAPTIVE_DIR" \
        --fractions 1.0 0.75 0.5 0.25 0.1
}

# ── Stage 7: security-metric report (3Bi4 W2: TPR@FPR / PR-AUC / CI) ───────
check_security_metrics() { [[ -f "$SECURITY_MD" ]]; }
do_security_metrics() {
    # shellcheck disable=SC2086
    run_py -m pma_shield.detector.scripts.report_security_metrics \
        --detection "$DETECTION_JSON:main_logistic:PMAShield" \
        --detection "$DETECTION_JSON:random_heads:Random heads (ablation)" \
        --detection "$DETECTION_JSON:top6:Top-6 heads (ablation)" \
        --detection "$DETECTION_JSON:query_attn:Query-attention (ablation)" \
        --baselines "$BASELINES_JSON" \
        --out "$SECURITY_MD" \
        --fpr-targets $FPR_TARGETS \
        --bootstrap "$BOOTSTRAP"
}

# ── Stage 8: Qwen3-4B cross-model reproduction (Nyqg W3) ────────────────────
check_secondary_patching() {
    [[ -f "$SECONDARY_PATCH_DIR/layer_attn_mlp.npz" && -f "$SECONDARY_PATCH_DIR/head_importance.npz" ]]
}
do_secondary_patching() {
    ensure_dir "$SECONDARY_PATCH_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" run_py -m pma_shield.interp.scripts.run_patching \
        --model "$SECONDARY_MODEL" --limit 40 \
        --out "$SECONDARY_PATCH_DIR" \
        --torch-dtype "$TORCH_DTYPE" \
        $LOAD_IN_8BIT $LOAD_IN_4BIT
}

# ── Stage 9: cross-model figures + concentration table ──────────────────────
check_figures() {
    [[ -f "figures/interp/fig_attn_vs_mlp_multi_model.pdf" && -f "figures/interp/tab_concentration.tex" ]]
}
do_figures() {
    run_py -m pma_shield.interp.scripts.make_figures
}

# ── Run everything ──────────────────────────────────────────────────────────
run_stage "data"                check_data                do_data
run_stage "capture"              check_capture             do_capture
run_stage "selection"            check_selection           do_selection
run_stage "detection"            check_detection           do_detection
run_stage "extras_capture"       check_extras_capture      do_extras_capture
run_stage "baselines"            check_baselines           do_baselines
run_stage "adaptive"             check_adaptive            do_adaptive
run_stage "security_metrics"     check_security_metrics    do_security_metrics
run_stage "secondary_patching"   check_secondary_patching  do_secondary_patching
run_stage "figures"              check_figures             do_figures

echo "=========================================================="
echo " All requested stages complete."
echo "   Detection report:   $DETECTION_JSON"
echo "   Baselines:           $BASELINES_JSON"
echo "   Adaptive sweep:       $ADAPTIVE_JSON"
echo "   Security metrics:     $SECURITY_MD"
echo "   Qwen3-4B patching:    $SECONDARY_PATCH_DIR/"
echo "   Figures:              figures/interp/"
echo "=========================================================="
