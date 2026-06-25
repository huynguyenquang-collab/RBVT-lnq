#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T}"
BITS="${BITS:-4}"
NUM_GROUPS="${NUM_GROUPS:-4}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/tinyllama_lnq_rbvt_ppl}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CD_CYCLES="${CD_CYCLES:-4}"
RBVT_POSITION="${RBVT_POSITION:-assignment_last}"
RBVT_MODE="${RBVT_MODE:-lnq_aware}"
RBVT_CALIB_DATASET="${RBVT_CALIB_DATASET:-c4}"
RBVT_N_CALIB="${RBVT_N_CALIB:-128}"
RBVT_MAX_LENGTH="${RBVT_MAX_LENGTH:-2048}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_BUDGET_P="${RBVT_BUDGET_P:-0.005}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-0.1}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-1}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_C4_SAMPLES="${EVAL_C4_SAMPLES:-500}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

timestamp="$(date +%Y%m%d-%H%M%S)"
model_slug="$(slugify "$MODEL")"
log_path="${LOG_DIR}/tinyllama_lnq_rbvt_ppl_${model_slug}_${BITS}bit_${RBVT_POSITION}_${RBVT_MODE}_${timestamp}.log"

exec > >(tee -a "$log_path") 2>&1

echo "=== TinyLlama LNQ + RBVT full PPL | position=${RBVT_POSITION} | target=${RBVT_MODE} | model=${MODEL} | bits=${BITS} ==="
echo "Log: ${log_path}"
echo "PPL only: WikiText-2 + C4 validation, lm-eval disabled"
echo "RBVT budget_p=${RBVT_BUDGET_P} target_ratio=${RBVT_TARGET_RATIO} mse_guard=${RBVT_MSE_GUARD}"

extra_rbvt_args=()
if [[ "$RBVT_MSE_GUARD" == "1" ]]; then
  extra_rbvt_args+=(--rbvt-mse-guard)
fi

"$PYTHON_BIN" main.py \
  --model-path "$MODEL" \
  --bits "$BITS" \
  --device "$DEVICE" \
  --cache-dir "$CACHE_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --dataset "$DATASET" \
  --seq-len "$SEQ_LEN" \
  --num-examples "$NUM_EXAMPLES" \
  --num-groups "$NUM_GROUPS" \
  --num-iterations "$NUM_ITERATIONS" \
  --cd-cycles "$CD_CYCLES" \
  --rbvt-calib-dataset "$RBVT_CALIB_DATASET" \
  --rbvt-n-calib "$RBVT_N_CALIB" \
  --rbvt-max-length "$RBVT_MAX_LENGTH" \
  --rbvt-topk "$RBVT_TOPK" \
  --rbvt-budget-p "$RBVT_BUDGET_P" \
  --rbvt-target-ratio "$RBVT_TARGET_RATIO" \
  "${extra_rbvt_args[@]}" \
  --eval-max-length "$EVAL_MAX_LENGTH" \
  --eval-stride "$EVAL_STRIDE" \
  --eval-c4-samples "$EVAL_C4_SAMPLES" \
  --rbvt-position "$RBVT_POSITION" \
  --rbvt-mode "$RBVT_MODE"
