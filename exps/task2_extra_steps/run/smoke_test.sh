#!/usr/bin/env bash
# ============================================================================
# Task2 extrapolation steps -- smoke test (very few steps, checks the pipeline runs)
# ----------------------------------------------------------------------------
# Runs preprocessing -> training -> evaluation -> resume once, with a tiny step count.
# Acceptance: training produces no nan, checkpoints save/restore, and evaluation
# emits the unified baseline 5 metrics (metrics_alltp/lasttp.json).
#
# The task is fixed to extrapolation_steps: the last N time points are the test set.
# Defaults to ZB / 2 extrapolation steps / 160 training steps, a few minutes.
#
# NOTE naming convention (matching the python side, two different kinds of "steps"):
#     --ex_steps     N = extrapolation steps (the last N time points are the test set).
#     --train-steps  M = training iterations (a tiny value for the smoke test).
#
# Usage:
#   bash smoke_test.sh
#   bash smoke_test.sh --dataset ZB --ex_steps 3 --train-steps 160 --gpu 0
#
# Options:
#   --dataset NAME   dataset (default ZB)
#   --ex_steps N     extrapolation steps, the last N time points are the test set (default 2, N in [1, T-1])
#   --train-steps N  training iterations (default 160)
#   --gpu    N       GPU index (default 0)
#   --vae-epochs N   epochs used when the VAE cache is missing (default 20, smoke test only)
#   --ot-type T      OT cost computation CPU/CPU_COST/CUDA (default CPU); used for both preprocessing and training
#   --batch-size N   batch override (default empty = use the dataset config; SC needs 8 to avoid a 24GB OOM)
#   --no-resume      skip the resume test
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/env.sh"

DATASET="ZB"
EX_STEPS=2
TRAIN_STEPS=160
GPU=0
VAE_EPOCHS=20
DO_RESUME=1
OT_TYPE="CPU"
BATCH_SIZE=""
RESULTS_ROOT="results/task2_extra_steps"

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)     DATASET="$2"; shift 2;;
    --ex_steps)    EX_STEPS="$2"; shift 2;;
    --train-steps) TRAIN_STEPS="$2"; shift 2;;
    --gpu)         GPU="$2"; shift 2;;
    --vae-epochs)  VAE_EPOCHS="$2"; shift 2;;
    --ot-type)     OT_TYPE="$2"; shift 2;;
    --batch-size)  BATCH_SIZE="$2"; shift 2;;
    --no-resume)   DO_RESUME=0; shift;;
    -h|--help)     usage; exit 0;;
    *) echo "unknown argument: $1"; usage; exit 1;;
  esac
done

DS_UP="$(echo "${DATASET}" | tr '[:lower:]' '[:upper:]')"
DS_LOW="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
OT_TYPE="$(echo "${OT_TYPE}" | tr '[:lower:]' '[:upper:]')"

VAL_EVERY=$(( TRAIN_STEPS / 2 ));   [[ "${VAL_EVERY}" -lt 1 ]] && VAL_EVERY=1
WARMUP=$(( TRAIN_STEPS / 4 ));      [[ "${WARMUP}" -lt 1 ]] && WARMUP=1
LENGTH=$(( TRAIN_STEPS * 2 ))
RESUME_STEPS=$(( TRAIN_STEPS + 10 ))

export CUDA_VISIBLE_DEVICES="${GPU}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RESULTS_ROOT}/smoke_${DS_UP}_extrapolation_steps_${EX_STEPS}_${STAMP}"

BATCH_ARGS=()
[[ -n "${BATCH_SIZE}" ]] && BATCH_ARGS+=(--training.batch_size "${BATCH_SIZE}")

echo "============================================================"
echo "=== Smoke test: ${DS_UP} ex_steps=${EX_STEPS}, ${TRAIN_STEPS} training steps (gpu=${GPU}) ==="
echo "    OUTPUT_DIR=${OUT}"
echo "============================================================"

echo "[smoke] making sure the VAE/OT caches are ready..."
bash "${SCRIPT_DIR}/preprocess.sh" \
    --dataset "${DS_UP}" --ex_steps "${EX_STEPS}" \
    --vae-epochs "${VAE_EPOCHS}" --ot-type "${OT_TYPE}" --device "cuda:0"

echo "[smoke] training ${TRAIN_STEPS} steps..."
"${PYTHON}" run/train_model.py \
    --dataset "${DS_LOW}" \
    --task extrapolation_steps \
    --ex_steps "${EX_STEPS}" \
    --seed 0 \
    --ot_type "${OT_TYPE}" \
    --output_dir "${OUT}" \
    --results_root "${RESULTS_ROOT}" \
    --smoke_test \
    --training.total_steps "${TRAIN_STEPS}" \
    --training.val_every "${VAL_EVERY}" \
    --training.warmup_steps "${WARMUP}" \
    --data.length "${LENGTH}" \
    --data.num_workers 0 \
    ${BATCH_ARGS[@]+"${BATCH_ARGS[@]}"}

CKPT="${OUT}/best.pt"
if [[ -f "${CKPT}" ]]; then
  echo "[smoke] evaluating ${CKPT} (num_repeats=3)..."
  "${PYTHON}" run/eval_model.py \
      --checkpoint "${CKPT}" \
      --dataset "${DS_UP}" \
      --task extrapolation_steps \
      --ex_steps "${EX_STEPS}" \
      --num_repeats 3 \
      --cfg_weight 3.0 \
      --output_dir "${OUT}"
else
  echo "error: best.pt not found at ${CKPT}"; exit 1
fi

if [[ "${DO_RESUME}" -eq 1 ]]; then
  echo "[smoke] resuming training (from best.pt -> ${RESUME_STEPS} steps)..."
  "${PYTHON}" run/train_model.py \
      --dataset "${DS_LOW}" \
      --task extrapolation_steps \
      --ex_steps "${EX_STEPS}" \
      --seed 0 \
      --ot_type "${OT_TYPE}" \
      --output_dir "${OUT}_resume" \
      --results_root "${RESULTS_ROOT}" \
      --smoke_test \
      --resume_from "${CKPT}" \
      --training.total_steps "${RESUME_STEPS}" \
      --training.val_every "${VAL_EVERY}" \
      --training.warmup_steps "${WARMUP}" \
      --data.length "${LENGTH}" \
      --data.num_workers 0 \
      ${BATCH_ARGS[@]+"${BATCH_ARGS[@]}"}
fi

echo "============================================================"
echo "=== Smoke test passed (${DS_UP} ex_steps=${EX_STEPS}) ==="
echo "============================================================"
