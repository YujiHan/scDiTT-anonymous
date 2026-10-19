#!/usr/bin/env bash
# ============================================================================
# Task1 time-series prediction -- diffusion model training (SCDiT + Flow Matching)
# ----------------------------------------------------------------------------
# Prerequisite: preprocess.sh has produced the VAE + OT caches for this dataset/task.
# Trains one model per (dataset x task x seed) combination and writes best.pt and
# the final ckpt.
#
# Usage:
#   bash train.sh --dataset ZB                          # interp+extrap, seed 0
#   bash train.sh --dataset ZB --task interpolation --seeds "0 1 2"
#   bash train.sh --dataset ZB --task extrapolation --gpu 1 --batch-size 32
#   bash train.sh --dataset ZB --min-snr                # enable min_snr weighting
#   bash train.sh --dataset ZB -- --diffusion.min_snr_gamma 3.0   # pass-through override
#
# Options (aligned with run/train_model.py):
#   --dataset NAME     dataset: ZB/DR/SC/GA/MA (required)
#   --task   T         interpolation | extrapolation | both (default both)
#   --seeds  "..."     random seeds, space-separated (default "0")
#   --total-steps N    training steps (default 40000)
#   --batch-size N     batch (default empty: use the value in configs/, which is 16 on
#                      every dataset -- the setting the paper reports; an explicit
#                      value overrides it, 32 fits on 40G+)
#   --gpu N            GPU index -> CUDA_VISIBLE_DEVICES (default 0)
#   --peak-lr F        peak learning rate (config default 1e-4)
#   --min-snr          enable min-SNR weighting (off by default)
#   --ot-type T        which OT cache to read: CPU / CPU_COST / CUDA (default CPU)
#                      must match --ot-type of preprocess.sh (the cache directory carries an _{OT_TYPE} suffix)
#   --output-base DIR  model output root (default results/task1_ts_pred; caches still read that experiment root)
#   --output-dir DIR   set the output directory of a single run directly (overrides output-base, single combination only)
#   --resume-from CKPT resume from a checkpoint
#   -- ...             everything after '--' is passed through verbatim to train_model (e.g. --training.xxx)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/env.sh"

DATASET=""
TASK="both"
SEEDS="0"
TOTAL_STEPS=40000
BATCH_SIZE=""
GPU=0
PEAK_LR=""
USE_MIN_SNR=0
OT_TYPE="CPU"
RESULTS_ROOT="results/task1_ts_pred"
OUTPUT_BASE="${RESULTS_ROOT}"
OUTPUT_DIR=""
RESUME_FROM=""
EXTRA=()

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DATASET="$2"; shift 2;;
    --task)         TASK="$2"; shift 2;;
    --seeds|--seed) SEEDS="$2"; shift 2;;
    --total-steps)  TOTAL_STEPS="$2"; shift 2;;
    --batch-size)   BATCH_SIZE="$2"; shift 2;;
    --gpu)          GPU="$2"; shift 2;;
    --peak-lr)      PEAK_LR="$2"; shift 2;;
    --min-snr)      USE_MIN_SNR=1; shift;;
    --ot-type)      OT_TYPE="$2"; shift 2;;
    --output-base)  OUTPUT_BASE="$2"; shift 2;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2;;
    --resume-from)  RESUME_FROM="$2"; shift 2;;
    -h|--help)      usage; exit 0;;
    --)             shift; EXTRA+=("$@"); break;;
    *) echo "unknown argument: $1 (put arbitrary train_model overrides after --)"; exit 1;;
  esac
done

[[ -z "${DATASET}" ]] && { echo "error: --dataset is required"; usage; exit 1; }
DS_UP="$(echo "${DATASET}" | tr '[:lower:]' '[:upper:]')"
DS_LOW="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
OT_TYPE="$(echo "${OT_TYPE}" | tr '[:lower:]' '[:upper:]')"

if [[ "${TASK}" == "both" ]]; then
  TASK_LIST=(interpolation extrapolation)
else
  TASK_LIST=("${TASK}")
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "============================================================"
echo "[train] dataset=${DS_UP} tasks='${TASK_LIST[*]}' seeds='${SEEDS}'"
echo "[train] total_steps=${TOTAL_STEPS} batch=${BATCH_SIZE} gpu=${GPU} min_snr=${USE_MIN_SNR}"
echo "============================================================"

for t in "${TASK_LIST[@]}"; do
  for s in ${SEEDS}; do
    VAE_CKPT="${RESULTS_ROOT}/vae_checkpoints/${DS_UP}_${t}_seed${s}.pt"
    LATENTS="${RESULTS_ROOT}/vae_checkpoints/${DS_UP}_${t}_seed${s}_latents.h5ad"
    COUPLING_DIR="${RESULTS_ROOT}/coupling_cache/${DS_UP}_${t}_seed${s}_${OT_TYPE}"
    if [[ ! -f "${VAE_CKPT}" || ! -f "${LATENTS}" \
          || ! -d "${COUPLING_DIR}" || -z "$(ls -A "${COUPLING_DIR}" 2>/dev/null)" ]]; then
      echo "error: missing the VAE or OT(${OT_TYPE}) cache for ${DS_UP}/${t}/seed${s}. Run first:"
      echo "  bash ${SCRIPT_DIR}/preprocess.sh --dataset ${DS_UP} --tasks \"${t}\" --seed ${s} --ot-type ${OT_TYPE} --device cuda:0"
      exit 1
    fi
    if ! "${PYTHON}" - "${DS_UP}" "${t}" "${s}" "${RESULTS_ROOT}" <<'PY'
import sys
sys.path.insert(0, ".")
from src.data.splits import vae_cache_valid
ds, task, seed, root = sys.argv[1:5]
sys.exit(0 if vae_cache_valid(ds, task, None, int(seed), results_root=root) else 1)
PY
    then
      echo "error: the VAE for ${DS_UP}/${t}/seed${s} does not match train_tps for this task; the latent space may"
      echo "      already contain test information. Rerun preprocessing:"
      echo "  bash ${SCRIPT_DIR}/preprocess.sh --dataset ${DS_UP} --tasks \"${t}\" --seed ${s} --ot-type ${OT_TYPE} --device cuda:0 --force"
      exit 1
    fi

    if [[ -n "${OUTPUT_DIR}" ]]; then
      out="${OUTPUT_DIR}"
    else
      out="${OUTPUT_BASE}/${DS_UP}_${t}_seed${s}"
    fi

    ARGS=(--dataset "${DS_LOW}" --task "${t}" --seed "${s}" --output_dir "${out}"
          --results_root "${RESULTS_ROOT}"
          --ot_type "${OT_TYPE}" --training.total_steps "${TOTAL_STEPS}")
    [[ -n "${BATCH_SIZE}" ]]     && ARGS+=(--training.batch_size "${BATCH_SIZE}")
    [[ "${USE_MIN_SNR}" -eq 1 ]] && ARGS+=(--diffusion.use_min_snr true)
    [[ -n "${PEAK_LR}" ]]        && ARGS+=(--training.peak_lr "${PEAK_LR}")
    [[ -n "${RESUME_FROM}" ]]    && ARGS+=(--resume_from "${RESUME_FROM}")
    ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

    echo "------------------------------------------------------------"
    echo "[train] ${DS_UP}/${t}/seed${s} -> ${out}"
    echo "[train] run/train_model.py ${ARGS[*]}"
    echo "------------------------------------------------------------"
    "${PYTHON}" run/train_model.py "${ARGS[@]}"
  done
done

echo "============================================================"
echo "[train] all done. checkpoints are in ${OUTPUT_BASE}/ (or wherever --output-dir points)"
echo "============================================================"
