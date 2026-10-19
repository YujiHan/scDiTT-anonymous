#!/usr/bin/env bash
# ============================================================================
# Task1 time-series prediction -- data preprocessing (VAE encoding + OT coupling)
# ----------------------------------------------------------------------------
# Builds the two caches later training needs for a given dataset. Both are keyed
# uniquely by dataset/task/seed so that randomness runs through the whole chain
# VAE->latents->OT->diffusion (the single source of truth is
# src/data/splits.py: cache_paths):
#   1) VAE: pre-trained only on the train_tps of that task, then used to encode
#      every cell (test tps are encoded only)
#        -> results/task1_ts_pred/vae_checkpoints/{DS}_{task}_seed{seed}.pt
#        -> results/task1_ts_pred/vae_checkpoints/{DS}_{task}_seed{seed}_latents.h5ad  (obsm['X_latent'])
#   2) OT coupling: one set of adjacent train_tp couplings per task (reading the latents of that seed)
#        -> results/task1_ts_pred/coupling_cache/{DS}_{task}_seed{seed}_{OT_TYPE}/pi_*.npz + meta.json
#
# The VAE is task-dependent (different train_tps -> a different VAE); the task
# label is part of correctness and VAEs must never be reused across tasks.
# Existing caches are skipped by default (force with --force / --force-vae /
# --force-ot); "already exists" for the VAE means splits.vae_cache_valid, which
# requires the train_tps recorded in the ckpt to match this task.
#
# Usage:
#   bash preprocess.sh --dataset ZB
#   bash preprocess.sh --dataset ZB --tasks "interpolation extrapolation" \
#        --device cuda:0 --epsilon 0.05 --max-cells 2000 --vae-epochs 150
#
# Options (aligned with run/preprocess_vae.py + run/preprocess_coupling.py):
#   --dataset NAME        dataset: ZB/DR/SC/GA/MA (required)
#   --tasks  "..."        space-separated list of tasks to compute OT for (default: both)
#   --seed N              random seed (default 0)
#   --device DEV          cpu / cuda:N, shared by the VAE and OT (default cuda:0)
#   --- VAE hyper-parameters (leave empty to use the config defaults) ---
#   --vae-epochs N        VAE training epochs (config default 150)
#   --latent-dim N        latent dimensionality (default 64)
#   --vae-lr F            VAE learning rate (default 1e-3)
#   --vae-batch-size N    VAE batch size (default 256)
#   --vae-beta F          KL weight beta (default 0.1)
#   --- OT hyper-parameters ---
#   --ot-type T           cost matrix computation (for the OT comparison, default CPU):
#                         CPU (528 classic 3-D broadcast) / CPU_COST (memory-saving CPU) / CUDA (GPU)
#                         CPU/CPU_COST run on the CPU; CUDA uses the GPU given by --device
#   --epsilon F           Sinkhorn entropic regularisation epsilon (default 0.05)
#   --max-cells N         per-side cell cap, <=0 disables it (default 2000, matching 528)
#   --no-normalize-cost   disable median normalisation of the cost
#   --- control ---
#   --force-vae / --force-ot / --force   force recomputation
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/env.sh"

DATASET=""
TASKS="interpolation extrapolation"
SEED=0
DEVICE="cuda:0"
VAE_EPOCHS=""
LATENT_DIM=""
VAE_LR=""
VAE_BATCH=""
VAE_BETA=""
OT_TYPE="CPU"
EPSILON="0.05"
MAX_CELLS="2000"
NORMALIZE_COST=1
FORCE_VAE=0
FORCE_OT=0

RESULTS_ROOT="results/task1_ts_pred"

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)           DATASET="$2"; shift 2;;
    --tasks)             TASKS="$2"; shift 2;;
    --seed)              SEED="$2"; shift 2;;
    --device)            DEVICE="$2"; shift 2;;
    --vae-epochs)        VAE_EPOCHS="$2"; shift 2;;
    --latent-dim)        LATENT_DIM="$2"; shift 2;;
    --vae-lr)            VAE_LR="$2"; shift 2;;
    --vae-batch-size)    VAE_BATCH="$2"; shift 2;;
    --vae-beta)          VAE_BETA="$2"; shift 2;;
    --ot-type)           OT_TYPE="$2"; shift 2;;
    --epsilon)           EPSILON="$2"; shift 2;;
    --max-cells)         MAX_CELLS="$2"; shift 2;;
    --no-normalize-cost) NORMALIZE_COST=0; shift;;
    --force-vae)         FORCE_VAE=1; shift;;
    --force-ot)          FORCE_OT=1; shift;;
    --force)             FORCE_VAE=1; FORCE_OT=1; shift;;
    -h|--help)           usage; exit 0;;
    *) echo "unknown argument: $1"; echo "run -h for help"; exit 1;;
  esac
done

[[ -z "${DATASET}" ]] && { echo "error: --dataset is required"; usage; exit 1; }
DS_UP="$(echo "${DATASET}" | tr '[:lower:]' '[:upper:]')"
OT_TYPE="$(echo "${OT_TYPE}" | tr '[:lower:]' '[:upper:]')"

echo "============================================================"
echo "[preprocess] dataset=${DS_UP} tasks='${TASKS}' seed=${SEED} ot_type=${OT_TYPE} device=${DEVICE}"
echo "[preprocess] root=${ROOT}"
echo "============================================================"

vae_cache_ok() {   # $1=dataset $2=task $3=seed $4=results_root [$5=ex_steps]
  "${PYTHON}" - "$@" <<'PY'
import sys
sys.path.insert(0, ".")
from src.data.splits import vae_cache_valid
ds, task, seed, root = sys.argv[1:5]
ex = int(sys.argv[5]) if len(sys.argv) > 5 else None
sys.exit(0 if vae_cache_valid(ds, task, ex, int(seed), results_root=root) else 1)
PY
}

ot_cache_ok() {    # $1=dataset $2=task $3=seed $4=results_root $5=ot_type [$6=ex_steps]
  "${PYTHON}" - "$@" <<'PY'
import sys
sys.path.insert(0, ".")
from src.data.splits import coupling_complete
ds, task, seed, root, ot = sys.argv[1:6]
ex = int(sys.argv[6]) if len(sys.argv) > 6 else None
sys.exit(0 if coupling_complete(ds, task, ex, int(seed),
                                ot_type=ot, results_root=root) else 1)
PY
}

for TASK in ${TASKS}; do
  TAG="${TASK}"
  VAE_CKPT="${RESULTS_ROOT}/vae_checkpoints/${DS_UP}_${TAG}_seed${SEED}.pt"
  LATENTS="${RESULTS_ROOT}/vae_checkpoints/${DS_UP}_${TAG}_seed${SEED}_latents.h5ad"
  COUPLING_DIR="${RESULTS_ROOT}/coupling_cache/${DS_UP}_${TAG}_seed${SEED}_${OT_TYPE}"

  if [[ "${FORCE_VAE}" -eq 0 ]] && \
     vae_cache_ok "${DS_UP}" "${TASK}" "${SEED}" "${RESULTS_ROOT}"; then
    echo "[VAE] ${TAG}/seed${SEED} cache exists and matches train_tps, skipping (--force-vae to recompute): ${VAE_CKPT}"
  else
    VAE_ARGS=(--dataset "${DS_UP}" --task "${TASK}" --seed "${SEED}" --device "${DEVICE}"
              --results_root "${RESULTS_ROOT}")
    [[ -n "${VAE_EPOCHS}" ]] && VAE_ARGS+=(--epochs "${VAE_EPOCHS}")
    [[ -n "${LATENT_DIM}" ]] && VAE_ARGS+=(--latent_dim "${LATENT_DIM}")
    [[ -n "${VAE_LR}"     ]] && VAE_ARGS+=(--lr "${VAE_LR}")
    [[ -n "${VAE_BATCH}"  ]] && VAE_ARGS+=(--batch_size "${VAE_BATCH}")
    [[ -n "${VAE_BETA}"   ]] && VAE_ARGS+=(--beta "${VAE_BETA}")
    echo "[VAE] run/preprocess_vae.py ${VAE_ARGS[*]}"
    "${PYTHON}" run/preprocess_vae.py "${VAE_ARGS[@]}"
  fi

  if [[ "${FORCE_OT}" -eq 0 ]] && \
     ot_cache_ok "${DS_UP}" "${TASK}" "${SEED}" "${RESULTS_ROOT}" "${OT_TYPE}"; then
    echo "[OT] ${TAG}/seed${SEED} cache is complete and not older than latents, skipping (--force-ot to recompute): ${COUPLING_DIR}"
  else
    OT_ARGS=(--dataset "${DS_UP}" --task "${TASK}" --seed "${SEED}" \
             --ot_type "${OT_TYPE}" --device "${DEVICE}" \
             --epsilon "${EPSILON}" --max_cells "${MAX_CELLS}" \
             --results_root "${RESULTS_ROOT}")
    [[ "${NORMALIZE_COST}" -eq 0 ]] && OT_ARGS+=(--no_normalize_cost)
    echo "[OT] run/preprocess_coupling.py ${OT_ARGS[*]}"
    "${PYTHON}" run/preprocess_coupling.py "${OT_ARGS[@]}"
  fi
done

echo "============================================================"
echo "[preprocess] done. caches (by dataset/task/seed):"
for TASK in ${TASKS}; do
  echo "  ${DS_UP}/${TASK}/seed${SEED}:"
  echo "    VAE ${RESULTS_ROOT}/vae_checkpoints/${DS_UP}_${TASK}_seed${SEED}.pt"
  echo "    OT  ${RESULTS_ROOT}/coupling_cache/${DS_UP}_${TASK}_seed${SEED}_${OT_TYPE}/"
done
echo "============================================================"
