#!/usr/bin/env bash
# ============================================================================
# 5-4 trajectory-construction ablation on real data -- one-command run.
# Produces Table S3: entropic OT / hard OT / random / k-NN matching on ZB and SC
# under the extrapolation split, five seeds each.
#
# Per (dataset, seed) the autoencoder is fitted ONCE and shared by all four arms,
# so the arms differ only in the coupling kernel. The entropic arm is scDiTT
# itself and its row reproduces the scDiTT entries of the extrapolation table.
#
# Stages:
#   1) fit the VAE for each (dataset, seed)            [shared by the four arms]
#   2) build one coupling cache per arm                [the only thing that varies]
#   3) train + evaluate each arm at the full 40k budget
#   4) aggregate into metrics/main_table.csv
#
# Usage:
#   bash run_all.sh                       # everything, resumable
#   bash run_all.sh --dry-run             # print the plan only
#   DATASETS="ZB" SEEDS="42" bash run_all.sh
#   ARMS="entropic knn" bash run_all.sh
#
# Environment (all optional, defaults in parentheses):
#   DATASETS   datasets to run (ZB SC)
#   SEEDS      random seeds (42 43 44 45 46)
#   ARMS       arms to run (entropic hard_ot random knn)
#   GPU        CUDA device index (0)
#   OT_TYPE    coupling cost backend, CPU / CPU_COST / CUDA (CPU)
#   PYTHON     interpreter; normally taken from env.sh at the repository root
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

DATASETS="${DATASETS:-ZB SC}"
SEEDS="${SEEDS:-42 43 44 45 46}"
ARMS="${ARMS:-entropic hard_ot random knn}"
TASK="extrapolation"
GPU="${GPU:-0}"
OT_TYPE="${OT_TYPE:-CPU}"

RES_REL="results/task5_ablation/5-4_OT_real"
OUT="${ROOT}/${RES_REL}"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"

DRY=0
for a in "$@"; do [[ "${a}" == "--dry-run" ]] && DRY=1; done

say() { echo -e "\n\033[1m===== $* =====\033[0m  $(date +%F\ %T)"; }
die() { echo -e "\033[31m[FATAL] $*\033[0m"; exit 1; }
RUN() {
    echo "  \$ $*"
    (( DRY == 1 )) && return 0
    "$@" || return 1
}

# Same, but sends the command's own output to a log file. The echoed command
# line still goes to the console, so --dry-run stays readable.
RUN_LOG() {
    local log="$1"; shift
    echo "  \$ $*"
    echo "    -> ${log}"
    (( DRY == 1 )) && return 0
    "$@" > "${log}" 2>&1 || return 1
}

[[ -n "${PYTHON:-}" && -x "${PYTHON}" ]] \
    || die "PYTHON is not set or not executable (edit ${ROOT}/env.sh, or pass PYTHON=...)"

cd "${ROOT}" || die "cannot cd to ${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

say "plan"
echo "  datasets : ${DATASETS}"
echo "  seeds    : ${SEEDS}"
echo "  arms     : ${ARMS}"
echo "  task     : ${TASK} (Table S3 is extrapolation only)"
echo "  output   : ${OUT}"

T0=$(date +%s)
FAILED=0

for DS in ${DATASETS}; do
  for SEED in ${SEEDS}; do
    TAG="${DS}_${TASK}_seed${SEED}"
    LATENTS="${OUT}/vae_checkpoints/${TAG}_latents.h5ad"

    say "stage 1/3  ${DS} seed${SEED}: autoencoder (shared by every arm)"
    if [[ -f "${LATENTS}" ]]; then
      echo "  [skip] ${LATENTS} exists"
    else
      RUN env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" run/preprocess_vae.py \
          --dataset "${DS}" --task "${TASK}" --seed "${SEED}" \
          --results_root "${RES_REL}" \
        || { echo "  [fail] VAE ${TAG}"; FAILED=$((FAILED+1)); continue; }
    fi

    say "stage 2/3  ${DS} seed${SEED}: couplings, one cache per arm"
    RUN "${PYTHON}" "${HERE}/build_couplings.py" \
        --dataset "${DS}" --task "${TASK}" --seed "${SEED}" \
        --latents "${LATENTS}" --arms ${ARMS} --ot-type "${OT_TYPE}" \
      || { echo "  [fail] couplings ${TAG}"; FAILED=$((FAILED+1)); continue; }

    for ARM in ${ARMS}; do
      RUNDIR="${OUT}/runs/${TAG}/${ARM}"
      if [[ -f "${RUNDIR}/metrics.json" ]]; then
        echo "  [skip] ${TAG}/${ARM} already evaluated"
        continue
      fi
      say "stage 3/3  ${DS} seed${SEED} arm=${ARM}: train + evaluate"
      # --dataset.coupling_cache_dir is the only per-arm difference; the VAE
      # checkpoint and the latents come from the shared results_root above.
      RUN_LOG "${LOGD}/train_${TAG}_${ARM}.log" \
          env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" run/train_model.py \
          --dataset "${DS}" --task "${TASK}" --seed "${SEED}" \
          --results_root "${RES_REL}" \
          --output_dir "${RUNDIR}" \
          --ot_type "${OT_TYPE}" \
          --dataset.coupling_cache_dir "${OUT}/couplings/${TAG}/${ARM}" \
        || { echo "  [fail] ${TAG}/${ARM} -- see ${LOGD}/train_${TAG}_${ARM}.log"; FAILED=$((FAILED+1)); }
    done
  done
done

if (( DRY == 1 )); then
  echo -e "\n[--dry-run] nothing was executed."
  exit 0
fi

say "aggregate"
RUN "${PYTHON}" "${HERE}/aggregate.py" 2>&1 | tee "${LOGD}/aggregate.log"

MIN=$(( ($(date +%s) - T0) / 60 ))
say "finished in ${MIN} min"
if (( FAILED > 0 )); then
  echo -e "\033[33m[warn] ${FAILED} step(s) failed; rerun this script to resume (finished work is skipped).\033[0m"
  exit 1
fi
echo "Artefacts:"
echo "  table : ${OUT}/metrics/main_table.csv  (+ main_table.md, runs_long.csv)"
echo "  runs  : ${OUT}/runs/<DS>_${TASK}_seed<seed>/<arm>/"
