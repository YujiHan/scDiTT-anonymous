#!/usr/bin/env bash
# ============================================================================
# Task1 time-series prediction -- UMAP visualisation (one model / one random seed)
# ----------------------------------------------------------------------------
# Draws the **interpolation + extrapolation** generated test points of one model on
# one dataset into a single two-panel figure, emitting one PNG per training seed
# (left Ground Truth / right the model).
# PCA+UMAP is fitted once on the real data and cached, so the background is
# point-for-point identical across seeds and across models.
#
# Images go to <out-dir>/<dataset>/<dataset>_<model>_seed<N>.png (one directory level per dataset).
#
# Usage:
#   bash run_umap.sh                                   # default scDiTT + all datasets + seeds 42/43/44
#   bash run_umap.sh --datasets "ZB DR"                # only some datasets
#   bash run_umap.sh --datasets ZB --model-name scDiTT --seeds "42 43 44"
#   bash run_umap.sh --refit                           # force the embedding cache to be recomputed
#
# Options:
#   --datasets "..."     dataset list, space-separated (default "ZB DR GA MA SC")
#   --model-name NAME    model name in the right-hand title and the filename (default scDiTT)
#   --seeds "..."        random seeds, space-separated (default "42 43 44")
#   --results-root DIR   read-only experiment results root (required; no default)
#   --out-dir DIR        image output root (default results/task1_ts_pred_visualization)
#   --refit              ignore the embedding cache and refit
#   -- ...               everything after '--' is passed through verbatim to umap_visualize.py
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/env.sh"

DATASETS="ZB DR GA MA SC"
MODEL_NAME="scDiTT"
SEEDS="42 43 44 45 46"
RESULTS_ROOT=""
OUT_DIR=""
REFIT=0
EXTRA=()

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets|--dataset) DATASETS="$2"; shift 2;;
    --model-name)   MODEL_NAME="$2"; shift 2;;
    --seeds|--seed) SEEDS="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --out-dir)      OUT_DIR="$2"; shift 2;;
    --refit)        REFIT=1; shift;;
    -h|--help)      usage; exit 0;;
    --)             shift; EXTRA+=("$@"); break;;
    *) echo "unknown argument: $1"; usage; exit 1;;
  esac
done

for DS in ${DATASETS}; do
  ARGS=(--dataset "${DS}" --model-name "${MODEL_NAME}" --seeds ${SEEDS})
  [[ -n "${RESULTS_ROOT}" ]] && ARGS+=(--results-root "${RESULTS_ROOT}")
  [[ -n "${OUT_DIR}" ]]      && ARGS+=(--out-dir "${OUT_DIR}")
  [[ "${REFIT}" -eq 1 ]]     && ARGS+=(--refit)
  ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

  echo "============================================================"
  echo "[umap] dataset=${DS} model=${MODEL_NAME} seeds='${SEEDS}'"
  echo "============================================================"
  "${PYTHON}" "${SCRIPT_DIR}/umap_visualize.py" "${ARGS[@]}"
done
