#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#   preprocess_vae -> preprocess_coupling -> train_model -> eval_model
#
#
#   DATASETS="ZB MA" SEEDS="42" bash exps/task1_ts_pred/run/run_all.sh
#   CUDA_VISIBLE_DEVICES=0 DATASETS="ZB" bash exps/task1_ts_pred/run/run_all.sh
#
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/env.sh"                 # -> ${PYTHON}
: "${PYTHON:?PYTHON in env.sh is empty: set the interpreter path there, or override with PYTHON=...}"

DATASETS="${DATASETS:-ZB DR SC GA MA}"
TASKS="${TASKS:-interpolation extrapolation}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

RUN () { if [ "${DRY}" = "1" ]; then echo "    + $*"; else "$@"; fi; }

for DS in ${DATASETS}; do
  for TASK in ${TASKS}; do
    for SEED in ${SEEDS}; do
      OUT="results/model_checkpoints/${DS}_${TASK}_seed${SEED}"
      if [ -f "${OUT}/metrics.json" ]; then
        echo "[skip] ${DS} ${TASK} seed${SEED}  (metrics.json already present)"
        continue
      fi
      echo "[run ] ${DS} ${TASK} seed${SEED}"
      RUN "${PYTHON}" run/preprocess_vae.py      --dataset "${DS}" --task "${TASK}" --seed "${SEED}"
      RUN "${PYTHON}" run/preprocess_coupling.py --dataset "${DS}" --task "${TASK}" --seed "${SEED}"
      RUN "${PYTHON}" run/train_model.py         --dataset "${DS}" --task "${TASK}" --seed "${SEED}"
      RUN "${PYTHON}" run/eval_model.py          --checkpoint "${OUT}/best.pt" \
                                                 --dataset "${DS}" --task "${TASK}"
    done
  done
done
echo "[done] Task1: all combinations finished."
