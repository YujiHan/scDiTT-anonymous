#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#   preprocess_vae -> preprocess_coupling -> train_model -> eval_model
#
#
#   DATASETS="ZB" EX_STEPS="0 4" SEEDS="42" bash exps/task2_extra_steps/run/run_all.sh
#
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/env.sh"                 # -> ${PYTHON}
: "${PYTHON:?PYTHON in env.sh is empty: set the interpreter path there, or override with PYTHON=...}"

DATASETS="${DATASETS:-ZB DR SC GA MA}"
EX_STEPS="${EX_STEPS:-0 1 2 3 4 5 6}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TASK=extrapolation_steps
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

RUN () { if [ "${DRY}" = "1" ]; then echo "    + $*"; else "$@"; fi; }

for DS in ${DATASETS}; do
  for EX in ${EX_STEPS}; do
    for SEED in ${SEEDS}; do
      OUT="results/model_checkpoints/${DS}_${TASK}_ex${EX}_seed${SEED}"
      if [ -f "${OUT}/metrics_alltp.json" ] && [ -f "${OUT}/metrics_lasttp.json" ]; then
        echo "[skip] ${DS} ex${EX} seed${SEED}  (both metrics files present)"
        continue
      fi
      echo "[run ] ${DS} ex${EX} seed${SEED}"
      RUN "${PYTHON}" run/preprocess_vae.py      --dataset "${DS}" --task "${TASK}" --ex_steps "${EX}" --seed "${SEED}"
      RUN "${PYTHON}" run/preprocess_coupling.py --dataset "${DS}" --task "${TASK}" --ex_steps "${EX}" --seed "${SEED}"
      RUN "${PYTHON}" run/train_model.py         --dataset "${DS}" --task "${TASK}" --ex_steps "${EX}" --seed "${SEED}"
      RUN "${PYTHON}" run/eval_model.py          --checkpoint "${OUT}/best.pt" \
                                                 --dataset "${DS}" --task "${TASK}" --ex_steps "${EX}"
    done
  done
done
echo "[done] Task2: all combinations finished."
