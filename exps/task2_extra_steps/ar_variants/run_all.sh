#!/usr/bin/env bash
# ============================================================================
# The two autoregressive variants of scDiTT -- one-command run.
# Produces the AR-prefix / AR-Markov rows of the extrapolation-horizon tables
# and of Figure 3 of the main paper.
#
# The comparison is run only on the horizon sweep (Appendix J): 5 datasets x
# 6 horizons x 5 seeds x 2 arms = 300 runs. Each arm shares the autoencoder and
# the optimal-transport couplings of the corresponding scDiTT run, so run the
# scDiTT sweep first (exps/task2_extra_steps/run/run_all.sh) and point
# SCDITT_CACHE_ROOT at its results root -- otherwise every arm refits its own.
#
# Usage:
#   bash run_all.sh                              # everything, resumable
#   bash run_all.sh --dry-run
#   DATASETS="ZB" HORIZONS="1 6" SEEDS="42" bash run_all.sh
#   ARMS="prefix" bash run_all.sh
#
# Environment (defaults in parentheses):
#   DATASETS            (ZB DR SC GA MA)
#   HORIZONS            (1 2 3 4 5 6)
#   SEEDS               (42 43 44 45 46)
#   ARMS                (prefix markov)
#   GPU                 CUDA device index (0)
#   SCDITT_CACHE_ROOT   scDiTT sweep results root to reuse (results/task2_extra_steps)
#   PYTHON              interpreter; normally taken from env.sh
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

DATASETS="${DATASETS:-ZB DR SC GA MA}"
HORIZONS="${HORIZONS:-1 2 3 4 5 6}"
SEEDS="${SEEDS:-42 43 44 45 46}"
ARMS="${ARMS:-prefix markov}"
GPU="${GPU:-0}"
export SCDITT_CACHE_ROOT="${SCDITT_CACHE_ROOT:-results/task2_extra_steps}"

OUT="${ROOT}/results/task2_extra_steps/ar_variants"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"

DRY=0
for a in "$@"; do [[ "${a}" == "--dry-run" ]] && DRY=1; done

say() { echo -e "\n\033[1m===== $* =====\033[0m  $(date +%F\ %T)"; }
die() { echo -e "\033[31m[FATAL] $*\033[0m"; exit 1; }

[[ -n "${PYTHON:-}" && -x "${PYTHON}" ]] \
    || die "PYTHON is not set or not executable (edit ${ROOT}/env.sh, or pass PYTHON=...)"

cd "${ROOT}" || die "cannot cd to ${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

say "self-test"
"${PYTHON}" "${HERE}/selftest.py" || die "the AR construction self-test failed"

N_TOTAL=0
for d in ${DATASETS}; do for h in ${HORIZONS}; do for s in ${SEEDS}; do
  for a in ${ARMS}; do N_TOTAL=$((N_TOTAL+1)); done; done; done; done

say "plan"
echo "  datasets  : ${DATASETS}"
echo "  horizons  : ${HORIZONS}"
echo "  seeds     : ${SEEDS}"
echo "  arms      : ${ARMS}"
echo "  runs      : ${N_TOTAL}"
echo "  reuse     : SCDITT_CACHE_ROOT=${SCDITT_CACHE_ROOT}"
echo "  output    : ${OUT}"

T0=$(date +%s); DONE=0; FAILED=0
for SEED in ${SEEDS}; do
  for DS in ${DATASETS}; do
    for H in ${HORIZONS}; do
      for ARM in ${ARMS}; do
        TAG="${DS}_ex${H}_seed${SEED}"
        LOG="${LOGD}/train_${TAG}_${ARM}.log"
        DONE=$((DONE+1))
        echo -e "\n[${DONE}/${N_TOTAL}] AR-${ARM}  ${DS}  horizon=${H}  seed=${SEED}"
        echo "  \$ ${PYTHON} ${HERE}/train_arm.py --arm ${ARM} --dataset ${DS} --horizon ${H} --seed ${SEED}"
        echo "    -> ${LOG}"
        (( DRY == 1 )) && continue
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/train_arm.py" \
            --arm "${ARM}" --dataset "${DS}" --horizon "${H}" --seed "${SEED}" \
            > "${LOG}" 2>&1 \
          || { echo "  [fail] see ${LOG}"; FAILED=$((FAILED+1)); }
      done
    done
  done
done

if (( DRY == 1 )); then
  echo -e "\n[--dry-run] nothing was executed."
  exit 0
fi

say "aggregate"
"${PYTHON}" "${HERE}/aggregate.py" 2>&1 | tee "${LOGD}/aggregate.log"

MIN=$(( ($(date +%s) - T0) / 60 ))
say "finished in ${MIN} min"
if (( FAILED > 0 )); then
  echo -e "\033[33m[warn] ${FAILED} run(s) failed; rerun to resume (finished runs are skipped).\033[0m"
  exit 1
fi
echo "Artefacts:"
echo "  tables : ${OUT}/metrics/ar_horizons.csv  (+ ar_horizons.md)"
echo "  runs   : ${OUT}/runs/<DS>_ex<l>_seed<seed>/<arm>/"
