#!/usr/bin/env bash
#
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

GPU="${GPU:-3}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TOTAL_STEPS="${TOTAL_STEPS:-8000}"
SIGMA="${SIGMA:-0.25}"
OUT="${ROOT}/results/task5_ablation/5-1_OT"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"
RUN() { echo ">>> $*"; CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "$@"; }

prep() {
    echo "======== stages A-C (GPU ${GPU}) $(date +%F_%T) ========"
    RUN "${HERE}/simulate.py"        --regime all --sigma all --seeds 42,43,44,45,46
    RUN "${HERE}/build_couplings.py" --regime all --sigma all --seeds 42,43,44,45,46 \
        --ot_type CUDA --device cuda:0
    RUN "${HERE}/ot_metrics.py"      --regime all --sigma all --seeds 42,43,44,45,46 \
        > "${LOGD}/ot_metrics_full.log" 2>&1
    echo "OT diagnostics table -> ${LOGD}/ot_metrics_full.log ; json -> ${OUT}/metrics/ot_quality.json"
}

train_arm() {
    local regime="$1" arm="$2" seed="$3"
    local done_marker="${OUT}/checkpoints/${regime}_s${SIGMA}/seed${seed}/${arm}/metrics.json"
    if [[ -f "${done_marker}" ]]; then echo "[skip] ${regime} ${arm} seed${seed} (already done)"; return; fi
    echo "[train] ${regime} s${SIGMA} ${arm} seed${seed} steps=${TOTAL_STEPS} $(date +%T)"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/train_eval.py" \
        --regime "${regime}" --sigma "${SIGMA}" --arm "${arm}" --seed "${seed}" \
        --total_steps "${TOTAL_STEPS}" --device cuda:0 --num_workers 6 \
        > "${LOGD}/train_${regime}_${arm}_seed${seed}.log" 2>&1
}

train() {
    echo "======== stage D training matrix (GPU ${GPU}, seeds='${SEEDS}', steps=${TOTAL_STEPS}) $(date +%F_%T) ========"
    for seed in ${SEEDS}; do
        for regime in bifurcation trifurcation; do
            for arm in true ot random; do
                train_arm "${regime}" "${arm}" "${seed}"
            done
        done
    done
    echo "======== stage D finished $(date +%F_%T) ========"
}

agg() {
    echo "======== stage E aggregation $(date +%F_%T) ========"
    RUN "${HERE}/aggregate.py" --sigma "${SIGMA}" --seeds "$(echo ${SEEDS} | tr ' ' ',')"
}

case "${1:-all}" in
    prep)  prep ;;
    train) train ;;
    agg)   agg ;;
    all)   prep; train; agg ;;
    *) echo "usage: bash run_all.sh {prep|train|agg|all}"; exit 1 ;;
esac
