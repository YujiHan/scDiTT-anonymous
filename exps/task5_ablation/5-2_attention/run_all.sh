#!/usr/bin/env bash
# ============================================================================
#
#
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

GPUS="${GPUS:-0 1}"
PREP_GPU="${PREP_GPU:-0}"
OUT="${ROOT}/results/task5_ablation/5-2_attention"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"

prep() {
    echo "======== stage A: VAE + OT cache (GPU ${PREP_GPU}) $(date +%F_%T) ========"
    CUDA_VISIBLE_DEVICES="${PREP_GPU}" "${PYTHON}" "${HERE}/preprocess.py" \
        --device cuda:0 2>&1 | tee "${LOGD}/preprocess.log"
}

train() {
    local wait_flag="${1:-}"
    echo "======== stage B: training matrix (GPUS='${GPUS}') $(date +%F_%T) ========"
    "${PYTHON}" "${HERE}/gen_jobs.py" --pending | sed 's/^/  pending: /'
    local pids=()
    for g in ${GPUS}; do
        bash "${HERE}/worker.sh" "${g}" ${wait_flag} 2>&1 | sed "s/^/[gpu${g}] /" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}"; done
    echo "======== stage B finished $(date +%F_%T) ========"
    status
}

probe() {
    local g="${PREP_GPU}"
    echo "======== stage C: chain B mechanism diagnostics (GPU ${g}) $(date +%F_%T) ========"
    CUDA_VISIBLE_DEVICES="${g}" "${PYTHON}" "${HERE}/probe_attention.py" \
        2>&1 | tee "${LOGD}/probe_attention.log"
    CUDA_VISIBLE_DEVICES="${g}" "${PYTHON}" "${HERE}/probe_conditioning.py" \
        2>&1 | tee "${LOGD}/probe_conditioning.log"
    CUDA_VISIBLE_DEVICES="${g}" "${PYTHON}" "${HERE}/bench_efficiency.py" \
        2>&1 | tee "${LOGD}/bench_efficiency.log"
}

agg() {
    echo "======== stage D: aggregation $(date +%F_%T) ========"
    "${PYTHON}" "${HERE}/aggregate.py" 2>&1 | tee "${LOGD}/aggregate.log"
}

status() {
    "${PYTHON}" "${HERE}/status.py"
}

case "${1:-}" in
    prep)   prep ;;
    train)  train "${2:-}" ;;
    probe)  probe ;;
    agg)    agg ;;
    status) status ;;
    all)    prep; train "${2:-}"; probe; agg ;;
    *) echo "usage: bash run_all.sh {prep|train [--wait]|probe|agg|status|all}"; exit 1 ;;
esac
