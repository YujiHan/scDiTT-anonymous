#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

GPU="${GPU:-3}"
SIGMA="0.25"
TOTAL_STEPS="${TOTAL_STEPS:-8000}"
SEED=42
TASK=extrap
OUT="${ROOT}/results/task5_ablation/5-1_OT"
LOGD="${OUT}/logs"
CKPT="${OUT}/checkpoints"
mkdir -p "${LOGD}"

INTERP_DONE="${CKPT}/linear_s0.25/seed0/random/metrics.json"
echo "[extrap] $(date +%F_%T) waiting for the interpolation matrix to finish (poll ${INTERP_DONE})"
WAITED=0
while [[ ! -f "${INTERP_DONE}" ]]; do
    sleep 120; WAITED=$((WAITED+120))
    if [[ ${WAITED} -gt 21600 ]]; then echo "[extrap] waited more than 6h, giving up"; exit 1; fi
done
echo "[extrap] $(date +%F_%T) interpolation done; waiting another 30s for the GPU to free up"; sleep 30

echo "[extrap] $(date +%F_%T) rebuilding the extrapolation couplings with CUDA"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/build_couplings.py" \
    --regime all --sigma "${SIGMA}" --seeds "${SEED}" --task extrap \
    --ot_type CUDA --device cuda:0 > "${LOGD}/build_extrap_cuda.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/ot_metrics.py" \
    --regime all --sigma "${SIGMA}" --seeds "${SEED}" --task extrap \
    > "${LOGD}/ot_metrics_extrap.log" 2>&1 || echo "[extrap] ot_metrics(extrap) skipped (--task may not be supported)"

train_arm() {
    local regime="$1" arm="$2"
    local dm="${CKPT}/${regime}_s${SIGMA}/seed${SEED}/${TASK}_${arm}/metrics.json"
    if [[ -f "${dm}" ]]; then echo "[skip/extrap] ${regime} ${arm}"; return; fi
    echo "[train/extrap] ${regime} ${arm} $(date +%T)"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/train_eval.py" \
        --regime "${regime}" --sigma "${SIGMA}" --arm "${arm}" --seed "${SEED}" \
        --task "${TASK}" --total_steps "${TOTAL_STEPS}" --device cuda:0 --num_workers 6 \
        > "${LOGD}/train_extrap_${regime}_${arm}_seed${SEED}.log" 2>&1
}
echo "[extrap] $(date +%F_%T) starting the extrapolation training matrix"
for arm in true ot random; do train_arm bifurcation "${arm}"; done
for arm in true ot random; do train_arm linear "${arm}"; done
echo "[extrap] $(date +%F_%T) extrapolation training matrix finished"
