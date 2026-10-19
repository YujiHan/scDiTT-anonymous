#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"
GPU="${GPU:-3}"; SIGMA="0.25"; STEPS="${TOTAL_STEPS:-8000}"
OUT="${ROOT}/results/task5_ablation/5-1_OT"; LOGD="${OUT}/logs"; CKPT="${OUT}/checkpoints"
mkdir -p "${LOGD}"
PY() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "$@"; }

echo "======== P1 multi-seed $(date +%F_%T) ========"

echo "[build] couplings $(date +%T)"
PY "${HERE}/build_couplings.py" --regime bifurcation --sigma ${SIGMA} --seeds 43,44,45,46 --task extrap --arms true,ot,random      >>"${LOGD}/build_p1p2.log" 2>&1
PY "${HERE}/build_couplings.py" --regime bifurcation --sigma ${SIGMA} --seeds 43,44,45,46 --task interp --arms true,ot,random      >>"${LOGD}/build_p1p2.log" 2>&1
echo "[build] done $(date +%T)"

train_one() {
    local regime="$1" task="$2" arm="$3" seed="$4"
    local sub="${arm}"; [[ "${task}" != "interp" ]] && sub="${task}_${arm}"
    local dm="${CKPT}/${regime}_s${SIGMA}/seed${seed}/${sub}/metrics.json"
    if [[ -f "${dm}" ]]; then echo "[skip] ${task} ${regime} ${arm} s${seed}"; return; fi
    echo "[train] ${task} ${regime} ${arm} s${seed} $(date +%T)"
    PY "${HERE}/train_eval.py" --regime "${regime}" --sigma ${SIGMA} --arm "${arm}" \
        --seed "${seed}" --task "${task}" --total_steps "${STEPS}" --device cuda:0 --num_workers 6 \
        > "${LOGD}/train_${task}_${regime}_${arm}_s${seed}.log" 2>&1
}

for seed in 42 43 44 45 46; do
    for task in interp extrap; do
        for arm in true ot random; do
            train_one bifurcation "${task}" "${arm}" "${seed}"
        done
    done
done
for task in interp extrap; do
    for arm in true ot random; do
        train_one linear "${task}" "${arm}" 42
    done
done
echo "======== P1+P2 training finished $(date +%F_%T) ========"
