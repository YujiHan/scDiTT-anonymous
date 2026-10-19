#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"
GPU="$1"; SPECS="$2"; SIGMA="0.25"; STEPS="${TOTAL_STEPS:-8000}"
OUT="${ROOT}/results/task5_ablation/5-1_OT"; LOGD="${OUT}/logs"; CKPT="${OUT}/checkpoints"
mkdir -p "${LOGD}"

train_one() {
    local regime="$1" task="$2" arm="$3" seed="$4"
    local sub="${arm}"; [[ "${task}" != "interp" ]] && sub="${task}_${arm}"
    local dm="${CKPT}/${regime}_s${SIGMA}/seed${seed}/${sub}/metrics.json"
    if [[ -f "${dm}" ]]; then echo "[skip gpu${GPU}] ${task} ${regime} ${arm} s${seed}"; return; fi
    echo "[train gpu${GPU}] ${task} ${regime} ${arm} s${seed} $(date +%T)"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/train_eval.py" \
        --regime "${regime}" --sigma ${SIGMA} --arm "${arm}" --seed "${seed}" \
        --task "${task}" --total_steps "${STEPS}" --device cuda:0 --num_workers 5 \
        > "${LOGD}/train_${task}_${regime}_${arm}_s${seed}.log" 2>&1
}

echo "[worker gpu${GPU}] start specs='${SPECS}' $(date +%F_%T)"
for spec in ${SPECS}; do
    regime="${spec%:*}"; seed="${spec#*:}"
    for task in interp extrap; do
        for arm in true ot random; do
            train_one "${regime}" "${task}" "${arm}" "${seed}"
        done
    done
done
echo "[worker gpu${GPU}] done $(date +%F_%T)"
