#!/usr/bin/env bash
# ============================================================================
#
#
#
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

GPU="${1:-0}"
WAIT_MODE="${2:-}"
SLOT="${SLOT:-a}"
TAG="gpu${GPU}${SLOT}"

MEM_PER_RUN="${MEM_PER_RUN:-10000}"
POLL_SEC="${POLL_SEC:-60}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
MAX_PASSES="${MAX_PASSES:-2}"

OUT="${ROOT}/results/task5_ablation/5-2_attention"
LOGD="${OUT}/logs"
CLAIMD="${OUT}/claims"
mkdir -p "${LOGD}" "${CLAIMD}"

FILTER=()
[[ -n "${ARMS:-}" ]]     && FILTER+=(--arms ${ARMS})
[[ -n "${DATASETS:-}" ]] && FILTER+=(--datasets ${DATASETS})
[[ -n "${TASKS:-}" ]]    && FILTER+=(--tasks ${TASKS})
[[ -n "${SEEDS:-}" ]]    && FILTER+=(--seeds ${SEEDS})

wait_for_room() {
    [[ "${WAIT_MODE}" != "--wait" ]] && return 0
    local announced=0
    while true; do
        read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
            --format=csv,noheader,nounits -i "${GPU}" | tr -d ',')
        local free=$(( total - used ))
        if (( free >= MEM_PER_RUN )); then
            (( announced )) && echo "[${TAG}] memory freed up (free=${free}MiB), starting $(date +%T)"
            return 0
        fi
        if (( announced == 0 )); then
            echo "[${TAG}] waiting for GPU memory (free=${free}MiB < ${MEM_PER_RUN}MiB), retrying every ${POLL_SEC}s... $(date +%T)"
            announced=1
        fi
        sleep "${POLL_SEC}"
    done
}

STEP_ARG=(); [[ -n "${TOTAL_STEPS}" ]] && STEP_ARG=(--total_steps "${TOTAL_STEPS}")
echo "[${TAG}] started | MEM_PER_RUN=${MEM_PER_RUN}MiB steps=${TOTAL_STEPS:-per-dataset} $(date +%F_%T)"

for (( pass = 1; pass <= MAX_PASSES; pass++ )); do
    mapfile -t JOBS < <("${PYTHON}" "${HERE}/gen_jobs.py" --pending ${FILTER[@]+"${FILTER[@]}"})
    (( ${#JOBS[@]} == 0 )) && { echo "[${TAG}] pass ${pass}: no pending jobs"; break; }
    echo "[${TAG}] pass ${pass}/${MAX_PASSES}, ${#JOBS[@]} pending"

    for job in "${JOBS[@]}"; do
        read -r ds task seed arm <<< "${job}"
        jid="${ds}_${task}_seed${seed}_${arm}"
        rdir="${OUT}/runs/${ds}_${task}_seed${seed}/${arm}"

        [[ -f "${rdir}/metrics.json" ]] && { echo "[${TAG}] [skip] ${jid}"; continue; }
        mkdir "${CLAIMD}/${jid}" 2>/dev/null || { echo "[${TAG}] [taken] ${jid}"; continue; }

        wait_for_room

        echo "[${TAG}] [start] ${jid} $(date +%T)"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${HERE}/train_arm.py" \
            --arm "${arm}" --dataset "${ds}" --task "${task}" --seed "${seed}" \
            "${STEP_ARG[@]}" \
            > "${LOGD}/${jid}.log" 2>&1
        rc=$?
        if (( rc != 0 )); then
            echo "[${TAG}] [FAILED rc=${rc}] ${jid} -> see ${LOGD}/${jid}.log (claim released, retried next pass)"
            rmdir "${CLAIMD}/${jid}" 2>/dev/null
            sleep 10
        else
            echo "[${TAG}] [done] ${jid} $(date +%T)"
        fi
    done
done

echo "[${TAG}] finished $(date +%F_%T)"
