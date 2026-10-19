#!/usr/bin/env bash
# ============================================================================
#
#   bash run_all.sh all         # prep -> train -> agg
#
#   SEEDS="42 43 44 45 46"
#   TASKS="interpolation extrapolation"
#
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"
cd "${ROOT}"

GPUS="${GPUS:-${GPU:-0}}"
ARMS="${ARMS:-mix random tail_only scatter_only}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DATASETS="${DATASETS:-ZB}"
TASKS="${TASKS:-interpolation extrapolation}"
TOTAL_STEPS="${TOTAL_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
OT_TYPE="CPU"

RESULTS_ROOT="results/task5_ablation/5-3_mask"
OUT="${ROOT}/${RESULTS_ROOT}"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}" "${OUT}/metrics"

run_sharded() {
    local fn="$1"; shift
    local jobs=("$@")
    local gpus=(${GPUS})
    local n=${#gpus[@]}
    local total=${#jobs[@]}
    local pids=()
    local w
    for ((w = 0; w < n && w < total; w++)); do
        (
            local gpu="${gpus[$w]}"
            local j
            for ((j = w; j < total; j += n)); do
                IFS=':' read -r f1 f2 f3 f4 <<< "${jobs[$j]}"
                "${fn}" "${gpu}" "$((j + 1))/${total}" "${f1}" "${f2}" "${f3}" "${f4}"
            done
        ) &
        pids+=($!)
    done
    wait "${pids[@]}" 2>/dev/null
}

stats() {
    echo "======== stage A: mask distribution diagnostics (CPU) $(date +%F_%T) ========"
    "${PYTHON}" "${HERE}/mask_stats.py" 2>&1 | tee "${LOGD}/mask_stats.log"
}

prep_cell() {
    local gpu="$1" tag="$2" ds="$3" task="$4" seed="$5"

    if "${PYTHON}" - "${ds}" "${task}" "${seed}" "${RESULTS_ROOT}" <<'PY'
import sys
sys.path.insert(0, ".")
from src.data.splits import vae_cache_valid
ds, task, seed, root = sys.argv[1:5]
sys.exit(0 if vae_cache_valid(ds, task, None, int(seed), results_root=root) else 1)
PY
    then
        echo "${tag} [gpu${gpu}][skip vae] ${ds}/${task}/seed${seed}"
    else
        echo "${tag} [gpu${gpu}][vae ] ${ds}/${task}/seed${seed} $(date +%T)"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" run/preprocess_vae.py \
            --dataset "${ds}" --task "${task}" --seed "${seed}" \
            --device cuda:0 --results_root "${RESULTS_ROOT}" \
            > "${LOGD}/prep_vae_${ds}_${task}_seed${seed}.log" 2>&1 \
          || { echo "${tag} [gpu${gpu}][FAIL vae] ${ds}/${task}/seed${seed} -- see ${LOGD}/prep_vae_${ds}_${task}_seed${seed}.log"; return 1; }
    fi

    if "${PYTHON}" - "${ds}" "${task}" "${seed}" "${OT_TYPE}" "${RESULTS_ROOT}" <<'PY'
import sys
sys.path.insert(0, ".")
from src.data.splits import coupling_complete
ds, task, seed, ot, root = sys.argv[1:6]
sys.exit(0 if coupling_complete(ds, task, None, int(seed),
                                ot_type=ot, results_root=root) else 1)
PY
    then
        echo "${tag} [gpu${gpu}][skip ot ] ${ds}/${task}/seed${seed}"
    else
        echo "${tag} [gpu${gpu}][ot  ] ${ds}/${task}/seed${seed} $(date +%T)"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" run/preprocess_coupling.py \
            --dataset "${ds}" --task "${task}" --seed "${seed}" \
            --ot_type "${OT_TYPE}" --device cuda:0 \
            --results_root "${RESULTS_ROOT}" \
            > "${LOGD}/prep_ot_${ds}_${task}_seed${seed}.log" 2>&1 \
          || { echo "${tag} [gpu${gpu}][FAIL ot ] ${ds}/${task}/seed${seed} -- see ${LOGD}/prep_ot_${ds}_${task}_seed${seed}.log"; return 1; }
    fi
    echo "${tag} [gpu${gpu}][ok  ] ${ds}/${task}/seed${seed} cache ready"
}

prep() {
    local jobs=()
    for ds in ${DATASETS}; do for task in ${TASKS}; do for seed in ${SEEDS}; do
        jobs+=("${ds}:${task}:${seed}")
    done; done; done
    echo "======== stage B: ${#jobs[@]} cells, VAE + OT cache (GPUS='${GPUS}') $(date +%F_%T) ========"
    run_sharded prep_cell "${jobs[@]}"
    echo "[prep] done. caches -> ${RESULTS_ROOT}/{vae_checkpoints,coupling_cache}/"
}

train_one() {
    local gpu="$1" tag="$2" arm="$3" ds="$4" task="$5" seed="$6"
    local d="${OUT}/checkpoints/${arm}/${ds}_${task}_seed${seed}"
    local log="${LOGD}/train_${arm}_${ds}_${task}_seed${seed}.log"

    if [[ -f "${d}/metrics.json" ]]; then
        echo "${tag} [gpu${gpu}][skip ] ${arm} ${ds}/${task}/seed${seed} (already done)"; return 0
    fi
    local t0; t0=$(date +%s)
    echo "${tag} [gpu${gpu}][train] ${arm} ${ds}/${task}/seed${seed} steps=${TOTAL_STEPS} $(date +%T)"
    local xargs=()
    [[ -n "${BATCH_SIZE}"  ]] && xargs+=(--batch_size "${BATCH_SIZE}")
    [[ -n "${NUM_WORKERS}" ]] && xargs+=(--num_workers "${NUM_WORKERS}")
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${HERE}/train_arm.py" \
        --arm "${arm}" --dataset "${ds}" --task "${task}" --seed "${seed}" \
        --total_steps "${TOTAL_STEPS}" "${xargs[@]+"${xargs[@]}"}" > "${log}" 2>&1
    local rc=$?
    local el=$(( $(date +%s) - t0 ))
    if [[ ${rc} -eq 0 && -f "${d}/metrics.json" ]]; then
        local w2
        w2=$("${PYTHON}" -c "import json;print(f\"{json.load(open('${d}/metrics.json'))['avg']['w2_avg']:.2f}\")" 2>/dev/null || echo "?")
        echo "${tag} [gpu${gpu}][ok   ] ${arm} ${ds}/${task}/seed${seed} w2_avg=${w2} ($(( el/60 ))m$(( el%60 ))s)"
    else
        echo "${tag} [gpu${gpu}][FAIL ] ${arm} ${ds}/${task}/seed${seed} -- see ${log}"
    fi
}

#
train() {
    local jobs=() done_n=0
    for seed in ${SEEDS}; do for ds in ${DATASETS}; do for task in ${TASKS}; do
        for arm in ${ARMS}; do
            if [[ -f "${OUT}/checkpoints/${arm}/${ds}_${task}_seed${seed}/metrics.json" ]]; then
                done_n=$((done_n + 1))
                echo "[skip ] ${arm} ${ds}/${task}/seed${seed} (already done)"
            else
                jobs+=("${arm}:${ds}:${task}:${seed}")
            fi
        done
    done; done; done

    local ngpu; ngpu=$(echo "${GPUS}" | wc -w)
    if [[ ${#jobs[@]} -eq 0 ]]; then
        echo "======== stage C: all ${done_n} runs already finished, nothing to train ========"
        return 0
    fi
    echo "======== stage C: ${#jobs[@]} runs to train (${done_n} already done), "\
"${ngpu} GPUs in parallel, about $(( (${#jobs[@]} + ngpu - 1) / ngpu )) waves $(date +%F_%T) ========"
    run_sharded train_one "${jobs[@]}"
    echo "[train] training matrix finished $(date +%F_%T)"
}

agg() { "${PYTHON}" "${HERE}/aggregate.py"; }

case "${1:-all}" in
    selftest)  "${PYTHON}" "${HERE}/selftest.py" ;;
    stats)     stats ;;
    prep)      prep ;;
    train)     train ;;
    agg)       agg ;;
    all)       prep && train && agg ;;
    *) echo "usage: bash run_all.sh {selftest|stats|prep|train|agg|all}"; exit 1 ;;
esac
