#!/usr/bin/env bash
# ============================================================================
#
#
#
#
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"

OUT="${ROOT}/results/task5_ablation/5-2_attention"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"

GPUS="${GPUS:-0 1}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
MEM_PER_RUN="${MEM_PER_RUN:-10000}"
STAGGER_SEC="${STAGGER_SEC:-180}"
POLL_SEC="${POLL_SEC:-60}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
PROBE_SEEDS="${PROBE_SEEDS:-42 43 44}"
SKIP_PROBE="${SKIP_PROBE:-0}"
TMUX_SESSION="${TMUX_SESSION:-ablation-attn}"

DRY=0
for a in "$@"; do
    [[ "${a}" == "--dry" ]] && DRY=1
    if [[ "${a}" == "--tmux" ]]; then
        command -v tmux >/dev/null || { echo "tmux is not installed: apt-get update && apt-get install -y tmux"; exit 1; }
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            echo "tmux session '${TMUX_SESSION}' already exists -- the experiment may already be running."
            echo "  view:    tmux attach -t ${TMUX_SESSION}   (detach: Ctrl-b then d)"
            echo "  restart: tmux kill-session -t ${TMUX_SESSION}"
            exit 1
        fi
        REST="$(printf '%s ' "$@" | sed 's/--tmux//g')"
        tmux new-session -d -s "${TMUX_SESSION}" \
            "cd '${HERE}' && GPUS='${GPUS}' SLOTS_PER_GPU='${SLOTS_PER_GPU}' \
             MEM_PER_RUN='${MEM_PER_RUN}' STAGGER_SEC='${STAGGER_SEC}' \
             TOTAL_STEPS='${TOTAL_STEPS}' PROBE_SEEDS='${PROBE_SEEDS}' \
             bash run_experiment.sh ${REST}; \
             echo; echo '[experiment finished] press any key to close this session'; read -n 1"
        echo "started in the background in tmux session '${TMUX_SESSION}' (GPU [${GPUS}] x ${SLOTS_PER_GPU} slots)."
        echo "  live output: tmux attach -t ${TMUX_SESSION}    (detach: Ctrl-b then d)"
        echo "  progress:    bash ${HERE}/run_all.sh status"
        echo "  stop:        tmux kill-session -t ${TMUX_SESSION}"
        exit 0
    fi
done

STAMP="$(date +%Y%m%d-%H%M%S)"
MAIN_LOG="${LOGD}/run_experiment_${STAMP}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

say() { echo -e "\n\033[1m========== $* ==========\033[0m  $(date +%F\ %T)"; }
die() { echo -e "\033[31m[FATAL] $*\033[0m"; echo "log: ${MAIN_LOG}"; exit 1; }

T0=$(date +%s)
echo "5-2_attention one-command run | started $(date +%F\ %T)"
echo "main log: ${MAIN_LOG}"

say "stage 0/4 preflight"

[[ -x "${PYTHON}" ]] || die "python not found: ${PYTHON} (edit ${ROOT}/env.sh)"
"${PYTHON}" -c "import torch, anndata, numpy" || die "the python environment is missing dependencies"
echo "  python: ${PYTHON}"
echo "  torch : $("${PYTHON}" -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())')"

[[ -f "${ROOT}/Data/ZB/zebrafish_hvg_annotated.h5ad" ]] \
    || die "missing ZB data: ${ROOT}/Data/ZB/zebrafish_hvg_annotated.h5ad"
echo "  data  : ZB h5ad present"

NJOBS=$("${PYTHON}" -c "import sys; sys.path.insert(0, '${HERE}'); import common; print(len(common.all_jobs()))")
NEED_GB=$(( (NJOBS * 400 + 1023) / 1024 ))
FREE_GB=$(df -BG --output=avail "${ROOT}" | tail -1 | tr -dc '0-9')
echo "  disk  : ${FREE_GB} GB free (best.pt for ${NJOBS} runs needs about ${NEED_GB} GB)"
(( FREE_GB < NEED_GB + 5 )) && die "less than $(( NEED_GB + 5 )) GB of disk space left"

NGPU=$(echo ${GPUS} | wc -w)
NPAR=$(( NGPU * SLOTS_PER_GPU ))
NEED=$(( SLOTS_PER_GPU * MEM_PER_RUN ))
for g in ${GPUS}; do
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits -i "${g}" | tr -d ',')
    echo "  GPU ${g} : ${total} MiB total, ${used} MiB used"
    if (( total < NEED )); then
        echo -e "\033[33m  [warn] GPU ${g}: ${total} MiB < ${SLOTS_PER_GPU}×${MEM_PER_RUN}=${NEED} MiB"
        echo -e "         the second run would wait for memory forever (no OOM, but it degrades to one run per GPU)."
        echo -e "         if a single run really uses less than ${MEM_PER_RUN} MiB, lower MEM_PER_RUN and rerun.\033[0m"
    fi
done
echo "  plan  : GPU [${GPUS}] x ${SLOTS_PER_GPU} each -> ${NPAR} runs at a time"

NCPU=$(nproc)
if [[ -z "${SCDITT_NUM_WORKERS:-}" ]]; then
    SCDITT_NUM_WORKERS=$(( NCPU / (NPAR > 0 ? NPAR : 1) - 1 ))
    (( SCDITT_NUM_WORKERS < 1 )) && SCDITT_NUM_WORKERS=1
    (( SCDITT_NUM_WORKERS > 4 )) && SCDITT_NUM_WORKERS=4
fi
export SCDITT_NUM_WORKERS
echo "  CPU   : ${NCPU} cores -> ${SCDITT_NUM_WORKERS} dataloader workers per run"

say "stage 1/4 skipping the import (open-source version: the full arm is trained locally)"

say "stage 2/4 filling in missing VAE / OT caches (normally everything is skipped)"
PREP_GPU="${GPUS%% *}"
CUDA_VISIBLE_DEVICES="${PREP_GPU}" "${PYTHON}" "${HERE}/preprocess.py" --device cuda:0 \
    || die "preprocessing failed"

say "stage 3/4 training the ablation arms"
"${PYTHON}" "${HERE}/status.py"

if (( DRY == 1 )); then
    echo -e "\n[--dry] would start ${SLOTS_PER_GPU} workers on each of GPU [${GPUS}] (${NPAR} runs at a time), jobs:"
    "${PYTHON}" "${HERE}/gen_jobs.py" --pending | sed 's/^/  /'
    echo -e "\n[--dry] dry run finished, nothing trained. Drop --dry to run for real."
    exit 0
fi

PENDING=$("${PYTHON}" "${HERE}/gen_jobs.py" --pending | wc -l)
if (( PENDING == 0 )); then
    echo "  no pending jobs, skipping training."
else
    echo "  ${PENDING} pending -> GPU [${GPUS}] x ${SLOTS_PER_GPU} slots = ${NPAR} at a time"
    SLOT_NAMES=(a b c d e f)
    pids=()
    for g in ${GPUS}; do
        for (( s = 0; s < SLOTS_PER_GPU; s++ )); do
            SLOT="${SLOT_NAMES[s]}" MEM_PER_RUN="${MEM_PER_RUN}" POLL_SEC="${POLL_SEC}" \
                TOTAL_STEPS="${TOTAL_STEPS}" DELAY=$(( s * STAGGER_SEC )) \
                bash -c 'sleep "${DELAY}"; exec bash "$0" "$1" --wait' \
                "${HERE}/worker.sh" "${g}" 2>&1 | sed "s/^/  /" &
            pids+=($!)
        done
    done
    for p in "${pids[@]}"; do wait "${p}"; done
fi

"${PYTHON}" "${HERE}/status.py"
LEFT=$("${PYTHON}" "${HERE}/gen_jobs.py" --pending | wc -l)
if (( LEFT > 0 )); then
    echo -e "\033[33m[warn] ${LEFT} jobs are still unfinished. See ${LOGD}/<job>.log for the cause;"
    echo -e "       rerun this script after fixing them to resume (finished jobs are skipped). Aggregation continues anyway.\033[0m"
fi

if [[ "${SKIP_PROBE}" == "1" ]]; then
    say "stage 4/4 aggregation (SKIP_PROBE=1, chain B skipped)"
else
    say "stage 4/4 chain B mechanism diagnostics (GPU ${PREP_GPU})"
    CUDA_VISIBLE_DEVICES="${PREP_GPU}" "${PYTHON}" "${HERE}/probe_attention.py" \
        --arms full no_cell no_time --seeds ${PROBE_SEEDS} \
        2>&1 | tee "${LOGD}/probe_attention.log"
    CUDA_VISIBLE_DEVICES="${PREP_GPU}" "${PYTHON}" "${HERE}/probe_conditioning.py" \
        --seeds ${PROBE_SEEDS} 2>&1 | tee "${LOGD}/probe_conditioning.log"
    CUDA_VISIBLE_DEVICES="${PREP_GPU}" "${PYTHON}" "${HERE}/bench_efficiency.py" \
        2>&1 | tee "${LOGD}/bench_efficiency.log"
fi

say "chain A aggregation"
"${PYTHON}" "${HERE}/aggregate.py" 2>&1 | tee "${LOGD}/aggregate.log"

MIN=$(( ($(date +%s) - T0) / 60 ))
say "all finished (took ${MIN} minutes)"
cat <<EOF
Artefacts:
  main table / delta table / figure : ${OUT}/metrics/{main_table.csv,delta_table.csv,fig_main.png}
  chain B diagnostics              : ${OUT}/probes/{attention_maps.json,conditioning.json,efficiency.csv,fig_*.png}
  per-run ckpt                     : ${OUT}/runs/<ZB_task_seed>/<arm>/
  main log                         : ${MAIN_LOG}

How to read it: main_table.csv is the ablation table itself (mean +/- std over seeds, one row per arm);
delta_table.csv gives each arm's paired degradation against the full model. aggregate.py prints both,
plus the provenance of the imported full arm and its agreement with the main experiment.
Key point: if the W2 of no_time does not collapse, do not panic -- look at the CSI in probes (should be 0) and the lineage consistency,
that is the direct evidence that "time-attn is the only conditioning path".
EOF
