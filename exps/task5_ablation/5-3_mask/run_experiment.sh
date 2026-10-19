#!/usr/bin/env bash
# ============================================================================
# 5-3_mask ablation -- run the whole experiment with one command (multi-GPU capable)
#
#   bash exps/task5_ablation/5-3_mask/run_experiment.sh --gpus "0 1 2 3 4 5"
#
# In the background (recommended, survives an ssh disconnect):
#   cd <repo_root>
#   nohup bash exps/task5_ablation/5-3_mask/run_experiment.sh --gpus "0 1 2 3 4 5" \
#         > /dev/null 2>&1 &
#   tail -f results/task5_ablation/5-3_mask/logs/run_experiment_latest.log
#
# It does 5 things in order (each is idempotent; after Ctrl-C just rerun this script
# and it continues from where it stopped):
#   0) preflight  check data / interpreter / free memory on each GPU (stops early rather than wasting a run)
#   1) selftest   wiring self-check (CPU)
#   2) stats      stage A: mask distribution diagnostics (CPU)
#   3) prep       stage B: 6 VAE + OT caches (ZB x 2 tasks x 3 seeds, shared by the four arms; multi-GPU)
#   4) train      stage C: 24 runs (4 arms x 2 tasks x 3 seeds; one worker per GPU, round-robin sharding)
#   5) agg        stage D: main table / matched-mismatched 2x2 / figures
#
# Options (all optional, defaults in parentheses):
#   --gpus "0 1 2..."  GPUs to use (0). One worker per GPU, serial within a GPU, parallel across GPUs.
#   --arms "..."       arms to run (mix random tail_only scatter_only)
#   --seeds "..."      random seeds (42 43 44 45 46)
#   --datasets "..."   datasets (ZB;  to add GA back: --datasets "ZB GA")
#   --tasks "..."      tasks (interpolation extrapolation)
#   --total-steps N    training steps; every arm runs the full budget (40000)
#   --batch-size N     override the batch (default: the 16 from configs/dataset/zb.yaml, needing about 17.5GB of GPU memory).
#                      **Lower it only when a new machine lacks memory**; all four arms use the same value, so the internal comparison stays clean.
#   --num-workers N    dataloader workers (default 4). Lower it when running multi-GPU on few CPU cores.
#   --min-free-mib N   minimum free memory required per GPU (18000; ZB batch=16 measures about 17.5GB)
#   --skip-prep        skip stage B (when the caches are ready)
#   --force            run even when GPU memory is short
#   --dry-run          only print the matrix that would be executed
#
# Runtime: about 40 minutes per run. On 6 GPUs, 24 runs take 4 waves, roughly 2.7 hours
# (plus about 0.5-1 hour of prep).
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/env.sh"
cd "${ROOT}"

GPUS="0"
ARMS="mix random tail_only scatter_only"
SEEDS="42 43 44 45 46"
DATASETS="ZB"
TASKS="interpolation extrapolation"
TOTAL_STEPS=40000
BATCH_SIZE=""
NUM_WORKERS=""
MIN_FREE_MIB=18000
SKIP_PREP=0
FORCE=0
DRY_RUN=0

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu)   GPUS="$2"; shift 2;;
    --arms)         ARMS="$2"; shift 2;;
    --seeds)        SEEDS="$2"; shift 2;;
    --datasets)     DATASETS="$2"; shift 2;;
    --tasks)        TASKS="$2"; shift 2;;
    --total-steps)  TOTAL_STEPS="$2"; shift 2;;
    --batch-size)   BATCH_SIZE="$2"; shift 2;;
    --num-workers)  NUM_WORKERS="$2"; shift 2;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2;;
    --skip-prep)    SKIP_PREP=1; shift;;
    --force)        FORCE=1; shift;;
    --dry-run)      DRY_RUN=1; shift;;
    -h|--help)      usage; exit 0;;
    *) echo "unknown argument: $1"; usage; exit 1;;
  esac
done

OUT="${ROOT}/results/task5_ablation/5-3_mask"
LOGD="${OUT}/logs"
mkdir -p "${LOGD}"
MASTER="${LOGD}/run_experiment_$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$(basename "${MASTER}")" "${LOGD}/run_experiment_latest.log"
exec > >(tee -a "${MASTER}") 2>&1

N_GPU=$(echo "${GPUS}" | wc -w)
N_ARMS=$(echo "${ARMS}" | wc -w)
N_CELLS=0
for d in ${DATASETS}; do for t in ${TASKS}; do for s in ${SEEDS}; do N_CELLS=$((N_CELLS+1)); done; done; done
N_RUNS=$((N_ARMS * N_CELLS))

N_DONE=0
for s_ in ${SEEDS}; do for d_ in ${DATASETS}; do for t_ in ${TASKS}; do for a_ in ${ARMS}; do
  [[ -f "${ROOT}/results/task5_ablation/5-3_mask/checkpoints/${a_}/${d_}_${t_}_seed${s_}/metrics.json" ]] \
    && N_DONE=$((N_DONE+1))
done; done; done; done
N_TODO=$((N_RUNS - N_DONE))
N_WAVES=$(( (N_TODO + N_GPU - 1) / N_GPU ))

hr() { printf '%.0s─' {1..76}; echo; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

on_interrupt() {
    hr
    say "!!  Interrupted. Every stage is idempotent -- just rerun this script to continue where it stopped:"
    say "    bash ${BASH_SOURCE[0]} --gpus \"${GPUS}\""
    say "log: ${MASTER}"
    exit 130
}
trap on_interrupt INT TERM

hr
echo "  5-3_mask ablation -- one-command run"
hr
say "GPUs (${N_GPU})   : ${GPUS}   -- one worker per GPU, serial within a GPU, parallel across GPUs"
say "arms (${N_ARMS})     : ${ARMS}"
say "datasets      : ${DATASETS}"
say "tasks         : ${TASKS}"
say "seeds        : ${SEEDS}"
say "cells         : ${N_CELLS}  (dataset x task x seed) -- the four arms share their VAE/OT cache"
say "training runs : ${N_RUNS} (${N_ARMS} arms x ${N_CELLS} cells); ${N_DONE} done -> **${N_TODO} remaining**, ${N_GPU} GPUs over ${N_WAVES} waves"
say "train steps   : ${TOTAL_STEPS} (no early stopping -- every arm runs the full budget)"
say "batch         : ${BATCH_SIZE:-configs/dataset/zb.yaml default (16)}"
say "num_workers   : ${NUM_WORKERS:-default (4)}"
say "est. train time: about $(( N_WAVES * 40 / 60 ))h $(( N_WAVES * 40 % 60 ))m"
say "artefacts     : ${OUT}"
say "main log      : ${MASTER}"
hr

# ---------------------------------------------------------------- 0) preflight
say "[0/5] preflight"
fail=0

[[ -x "${PYTHON}" ]] && say "  ✓ python: ${PYTHON}" \
  || { say "  x python not found: ${PYTHON}  -> edit that line in ${ROOT}/env.sh"; fail=1; }

for ds in ${DATASETS}; do
    h5=$("${PYTHON}" - "$ds" <<'PY' 2>/dev/null
import sys, yaml, pathlib
ds = sys.argv[1].lower()
print(yaml.safe_load(pathlib.Path(f"configs/dataset/{ds}.yaml").read_text())["dataset"]["h5ad_path"])
PY
)
    if [[ -n "${h5}" && -f "${h5}" ]]; then say "  ok data: ${h5}"
    else say "  x missing data: ${h5:-could not read configs/dataset/${ds,,}.yaml}"; fail=1; fi
done

MIN_FREE_GB=20
free_gb=$(df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -z "${free_gb}" ]]; then
    say "  . disk: cannot read free space (check skipped)"
elif [[ "${free_gb}" -lt "${MIN_FREE_GB}" ]]; then
    say "  x disk: ${ROOT} has only ${free_gb}GB left < the recommended ${MIN_FREE_GB}GB"
    say "    (checkpoints + prediction output for ${N_RUNS} runs need about 10-15GB)"
    say "    point the results root at a bigger disk, or force it with --force"
    [[ "${FORCE}" -eq 1 || "${DRY_RUN}" -eq 1 ]] || fail=1
else
    say "  ok disk: ${ROOT} has ${free_gb}GB free"
fi

N_VISIBLE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
say "  . ${N_VISIBLE} GPUs visible on this machine"
for g in ${GPUS}; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}" 2>/dev/null | tr -d ' ')
    if ! [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        say "  x GPU ${g}: cannot read memory -- does that index exist? (this machine has ${N_VISIBLE}, numbered 0..$((N_VISIBLE-1)))"
        fail=1
    elif [[ "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
        say "  x GPU ${g}: ${free_mib} MiB free < the required ${MIN_FREE_MIB} MiB (the GPU is busy)"
        [[ "${FORCE}" -eq 1 || "${DRY_RUN}" -eq 1 ]] || fail=1
    else
        say "  ok GPU ${g}: ${free_mib} MiB free"
    fi
done

if [[ "${fail}" -ne 0 && "${DRY_RUN}" -ne 1 ]]; then
    hr; say "preflight failed, stopped (no compute wasted)."
    say "to push on despite low GPU memory: add --force (may OOM); for smaller GPUs tune --min-free-mib"
    exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    hr; say "--dry-run: ${N_RUNS} runs (${N_DONE} done, ${N_TODO} remaining) spread over ${N_GPU} GPUs -"
    gpus=(${GPUS}); i=0
    for s in ${SEEDS}; do for d in ${DATASETS}; do for t in ${TASKS}; do for a in ${ARMS}; do
        g="${gpus[$(( i % N_GPU ))]}"
        mark=""; [[ -f "${OUT}/checkpoints/${a}/${d}_${t}_seed${s}/metrics.json" ]] && mark="  already done, will skip"
        printf "    [%2d/%2d] gpu%s  %-13s %s/%s/seed%s%s\n" $((i+1)) "${N_RUNS}" "${g}" "${a}" "${d}" "${t}" "${s}" "${mark}"
        i=$((i+1))
    done; done; done; done
    exit 0
fi

T_ALL=$(date +%s)
RA="${HERE}/run_all.sh"
export GPUS ARMS SEEDS DATASETS TASKS TOTAL_STEPS BATCH_SIZE NUM_WORKERS

stage() {
    local no="$1" name="$2"; shift 2
    hr; say "[${no}/5] ${name}"
    local t0 t1 rc
    t0=$(date +%s); "$@"; rc=$?; t1=$(date +%s)
    say "[${no}/5] ${name} finished (took $(( (t1-t0)/60 ))m $(( (t1-t0)%60 ))s, rc=${rc})"
    return ${rc}
}

# ---------------------------------------------------------------- 1) selftest
stage 1 "wiring self-check (CPU)" bash "${RA}" selftest \
  || { hr; say "x self-check failed -- fix it before running (no compute wasted)."; exit 1; }

# ---------------------------------------------------------------- 2) stats
stage 2 "stage A: mask distribution diagnostics (CPU)" bash "${RA}" stats

# ---------------------------------------------------------------- 3) prep
if [[ "${SKIP_PREP}" -eq 1 ]]; then
    hr; say "[3/5] stage B: skipped (--skip-prep)"
else
    stage 3 "stage B: ${N_CELLS} cells, VAE + OT cache (${N_GPU} GPUs in parallel, shared by the four arms)" bash "${RA}" prep
fi

say "re-checking ${N_CELLS} caches ..."
if ! "${PYTHON}" - <<PY
import sys
sys.path.insert(0, ".")
from pathlib import Path
from src.data.splits import cache_paths, coupling_complete, vae_cache_valid
root, bad = "results/task5_ablation/5-3_mask", []
for ds in "${DATASETS}".split():
    for task in "${TASKS}".split():
        for seed in [int(s) for s in "${SEEDS}".split()]:
            p = cache_paths(ds, task, None, seed, ot_type="CPU", results_root=root)
            if not Path(p["vae_ckpt_path"]).exists():  bad.append(f"{ds}/{task}/seed{seed}: missing VAE")
            elif not Path(p["latents_path"]).exists(): bad.append(f"{ds}/{task}/seed{seed}: missing latents")
            elif not vae_cache_valid(ds, task, None, seed, results_root=root):
                bad.append(f"{ds}/{task}/seed{seed}: the VAE train_tps do not match this split")
            elif not coupling_complete(ds, task, None, seed, ot_type="CPU", results_root=root):
                bad.append(f"{ds}/{task}/seed{seed}: OT cache incomplete")
for b in bad:
    print("  ✗", b)
print("  all ready" if not bad else f"  {len(bad)} missing/incomplete")
sys.exit(1 if bad else 0)
PY
then
    hr
    say "x cache incomplete -- training would fail while loading data. Check ${LOGD}/prep_*.log first,"
    say "  then rerun this script (the parts already done are skipped)."
    exit 1
fi

# ---------------------------------------------------------------- 4) train
hr
say "[4/5] stage C: ${N_TODO} runs left to train, ${N_GPU} GPUs in parallel (about ${N_WAVES} waves x 40 min)"
say "        progress: tail -f ${MASTER}   (each line carries [i/${N_RUNS}] and the gpu index)"
T0=$(date +%s)
bash "${RA}" train
T1=$(date +%s)
say "[4/5] training finished (took $(( (T1-T0)/3600 ))h $(( ((T1-T0)%3600)/60 ))m)"

# ---------------------------------------------------------------- 5) agg
stage 5 "stage D: aggregate main table / 2x2 / figures" bash "${RA}" agg

DONE=$(find "${OUT}/checkpoints" -name metrics.json 2>/dev/null | wc -l)
EL=$(( $(date +%s) - T_ALL ))
hr
say "all finished. ${DONE}/${N_RUNS} runs completed, total time $(( EL/3600 ))h $(( (EL%3600)/60 ))m."
if [[ "${DONE}" -lt "${N_RUNS}" ]]; then
    say "!!  $(( N_RUNS - DONE )) runs did not finish -- check ${LOGD}/train_*.log,"
    say "    then rerun this script (finished runs are skipped)."
fi
hr
echo "results:"
echo "  main table (with the matched/mismatched 2x2 and the mix calibration): ${OUT}/metrics/main_table.md"
echo "  stage A diagnostics (mask distribution, the mechanism behind downstream differences): ${OUT}/metrics/mask_stats.md"
echo "  figures: ${OUT}/metrics/fig_main.png  /  fig_mask_stats.png"
echo "  per-run detail: ${OUT}/metrics/runs_long.csv"
hr
