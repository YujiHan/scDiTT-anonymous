#!/usr/bin/env bash
#
#
#
#
#
#   PER_GPU=2 EPOCHS=1000 bash launch_baselines.sh
set -uo pipefail
cd "$(dirname "$0")"

CONDA_SH="${CONDA_SH:?set this to your conda.sh path, e.g. ~/miniconda3/etc/profile.d/conda.sh}"
EPOCHS="${EPOCHS:-1000}"
PER_GPU="${PER_GPU:-3}"
GPUS="${GPUS:-0 1 2 3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASETS="${DATASETS:-GA MA ZB DR SC}"
declare -A EPOCHS_OF=( [MIOFlow]=200 )
HEAVY="scDiffEq MIOFlow"

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(scIMF PI-SDE scDiffEq PRESCIENT)
fi

LOGD=../logs/baseline_div
mkdir -p "$LOGD"
QUEUE="$LOGD/jobs.txt"
CURSOR="$LOGD/cursor"
: > "$QUEUE"
for ds in $DATASETS; do
  for task in interpolation extrapolation; do
    for m in "${METHODS[@]}"; do
      echo "$m $ds $task" >> "$QUEUE"
    done
  done
done
echo 0 > "$CURSOR"
N=$(wc -l < "$QUEUE")
echo "[launch] $N jobs | dataset order: $DATASETS | GPUs: $GPUS x $PER_GPU"
echo "[launch] epochs: default $EPOCHS; overrides: $(for k in "${!EPOCHS_OF[@]}"; do echo -n "$k=${EPOCHS_OF[$k]} "; done)"
echo "[launch] heavy methods, one at a time per GPU: $HEAVY"

is_heavy() { case " $HEAVY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

worker() {
  local gpu=$1 slot=$2
  while true; do
    local idx
    idx=$(flock "$CURSOR" bash -c "i=\$(cat '$CURSOR'); echo \$((i+1)) > '$CURSOR'; echo \$i")
    [ "$idx" -ge "$N" ] && break
    local line; line=$(sed -n "$((idx+1))p" "$QUEUE")
    local m ds task; read -r m ds task <<< "$line"
    local ep="${EPOCHS_OF[$m]:-$EPOCHS}"
    local log="$LOGD/${m}_${ds}_${task}.log"
    echo "[gpu$gpu.$slot] ($((idx+1))/$N) $m $ds/$task ep=$ep -> $log"
    (
      if is_heavy "$m"; then exec 9>"$LOGD/heavy_gpu${gpu}.lock"; flock 9; fi
      source "$CONDA_SH" && conda activate "$m" && \
      python baseline_diversity.py --method "$m" --ds "$ds" --task "$task" \
             --device "cuda:$gpu" --epochs "$ep" --train_seed "$TRAIN_SEED"
    ) > "$log" 2>&1
    local rc=$?
    echo "[gpu$gpu.$slot] ($((idx+1))/$N) $m $ds/$task exit=$rc"
    [ $rc -ne 0 ] && echo "$m $ds $task rc=$rc" >> "$LOGD/FAILED.txt"
  done
  echo "[gpu$gpu.$slot] drained"
}

for g in $GPUS; do
  for s in $(seq 1 "$PER_GPU"); do worker "$g" "$s" & done
done
wait
echo "[launch] all finished. failures:"
cat "$LOGD/FAILED.txt" 2>/dev/null || echo "  (none)"
