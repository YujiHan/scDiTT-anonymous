#!/usr/bin/env bash
#
#
set -uo pipefail
cd "$(dirname "$0")"

CONDA_SH="${CONDA_SH:?set this to your conda.sh path, e.g. ~/miniconda3/etc/profile.d/conda.sh}"
EPOCHS="${EPOCHS:-200}"
GPUS="${GPUS:-0 1 2 3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASETS="${DATASETS:-GA MA ZB DR SC}"

LOGD=../logs/baseline_div
mkdir -p "$LOGD"
QUEUE="$LOGD/jobs_mioflow.txt"
CURSOR="$LOGD/cursor_mioflow"
: > "$QUEUE"
for ds in $DATASETS; do
  for task in interpolation extrapolation; do
    echo "MIOFlow $ds $task" >> "$QUEUE"
  done
done
echo 0 > "$CURSOR"
N=$(wc -l < "$QUEUE")
echo "[mioflow] $N jobs | epochs=$EPOCHS | dataset order: $DATASETS"

worker() {
  local gpu=$1
  while true; do
    local idx
    idx=$(flock "$CURSOR" bash -c "i=\$(cat '$CURSOR'); echo \$((i+1)) > '$CURSOR'; echo \$i")
    [ "$idx" -ge "$N" ] && break
    local line; line=$(sed -n "$((idx+1))p" "$QUEUE")
    local m ds task; read -r m ds task <<< "$line"
    local log="$LOGD/${m}_${ds}_${task}.log"
    echo "[gpu$gpu] ($((idx+1))/$N) $m $ds/$task ep=$EPOCHS -> $log"
    (
      exec 9>"$LOGD/heavy_gpu${gpu}.lock"; flock 9
      source "$CONDA_SH" && conda activate MIOFlow && \
      python baseline_diversity.py --method MIOFlow --ds "$ds" --task "$task" \
             --device "cuda:$gpu" --epochs "$EPOCHS" --train_seed "$TRAIN_SEED"
    ) > "$log" 2>&1
    local rc=$?
    echo "[gpu$gpu] ($((idx+1))/$N) $m $ds/$task exit=$rc"
    [ $rc -ne 0 ] && echo "MIOFlow $ds $task rc=$rc" >> "$LOGD/FAILED.txt"
  done
  echo "[gpu$gpu] drained"
}

for g in $GPUS; do worker "$g" & done
wait
echo "[mioflow] finished. failures:"
grep MIOFlow "$LOGD/FAILED.txt" 2>/dev/null || echo "  (none)"
