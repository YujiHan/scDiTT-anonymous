#!/usr/bin/env bash
cd "$(dirname "$0")"
source "${CONDA_SH:?set this to your conda.sh path, e.g. ~/miniconda3/etc/profile.d/conda.sh}"
conda activate scNODE
mkdir -p ../logs

i=0
for ds in ZB DR SC GA MA; do
  for task in interpolation extrapolation; do
    gpu=$((i % 4))
    nohup python baseline_scnode_diversity.py --ds "$ds" --task "$task" \
        --device "cuda:$gpu" > "../logs/scnode_${ds}_${task}.log" 2>&1 &
    echo "[launch] scNODE $ds/$task -> cuda:$gpu"
    i=$((i + 1))
    sleep 2
  done
done
wait
echo "[launch] all scNODE runs finished"
