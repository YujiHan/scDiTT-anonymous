#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#
#
#   SCDITT_ARCHIVE_ROOT=... BASELINE_RESULTS_ROOT=... CONDA_SH=... bash run_all.sh
#
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs

: "${SCDITT_ARCHIVE_ROOT:?point this at the root of the scDiTT training artefacts}"
PY="${PYTHON:-python}"

"${PY}" verify_driver.py --ds ZB --task extrapolation --gpu 0

for pair in "ZB 0" "DR 1" "SC 2" "GA 3" "MA 0"; do
  set -- $pair
  "${PY}" compute_floor.py --ds "$1" --gpu "$2" > "../logs/floor_$1.log" 2>&1 &
done; wait

"${PY}" launch_scditt.py --gpus 0 1 2 3 --per_gpu 1

"${PY}" launch_scditt.py --gpus 0 1 2 3 --per_gpu 1 \
        --script compute_self_floor.py --tag selffloor

# --- 5. Baseline -----------------------------------------------------------
"${PY}" compute_baseline_metrics.py --gpu 0 --all_seeds 42 > ../logs/baseline_fid.log 2>&1
bash launch_scnode.sh                        # scNODE
bash launch_baselines.sh                     # scIMF / PI-SDE / scDiffEq / PRESCIENT
bash launch_mioflow.sh                       # MIOFlow

"${PY}" compute_metrics.py --gpu 0
for m in scNODE scIMF PI-SDE scDiffEq PRESCIENT MIOFlow; do
  "${PY}" compute_metrics.py --gpu 0 --method "${m}"
done

"${PY}" aggregate.py                         # -> metrics/aggregated.csv
"${PY}" rank_normalize.py                    # -> metrics/norm_compare.csv (rank)

"${PY}" make_fig4.py
"${PY}" report_cfg_sweep.py

echo "[run_all] done"
