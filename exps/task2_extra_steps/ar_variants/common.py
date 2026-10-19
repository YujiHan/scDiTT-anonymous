"""Shared configuration for the two autoregressive variants of scDiTT.

Paper reference
---------------
Supplementary material, "Autoregressive Variants of scDiTT" (Appendix J).

The comparison is run only on the extrapolation-horizon sweep, the one task that
reaches a self-feeding depth worth measuring: under interpolation every held-out
stage is immediately preceded by an observed one, and the fixed extrapolation
split reaches a depth of three at most.

Each arm shares the autoencoder and the optimal-transport couplings of the
corresponding scDiTT run, so `SCDITT_CACHE_ROOT` should point at the results root
of task2_extra_steps. Without it the caches are rebuilt from scratch under this
experiment's own results root, which is correct but wasteful.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../exps/task2_extra_steps/ar_variants
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_ROOT_REL = "results/task2_extra_steps/ar_variants"
OUT_ROOT = REPO_ROOT / RESULTS_ROOT_REL

RUNS_DIR = OUT_ROOT / "runs"                    # runs/<DS>_ex<l>_seed<seed>/<arm>/
METRIC_DIR = OUT_ROOT / "metrics"
LOG_DIR = OUT_ROOT / "logs"

# Where the scDiTT horizon-sweep VAE/OT caches live. Set it to reuse them.
CACHE_ROOT = os.environ.get("SCDITT_CACHE_ROOT", "results/task2_extra_steps")

ARMS = ["prefix", "markov"]
ARM_LABEL = {"prefix": "AR-prefix", "markov": "AR-Markov"}

DATASETS = ["ZB", "DR", "SC", "GA", "MA"]
HORIZONS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

TASK = "extrapolation_steps"
TOTAL_STEPS = 40000
OT_TYPE = "CPU"

# The horizon tables report W2 at the furthest withheld stage, which the standard
# evaluator writes to metrics_lasttp.json under the "lasttp" scheme.
METRICS_FILE = "metrics_lasttp.json"
REPORT_KEY = "w2_avg"


def run_tag(ds: str, horizon: int, seed: int) -> str:
    return f"{ds.upper()}_ex{int(horizon)}_seed{int(seed)}"


def run_dir(ds: str, horizon: int, seed: int, arm: str) -> Path:
    return RUNS_DIR / run_tag(ds, horizon, seed) / arm


def is_done(ds: str, horizon: int, seed: int, arm: str) -> bool:
    return (run_dir(ds, horizon, seed, arm) / METRICS_FILE).exists()


def all_jobs() -> list[tuple[str, int, int, str]]:
    return [(ds, h, seed, arm)
            for seed in SEEDS
            for ds in DATASETS
            for h in HORIZONS
            for arm in ARMS]
