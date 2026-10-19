"""Shared configuration for the trajectory-construction ablation on real data.

Paper reference
---------------
Supplementary material, Ablation Studies -> "Effect of OT-Based Trajectory
Construction" (Table S3).

Four ways of turning two adjacent unpaired snapshots into training pairs are
compared on ZB and SC under the extrapolation split. Only the kernel varies:
the autoencoder, the latents, the architecture, the objective, the masking
curriculum, the optimiser and the evaluation are identical across the arms, and
every arm reads the *same* autoencoder checkpoint (see run_all.sh).

The companion experiment 5-1_OT runs the same comparison on synthetic branching
courses, where the true pairing is known and supplies an oracle (Table S4).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../exps/task5_ablation/5-4_OT_real
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_ROOT_REL = "results/task5_ablation/5-4_OT_real"
OUT_ROOT = REPO_ROOT / RESULTS_ROOT_REL

RUNS_DIR = OUT_ROOT / "runs"                    # runs/<DS>_<task>_seed<seed>/<arm>/
COUPLING_DIR = OUT_ROOT / "couplings"           # couplings/<DS>_<task>_seed<seed>/<arm>/
METRIC_DIR = OUT_ROOT / "metrics"
LOG_DIR = OUT_ROOT / "logs"

# The rows of Table S3, in the order the table prints them. "entropic" is scDiTT
# itself: its row reproduces the scDiTT entries of the extrapolation table.
ARMS = ["entropic", "hard_ot", "random", "knn"]
ARM_LABEL = {
    "entropic": "entropic OT (ours)",
    "hard_ot":  "hard OT (argmax)",
    "random":   "random",
    "knn":      "k-NN matching",
}

DATASETS = ["ZB", "SC"]
TASKS = ["extrapolation"]
SEEDS = [42, 43, 44, 45, 46]

# The same budget as every other run in the paper; see the Training Details
# appendix. Nothing here is tuned per arm.
TOTAL_STEPS = 40000
OT_TYPE = "CPU"

# The two columns Table S3 reports. The remaining three unified metrics are
# still written to metrics.json by the standard evaluator.
REPORT_KEYS = ["w2_avg", "sampled_pearson"]
LOWER_IS_BETTER = {"w2_avg": True, "sampled_pearson": False}


def run_tag(ds: str, task: str, seed: int) -> str:
    return f"{ds.upper()}_{task}_seed{int(seed)}"


def run_dir(ds: str, task: str, seed: int, arm: str) -> Path:
    return RUNS_DIR / run_tag(ds, task, seed) / arm


def coupling_dir(ds: str, task: str, seed: int, arm: str) -> Path:
    return COUPLING_DIR / run_tag(ds, task, seed) / arm


def is_done(ds: str, task: str, seed: int, arm: str) -> bool:
    return (run_dir(ds, task, seed, arm) / "metrics.json").exists()


def all_jobs() -> list[tuple[str, str, int, str]]:
    return [(ds, task, seed, arm)
            for seed in SEEDS
            for ds in DATASETS
            for task in TASKS
            for arm in ARMS]
