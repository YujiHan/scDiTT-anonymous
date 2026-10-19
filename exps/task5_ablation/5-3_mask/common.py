from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent                 # .../exps/task5_ablation/5-3_mask
REPO_ROOT = HERE.parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"

RESULTS_ROOT_REL = "results/task5_ablation/5-3_mask"
OUT_ROOT = REPO_ROOT / RESULTS_ROOT_REL

CKPT_DIR = OUT_ROOT / "checkpoints"       # checkpoints/<arm>/<DS>_<task>_seed<seed>/
METRIC_DIR = OUT_ROOT / "metrics"
LOG_DIR = OUT_ROOT / "logs"

REFERENCE_CSV_DIR = REPO_ROOT / "results" / "reference_csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ["ZB"]
TASKS = ["interpolation", "extrapolation"]
SEEDS = [42, 43, 44, 45, 46]
ARMS = ["mix", "random", "tail_only", "scatter_only"]

OT_TYPE = "CPU"
TOTAL_STEPS = 40000

METRIC_KEYS = ["w2_avg", "mmd_avg", "pearson_mean", "sampled_pearson", "pg_w1_mean"]
PRIMARY_METRIC = "w2_avg"
LOWER_IS_BETTER = {"w2_avg": True, "mmd_avg": True, "pearson_mean": False,
                   "sampled_pearson": False, "pg_w1_mean": True}


def run_dir(arm: str, ds: str, task: str, seed: int) -> Path:
    return CKPT_DIR / arm / f"{ds.upper()}_{task}_seed{int(seed)}"


def cells(datasets: list[str] | None = None,
          tasks: list[str] | None = None,
          seeds: list[int] | None = None) -> Iterator[tuple[str, str, int]]:
    for ds in (datasets or DATASETS):
        for task in (tasks or TASKS):
            for seed in (seeds or SEEDS):
                yield ds.upper(), task, int(seed)


def get_split(ds: str, task: str) -> tuple[list[int], list[int]]:
    from src.data.splits import get_tp_split
    return get_tp_split(ds.upper(), task)


def get_T(ds: str) -> int:
    from src.data.splits import ALL_TPS
    return len(ALL_TPS[ds.upper()])
