#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ARMS, DATASETS, SEEDS, TASKS, all_jobs, is_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, choices=SEEDS)
    ap.add_argument("--pending", action="store_true", help="list only unfinished jobs")
    args = ap.parse_args()

    for ds, task, seed, arm in all_jobs():
        if arm not in args.arms or ds not in args.datasets \
                or task not in args.tasks or seed not in args.seeds:
            continue
        if args.pending and is_done(ds, task, seed, arm):
            continue
        print(f"{ds} {task} {seed} {arm}")


if __name__ == "__main__":
    main()
