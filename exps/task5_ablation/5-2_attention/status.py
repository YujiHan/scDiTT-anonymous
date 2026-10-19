#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ARMS, CACHE_ROOT, DATASETS, SEEDS, TASKS, all_jobs,
                    cache_ready, is_done, run_dir)


def main():
    jobs = all_jobs()
    done = [j for j in jobs if is_done(*j)]
    todo = [j for j in jobs if not is_done(*j)]

    print(f"cache root: {CACHE_ROOT}")
    miss = [(ds, task, seed) for seed in SEEDS for ds in DATASETS for task in TASKS
            if not cache_ready(ds, task, seed)]
    n_cache = len(DATASETS) * len(TASKS) * len(SEEDS)
    print(f"cache (VAE+OT): {n_cache - len(miss)}/{n_cache} ready"
          + (f"; missing: {[f'{d}/{t}/s{s}' for d, t, s in miss]}" if miss else ""))

    print(f"\ntraining: {len(done)}/{len(jobs)} done")
    for arm in ARMS:
        d = sum(1 for j in done if j[3] == arm)
        print(f"  {arm:8s}: {d:2d}/{len(DATASETS)*len(TASKS)*len(SEEDS)}")

    if todo:
        print(f"\npending ({len(todo)}):")
        for ds, task, seed, arm in todo[:16]:
            print(f"  {ds}/{task}/seed{seed}/{arm}")
        if len(todo) > 16:
            print(f"  ... and {len(todo)-16} more")
    else:
        print("\nall training finished -> you can run `bash run_all.sh probe` and `bash run_all.sh agg`")


if __name__ == "__main__":
    main()
