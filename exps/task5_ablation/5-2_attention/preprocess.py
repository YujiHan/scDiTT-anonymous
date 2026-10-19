#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CACHE_ROOT, DATASETS, MAX_CELLS, OT_TYPE, REPO_ROOT, SEEDS,
                    TASKS, cache_ready, cfg_paths)

PYTHON = sys.executable


def _run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def prep_one(ds: str, task: str, seed: int, cache_root: str,
             device: str = "cuda:0") -> bool:
    p = cfg_paths(ds, task, seed, cache_root)

    from src.data.splits import vae_cache_valid
    if vae_cache_valid(ds.upper(), task, None, int(seed),
                       vae_ckpt_path=p["vae_ckpt_path"],
                       latents_path=p["latents_path"]):
        print(f"  [skip] VAE already exists: {p['vae_ckpt_path']}")
    else:
        rc = _run([PYTHON, "run/preprocess_vae.py", "--dataset", ds,
                   "--task", task, "--seed", str(seed), "--device", device,
                   "--results_root", cache_root])
        if rc != 0:
            print(f"  [FAIL] preprocess_vae {ds}/{task}/seed{seed}")
            return False

    from src.data.splits import coupling_complete
    if coupling_complete(ds, task, None, seed, ot_type=OT_TYPE,
                         coupling_cache_dir=p["coupling_cache_dir"],
                         latents_path=p["latents_path"]):
        print(f"  [skip] OT cache already complete: {p['coupling_cache_dir']}")
    else:
        rc = _run([PYTHON, "run/preprocess_coupling.py", "--dataset", ds,
                   "--task", task, "--seed", str(seed), "--ot_type", OT_TYPE,
                   "--max_cells", str(MAX_CELLS), "--device", device,
                   "--results_root", cache_root])
        if rc != 0:
            print(f"  [FAIL] preprocess_coupling {ds}/{task}/seed{seed}")
            return False

    ok = cache_ready(ds, task, seed, cache_root)
    print(f"  [{'OK' if ok else 'INCOMPLETE'}] {ds}/{task}/seed{seed}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"=== stage A: VAE + OT cache -> {args.cache_root} ===")
    t0 = time.time()
    failed = []
    for seed in args.seeds:
        for ds in args.datasets:
            for task in args.tasks:
                print(f"--- {ds}/{task}/seed{seed} ---")
                if not prep_one(ds.upper(), task, int(seed),
                                args.cache_root, args.device):
                    failed.append(f"{ds}/{task}/seed{seed}")

    print(f"=== stage A finished in {(time.time()-t0)/60:.1f} min ===")
    if failed:
        print(f"[FAILED] {len(failed)} groups not ready: {failed}")
        sys.exit(1)
    print("all 12 caches are ready; training can start.")


if __name__ == "__main__":
    main()
