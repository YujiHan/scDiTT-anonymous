#!/usr/bin/env python
"""Train and evaluate one autoregressive arm at one (dataset, horizon, seed).

Paper reference
---------------
Supplementary material, "Autoregressive Variants of scDiTT" (Appendix J).

The arm is produced entirely by patching two names before the standard pipeline
runs (see ar_patch.py): the dataset's conditioning pattern and the sampling loop.
Everything else -- architecture, objective, optimiser, schedule, per-dataset N and
learning rate, the 40k-step budget, the autoencoder and the couplings -- is the
scDiTT configuration, unmodified.

Usage:
    PYTHONPATH=. python exps/task2_extra_steps/ar_variants/train_arm.py \
        --arm prefix --dataset ZB --horizon 3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402  (inserts REPO_ROOT into sys.path)

os.chdir(C.REPO_ROOT)


def build_cfg(ds: str, horizon: int, seed: int, arm: str,
              total_steps: int, cache_root: str) -> dict:
    from src.data.splits import cache_paths
    from src.utils.config import build_config, get_argparser

    argv = ["--dataset", ds.upper(), "--task", C.TASK,
            "--ex_steps", str(horizon), "--seed", str(seed),
            "--results_root", C.RESULTS_ROOT_REL,
            "--output_dir", str(C.run_dir(ds, horizon, seed, arm)),
            "--ot_type", C.OT_TYPE]
    args, extras = get_argparser().parse_known_args(argv)
    cfg = build_config(args, extras=extras)
    cfg["training"]["total_steps"] = int(total_steps)

    # Appendix J: "the same autoencoder checkpoint and optimal-transport
    # couplings as the corresponding scDiTT run". Point at the scDiTT cache when
    # one is available so the two arms and scDiTT share them exactly.
    if cache_root:
        p = cache_paths(ds.upper(), C.TASK, horizon, seed,
                        ot_type=C.OT_TYPE, results_root=cache_root)
        if Path(p["latents_path"]).is_file() and Path(p["vae_ckpt_path"]).is_file():
            cfg["dataset"]["vae_ckpt_path"] = p["vae_ckpt_path"]
            cfg["dataset"]["latents_path"] = p["latents_path"]
            cfg["dataset"]["coupling_cache_dir"] = p["coupling_cache_dir"]
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=C.ARMS)
    ap.add_argument("--dataset", required=True,
                    choices=C.DATASETS + [d.lower() for d in C.DATASETS])
    ap.add_argument("--horizon", type=int, required=True, choices=C.HORIZONS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total_steps", type=int, default=C.TOTAL_STEPS)
    ap.add_argument("--cache_root", default=C.CACHE_ROOT,
                    help="results root of the scDiTT horizon sweep, whose VAE/OT "
                         "caches this arm reuses; '' to build its own")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds, h, seed, arm = args.dataset.upper(), args.horizon, args.seed, args.arm
    out = C.run_dir(ds, h, seed, arm)

    if C.is_done(ds, h, seed, arm) and not args.force:
        print(f"[skip] {ds}/ex{h}/seed{seed}/{arm} already finished "
              f"({out / C.METRICS_FILE})")
        return 0

    # Patch BEFORE importing the training loop, so the loop picks up the AR
    # dataset and the AR sampler.
    from ar_patch import patch_all
    patch_all(arm)

    from src.training.train import train_model
    from src.utils.config import save_config

    cfg = build_cfg(ds, h, seed, arm, args.total_steps, args.cache_root)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    print(f"=== [AR-{arm}] {ds}/ex{h}/seed{seed} steps={cfg['training']['total_steps']} "
          f"N={cfg['data']['N']} lr={cfg['training']['peak_lr']} ===")
    print(f"    latents  : {cfg['dataset']['latents_path']}")
    print(f"    coupling : {cfg['dataset']['coupling_cache_dir']}")
    print(f"    output   : {out}")

    t0 = time.time()
    train_model(cfg)
    wall = time.time() - t0

    (out / "arm.json").write_text(json.dumps({
        "arm": arm, "variant": f"scDiTT-AR-{arm}",
        "dataset": ds, "task": C.TASK, "horizon": h, "seed": seed,
        "total_steps": cfg["training"]["total_steps"],
        "N": cfg["data"]["N"], "peak_lr": cfg["training"]["peak_lr"],
        "batch_size": cfg["training"]["batch_size"],
        "nfe_per_draw": 2 * cfg["inference"]["num_steps"] * h,
        "latents_path": cfg["dataset"]["latents_path"],
        "coupling_cache_dir": cfg["dataset"]["coupling_cache_dir"],
        "wall_sec": wall,
    }, indent=2))

    ok = (out / C.METRICS_FILE).exists()
    print(f"=== done AR-{arm} {ds}/ex{h}/seed{seed} wall={wall / 60:.1f}min "
          f"{C.METRICS_FILE}={'OK' if ok else 'MISSING'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
