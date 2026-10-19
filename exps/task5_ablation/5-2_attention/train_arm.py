#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ABLATION_STEPS, ARMS, CACHE_ROOT, DATASETS, REPO_ROOT,
                    SEEDS, TASKS, build_cfg, cache_ready, cfg_paths, is_done,
                    run_dir)

os.chdir(REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--total_steps", type=int, default=None,
                    help=f"default None = every arm uses the shared {ABLATION_STEPS}-step budget "
                         f"(common.ABLATION_STEPS); an explicit value overrides it")
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="rerun even if metrics.json already exists")
    args = ap.parse_args()

    ds, task, seed, arm = args.dataset.upper(), args.task, args.seed, args.arm
    steps = args.total_steps if args.total_steps is not None else ABLATION_STEPS
    out = run_dir(ds, task, seed, arm)

    if is_done(ds, task, seed, arm) and not args.force:
        print(f"[skip] {ds}/{task}/seed{seed}/{arm} already finished ({out/'metrics.json'})")
        return

    if not cache_ready(ds, task, seed, args.cache_root):
        p = cfg_paths(ds, task, seed, args.cache_root)
        print(f"[error] {ds}/{task}/seed{seed} has an incomplete VAE/OT cache; run preprocess.sh first\n"
              f"        vae      : {p['vae_ckpt_path']}\n"
              f"        latents  : {p['latents_path']}\n"
              f"        coupling : {p['coupling_cache_dir']}")
        sys.exit(1)

    from blocks_ablation import patch_dit
    patch_dit(arm)

    from src.training.train import train_model
    from src.utils.config import save_config

    cfg = build_cfg(ds, task, seed, arm, total_steps=steps,
                    cache_root=args.cache_root)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    print(f"=== [5-2_attention] arm={arm} {ds}/{task}/seed{seed} "
          f"steps={steps} N={cfg['data']['N']} "
          f"lr={cfg['training']['peak_lr']} bs={cfg['training']['batch_size']} ===")
    print(f"    latents  : {cfg['dataset']['latents_path']}")
    print(f"    coupling : {cfg['dataset']['coupling_cache_dir']}")
    print(f"    output   : {out}")

    t0 = time.time()
    train_model(cfg)
    wall = time.time() - t0

    from src.model.dit import SCDiT
    from src.model.model_utils import count_params
    m = SCDiT(D_latent=cfg["model"]["D_latent"], D_model=cfg["model"]["D_model"],
              num_blocks=cfg["model"]["num_blocks"],
              num_heads=cfg["model"]["num_heads"],
              T_max=max(32, cfg["dataset"]["T"]))
    (out / "arm.json").write_text(json.dumps({
        "arm": arm, "dataset": ds, "task": task, "seed": seed,
        "params_total": count_params(m)["total"],
        "total_steps": steps,
        "N": cfg["data"]["N"], "peak_lr": cfg["training"]["peak_lr"],
        "batch_size": cfg["training"]["batch_size"],
        "latents_path": cfg["dataset"]["latents_path"],
        "coupling_cache_dir": cfg["dataset"]["coupling_cache_dir"],
        "wall_sec": wall,
    }, indent=2, ensure_ascii=False))

    ok = (out / "metrics.json").exists()
    print(f"=== done arm={arm} {ds}/{task}/seed{seed} "
          f"wall={wall/60:.1f}min metrics.json={'OK' if ok else 'MISSING'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
