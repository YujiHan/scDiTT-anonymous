#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import common
from common import (ARMS, CONFIG_ROOT, DATASETS, OT_TYPE, REPO_ROOT,
                    RESULTS_ROOT_REL, SEEDS, TASKS, TOTAL_STEPS, get_split,
                    run_dir)
from mask_arms import ARM_DESC, ARM_WEIGHTS, sampler_factory

import src.training.train as train_mod
from src.utils.config import build_config, save_config


def main() -> None:
    p = argparse.ArgumentParser(description="mask ablation: train a single arm / single cell")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--batch_size", type=int, default=None,
                   help="override the batch in configs/dataset/<ds>.yaml (ZB=16, needs about 17.5GB of GPU memory)."
                        "Reduce it only when a new machine lacks memory -- all four arms must use the same value or the comparison is not clean.")
    p.add_argument("--num_workers", type=int, default=None,
                   help="override the number of dataloader workers (default 4); reduce it when running on many GPUs with few CPU cores.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="default: results/task5_ablation/5-3_mask/checkpoints/<arm>/<DS>_<task>_seed<seed>")
    args = p.parse_args()

    os.chdir(REPO_ROOT)

    out = Path(args.output_dir) if args.output_dir else run_dir(
        args.arm, args.dataset, args.task, args.seed)
    out.mkdir(parents=True, exist_ok=True)

    ns = SimpleNamespace(
        dataset=args.dataset.lower(), task=args.task, seed=int(args.seed),
        output_dir=str(out), results_root=RESULTS_ROOT_REL, ot_type=OT_TYPE,
        ex_steps=None, resume_from=None, smoke_test=False,
    )
    extras = ["--training.total_steps", str(args.total_steps)]
    if args.batch_size is not None:
        extras += ["--training.batch_size", str(args.batch_size)]
    if args.num_workers is not None:
        extras += ["--data.num_workers", str(args.num_workers)]
    cfg = build_config(ns, extras=extras, cfg_root=CONFIG_ROOT)

    train_tps, test_tps = get_split(args.dataset, args.task)
    cfg["mask_ablation"] = {
        "arm": args.arm,
        "weights": ARM_WEIGHTS[args.arm],
        "desc": ARM_DESC[args.arm],
        "train_tps": train_tps,
        "test_tps": test_tps,
    }
    save_config(cfg, out / "config.yaml")
    (out / "arm.json").write_text(
        json.dumps(cfg["mask_ablation"], ensure_ascii=False, indent=2))

    print("=" * 68)
    print(f"[mask-ablation] arm={args.arm} :: {ARM_DESC[args.arm]}")
    print(f"[mask-ablation] {args.dataset}/{args.task}/seed{args.seed} -> {out}")
    print(f"[mask-ablation] train_tps={train_tps}  test_tps={test_tps}")
    print(f"[mask-ablation] latents  = {cfg['dataset']['latents_path']}")
    print(f"[mask-ablation] coupling = {cfg['dataset']['coupling_cache_dir']}")
    print("=" * 68)

    train_mod.MaskSampler = sampler_factory(args.arm)
    # ===========================================================

    train_mod.train_model(cfg)


if __name__ == "__main__":
    main()
