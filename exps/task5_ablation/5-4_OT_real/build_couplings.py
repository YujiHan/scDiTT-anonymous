#!/usr/bin/env python
"""Build one coupling cache per ablation arm from a shared set of latents.

Paper reference
---------------
Supplementary material, Ablation Studies -> "Effect of OT-Based Trajectory
Construction" (Table S3).

The autoencoder is fitted once per (dataset, task, seed) by run/preprocess_vae.py
and shared by all four arms, so the latents every arm sees are byte-identical and
the only thing that differs is how those latents are strung into trajectories.
Subsampling to `coupling.max_cells` consumes the rng identically in every arm
(see src/data/coupling.py::compute_coupling), so the arms even see the same
subsample of cells at every stage pair.

Usage:
    PYTHONPATH=. python exps/task5_ablation/5-4_OT_real/build_couplings.py \
        --dataset ZB --task extrapolation --seed 42 --latents <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as C  # noqa: E402  (inserts REPO_ROOT into sys.path)

from src.data.coupling import KERNELS, precompute_all_couplings  # noqa: E402
from src.data.splits import get_tp_split  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def build_arm(ds: str, task: str, seed: int, arm: str, latents: Path,
              epsilon: float, max_cells: int, normalize_cost: bool,
              ot_type: str, device: str, force: bool, logger) -> Path:
    out = C.coupling_dir(ds, task, seed, arm)
    train_tps, _ = get_tp_split(ds, task)
    expected = [out / f"pi_{a}_to_{b}.npz"
                for a, b in zip(train_tps[:-1], train_tps[1:])]
    if not force and out.is_dir() and all(p.exists() for p in expected):
        logger.info(f"[skip] {ds}/{task}/seed{seed}/{arm}: {len(expected)} couplings present")
        return out

    logger.info(f"[build] {ds}/{task}/seed{seed}/{arm}: kernel={arm} "
                f"over {len(expected)} adjacent stage pairs")
    precompute_all_couplings(
        latents_path=latents,
        train_tps=train_tps,
        output_dir=out,
        epsilon=epsilon,
        normalize_cost=normalize_cost,
        max_cells=max_cells,
        seed=seed,
        ot_type=ot_type,
        device=device,
        kernel=arm,
        logger=logger,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS + ["zb", "sc"])
    ap.add_argument("--task", default="extrapolation", choices=C.TASKS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--latents", required=True,
                    help="the *_latents.h5ad shared by every arm of this (dataset, task, seed)")
    ap.add_argument("--arms", nargs="+", default=C.ARMS, choices=C.ARMS)
    ap.add_argument("--ot-type", default=C.OT_TYPE, choices=["CPU", "CPU_COST", "CUDA"])
    ap.add_argument("--device", default="cuda:0", help="only used with --ot-type CUDA")
    ap.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    args = ap.parse_args()

    ds = args.dataset.upper()
    latents = Path(args.latents)
    if not latents.is_file():
        raise SystemExit(
            f"missing latents: {latents}\n"
            f"Run run/preprocess_vae.py for {ds}/{args.task}/seed{args.seed} first "
            f"(run_all.sh does this for you).")

    # The coupling hyperparameters are the shared ones; nothing is tuned per arm.
    cfg = load_yaml(C.REPO_ROOT / "configs" / "default.yaml").get("coupling", {})
    epsilon = float(cfg.get("epsilon", 0.05))
    max_cells = cfg.get("max_cells", 2000)
    normalize_cost = bool(cfg.get("normalize_cost", True))

    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger("scdit.5-4.couplings",
                        C.LOG_DIR / f"couplings_{C.run_tag(ds, args.task, args.seed)}.log")
    logger.info(f"epsilon={epsilon} max_cells={max_cells} "
                f"normalize_cost={normalize_cost} ot_type={args.ot_type}")

    for arm in args.arms:
        if arm not in KERNELS:
            raise SystemExit(f"arm {arm!r} is not a kernel known to src/data/coupling.py")
        build_arm(ds, args.task, args.seed, arm, latents,
                  epsilon, max_cells, normalize_cost,
                  args.ot_type, args.device, args.force, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
