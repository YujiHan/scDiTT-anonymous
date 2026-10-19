#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import TIME_KEY, cache_paths, get_tp_split
from src.data.coupling import precompute_all_couplings
from src.utils.logging import get_logger
from src.utils.seeds import set_all_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=["ZB", "DR", "SC", "GA", "MA",
                            "zb", "dr", "sc", "ga", "ma"])
    p.add_argument("--task", type=str, required=True,
                   choices=["interpolation", "extrapolation",
                            "extrapolation_steps", "all"])
    p.add_argument("--ex_steps", type=int, default=None,
                   help="only for task=extrapolation_steps: number of trailing steps held out as the test set")
    p.add_argument("--ot_type", type=str, default="CPU",
                   choices=["CPU", "CPU_COST", "CUDA"],
                   help="cost matrix computation: CPU (528 classic, default) / CPU_COST (memory-saving) / "
                        "CUDA (GPU). With CUDA, select the GPU via --device")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--max_cells", type=int, default=2000,
                   help="per-side cell cap (default 2000, matching 528); <=0 means no cap and OT over all cells")
    p.add_argument("--no_normalize_cost", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0",
                   help="only used with --ot_type CUDA: select the GPU (e.g. cuda:0/cuda:1)")
    p.add_argument("--results_root", type=str, default="results",
                   help="output/cache root directory, used to isolate experiments (e.g. results/task1_ts_pred); "
                        "OT couplings land under {results_root}/coupling_cache/, "
                        "and latents are read from the same root")
    args = p.parse_args()

    ds_up = args.dataset.upper()
    time_key = TIME_KEY[ds_up]

    if args.task == "extrapolation_steps" and args.ex_steps is None:
        p.error("task 'extrapolation_steps' requires --ex_steps N (integer, 0..T-1).")
    train_tps, test_tps = get_tp_split(ds_up, args.task, n_steps=args.ex_steps)
    _paths = cache_paths(ds_up, args.task, args.ex_steps, args.seed,
                         ot_type=args.ot_type, results_root=args.results_root)
    latents_path = _paths["latents_path"]
    out_dir = Path(_paths["coupling_cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    max_cells = None if args.max_cells <= 0 else args.max_cells

    set_all_seeds(args.seed)
    log = get_logger("scdit.preprocess_coupling", out_dir / "preprocess_coupling.log")
    log.info(f"Dataset={ds_up} task={args.task} eps={args.epsilon} "
             f"max_cells={max_cells} ot_type={args.ot_type} device={args.device}")
    log.info(f"train_tps={train_tps}")
    log.info(f"Output dir: {out_dir}")

    precompute_all_couplings(
        latents_path=latents_path,
        train_tps=train_tps,
        output_dir=str(out_dir),
        time_key=time_key,
        epsilon=args.epsilon,
        normalize_cost=not args.no_normalize_cost,
        max_cells=max_cells,
        seed=args.seed,
        ot_type=args.ot_type,
        device=args.device,
        logger=log,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
