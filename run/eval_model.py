#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True,
                   choices=["ZB", "DR", "SC", "GA", "MA",
                            "zb", "dr", "sc", "ga", "ma"])
    p.add_argument("--task", type=str, required=True,
                   choices=["interpolation", "extrapolation",
                            "extrapolation_steps", "all"])
    p.add_argument("--ex_steps", type=int, default=None,
                   help="only for task=extrapolation_steps: number of trailing steps held out as the test set (0..T-1)")
    p.add_argument("--num_repeats", type=int, default=12)
    p.add_argument("--cfg_weight", type=float, default=3.0)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--N_eval", type=int, default=256)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--n_pred_cells", type=int, default=2000,
                   help="number of predicted cells saved/evaluated per time point (second dimension of pred_genes.npy; "
                        "the default of 2000 matches the baseline n_sim_cells)")
    p.add_argument("--no_save_predictions", action="store_true",
                   help="do not save pred_genes.npy (by default predictions for all time points are saved)")
    args = p.parse_args()

    ds_up = args.dataset.upper()
    if args.task == "extrapolation_steps" and args.ex_steps is None:
        p.error("task 'extrapolation_steps' requires --ex_steps N (integer, 0..T-1).")
    records = evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_name=ds_up,
        task_name=args.task,
        num_repeats=args.num_repeats,
        cfg_weight=args.cfg_weight,
        num_steps=args.num_steps,
        use_ema=not args.no_ema,
        N_eval=args.N_eval,
        eval_seed=args.eval_seed,
        n_steps=args.ex_steps,
        output_dir=args.output_dir,
        n_pred_cells=args.n_pred_cells,
        save_predictions=not args.no_save_predictions,
    )
    keys = ["w2_avg", "mmd_avg", "pearson_mean", "sampled_pearson", "pg_w1_mean"]
    for stem, rec in records.items():
        avg = rec["avg"]
        print(f"[{stem}] " + "  ".join(f"{k}={avg[k]:.4f}" for k in keys))


if __name__ == "__main__":
    main()
