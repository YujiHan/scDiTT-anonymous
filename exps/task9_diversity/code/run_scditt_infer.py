from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cfg", nargs="*", type=float, default=C.CFG_WEIGHTS)
    ap.add_argument("--eval_seeds", nargs="*", type=int, default=C.EVAL_SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = C.set_gpu(args.gpu)
    from infer_ctx import InferCtx, save_preds  # noqa: E402

    todo = [(w, e) for w in args.cfg for e in args.eval_seeds
            if args.force or not C.pred_dir(args.ds, args.task, w,
                                            args.train_seed, e).joinpath("preds.npz").exists()]
    if not todo:
        print(f"[{args.ds}/{args.task}] nothing to do")
        return 0

    t0 = time.time()
    ctx = InferCtx(args.ds, args.task, args.train_seed, device=device)
    print(f"[{args.ds}/{args.task}] ctx ready in {time.time()-t0:.1f}s | "
          f"T={ctx.T} weights={ctx.weights} test_tps={ctx.test_tps} "
          f"gene_dim={ctx.vae.gene_dim} | {len(todo)} runs", flush=True)

    for i, (w, e) in enumerate(todo):
        t1 = time.time()
        preds = ctx.generate(cfg_weight=w, eval_seed=e)
        out = C.pred_dir(args.ds, args.task, w, args.train_seed, e)
        arr = save_preds(out, preds, ctx.test_tps)
        with open(out / "meta.json", "w") as fh:
            json.dump({
                "dataset": args.ds, "task": args.task,
                "train_seed": args.train_seed, "eval_seed": e,
                "cfg_weight": w, "num_steps": C.NUM_STEPS,
                "num_repeats": C.NUM_REPEATS, "N_eval": C.N_EVAL,
                "n_pred_cells": C.N_PRED_CELLS, "use_ema": ctx.weights == "ema",
                "test_tps": [int(t) for t in ctx.test_tps],
                "pred_shape": list(arr.shape),
                "checkpoint": str(C.ckpt_path(args.ds, args.task, args.train_seed)),
            }, fh, indent=2)
        print(f"[{args.ds}/{args.task}] {i+1}/{len(todo)} cfg={w:g} "
              f"evalseed={e} shape={arr.shape} ({time.time()-t1:.1f}s)", flush=True)

    print(f"[{args.ds}/{args.task}] done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
