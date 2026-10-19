from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

SELF_FLOOR_REPEATS = 36
SELF_FLOOR_R = 20
SELF_FLOOR_MAX_BLOCKS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--cfg", nargs="*", type=float, default=C.CFG_WEIGHTS)
    ap.add_argument("--eval_seed", type=int, default=1000,
                    help="dedicated eval_seed, disjoint from the main experiment's 0..K-1")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = C.FLOOR_DIR / args.ds / args.task / f"self_floor_trainseed{args.train_seed}.json"
    if out.exists() and not args.force:
        print(f"[skip] {out} exists")
        return 0

    device_str = C.set_gpu(args.gpu)
    import metrics_lib as ML  # noqa: E402
    from infer_ctx import InferCtx  # noqa: E402
    device = ML.get_device(device_str)

    with open(C.FLOOR_DIR / args.ds / args.task / "floor.json") as fh:
        real_floor = json.load(fh)
    test_tps = real_floor["test_tps"]

    pca = C.get_pca(args.ds)
    n_unique = SELF_FLOOR_REPEATS * C.N_EVAL
    ctx = InferCtx(args.ds, args.task, args.train_seed, device=device_str)

    rec = {"dataset": args.ds, "task": args.task, "train_seed": args.train_seed,
           "eval_seed": args.eval_seed, "num_repeats": SELF_FLOOR_REPEATS,
           "n_unique_cells": n_unique, "R": SELF_FLOOR_R,
           "max_blocks": SELF_FLOOR_MAX_BLOCKS,
           "test_tps": [int(t) for t in test_tps], "cfg": {}}

    for w in args.cfg:
        t0 = time.time()
        preds = ctx.generate(cfg_weight=w, eval_seed=args.eval_seed,
                             num_repeats=SELF_FLOOR_REPEATS,
                             n_eval_cells=n_unique)
        assert preds[test_tps[0]].shape[0] == n_unique
        per_stage = {}
        for t in test_tps:
            n_t = real_floor["stages"][str(t)]["n"]
            f = ML.block_floor(preds[int(t)], n_t, pca, device,
                               R=SELF_FLOOR_R, base_seed=4242,
                               max_blocks=SELF_FLOOR_MAX_BLOCKS)
            per_stage[str(int(t))] = f
            print(f"[self] {args.ds}/{args.task} cfg={w:g} tp{t}: n={n_t} "
                  f"M={f['M_blocks']}(use {f['M_used']}) pairs={f['n_pairs']} "
                  f"F_self={f['F']:.4f} +- {f['F_sem']:.4f} | "
                  f"F_real={real_floor['stages'][str(t)]['F']:.4f}", flush=True)
        rec["cfg"][f"{w:g}"] = per_stage
        print(f"[self] {args.ds}/{args.task} cfg={w:g} done ({time.time()-t0:.0f}s)",
              flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[self] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
