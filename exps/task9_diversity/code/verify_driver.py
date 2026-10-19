from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="ZB")
    ap.add_argument("--task", default="extrapolation")
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cfg_weight", type=float, default=3.0)
    ap.add_argument("--eval_seed", type=int, default=0)
    args = ap.parse_args()

    device = C.set_gpu(args.gpu)
    import numpy as np
    from infer_ctx import InferCtx  # noqa: E402
    from src.evaluation.inference import predict_test_tps_from_adata

    ctx = InferCtx(args.ds, args.task, args.train_seed, device=device)
    print(f"[ctx] ds={args.ds} task={args.task} T={ctx.T} weights={ctx.weights}")
    print(f"[ctx] train_tps={ctx.train_tps}  test_tps={ctx.test_tps}")
    print(f"[ctx] gene_dim={ctx.vae.gene_dim}  D_latent={ctx.D_latent}")

    mine = ctx.generate(cfg_weight=args.cfg_weight, eval_seed=args.eval_seed)

    ref = predict_test_tps_from_adata(
        model=ctx.model, vae=ctx.vae,
        latents_path=ctx.cfg["dataset"]["latents_path"],
        h5ad_path=ctx.cfg["dataset"]["h5ad_path"],
        train_tps=ctx.train_tps, test_tps=ctx.test_tps,
        T=ctx.T, time_key=ctx.time_key, scheduler=ctx.scheduler,
        N=C.N_EVAL, num_repeats=C.NUM_REPEATS,
        cfg_weight=args.cfg_weight, num_steps=C.NUM_STEPS,
        device=ctx.device, eval_seed=args.eval_seed,
        n_eval_cells=C.N_PRED_CELLS, predict_all_tps=True,
    )["predictions_gene"]

    ok = True
    for t in range(ctx.T):
        a, b = mine[t], ref[t]
        if a.shape != b.shape:
            print(f"  tp{t}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        exact = np.array_equal(a, b)
        mx = float(np.abs(a - b).max())
        flag = "EXACT" if exact else ("close" if mx < 1e-4 else "DIFF")
        if flag == "DIFF":
            ok = False
        print(f"  tp{t}: {flag} max|Δ|={mx:.3e} shape={a.shape}")

    assert mine[0].shape[-1] > 1000, "gene dim <= 1000 -- the VAE was not loaded (pitfall 8)"
    print(f"\n[assert] pred gene dim = {mine[0].shape[-1]} > 1000  OK")
    print(f"[result] driver-vs-reference: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
