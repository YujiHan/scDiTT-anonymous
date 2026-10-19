#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocks_ablation import build_model  # noqa: E402
from common import ARMS, DATASETS, PROBE_DIR, TASKS, build_cfg  # noqa: E402
from src.diffusion.losses import diffusion_loss  # noqa: E402
from src.diffusion.schedulers import build_scheduler  # noqa: E402
from src.diffusion.sampling import masked_diffusion_inpaint  # noqa: E402
from src.model.model_utils import count_params  # noqa: E402


def bench_arm(arm: str, ds: str, task: str, device: str, iters: int = 20) -> dict:
    cfg = build_cfg(ds, task, 42, arm)
    T = cfg["dataset"]["T"]
    N = cfg["data"]["N"]
    B = cfg["training"]["batch_size"]
    D = cfg["model"]["D_latent"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(arm, cfg, device=device)
    params = count_params(model)["total"]

    sched = build_scheduler(cfg["diffusion"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x0 = torch.randn(B, T, N, D, device=device)
    obs_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    loss_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    t_real = torch.arange(T, dtype=torch.float32, device=device)

    def train_step():
        tau = sched.sample_timesteps(B, torch.device(device))
        noise = torch.randn_like(x0)
        xt = sched.add_noise(x0, noise, tau)
        v_t = sched.compute_target(x0, noise, tau)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v = model(xt, tau, obs_mask, t_real)
            loss = diffusion_loss(v, v_t, loss_mask)["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(5):            # warmup
        train_step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        train_step()
    torch.cuda.synchronize()
    ms_per_step = (time.time() - t0) / iters * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    model.eval()
    N_eval = cfg["inference"]["N_eval"]
    x_obs = torch.randn(1, T, N_eval, D, device=device)
    om = torch.zeros(1, T, dtype=torch.bool, device=device)
    om[0, : T - 2] = True
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        masked_diffusion_inpaint(
            model=model, x_obs=x_obs, obs_mask=om, t_real=t_real, scheduler=sched,
            num_steps=cfg["inference"]["num_steps"],
            cfg_weight=cfg["inference"]["cfg_weight"], device=torch.device(device))
    torch.cuda.synchronize()
    infer_s = time.time() - t0

    del model, opt
    torch.cuda.empty_cache()

    return {"arm": arm, "dataset": ds, "T": T, "N": N, "batch": B,
            "params_M": round(params / 1e6, 2),
            "ms_per_step": round(ms_per_step, 1),
            "peak_mem_GB": round(peak_gb, 2),
            "infer_s_per_repeat": round(infer_s, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--task", default=TASKS[0], choices=TASKS,
                    help="efficiency is task-independent; this is only used to get the per-dataset N/batch")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        for arm in args.arms:
            r = bench_arm(arm, ds, args.task, args.device)
            rows.append(r)
            print(f"[bench] {ds:3s} {arm:8s} params={r['params_M']:6.2f}M "
                  f"{r['ms_per_step']:7.1f} ms/step  peak={r['peak_mem_GB']:5.2f} GB  "
                  f"infer={r['infer_s_per_repeat']:5.2f} s")

    out = PROBE_DIR / "efficiency.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== capacity relative to full ===")
    for ds in args.datasets:
        full = next((r for r in rows if r["dataset"] == ds and r["arm"] == "full"), None)
        if not full:
            continue
        for r in [r for r in rows if r["dataset"] == ds]:
            print(f"  {ds} {r['arm']:8s}: {r['params_M']:6.2f}M "
                  f"({100*r['params_M']/full['params_M']:5.1f}% of full), "
                  f"{100*r['ms_per_step']/full['ms_per_step']:5.1f}% step time")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
