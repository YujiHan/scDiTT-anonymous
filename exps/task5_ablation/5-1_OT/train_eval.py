from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (T, K, D, TIME_KEY, SIGMA_MAIN, sim_h5ad_path,
                    coupling_dir, ckpt_dir, get_split, tag)
from src.data.dataset import SCTrajectoryDataset
from src.data.mask_sampler import MaskSampler
from src.model.dit import SCDiT
from src.training.callbacks import TrainLossProbe, build_probe_batches
from src.training.ema import EMA
from src.training.optim import build_optimizer
from src.training.optim import build_scheduler as build_lr_scheduler
from src.diffusion.schedulers import build_scheduler as build_diff_scheduler
from src.diffusion.losses import diffusion_loss
from src.evaluation.inference import predict_test_tps_from_adata
from src.evaluation.metrics import compute_wd


def make_config(total_steps: int, N: int, num_workers: int) -> dict:
    return {
        "model": {"D_latent": D, "D_model": 384, "num_blocks": 12,
                  "num_heads": 6},
        "data": {"N": N, "num_workers": num_workers, "length": 100000},
        "training": {"total_steps": total_steps, "batch_size": 16,
                     "peak_lr": 1.0e-4, "warmup_steps": 1000,
                     "min_lr_ratio": 0.1, "weight_decay": 0.01,
                     "grad_clip": 1.0, "ema_decay": 0.999,
                     "val_every": 1000, "use_bf16": True,
                     "probe_batches": 8},
        "diffusion": {"num_train_timesteps": 1000, "use_min_snr": False},
        "inference": {"num_steps": 20, "cfg_weight": 3.0, "num_repeats": 12,
                      "N_eval": 256, "eval_seed": 0},
    }


def _worker_init_fn(worker_id: int):
    np.random.seed(torch.initial_seed() % 2**32)


def _infinite(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def eval_latent_w2(model, ema, sim_path, diff_sched, cfg, device,
                   train_tps, test_tps) -> tuple:
    backup = ema.store(model)
    ema.apply_to(model)
    res = predict_test_tps_from_adata(
        model=model, vae=None, latents_path=str(sim_path),
        h5ad_path=str(sim_path),
        train_tps=train_tps, test_tps=test_tps, T=T, time_key=TIME_KEY,
        scheduler=diff_sched, N=cfg["inference"]["N_eval"],
        num_repeats=cfg["inference"]["num_repeats"],
        cfg_weight=cfg["inference"]["cfg_weight"],
        num_steps=cfg["inference"]["num_steps"], device=device,
        eval_seed=cfg["inference"]["eval_seed"], n_eval_cells=K,
    )
    ema.restore(model, backup)
    w2 = {}
    for t in test_tps:
        pred = res["predictions_latent"][t]        # [n_eval_cells, D]
        true = res["true_gene_list"][t]            # [K, D] (latent, vae=None)
        w2[int(t)] = float(compute_wd(pred, true, device=device))
    avg = float(np.mean(list(w2.values())))
    return avg, w2


def train_one(regime, sigma, arm, seed, total_steps, device_str="cuda:0",
              num_workers=4, task="interp"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    cfg = make_config(total_steps, N=128, num_workers=num_workers)
    train_tps, test_tps, _ = get_split(task)

    sim_path = sim_h5ad_path(regime, sigma, seed)
    cdir = coupling_dir(regime, sigma, seed, arm, task)
    out_dir = ckpt_dir(regime, sigma, seed, arm, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== train {tag(regime, sigma)} seed{seed} arm={arm} task={task} "
        f"steps={total_steps} device={device} ===")
    log(f"train_tps={train_tps} test_tps={test_tps}")

    mask_sampler = MaskSampler(train_tps=train_tps, T=T,
                               rng=np.random.default_rng(seed))
    ds = SCTrajectoryDataset(
        latents_path=str(sim_path), coupling_cache_dir=str(cdir),
        train_tps=train_tps, test_tps=test_tps, T=T, N=cfg["data"]["N"],
        t_real=[float(t) for t in range(T)], time_key=TIME_KEY,
        mask_sampler=mask_sampler, length=cfg["data"]["length"], seed=seed)
    loader = DataLoader(
        ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=num_workers, drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0)

    # --- model / ema / optim / schedulers ---
    model = SCDiT(D_latent=D, D_model=cfg["model"]["D_model"],
                  num_blocks=cfg["model"]["num_blocks"],
                  num_heads=cfg["model"]["num_heads"],
                  T_max=max(32, T)).to(device)
    log(f"model params: {model.num_parameters()/1e6:.2f}M")
    ema = EMA(model, decay=cfg["training"]["ema_decay"])
    optimizer = build_optimizer(model.parameters(),
                                peak_lr=cfg["training"]["peak_lr"],
                                weight_decay=cfg["training"]["weight_decay"])
    lr_sched = build_lr_scheduler(optimizer,
                                  warmup_steps=cfg["training"]["warmup_steps"],
                                  total_steps=total_steps,
                                  min_ratio=cfg["training"]["min_lr_ratio"])
    diff_sched = build_diff_scheduler(cfg["diffusion"])

    use_bf16 = cfg["training"]["use_bf16"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    grad_clip = cfg["training"]["grad_clip"]

    probe = TrainLossProbe(
        batches=build_probe_batches(
            ds, n_batches=cfg["training"].get("probe_batches", 8),
            batch_size=cfg["training"]["batch_size"], seed=seed + 777),
        scheduler=diff_sched, device=device, seed=seed + 999,
        use_bf16=use_bf16, amp_dtype=amp_dtype)

    best_avg = float("inf")
    best_w2 = None
    sel_avg, sel_w2 = float("nan"), {}
    val_hist = []
    data_iter = _infinite(loader)

    model.train()
    t0 = time.time()
    for step in range(total_steps):
        batch = next(data_iter)
        x0 = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        t_real = batch["t_real"][0].to(device)
        B = x0.shape[0]

        tau = diff_sched.sample_timesteps(B, device)
        noise = torch.randn_like(x0)
        xt = diff_sched.add_noise(x0, noise, tau)
        v_target = diff_sched.compute_target(x0, noise, tau)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_bf16):
            v_pred = model(xt, tau, obs_mask, t_real)
            loss = diffusion_loss(v_pred, v_target, loss_mask)["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
        ema.update(model)

        if step % 200 == 0:
            log(f"step {step:6d} loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"({(time.time()-t0)/max(step,1)*1000:.0f} ms/step)")

        do_val = (step > 0 and step % cfg["training"]["val_every"] == 0) \
            or step == total_steps - 1
        if do_val:
            probe_loss = probe.run(model, ema, step)
            improved = probe.update_best(probe_loss, step)
            if improved:
                torch.save({"ema": ema.state_dict(), "step": step,
                            "selection_metric": "probe_train_loss",
                            "best_probe_loss": probe_loss}, out_dir / "best.pt")
            log(f"[probe step={step}] train_loss={probe_loss:.6f} "
                f"(best={probe.best:.6f} @ step {probe.best_step})")

            avg, w2 = eval_latent_w2(model, ema, sim_path, diff_sched, cfg,
                                     device, train_tps, test_tps)
            val_hist.append({"step": step, "avg_w2": avg, "w2": w2,
                             "probe_train_loss": probe_loss})
            log(f"[val-diagnostic step={step}] avg_latent_w2={avg:.4f} "
                + " ".join(f"t{t}={v:.4f}" for t, v in w2.items()))
            if improved:
                sel_avg, sel_w2 = avg, w2
            best_avg = min(best_avg, avg)
            if best_w2 is None or avg <= best_avg:
                best_w2 = w2
            model.train()

    metrics = {"regime": regime, "sigma": sigma, "arm": arm, "seed": seed,
               "task": task, "total_steps": total_steps,
               "selection_metric": "probe_train_loss",
               "best_probe_loss": probe.best, "best_step": probe.best_step,
               "best_avg_w2": sel_avg, "best_w2": sel_w2,
               "curve_min_avg_w2": best_avg, "curve_min_w2": best_w2,
               "val_hist": val_hist, "wall_sec": time.time() - t0}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"=== done selected_avg_latent_w2={sel_avg:.4f} "
        f"(step {probe.best_step}, curve_min={best_avg:.4f}) "
        f"wall={metrics['wall_sec']/60:.1f}min ===")
    log_f.close()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True,
                    choices=["linear", "bifurcation", "trifurcation"])
    ap.add_argument("--sigma", type=float, default=SIGMA_MAIN)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    train_one(args.regime, args.sigma, args.arm, args.seed,
              args.total_steps, args.device, args.num_workers, task=args.task)


if __name__ == "__main__":
    main()
