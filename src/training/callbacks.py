"""Validation / checkpointing callbacks.

`TrainLossProbe` provides the signal for model selection (best.pt) and only sees training
data; `ValidationCallback` computes test W2, is purely diagnostic, and only goes into train.log to draw the
convergence curve. The two are not interchangeable.

Paper reference
---------------
Experiments -> "Evaluation protocol and metrics" ("no held-out stages are used
during training or model selection").

`TrainLossProbe` is the single signal that drives checkpoint selection: the
masked flow-matching loss on a fixed probe set drawn from the *training*
distribution, with batches, masking patterns, flow times tau and noise all
frozen up front, evaluated under the current EMA weights. Freezing every source
of stochasticity is what makes the curve comparable across steps; the step-wise
training loss is unsuitable because its jitter exceeds the total decrease over
training. Selection is the probe's only job -- training always runs the full
`training.total_steps` budget, with no early stopping.

The separation enforced here is what makes the claim above true:
`ValidationCallback` computes test W2 on held-out stages and is logged for
diagnosis only -- it must never feed best.pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from ..diffusion.losses import diffusion_loss, min_snr_weight


def build_probe_batches(dataset, n_batches: int, batch_size: int,
                        seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    saved_rng, saved_mask_rng = dataset._rng, dataset.mask_sampler.rng
    dataset._rng = rng
    dataset.mask_sampler.rng = rng
    try:
        return [default_collate([dataset[i] for i in range(batch_size)])
                for _ in range(n_batches)]
    finally:
        dataset._rng = saved_rng
        dataset.mask_sampler.rng = saved_mask_rng


class TrainLossProbe:

    def __init__(self, batches: list[dict], scheduler, device: torch.device,
                 seed: int = 0, use_min_snr: bool = False,
                 min_snr_gamma: float = 5.0,
                 use_bf16: bool = False,
                 amp_dtype: torch.dtype = torch.float32):
        self.batches = batches
        self.scheduler = scheduler
        self.device = device
        self.seed = int(seed)
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.use_bf16 = use_bf16
        self.amp_dtype = amp_dtype
        self.best: float = float("inf")
        self.best_step: Optional[int] = None
        self.history: list[tuple[int, float]] = []

    @torch.no_grad()
    def run(self, model: nn.Module, ema, step: int) -> float:
        backup = None
        if ema is not None:
            backup = ema.store(model)
            ema.apply_to(model)
        model.eval()
        try:
            total, n = 0.0, 0
            for i, batch in enumerate(self.batches):
                x0 = batch["x"].to(self.device)              # [B, T, N, D]
                obs_mask = batch["obs_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                t_real = batch["t_real"][0].to(self.device)
                B = x0.shape[0]

                g = torch.Generator(device=self.device)
                g.manual_seed(self.seed + i)
                tau = self.scheduler.sample_timesteps(B, self.device, generator=g)
                noise = torch.randn(x0.shape, generator=g, device=self.device,
                                    dtype=x0.dtype)
                xt = self.scheduler.add_noise(x0, noise, tau)
                v_target = self.scheduler.compute_target(x0, noise, tau)

                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_bf16):
                    v_pred = model(xt, tau, obs_mask, t_real)
                    snr_w = None
                    if self.use_min_snr:
                        snr_w = min_snr_weight(
                            self.scheduler.snr(tau), gamma=self.min_snr_gamma,
                            prediction_type="v_prediction")
                    ld = diffusion_loss(v_pred, v_target, loss_mask, snr_w)
                total += float(ld["total"]) * B
                n += B
        finally:
            if ema is not None and backup is not None:
                ema.restore(model, backup)
            model.train()

        value = total / max(n, 1)
        self.history.append((int(step), value))
        return value

    def update_best(self, value: float, step: int) -> bool:
        if value < self.best:
            self.best = value
            self.best_step = int(step)
            return True
        return False


class ValidationCallback:

    def __init__(self, dataset_name: str, task_type: str,
                 num_repeats: int = 4, cfg_weight: float = 3.0,
                 num_steps: int = 20,
                 N_eval: int = 256, eval_seed: int = 0,
                 device: str = "cuda"):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.num_repeats = num_repeats
        self.cfg_weight = cfg_weight
        self.num_steps = num_steps
        self.N_eval = N_eval
        self.eval_seed = eval_seed
        self.device = device
        self.best: float = float("inf")

    def run(self, model: nn.Module, ema, step: int,
            evaluate_fn, **kwargs) -> dict:
        """evaluate_fn: callable(model_for_eval, **kwargs) -> dict with 'avg_wd'."""
        model_backup = None
        if ema is not None:
            model_backup = ema.store(model)
            ema.apply_to(model)

        model.eval()
        try:
            metrics = evaluate_fn(
                model=model,
                num_repeats=self.num_repeats,
                cfg_weight=self.cfg_weight,
                num_steps=self.num_steps,
                N_eval=self.N_eval,
                eval_seed=self.eval_seed,
                **kwargs,
            )
        finally:
            if ema is not None and model_backup is not None:
                ema.restore(model, model_backup)
            model.train()

        return {"val/avg_wd": float(metrics.get("avg_wd", float("nan"))),
                **{f"val/wd_t{t}": float(v)
                   for t, v in metrics.get("per_tp_wd", {}).items()}}
