"""Flow-matching scheduler: rectified flow with uniform timestep sampling.

Training-side interface:
    tau = sched.sample_timesteps(B, device)    # [B] float in (0, 1)
    x_t = sched.add_noise(x0, noise, tau)
    v_target = sched.compute_target(x0, noise, tau)

The inference-side solver (Flow-Euler) lives in sampling.py.

Paper reference
---------------
Methodology -> "Masked flow matching objective".

`add_noise` is the linear interpolation z_tau = (1 - tau) * eps + tau * z_0 and
`compute_target` the constant regression velocity v* = z_0 - eps that the
diffusion transformer predicts. One tau is drawn per example and all T slots are
corrupted at that shared level; which slots are evidence and which are targets
is communicated solely through the mask embedding ME(m_k) of Eq. (3), never
through the noise level.
"""
from __future__ import annotations

from typing import Optional

import torch


def _expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape v: [B] so that it broadcasts against x [B, T, N, D_latent]."""
    while v.dim() < x.dim():
        v = v.unsqueeze(-1)
    return v


class FlowMatchingScheduler:
    """Rectified flow:
        x_t = (1 - t) * noise + t * x0   (t ∈ (0, 1); t=0 is pure noise, t=1 is clean)
        v_target = x0 - noise            (a time-invariant velocity)

    Convention: tau=1 corresponds to clean data and tau=0 to pure noise.
    Timesteps are always sampled uniformly from [1e-5, 1-1e-5].

    Paper: the uniform draw is deliberate and is *not* the logit-normal
    distribution favoured in image synthesis. Masked completion depends
    disproportionately on the clean end of the path, where information
    propagates from the evidence slots into the generated ones; and at test time
    the observed slots hold clean latents throughout sampling, so training and
    sampling conditions coincide only as tau -> 1. Down-weighting tau ~ 1
    measurably degrades held-out accuracy (supplementary material, Training
    Details -> "Masking Curriculum").
    """

    def __init__(self, num_train_timesteps: int = 1000, **_kwargs):
        self.num_train_timesteps = num_train_timesteps

    def sample_timesteps(self, batch_size: int, device: torch.device,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.rand(batch_size, device=device,
                          generator=generator).clamp(1e-5, 1.0 - 1e-5)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  tau: torch.Tensor) -> torch.Tensor:
        t = _expand_like(tau, x0)
        return (1.0 - t) * noise + t * x0

    def compute_target(self, x0: torch.Tensor, noise: torch.Tensor,
                       tau: torch.Tensor) -> torch.Tensor:
        return x0 - noise

    def v_to_x0(self, v: torch.Tensor, xt: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        """From v = x0 - noise and xt = (1-t) noise + t x0 it follows that
           x0 = xt + (1-t) v."""
        t = _expand_like(tau, xt)
        return xt + (1.0 - t) * v

    def v_to_eps(self, v: torch.Tensor, xt: torch.Tensor,
                 tau: torch.Tensor) -> torch.Tensor:
        t = _expand_like(tau, xt)
        return xt - t * v

    def snr(self, tau: torch.Tensor) -> torch.Tensor:
        """Under flow matching, SNR(t) ≈ (t/(1-t))^2."""
        t = tau.clamp(1e-5, 1.0 - 1e-5)
        return (t / (1.0 - t)) ** 2


def build_scheduler(cfg_diffusion: dict) -> FlowMatchingScheduler:
    """Instantiate the scheduler from a config dict (always a FlowMatchingScheduler)."""
    return FlowMatchingScheduler(
        num_train_timesteps=cfg_diffusion.get("num_train_timesteps", 1000),
    )
