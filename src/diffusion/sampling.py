"""Masked diffusion inpainting with a Flow-Euler solver.

At each step the model predicts a velocity (with classifier-free guidance), an
Euler step advances x_tau to x_{tau+dt}, and the observed timeslots are re-injected
as clean latents by direct replacement.

Paper reference
---------------
Algorithm S1 (supplementary material); Methodology -> "Inference by Joint
Temporal Inpainting".

`masked_diffusion_inpaint` is Algorithm S1 line for line. Defaults match the
paper: 20 Euler steps and classifier-free guidance weight 3.0 (supplementary
material, Training Details -> "Couplings and Sampling").

The property the paper's long-horizon claim rests on is structural and visible
here: *all* missing stages are produced within this single conditional pass, so
the path from evidence to target has length one whatever the horizon, and cost
depends on how many stages are generated rather than on how far they lie from
the observed measurements. The autoregressive controls of Figure 3 and Table S2
(scDiTT-AR-prefix / -Markov) differ from this function only in the generation
schedule -- they call the same backbone stage by stage.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cfg import cfg_predict
from .schedulers import FlowMatchingScheduler, _expand_like


def flow_euler_step(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    tau_cur: torch.Tensor,
    tau_next: torch.Tensor,
) -> torch.Tensor:
    """Rectified-flow Euler step:
        x_{t+dt} = x_t + dt * v
    tau increases from tau_cur to tau_next, i.e. from noise (tau=0) to clean (tau=1).

    Paper: Algorithm S1 line 8.
    """
    dt = _expand_like(tau_next - tau_cur, x_t)
    return x_t + dt * v_pred


@torch.no_grad()
def masked_diffusion_inpaint(
    model: nn.Module,
    x_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    t_real: torch.Tensor,
    scheduler: FlowMatchingScheduler,
    num_steps: int = 20,
    cfg_weight: float = 3.0,
    device: Optional[torch.device] = None,
    rng_generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Masked diffusion inpainting with the Flow-Euler solver.

    Args:
        x_obs    : [B, T, N, D_latent] — clean latents at the timepoints where
                   obs_mask[t] is True
        obs_mask : [B, T] bool
        t_real   : [T] float
        scheduler: FlowMatchingScheduler

    Returns:
        x_final: [B, T, N, D_latent] — fully denoised; observed slots keep their
        clean values.
    """
    if device is None:
        device = x_obs.device
    B, T, N, D = x_obs.shape

    # Flow matching: tau runs from 0 (pure noise) to 1 (clean data).
    taus = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    taus = taus.clamp(1e-4, 1.0 - 1e-4)

    # Initial state: tau=0 corresponds to pure noise.
    # Paper: Algorithm S1 line 1 -- the held-out slots are initialised with
    # Gaussian noise.
    x = torch.randn(B, T, N, D, device=device, generator=rng_generator)

    # Clean latents at the observed positions (only these are re-injected).
    # Paper: Algorithm S1 line 2 -- the clean latents of the observed stages are
    # placed into their own slots as conditioning tokens.
    obs_mask_f = obs_mask.view(B, T, 1, 1).float()
    x = x * (1.0 - obs_mask_f) + x_obs * obs_mask_f

    # Paper: Algorithm S1 lines 3-10 -- integrate the learned velocity field
    # from tau = 0 to tau = 1.
    for i in range(num_steps):
        tau_cur = taus[i].expand(B)
        tau_next = taus[i + 1].expand(B)

        # Paper: Algorithm S1 lines 5-7 (conditional and unconditional branches
        # combined by classifier-free guidance); see diffusion/cfg.py.
        v_pred = cfg_predict(model, x, tau_cur, obs_mask, t_real, cfg_weight)
        # Paper: Algorithm S1 line 8 (Euler step).
        x_new = flow_euler_step(x, v_pred, tau_cur, tau_next)

        # Re-inject the clean latents at observed positions (simple replacement).
        # Paper: Algorithm S1 line 9. Re-injecting after *every* step is what
        # keeps the conditioning from being overwritten by the integration.
        x_new = x_new * (1.0 - obs_mask_f) + x_obs * obs_mask_f
        x = x_new

    return x
