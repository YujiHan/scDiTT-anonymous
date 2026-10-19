"""Classifier-free guidance: training-time dropout and an inference-time helper.

Paper reference
---------------
Methodology -> "Temporal mask strategy" (training) and "Inference by Joint
Temporal Inpainting" (test time); Algorithm S1, lines 5-7.

The unconditional branch is obtained by setting the observation mask to all
zeros -- exactly the `full` pattern the mask sampler draws with probability 0.05
during training, so the null condition is one the model has actually seen rather
than an out-of-distribution input.
"""
from __future__ import annotations

import torch


def apply_cfg_dropout(obs_mask: torch.Tensor,
                      dropout_prob: float = 0.0) -> torch.Tensor:
    """During training, set the entire obs_mask of a sample to False with
    probability dropout_prob.

    Note that the `full` pattern of the mask sampler already supplies the null
    condition during training, so dropout_prob is normally 0.0 (a no-op).
    """
    if dropout_prob <= 0.0:
        return obs_mask
    B = obs_mask.shape[0]
    drop = torch.rand(B, device=obs_mask.device) < dropout_prob
    obs_mask = obs_mask.clone()
    obs_mask[drop] = False
    return obs_mask


@torch.no_grad()
def cfg_predict(model, x, tau_diff, obs_mask, t_real,
                cfg_weight: float) -> torch.Tensor:
    """Classifier-free guidance:
        v_guided = v_uncond + cfg_weight * (v_cond - v_uncond)

    cfg_weight == 1.0 reduces to the purely conditional prediction; cfg_weight == 0
    reduces to the purely unconditional one.

    Paper: Algorithm S1 lines 5-7. `null_mask` below is the all-zero observation
    mask, i.e. the `full` masking pattern of training. The weight used for every
    reported result is w = 3.0 (supplementary material -> "Couplings and
    Sampling").
    """
    v_cond = model(x, tau_diff, obs_mask, t_real)
    if cfg_weight == 1.0:
        return v_cond
    null_mask = torch.zeros_like(obs_mask)
    v_uncond = model(x, tau_diff, null_mask, t_real)
    if cfg_weight == 0.0:
        return v_uncond
    return v_uncond + cfg_weight * (v_cond - v_uncond)
