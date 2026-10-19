"""Diffusion loss and optional min-SNR weighting.

Paper reference
---------------
Eq. (4), Methodology -> "Masked flow matching objective".

`diffusion_loss` is the masked rectified-flow regression the paper trains on.
`min_snr_weight` is an optional extra not used for any reported result (the
paper's objective is the unweighted Eq. (4)); it sits behind
`diffusion.use_min_snr`, which is `false` in configs/default.yaml and is not
overridden by any per-dataset config.
"""
from __future__ import annotations

from typing import Optional

import torch


def diffusion_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    loss_mask: torch.Tensor,
    snr_weight: Optional[torch.Tensor] = None,
) -> dict:
    """Masked L2 loss: mean over cells and latent dimensions, then mean over the
    masked timeslots.

    v_pred, v_target: [B, T, N, D_latent]
    loss_mask: [B, T] bool
    snr_weight: [B] or None

    Paper: Eq. (4). The three reductions below map term by term onto it --
    `.mean(dim=(2, 3))` is the 1/(Nd) inner normalisation over cells and latent
    dimensions, multiplying by `loss_mask` is the l_k factor that restricts
    supervision to the generated stages, and dividing by the mask count is the
    1/|l| outer normalisation. Stages that are evidence, and the truly held-out
    stages, contribute nothing to the gradient.
    """
    assert v_pred.shape == v_target.shape
    err = (v_pred - v_target).pow(2)           # [B, T, N, D]
    err = err.mean(dim=(2, 3))                 # [B, T]
    mask_f = loss_mask.float()
    err = err * mask_f
    denom = mask_f.sum(dim=1).clamp(min=1.0)   # [B]
    loss_per_sample = err.sum(dim=1) / denom   # [B]
    if snr_weight is not None:
        loss_per_sample = loss_per_sample * snr_weight
    return {"total": loss_per_sample.mean(),
            "loss_per_sample": loss_per_sample}


def min_snr_weight(snr: torch.Tensor, gamma: float = 5.0,
                   prediction_type: str = "v_prediction") -> torch.Tensor:
    """Min-SNR-gamma weighting (Hang et al., 2023).

    For v-prediction:   w = min(SNR, gamma) / (SNR + 1)
    For eps-prediction: w = min(SNR, gamma) / SNR
    For x0-prediction:  w = min(SNR, gamma)
    Under flow matching we reuse the v-prediction form.
    """
    capped = torch.clamp(snr, max=gamma)
    if prediction_type == "v_prediction":
        return capped / (snr + 1.0)
    if prediction_type == "eps_prediction":
        return capped / (snr + 1e-8)
    if prediction_type == "x0_prediction":
        return capped
    return capped / (snr + 1.0)
