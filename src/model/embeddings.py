"""Embeddings: diffusion timestep, real sampling time, and observation mask.

Paper reference
---------------
Methodology -> "Population-aware diffusion transformer".

The two conditioning signals of Eq. (3) are built here: `TimePositionEmbedding`
is PE(t_k), a sinusoidal-plus-MLP encoding of a *continuous* scalar per stage --
so unevenly spaced stages receive correspondingly unevenly spaced encodings,
which is how irregular sampling is represented. Which scalar that is, is named
per dataset by `real_time_key` (see configs/dataset/zb.yaml and
`src/data/splits.py::t_real_from_obs`): the physical acquisition time of the
stage in all five shipped configs, or the stage index if it is set to null.
`MaskEmbedding` is ME(m_k),
which marks a stage as evidence or as a slot to be generated. Both are shared
across the N cells of a stage. `TimestepEmbedding` is separate: it encodes the
flow time tau, which reaches the network only through AdaLN-Zero modulation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(x: torch.Tensor, dim: int,
                         max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding.

    Input:  x ∈ [*], an arbitrary floating-point tensor
    Output: [*, dim]
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    device = x.device
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # Broadcast over the trailing dimension.
    args = x.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """
    Diffusion timestep τ ∈ [0, 1] (or an integer in [0, num_train_timesteps]),
    mapped by sinusoidal features and an MLP to D_model.

    Input:  τ ∈ [B]
    Output: emb ∈ [B, D_model]

    Paper: the flow time tau of Methodology -> "Masked flow matching objective".
    Unlike PE(t_k) and ME(m_k) this is *not* added to the tokens of Eq. (3); it
    is injected through AdaLN-Zero only (see `model/blocks.py`).
    """

    def __init__(self, D_model: int, freq_dim: int = 256,
                 max_period: float = 10000.0):
        super().__init__()
        self.D_model = D_model
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, D_model),
            nn.SiLU(),
            nn.Linear(D_model, D_model),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(tau, self.freq_dim, self.max_period)
        return self.mlp(emb)


class TimePositionEmbedding(nn.Module):
    """
    Positional encoding of the sampling time t_real[w] (a continuous value),
    mapped by sinusoidal features and an MLP to D_model.

    Input:  t ∈ [T]   float
    Output: emb ∈ [T, D_model]

    Paper: PE(t_k) in Eq. (3). The input is a continuous scalar rather than a
    discrete position, so the encoding of two stages moves apart in proportion
    to the gap between their time values -- this is what lets the model
    represent irregular sampling intervals (Methodology -> "Population-aware
    diffusion transformer"). The scalar is named per dataset by `real_time_key`,
    which every shipped config points at that stage's physical acquisition time
    (hours post fertilisation on ZB, the collection-window midpoint on DR,
    reprogramming day on SC, post-conception weeks on GA/MA); `null` would fall
    back on the stage index.
    Ablating this embedding is the "w/o attn & time-PE" row of Table S5.

    Note that `max_period` is 100.0 here, not the 10000.0 of `TimestepEmbedding`:
    the input ranges over stage indices or hours/days/weeks (single or double
    digits), not over a diffusion timestep grid.
    """

    def __init__(self, D_model: int, freq_dim: int = 256,
                 max_period: float = 100.0):
        super().__init__()
        self.D_model = D_model
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, D_model),
            nn.SiLU(),
            nn.Linear(D_model, D_model),
        )

    def forward(self, t_real: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t_real, self.freq_dim, self.max_period)
        return self.mlp(emb)


class MaskEmbedding(nn.Module):
    """
    Two learned vectors, one for observed timepoints and one for masked timepoints.

    Input:  obs_mask ∈ [B, T]  bool
    Output: emb ∈ [B, T, D_model]

    Paper: ME(m_k) in Eq. (3), the observation embedding that marks stage k as
    evidence or as a slot to be generated according to the binary mask
    m ∈ {0,1}^T. This is the only channel through which the model is told which
    stages to condition on, so the same weights serve interpolation,
    extrapolation and the unconditional branch of classifier-free guidance.
    """

    def __init__(self, D_model: int):
        super().__init__()
        self.e_obs = nn.Parameter(torch.zeros(D_model))
        self.e_mask = nn.Parameter(torch.zeros(D_model))
        nn.init.normal_(self.e_obs, std=0.02)
        nn.init.normal_(self.e_mask, std=0.02)

    def forward(self, obs_mask: torch.Tensor) -> torch.Tensor:
        # obs_mask: [B, T] bool
        # Output: e_obs where True, e_mask where False -> [B, T, D_model]
        mask_f = obs_mask.unsqueeze(-1).float()  # [B, T, 1]
        return mask_f * self.e_obs + (1.0 - mask_f) * self.e_mask
