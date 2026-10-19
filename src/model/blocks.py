"""DiT building blocks: AdaLN-Zero modulation, cell/time attention, FFN, DiTBlock.

Paper reference
---------------
Methodology -> "Population-aware diffusion transformer"; supplementary material,
Training Details -> "Architecture".

This module holds the two-axis attention factorisation that adapts the diffusion
transformer from grids to cell populations: `CellAttention` is equivariant to
permutations of the cells within a stage, `TimeAttention` is order-aware only
through the continuous-time encoding added at the input. Ablating either axis is
Table S5.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    """Compute x * (1 + scale) + shift, broadcasting shift/scale to the shape of x.

    x: [B, ..., D_model], shift/scale: [B, D_model]
    """
    # Build a broadcastable shape: [B, 1, ..., 1, D_model]
    extra_dims = x.dim() - 2
    shape = [shift.shape[0]] + [1] * extra_dims + [shift.shape[-1]]
    shift = shift.view(shape)
    scale = scale.view(shape)
    return x * (1.0 + scale) + shift


def broadcast_gate(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape gate [B, D_model] so that it broadcasts against x [B, ..., D_model]."""
    extra_dims = x.dim() - 2
    shape = [gate.shape[0]] + [1] * extra_dims + [gate.shape[-1]]
    return gate.view(shape)


class AdaLNZero(nn.Module):
    """
    Produce num_mod_groups * 3 modulation parameters (scale, shift, gate) for one
    block. Each parameter has shape [B, D_model].

    The default num_mod_groups=3 corresponds to cell_attn / time_attn / ffn.

    Paper: supplementary material, Training Details -> "Architecture" -- the MLP
    that maps the flow-time embedding c to nine parameters per block, a shift
    b_i, a scale s_i and a gate g_i for each of the three sublayers. The final
    layer is zero-initialised so that each block starts as the identity.
    """

    def __init__(self, D_model: int, num_mod_groups: int = 3):
        super().__init__()
        self.num_mod_groups = num_mod_groups
        self.D_model = D_model
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_model, 3 * num_mod_groups * D_model, bias=True),
        )
        # Zero initialisation makes the block identity-like at initialisation.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of length 3*num_mod_groups, each entry [B, D_model].

        Order: (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2, ...)
        """
        out = self.mlp(c)  # [B, 3 * num_mod_groups * D_model]
        chunks = out.chunk(3 * self.num_mod_groups, dim=-1)
        return list(chunks)


class CellAttention(nn.Module):
    """
    Self-attention on the cell (N) dimension, treating (B, T) as batch.
    No positional encoding on N, hence permutation-equivariant.

    Input:  x ∈ [B, T, N, D_model]
    Output: y ∈ [B, T, N, D_model]

    Paper: MHA(h[k,:,:]) of Methodology -> "Population-aware diffusion
    transformer". Carrying no positional encoding along N is what makes the
    operation equivariant to permutations of the cells -- the correct symmetry
    for a stage that is an unordered set rather than a grid. Removing this axis
    is the "w/o time-attn & cell-attn" row of Table S5.
    """

    def __init__(self, D_model: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_model, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        y, _ = self.attn(x_flat, x_flat, x_flat, need_weights=False)
        return y.reshape(B, T, N, D)


class TimeAttention(nn.Module):
    """
    Self-attention on the time (T) dimension, treating (B, N) as batch.
    The time positional encoding is already added at the input layer.

    Input:  x ∈ [B, T, N, D_model]
    Output: y ∈ [B, T, N, D_model]

    Paper: MHA(h[:,n,:]) of Methodology -> "Population-aware diffusion
    transformer". This is the mechanism by which a missing stage is conditioned
    directly on *every* observed stage at any distance, rather than through a
    chain of local transitions; ordering reaches it exclusively through the
    continuous-time encoding PE(t_k) added at the input (Eq. (3)). Removing it
    is the single largest degradation in Table S5.
    """

    def __init__(self, D_model: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_model, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # [B, T, N, D] -> [B, N, T, D] -> [B*N, T, D]
        x_swap = x.transpose(1, 2).contiguous().reshape(B * N, T, D)
        y, _ = self.attn(x_swap, x_swap, x_swap, need_weights=False)
        y = y.reshape(B, N, T, D).transpose(1, 2).contiguous()
        return y


class FFN(nn.Module):
    """
    Two-layer MLP with SiLU activation and hidden width 4 * D_model.

    Input/Output: [B, T, N, D_model]
    """

    def __init__(self, D_model: int, expansion: int = 4):
        super().__init__()
        hidden = expansion * D_model
        self.fc1 = nn.Linear(D_model, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, D_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """
    Factorized DiT block: cell_attn -> time_attn -> ffn, each under AdaLN-Zero.

    Forward:
        x ∈ [B, T, N, D_model]
        c ∈ [B, D_model]  (conditioning, typically the timestep embedding)
    Output:
        y ∈ [B, T, N, D_model]

    Paper: Methodology -> "Population-aware diffusion transformer" ("Each
    transformer block contains cell-attention, time-attention, and feed-forward
    layers"); the three residual sublayers are applied in that fixed order, each
    wrapped in AdaLN-Zero (supplementary material, Training Details ->
    "Architecture"). Alternating the two axes is the population-data counterpart
    of the spatial-temporal attention split of video diffusion transformers.
    """

    def __init__(self, D_model: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.adaln = AdaLNZero(D_model, num_mod_groups=3)

        self.norm1 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
        self.cell_attn = CellAttention(D_model, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
        self.time_attn = TimeAttention(D_model, num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN(D_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (shift1, scale1, gate1,
         shift2, scale2, gate2,
         shift3, scale3, gate3) = self.adaln(c)

        # 1. Cell attention
        h = modulate(self.norm1(x), shift1, scale1)
        x = x + broadcast_gate(gate1, x) * self.cell_attn(h)

        # 2. Time attention
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + broadcast_gate(gate2, x) * self.time_attn(h)

        # 3. FFN
        h = modulate(self.norm3(x), shift3, scale3)
        x = x + broadcast_gate(gate3, x) * self.ffn(h)

        return x
