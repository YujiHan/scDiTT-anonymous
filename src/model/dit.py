"""Main model: a factorized DiT for single-cell temporal diffusion.

Paper reference
---------------
Methodology -> "Population-aware diffusion transformer"; supplementary material,
Training Details -> "Architecture".

`SCDiT.forward` implements Eq. (3) (tokenisation and conditioning) followed by
`num_blocks` factorised DiT blocks. The default configuration is the one used
for every dataset and every task in the paper: 12 blocks, D_model = 384,
6 heads, D_latent = 64, 45.2M parameters (supplementary material, Training
Details -> "Architecture"; the same count appears in the model-size Table S16).
Removing time-attention, cell-attention and the continuous-time encoding from
this backbone is the architecture ablation of Table S5.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DiTBlock, modulate
from .embeddings import (MaskEmbedding, TimePositionEmbedding,
                         TimestepEmbedding)


class SCDiT(nn.Module):
    """
    Forward:
        x ∈ [B, T, N, D_latent]   (noisy latent)
        tau_diff ∈ [B]             (diffusion timestep, continuous or integer)
        obs_mask ∈ [B, T]          (bool)
        t_real ∈ [T]               (float)
    Output:
        v ∈ [B, T, N, D_latent]    (v-prediction or flow-matching target)
    """

    def __init__(
        self,
        D_latent: int = 64,
        D_model: int = 384,
        num_blocks: int = 12,
        num_heads: int = 6,
        T_max: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.D_latent = D_latent
        self.D_model = D_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.T_max = T_max

        self.input_proj = nn.Linear(D_latent, D_model)
        self.time_pe = TimePositionEmbedding(D_model)
        self.mask_emb = MaskEmbedding(D_model)
        self.timestep_emb = TimestepEmbedding(D_model)

        self.blocks = nn.ModuleList([
            DiTBlock(D_model, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_model, 2 * D_model, bias=True),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        self.output_proj = nn.Linear(D_model, D_latent)
        # Zero-initialise the final projection (standard DiT practice).
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        tau_diff: torch.Tensor,
        obs_mask: torch.Tensor,
        t_real: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, D_latent = x.shape
        assert obs_mask.shape == (B, T), \
            f"obs_mask shape mismatch: {obs_mask.shape} vs (B={B}, T={T})"
        assert tau_diff.shape == (B,), \
            f"tau_diff shape mismatch: {tau_diff.shape}"
        assert t_real.shape == (T,), \
            f"t_real shape mismatch: {t_real.shape}"
        assert T <= self.T_max, f"T={T} exceeds T_max={self.T_max}"

        # 1. Input projection
        # Paper: Eq. (3), the W_in z[k,n] term.
        h = self.input_proj(x)  # [B, T, N, D_model]

        # 2. Time positional encoding + mask embedding
        # Paper: Eq. (3), the PE(t_k) and ME(m_k) terms. Both are shared across
        # the N cells of a stage, hence the broadcast over the cell axis.
        t_pe = self.time_pe(t_real)          # [T, D_model]
        m_emb = self.mask_emb(obs_mask)      # [B, T, D_model]
        # Broadcast: t_pe -> [1, T, 1, D_model]; m_emb -> [B, T, 1, D_model]
        h = h + t_pe.unsqueeze(0).unsqueeze(2) + m_emb.unsqueeze(2)

        # 3. Timestep conditioning
        # Paper: the flow time tau is embedded separately and enters *only*
        # through AdaLN-Zero, following DiT (Methodology -> "Population-aware
        # diffusion transformer"). It is never added to the tokens.
        c = self.timestep_emb(tau_diff)  # [B, D_model]

        # 4. Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # 5. Final output
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        v = self.output_proj(h)  # [B, T, N, D_latent]
        return v

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
