from __future__ import annotations

import torch
import torch.nn as nn

from common import ARM_SPECS
import src.model.dit as _dit_mod
from src.model.blocks import (AdaLNZero, CellAttention, DiTBlock, FFN,
                              TimeAttention, broadcast_gate, modulate)

NO_TIME_PE_ARMS = {"no_tpe"}
_ORIG_TIME_PE = _dit_mod.TimePositionEmbedding


class ZeroTimePE(nn.Module):

    def __init__(self, D_model: int, *args, **kwargs):
        super().__init__()
        self.D_model = D_model

    def forward(self, t_real: torch.Tensor) -> torch.Tensor:
        return torch.zeros(t_real.shape[0], self.D_model,
                           device=t_real.device, dtype=torch.float32)


class AblationDiTBlock(nn.Module):
    """DiTBlock pruned by (use_cell, use_time). The factory `block_class()` fills in USE_CELL/USE_TIME."""

    USE_CELL: bool = True
    USE_TIME: bool = True

    def __init__(self, D_model: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        n_groups = int(self.USE_CELL) + int(self.USE_TIME) + 1  # +1 = FFN
        self.adaln = AdaLNZero(D_model, num_mod_groups=n_groups)

        if self.USE_CELL:
            self.norm1 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
            self.cell_attn = CellAttention(D_model, num_heads, dropout=dropout)

        if self.USE_TIME:
            self.norm2 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
            self.time_attn = TimeAttention(D_model, num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(D_model, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN(D_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mods = self.adaln(c)
        i = 0

        if self.USE_CELL:
            shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
            i += 3
            h = modulate(self.norm1(x), shift, scale)
            x = x + broadcast_gate(gate, x) * self.cell_attn(h)

        if self.USE_TIME:
            shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
            i += 3
            h = modulate(self.norm2(x), shift, scale)
            x = x + broadcast_gate(gate, x) * self.time_attn(h)

        shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
        h = modulate(self.norm3(x), shift, scale)
        x = x + broadcast_gate(gate, x) * self.ffn(h)
        return x


def block_class(arm: str) -> type:
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm '{arm}'; available: {list(ARM_SPECS)}")
    if arm == "full":
        return DiTBlock
    use_cell, use_time = ARM_SPECS[arm]
    return type(f"DiTBlock_{arm}", (AblationDiTBlock,),
                {"USE_CELL": use_cell, "USE_TIME": use_time})


def patch_dit(arm: str) -> type:
    import src.model.dit as dit_mod
    dit_mod.DiTBlock = block_class(arm)
    dit_mod.TimePositionEmbedding = (ZeroTimePE if arm in NO_TIME_PE_ARMS
                                     else _ORIG_TIME_PE)
    return dit_mod.DiTBlock


def build_model(arm: str, cfg: dict, device="cuda") -> nn.Module:
    patch_dit(arm)
    from src.model.dit import SCDiT
    return SCDiT(
        D_latent=cfg["model"]["D_latent"],
        D_model=cfg["model"]["D_model"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        T_max=max(32, cfg["dataset"]["T"]),
    ).to(device)


def load_arm_model(arm: str, cfg: dict, ckpt_path, device="cuda",
                   use_ema: bool = True) -> nn.Module:
    from src.model.model_utils import load_checkpoint
    from src.training.ema import EMA

    model = build_model(arm, cfg, device=device)
    ckpt = load_checkpoint(str(ckpt_path), map_location=device)
    if use_ema and ckpt.get("ema") is not None:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
class Recorder:

    def __init__(self):
        self.on = False
        self.step = -1
        self.cell: dict[tuple[int, int], torch.Tensor] = {}   # (block, step) -> [T, N, N]
        self.time: dict[tuple[int, int], torch.Tensor] = {}   # (block, step) -> [T, T]

    def clear(self):
        self.cell.clear()
        self.time.clear()


class RecordingCellAttention(nn.Module):

    def __init__(self, base: CellAttention, rec: Recorder, block_idx: int):
        super().__init__()
        self.base = base
        self.rec = rec
        self.block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        xf = x.reshape(B * T, N, D)
        need = self.rec.on
        y, w = self.base.attn(xf, xf, xf, need_weights=need,
                              average_attn_weights=True)
        if need and w is not None:
            self.rec.cell[(self.block_idx, self.rec.step)] = \
                w.detach().float().reshape(B, T, N, N)[0].cpu()
        return y.reshape(B, T, N, D)


class RecordingTimeAttention(nn.Module):

    def __init__(self, base: TimeAttention, rec: Recorder, block_idx: int):
        super().__init__()
        self.base = base
        self.rec = rec
        self.block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        xs = x.transpose(1, 2).contiguous().reshape(B * N, T, D)
        need = self.rec.on
        y, w = self.base.attn(xs, xs, xs, need_weights=need,
                              average_attn_weights=True)
        if need and w is not None:
            self.rec.time[(self.block_idx, self.rec.step)] = \
                w.detach().float().mean(dim=0).cpu()
        y = y.reshape(B, N, T, D).transpose(1, 2).contiguous()
        return y


def instrument(model: nn.Module) -> Recorder:
    rec = Recorder()
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, "cell_attn"):
            blk.cell_attn = RecordingCellAttention(blk.cell_attn, rec, i)
        if hasattr(blk, "time_attn"):
            blk.time_attn = RecordingTimeAttention(blk.time_attn, rec, i)
    return rec
