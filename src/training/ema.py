"""EMA wrapper: shadow weights for inference quality."""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.shadow[k].dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def store(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}
