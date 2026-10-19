"""Optimizer + lr scheduler helpers."""
from __future__ import annotations

import math

import torch


def build_optimizer(params, peak_lr: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=peak_lr, betas=betas,
                             weight_decay=weight_decay)


def warmup_cosine_lambda(step: int, warmup_steps: int,
                         total_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer,
                    warmup_steps: int = 1000,
                    total_steps: int = 150000,
                    min_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    def fn(step: int) -> float:
        return warmup_cosine_lambda(step, warmup_steps, total_steps, min_ratio)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
