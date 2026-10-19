"""Model helpers: parameter counting, checkpoint serialisation, etc."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema,
    optimizer,
    scheduler,
    step: int,
    config: dict,
    extra: dict | None = None,
    save_optimizer: bool = True,
) -> None:
    """Save a checkpoint.

    With save_optimizer=False the optimiser state is omitted (the two AdamW moment
    buffers are together roughly twice the model size). Since best.pt is used only
    for inference and evaluation, and never for resuming, it is saved with
    save_optimizer=False by default, which halves the file size
    (~723 MB -> ~362 MB). Checkpoints named step_*.pt still store the full state so
    that training can be resumed from them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_opt = save_optimizer and optimizer is not None
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if save_opt else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(str(path), map_location=map_location)
