from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str = "scdit", log_file: Optional[str | Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class TBLogger:

    def __init__(self, log_dir: Optional[str | Path], enabled: bool = True):
        self.enabled = enabled and log_dir is not None
        if self.enabled:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag: str, values: dict[str, Any], step: int) -> None:
        if self.enabled:
            self.writer.add_scalars(tag, values, step)

    def close(self) -> None:
        if self.enabled:
            self.writer.close()
