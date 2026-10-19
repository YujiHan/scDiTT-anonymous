#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train_model
from src.utils.config import build_config, get_argparser, save_config


def main():
    parser = get_argparser()
    args, extras = parser.parse_known_args()

    cfg = build_config(args, extras=extras)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    train_model(cfg)


if __name__ == "__main__":
    main()
