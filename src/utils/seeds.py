from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python random / numpy / torch (cpu + cuda) together.

    This task involves 4 sources of randomness:
    1) diffusion timestep sampling
    2) noise epsilon
    3) categorical sampling of cells under the OT coupling
    4) training mask sampling
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)
