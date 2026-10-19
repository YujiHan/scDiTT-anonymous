from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_CAP = 2000
MIN_CELLS = 300


def n_star(n_real: int, cap: int = DEFAULT_CAP) -> int:
    return int(min(cap, n_real // 2))


def plan(real_by_tp: Dict[int, np.ndarray], cap: int = DEFAULT_CAP,
         min_cells: int = MIN_CELLS) -> Tuple[Dict[int, int], Dict[int, str]]:
    keep, skip = {}, {}
    for t, X in sorted(real_by_tp.items()):
        n = n_star(len(X), cap)
        if n < min_cells:
            skip[t] = f"n_star={n} < {min_cells} (real cells {len(X)})"
        else:
            keep[t] = n
    return keep, skip


def subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n >= len(X):
        return np.ascontiguousarray(X)
    idx = rng.choice(len(X), size=n, replace=False)
    return np.ascontiguousarray(X[idx])


def split_half(X: np.ndarray, n: int, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) < 2 * n:
        raise ValueError(f"the positive control needs 2x{n}={2*n} cells, but only {len(X)}; "
                         "use sampling.n_star() to compute n")
    idx = rng.permutation(len(X))
    return (np.ascontiguousarray(X[idx[:n]]),
            np.ascontiguousarray(X[idx[n:2 * n]]))


def align_arms(arms: Dict[str, np.ndarray], n: int,
               seed: int = 0) -> Dict[str, np.ndarray]:
    out = {}
    for i, (name, X) in enumerate(sorted(arms.items())):
        out[name] = subsample(X, n, np.random.default_rng(seed + 1000 * i))
    return out


__all__ = ["DEFAULT_CAP", "MIN_CELLS", "n_star", "plan",
           "subsample", "split_half", "align_arms"]
