from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from . import sampling


def negative_mean_noise(real: np.ndarray, n: int, rng: np.random.Generator,
                        noise_scale: float = 1.0) -> np.ndarray:
    mu = real.mean(axis=0)
    sd = real.std(axis=0) * float(noise_scale)
    return (mu[None, :] + rng.standard_normal((n, real.shape[1])).astype(np.float32)
            * sd[None, :]).astype(np.float32)


def negative_shuffle_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(labels).copy()
    rng.shuffle(out)
    return out


def negative_shuffle_tp(by_tp: Dict[int, np.ndarray],
                        rng: np.random.Generator) -> Dict[int, np.ndarray]:
    tps = sorted(by_tp)
    sizes = [len(by_tp[t]) for t in tps]
    pool = np.concatenate([by_tp[t] for t in tps], axis=0)
    perm = rng.permutation(len(pool))
    out, off = {}, 0
    for t, k in zip(tps, sizes):
        out[t] = np.ascontiguousarray(pool[perm[off:off + k]])
        off += k
    return out


def positive_split_half(real: np.ndarray, n: int, rng: np.random.Generator):
    return sampling.split_half(real, n, rng)


def build_arms(real: np.ndarray, n: int, seed: int = 0,
               noise_scale: float = 1.0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b = positive_split_half(real, n, rng)
    return {
        "neg_mean_noise": negative_mean_noise(real, n, rng, noise_scale),
        "pos_half_a": a,
        "pos_half_b": b,
    }


#:
ARM_ORDER: Sequence[str] = (
    "neg_mean_noise", "neg_shuffle",
    "PI-SDE", "scIMF", "scNODE", "MIOFlow", "PRESCIENT", "scDiffEq",
    "scDiTT",
    "vae_recon",
    "pos_split_half",
)


def sort_arms(names: Sequence[str]) -> list:
    idx = {k: i for i, k in enumerate(ARM_ORDER)}
    return sorted(names, key=lambda x: (idx.get(x, len(idx)), x))


__all__ = ["negative_mean_noise", "negative_shuffle_labels", "negative_shuffle_tp",
           "positive_split_half", "build_arms", "ARM_ORDER", "sort_arms"]
