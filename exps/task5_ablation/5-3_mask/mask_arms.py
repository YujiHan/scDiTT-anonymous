from __future__ import annotations

import numpy as np

from src.data.mask_sampler import DEFAULT_WEIGHTS, MaskSampler

FULL_W: float = DEFAULT_WEIGHTS["full"]      # 0.05
REST_W: float = 1.0 - FULL_W                 # 0.95

PATTERNS = ["uniform", "tail", "block", "strided", "full"]


def _w(uniform: float = 0.0, tail: float = 0.0, block: float = 0.0,
       strided: float = 0.0, full: float = FULL_W) -> dict[str, float]:
    return {"uniform": uniform, "tail": tail, "block": block,
            "strided": strided, "full": full}


ARM_WEIGHTS: dict[str, dict[str, float]] = {
    "mix": dict(DEFAULT_WEIGHTS),
    "random": _w(uniform=REST_W),
    "tail_only": _w(tail=REST_W),
    "scatter_only": _w(strided=REST_W),
}

ARM_DESC: dict[str, str] = {
    "mix": "full model (ours): uniform .30 / tail .25 / block .25 / strided .15 / full .05",
    "random": "purely random mask (MAE style): uniform .95 / full .05",
    "tail_only": "extrapolation-specific: tail .95 / full .05",
    "scatter_only": "interpolation-specific: pure-strided .95 / full .05 (tail union removed)",
}

ARM_MATCH: dict[str, dict[str, str]] = {
    "mix":          {"interpolation": "mix",        "extrapolation": "mix"},
    "random":       {"interpolation": "agnostic",   "extrapolation": "agnostic"},
    "tail_only":    {"interpolation": "mismatched", "extrapolation": "matched"},
    "scatter_only": {"interpolation": "matched",    "extrapolation": "mismatched"},
}


class PureStridedMaskSampler(MaskSampler):

    def _sample_strided(self) -> np.ndarray:
        L = len(self.train_tps)
        mask = np.zeros(self.T, dtype=bool)
        if L == 0:
            return mask
        stride = int(self.rng.choice([2, 3]))
        offset = int(self.rng.integers(0, stride))
        for i, t in enumerate(self.train_tps):
            if (i + offset) % stride == 0:
                mask[t] = True
        if not mask.any():
            mask[self.train_tps[int(self.rng.integers(0, L))]] = True
        return mask


def build_sampler(arm: str, train_tps: list[int], T: int,
                  rng: np.random.Generator | None = None) -> MaskSampler:
    if arm not in ARM_WEIGHTS:
        raise ValueError(f"unknown arm: {arm} (available: {list(ARM_WEIGHTS)})")
    cls = PureStridedMaskSampler if arm == "scatter_only" else MaskSampler
    return cls(train_tps=list(train_tps), T=int(T),
               weights=ARM_WEIGHTS[arm], rng=rng)


def sampler_factory(arm: str):
    def _make(train_tps, T, weights=None, rng=None):  # noqa: ANN001
        return build_sampler(arm, train_tps=train_tps, T=T, rng=rng)
    _make.__name__ = f"MaskSampler[{arm}]"
    return _make
