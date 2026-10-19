"""Training-time mask sampling: a weighted mixture of 5 patterns.

Emits training_mask of shape [T], bool:
    True  = this train_tps slot is masked out for this sample (participates in the loss)
    False = this train_tps slot stays observed for this sample, or it is a test_tps slot

test_tps positions are permanently set to False at the obs_mask level by the Dataset; that is not handled
here.

Paper reference
---------------
Methodology -> "Temporal mask strategy"; supplementary material, Training
Details -> "Masking Curriculum".

The mixture below is what makes one objective serve both interpolation and
extrapolation: a model trained this way is never specialised to a particular
hole pattern, so a single checkpoint covers both tasks and horizons it was not
explicitly trained for. `DEFAULT_WEIGHTS` are the fixed weights of the paper,
used unchanged across all five datasets and both tasks. Comparing this mixture
against purely random masking and against task-specific masking is Table S6.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# Paper: the fixed mixture of supplementary material -> "Masking Curriculum",
# used unchanged across all five datasets and both tasks. Each pattern supplies
# a distinct training signal (Methodology -> "Temporal mask strategy"):
#   uniform — hides observed stages independently, general conditional generation
#   tail    — a trailing run, the extrapolation signal
#   block   — an interior run, the interpolation signal
#   strided — forces long-range dependence on distant observed stages
#   full    — empties the mask, supplying the unconditional branch that
#             classifier-free guidance requires at test time
DEFAULT_WEIGHTS: dict[str, float] = {
    "uniform": 0.30,
    "tail":    0.25,
    "block":   0.25,
    "strided": 0.15,
    "full":    0.05,
}


class MaskSampler:
    def __init__(
        self,
        train_tps: list[int],
        T: int,
        weights: Optional[dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.train_tps = list(train_tps)
        self.T = T
        w = DEFAULT_WEIGHTS if weights is None else weights
        self.pattern_names = ["uniform", "tail", "block", "strided", "full"]
        w_arr = np.array([w[n] for n in self.pattern_names], dtype=np.float64)
        self.pattern_probs = w_arr / w_arr.sum()
        self.rng = rng if rng is not None else np.random.default_rng()

    def _sample_uniform(self) -> np.ndarray:
        # Paper: hides each observed stage independently with p ~ U(0.3, 0.8)
        # (supplementary material -> "Masking Curriculum").
        p = self.rng.uniform(0.3, 0.8)
        mask = np.zeros(self.T, dtype=bool)
        for t in self.train_tps:
            if self.rng.random() < p:
                mask[t] = True
        return mask

    def _sample_tail(self) -> np.ndarray:
        L = len(self.train_tps)
        if L <= 1:
            mask = np.zeros(self.T, dtype=bool)
            if L == 1:
                mask[self.train_tps[0]] = True
            return mask
        k_min = (L + 1) // 2
        k_max = L - 1
        k = int(self.rng.integers(k_min, k_max + 1))
        cutoff_idx = k
        mask = np.zeros(self.T, dtype=bool)
        for t in self.train_tps[cutoff_idx:]:
            mask[t] = True
        return mask

    def _sample_block(self) -> np.ndarray:
        L = len(self.train_tps)
        max_len = max(2, L // 2)
        if L < 2:
            mask = np.zeros(self.T, dtype=bool)
            if L == 1:
                mask[self.train_tps[0]] = True
            return mask
        length = int(self.rng.integers(2, max_len + 1))
        start = int(self.rng.integers(0, L - length + 1))
        mask = np.zeros(self.T, dtype=bool)
        for t in self.train_tps[start:start + length]:
            mask[t] = True
        return mask

    def _sample_strided(self) -> np.ndarray:
        L = len(self.train_tps)
        stride = int(self.rng.choice([2, 3]))
        offset = int(self.rng.integers(0, stride))
        mask = np.zeros(self.T, dtype=bool)
        for i, t in enumerate(self.train_tps):
            if (i + offset) % stride == 0:
                mask[t] = True
        if self.rng.random() < 0.5 and L >= 2:
            mask = mask | self._sample_tail()
        return mask

    def _sample_full(self) -> np.ndarray:
        mask = np.zeros(self.T, dtype=bool)
        for t in self.train_tps:
            mask[t] = True
        return mask

    def sample(self) -> np.ndarray:
        """Returns training_mask ∈ [T] bool."""
        idx = int(self.rng.choice(len(self.pattern_names), p=self.pattern_probs))
        name = self.pattern_names[idx]
        fn = getattr(self, f"_sample_{name}")
        return fn()

    def sample_with_name(self) -> tuple[np.ndarray, str]:
        idx = int(self.rng.choice(len(self.pattern_names), p=self.pattern_probs))
        name = self.pattern_names[idx]
        fn = getattr(self, f"_sample_{name}")
        return fn(), name
