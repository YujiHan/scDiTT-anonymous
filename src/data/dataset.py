"""PyTorch Dataset: on-the-fly stochastic trajectory sampling.

Each __getitem__ runs one pass of "start from t_0 and sample categorically along the train_tps chain using
the OT coupling", returning one [T, N, D_latent] trajectory + mask.

Paper reference
---------------
Methodology -> "Stochastic temporal alignment"; supplementary material, Training
Details -> "Couplings and Sampling".

This is where unpaired snapshots become the tensor z in R^{T x N x d} that
Eq. (1) presupposes. Seed cells are drawn uniformly from the first observed
stage and carried forward one stage at a time, each successor sampled from the
corresponding row of the transport kernel pi_bar built in `data/coupling.py`.
A fresh trajectory is drawn at *every* access (hence on-the-fly rather than
cached), so the model never trains on a fixed pairing but on samples from the
trajectory distribution the couplings induce -- this is what "stochastic
alignment prior" means operationally, and it is why the paper reports that
different draws from the same coupling teach population-level temporal
variation rather than a fixed lineage structure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from .mask_sampler import MaskSampler
from .splits import load_t_real
from .coupling import load_coupling


def _sample_categorical_batch(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # clip negative due to numerical noise
    probs = np.clip(probs, 0.0, None)
    # re-normalize if row sum deviates
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-30, 1.0, row_sum)
    probs = probs / row_sum
    cdf = np.cumsum(probs, axis=1)
    r = rng.random(probs.shape[0])[:, None]
    idx = (cdf < r).sum(axis=1)
    idx = np.clip(idx, 0, probs.shape[1] - 1)
    return idx


class SCTrajectoryDataset(Dataset):
    """
    Each __getitem__ returns:
        {
            'x':        [T, N, D_latent]  float32
            'obs_mask': [T]               bool   (effective observed)
            'loss_mask': [T]              bool   (participates in the diffusion loss)
            't_real':   [T]               float32
        }
    """

    def __init__(
        self,
        latents_path: str | Path,
        coupling_cache_dir: str | Path,
        train_tps: list[int],
        test_tps: list[int],
        T: int,
        N: int = 256,
        t_real: Optional[list[float]] = None,
        time_key: str = "tp",
        real_time_key: Optional[str] = None,
        mask_sampler: Optional[MaskSampler] = None,
        pattern_sampler: Optional[Callable[..., tuple[np.ndarray, np.ndarray]]] = None,
        length: int = 100000,
        seed: int = 0,
    ):
        super().__init__()
        self.latents_path = Path(latents_path)
        self.coupling_cache_dir = Path(coupling_cache_dir)
        self.train_tps = list(train_tps)
        self.test_tps = list(test_tps)
        self.T = T
        self.N = N
        self.time_key = time_key
        self.length = length

        if t_real is None:
            t_real = load_t_real(self.latents_path, time_key=time_key,
                                 real_time_key=real_time_key)
        if len(t_real) != T:
            print(f"[Dataset] WARN: len(t_real)={len(t_real)} != T={T}. "
                  f"Using first T values.")
            t_real = list(t_real)[:T]
        self.t_real = np.asarray(t_real, dtype=np.float32)

        adata = ad.read_h5ad(str(self.latents_path))
        if "X_latent" in adata.obsm:
            Z_all = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
        else:
            Z_all = np.asarray(adata.X, dtype=np.float32)
        self.D_latent = Z_all.shape[1]
        tps_col = np.asarray(adata.obs[time_key].values).astype(float)

        self.Z: dict[int, np.ndarray] = {}
        self.global_idx: dict[int, np.ndarray] = {}
        for t in range(T):
            mask_t = np.isclose(tps_col, float(t))
            gidx = np.where(mask_t)[0]
            if len(gidx) == 0:
                mask_t = np.isclose(tps_col, float(self.t_real[t]))
                gidx = np.where(mask_t)[0]
            self.global_idx[t] = gidx.astype(np.int64)
            self.Z[t] = Z_all[gidx]

        # self.pi[(t_a, t_b)] = [N_a', N_b']
        self.pi: dict[tuple[int, int], np.ndarray] = {}
        self.idx_global_to_sub_a: dict[tuple[int, int], np.ndarray] = {}
        self.idx_sub_to_global_b: dict[tuple[int, int], np.ndarray] = {}

        meta_path = self.coupling_cache_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = None

        for (t_a, t_b) in zip(self.train_tps[:-1], self.train_tps[1:]):
            npz_path = self.coupling_cache_dir / f"pi_{t_a}_to_{t_b}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Missing coupling {npz_path}. "
                    f"Did you run run/preprocess_coupling.py for this (dataset, task)?"
                )
            d = load_coupling(npz_path)
            pi = d["pi"].astype(np.float32)
            idx_a_global = d["idx_a_global"].astype(np.int64)
            idx_b_global = d["idx_b_global"].astype(np.int64)
            self.pi[(t_a, t_b)] = pi

            n_global = Z_all.shape[0]
            g2sub = -np.ones(n_global, dtype=np.int64)
            for sub_i, g in enumerate(idx_a_global):
                g2sub[g] = sub_i
            self.idx_global_to_sub_a[(t_a, t_b)] = g2sub
            self.idx_sub_to_global_b[(t_a, t_b)] = idx_b_global

        self._Z_all = Z_all

        # mask sampler
        if mask_sampler is None:
            mask_sampler = MaskSampler(train_tps=self.train_tps, T=T,
                                       rng=np.random.default_rng(seed))
        self.mask_sampler = mask_sampler

        # Optional override of the observation/generation split. scDiTT itself
        # leaves this None and uses `mask_sampler`, whose two masks partition the
        # observed stages. The autoregressive variants need a third category --
        # stages that are neither evidence nor target -- so they supply the two
        # masks directly and the remaining slots are zeroed before noise is
        # added, exactly as they are empty at inference time.
        self.pattern_sampler = pattern_sampler

        self._base_seed = seed
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return self.length

    def _get_rng(self, idx: int) -> np.random.Generator:
        if self._rng is None:
            # torch worker info
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self._base_seed + wid * 10007)
            self.mask_sampler.rng = self._rng
        return self._rng

    def __getitem__(self, idx: int) -> dict:
        rng = self._get_rng(idx)

        # Paper: the N seed cells are drawn uniformly from the first observed
        # stage, and successors are then sampled recursively from the transport
        # kernels (Methodology -> "Stochastic temporal alignment"; the uniform
        # draw is what the marginal-preservation argument of the supplementary
        # material, Training Details -> "Couplings and Sampling", rests on). N is
        # the per-dataset cells-per-stage of Training Details -> "Per-Dataset
        # Settings", listed per dataset in the runtime Table S17.
        t0 = self.train_tps[0]
        Z_t0 = self.Z[t0]
        N_t0 = Z_t0.shape[0]
        replace = self.N > N_t0
        local_seed = rng.choice(N_t0, self.N, replace=replace)
        current_global = self.global_idx[t0][local_seed]

        traj = np.zeros((self.T, self.N, self.D_latent), dtype=np.float32)
        traj[t0] = self._Z_all[current_global]

        # Paper: each seed cell is carried forward one stage at a time, its
        # successor at k' drawn from the kernel row pi_bar[i, :] of Eq. (2).
        # Stacking the visited latents gives one trajectory z ∈ R^{T x N x d},
        # the tensor Eq. (1) presupposes, whose k-th slice is an unordered set
        # of N cells.
        for (t_a, t_b) in zip(self.train_tps[:-1], self.train_tps[1:]):
            pi = self.pi[(t_a, t_b)]
            g2sub_a = self.idx_global_to_sub_a[(t_a, t_b)]
            sub2g_b = self.idx_sub_to_global_b[(t_a, t_b)]

            rows = g2sub_a[current_global]

            bad = (rows < 0)
            if bad.any():
                rows = rows.copy()
                fallback = rng.integers(0, pi.shape[0], size=int(bad.sum()))
                rows[bad] = fallback

            probs_rows = pi[rows]  # [N, N_b']
            next_sub = _sample_categorical_batch(probs_rows, rng)  # [N]
            next_global = sub2g_b[next_sub]  # [N]

            traj[t_b] = self._Z_all[next_global]
            current_global = next_global


        if self.pattern_sampler is not None:
            effective_obs_mask, loss_mask = self.pattern_sampler(
                rng, self.train_tps, self.T)
            effective_obs_mask = np.asarray(effective_obs_mask, dtype=bool)
            loss_mask = np.asarray(loss_mask, dtype=bool)
            if (effective_obs_mask & loss_mask).any():
                raise ValueError(
                    "pattern_sampler returned overlapping obs/loss masks; a stage "
                    "is either evidence or a target, never both")
            # Stages in neither role carry no information at inference, so they
            # carry none here either: zeroed before the noise is added.
            traj[~(effective_obs_mask | loss_mask)] = 0.0
        else:
            training_mask = self.mask_sampler.sample()  # [T] bool

            obs_mask = np.zeros(self.T, dtype=bool)
            for t in self.train_tps:
                obs_mask[t] = True

            # Paper: the observation mask m and the generation/loss mask l of
            # Eq. (4) are complementary by construction (l AND m = 0,
            # l OR m = 1_O), so every stage is either used as conditioning
            # information or reconstructed by the model, never both; the truly
            # held-out stages appear in neither.
            effective_obs_mask = obs_mask & ~training_mask
            loss_mask = obs_mask & training_mask

        return {
            "x": torch.from_numpy(traj),                                # [T, N, D_latent]
            "obs_mask": torch.from_numpy(effective_obs_mask),            # [T]
            "loss_mask": torch.from_numpy(loss_mask),                    # [T]
            "t_real": torch.from_numpy(self.t_real.copy()),              # [T]
        }
