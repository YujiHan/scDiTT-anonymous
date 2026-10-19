from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

from src.data.dataset import SCTrajectoryDataset
from src.data.mask_sampler import MaskSampler
from src.data.coupling import precompute_all_couplings


@pytest.fixture
def synthetic_latents(tmp_path: Path):
    T = 6
    D = 8
    n_per_tp = 50
    rng = np.random.default_rng(0)
    Z_parts = []
    tps = []
    for t in range(T):
        base = rng.normal(loc=float(t), scale=0.5, size=(n_per_tp, D)).astype(np.float32)
        Z_parts.append(base)
        tps.extend([t] * n_per_tp)
    Z = np.concatenate(Z_parts, axis=0)
    import pandas as pd
    adata = ad.AnnData(X=Z,
                       obs=pd.DataFrame({"tp": tps}),
                       obsm={"X_latent": Z})
    latents_path = tmp_path / "toy_latents.h5ad"
    adata.write_h5ad(latents_path)
    return latents_path, T, D


@pytest.fixture
def synthetic_coupling_cache(synthetic_latents, tmp_path):
    latents_path, T, D = synthetic_latents
    train_tps = [0, 2, 4]
    coupling_dir = tmp_path / "coupling"
    coupling_dir.mkdir(exist_ok=True)
    precompute_all_couplings(
        latents_path=str(latents_path),
        train_tps=train_tps,
        output_dir=str(coupling_dir),
        time_key="tp",
        epsilon=0.05,
        normalize_cost=True,
        max_cells=100,
        seed=0,
    )
    return latents_path, coupling_dir, train_tps, T, D


def test_dataset_shape(synthetic_coupling_cache):
    latents_path, coupling_dir, train_tps, T, D = synthetic_coupling_cache
    test_tps = [1, 3, 5]
    N = 12
    ds = SCTrajectoryDataset(
        latents_path=str(latents_path),
        coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps,
        test_tps=test_tps,
        T=T,
        N=N,
        time_key="tp",
        length=50,
        seed=0,
    )
    item = ds[0]
    assert item["x"].shape == (T, N, D)
    assert item["x"].dtype == torch.float32
    assert item["obs_mask"].shape == (T,)
    assert item["obs_mask"].dtype == torch.bool
    assert item["loss_mask"].shape == (T,)
    assert item["t_real"].shape == (T,)


def test_mask_disjoint_and_test_tps_never_observed(synthetic_coupling_cache):
    latents_path, coupling_dir, train_tps, T, D = synthetic_coupling_cache
    test_tps = [1, 3, 5]
    ds = SCTrajectoryDataset(
        latents_path=str(latents_path),
        coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps,
        test_tps=test_tps,
        T=T,
        N=8,
        time_key="tp",
        length=50,
        seed=1,
    )
    for i in range(20):
        item = ds[i]
        obs_m = item["obs_mask"].numpy()
        loss_m = item["loss_mask"].numpy()
        assert not np.any(obs_m & loss_m)
        for t in range(T):
            if t in test_tps:
                assert not obs_m[t]
                assert not loss_m[t]


def test_stochasticity(synthetic_coupling_cache):
    latents_path, coupling_dir, train_tps, T, D = synthetic_coupling_cache
    ds = SCTrajectoryDataset(
        latents_path=str(latents_path),
        coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps,
        test_tps=[1, 3, 5],
        T=T,
        N=8,
        time_key="tp",
        length=50,
        seed=7,
    )
    a = ds[0]["x"]
    b = ds[0]["x"]
    assert not torch.allclose(a, b)


def test_full_mask_gives_empty_effective_obs(synthetic_coupling_cache):
    latents_path, coupling_dir, train_tps, T, D = synthetic_coupling_cache
    ms = MaskSampler(train_tps, T, weights={"uniform": 0, "tail": 0,
                                             "block": 0, "strided": 0, "full": 1},
                     rng=np.random.default_rng(0))
    ds = SCTrajectoryDataset(
        latents_path=str(latents_path),
        coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps,
        test_tps=[1, 3, 5],
        T=T,
        N=8,
        time_key="tp",
        mask_sampler=ms,
        length=50,
        seed=0,
    )
    item = ds[0]
    assert not item["obs_mask"].any()
    for t in train_tps:
        assert item["loss_mask"][t]
