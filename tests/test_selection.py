from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.coupling import precompute_all_couplings
from src.data.dataset import SCTrajectoryDataset
from src.data.mask_sampler import MaskSampler
from src.diffusion.schedulers import FlowMatchingScheduler
from src.model.dit import SCDiT
from src.training.callbacks import TrainLossProbe, build_probe_batches
from src.training.ema import EMA


@pytest.fixture
def ds_and_model(tmp_path: Path):
    torch.manual_seed(0)
    T, D, n_per_tp = 6, 8, 40
    rng = np.random.default_rng(0)
    Z = np.concatenate([rng.normal(float(t), 0.3, size=(n_per_tp, D))
                        for t in range(T)]).astype(np.float32)
    tps = np.repeat(np.arange(T), n_per_tp)
    latents_path = tmp_path / "lat.h5ad"
    ad.AnnData(X=Z, obs=pd.DataFrame({"tp": tps}),
               obsm={"X_latent": Z}).write_h5ad(latents_path)

    train_tps, test_tps = [0, 2, 4], [1, 3, 5]
    coupling_dir = tmp_path / "coupling"
    coupling_dir.mkdir()
    precompute_all_couplings(
        latents_path=str(latents_path), train_tps=train_tps,
        output_dir=str(coupling_dir), time_key="tp",
        epsilon=0.1, normalize_cost=True, max_cells=60, seed=0)

    ds = SCTrajectoryDataset(
        latents_path=str(latents_path), coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps, test_tps=test_tps, T=T, N=12, time_key="tp",
        mask_sampler=MaskSampler(train_tps, T, rng=np.random.default_rng(1)),
        length=100, seed=0)
    model = SCDiT(D_latent=D, D_model=32, num_blocks=2, num_heads=4, T_max=16)
    return ds, model


def _probe(ds, seed=0, n_batches=2, batch_size=2):
    return TrainLossProbe(
        batches=build_probe_batches(ds, n_batches, batch_size, seed=seed),
        scheduler=FlowMatchingScheduler(), device=torch.device("cpu"), seed=123)


def test_build_probe_batches_restores_dataset_rng(ds_and_model):
    ds, _ = ds_and_model
    assert ds._rng is None
    batches = build_probe_batches(ds, n_batches=2, batch_size=2, seed=0)
    assert len(batches) == 2
    assert batches[0]["x"].shape[0] == 2
    assert ds._rng is None, "_rng was not restored after probe sampling -> worker seeding would break"


def test_probe_is_reproducible_and_weight_sensitive(ds_and_model):
    ds, model = ds_and_model
    ema = EMA(model, decay=0.9)
    probe = _probe(ds)

    a = probe.run(model, ema, step=0)
    b = probe.run(model, ema, step=1)
    assert a == b, "the probe is not reproducible -> the curve is not comparable across steps and argmin is meaningless"

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ema.update(model)
    c = probe.run(model, ema, step=2)
    assert c != a, "the weights changed but the probe did not -> it is not measuring the parameters that get saved"


def test_probe_does_not_disturb_model_state(ds_and_model):
    ds, model = ds_and_model
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    _probe(ds).run(model, ema, step=0)

    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} was modified after the probe"
    assert model.training, "the model should return to train mode after the probe"


def test_update_best_tracks_minimum(ds_and_model):
    ds, _ = ds_and_model
    probe = _probe(ds)
    assert probe.update_best(1.0, step=10) is True
    assert probe.update_best(2.0, step=20) is False
    assert probe.update_best(0.5, step=30) is True
    assert probe.best == 0.5 and probe.best_step == 30
