from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import SCTrajectoryDataset
from src.data.mask_sampler import MaskSampler
from src.data.coupling import precompute_all_couplings
from src.diffusion.losses import diffusion_loss
from src.diffusion.sampling import masked_diffusion_inpaint
from src.diffusion.schedulers import FlowMatchingScheduler
from src.model.dit import SCDiT


@pytest.fixture
def setup(tmp_path: Path):
    torch.manual_seed(0)
    np.random.seed(0)
    T, D, n_per_tp = 6, 8, 40
    rng = np.random.default_rng(0)
    Z_parts, tps = [], []
    for t in range(T):
        Z_parts.append(rng.normal(loc=float(t), scale=0.3,
                                  size=(n_per_tp, D)).astype(np.float32))
        tps.extend([t] * n_per_tp)
    Z = np.concatenate(Z_parts, axis=0)
    latents_path = tmp_path / "lat.h5ad"
    ad.AnnData(X=Z, obs=pd.DataFrame({"tp": tps}),
               obsm={"X_latent": Z}).write_h5ad(latents_path)

    train_tps, test_tps = [0, 2, 4], [1, 3, 5]
    coupling_dir = tmp_path / "coupling"
    coupling_dir.mkdir()
    precompute_all_couplings(
        latents_path=str(latents_path),
        train_tps=train_tps,
        output_dir=str(coupling_dir),
        time_key="tp",
        epsilon=0.1, normalize_cost=True, max_cells=60, seed=0,
    )
    return latents_path, coupling_dir, train_tps, test_tps, T, D


def test_training_step_flow_matching(setup):
    latents_path, coupling_dir, train_tps, test_tps, T, D = setup
    device = torch.device("cpu")
    ms = MaskSampler(train_tps, T, rng=np.random.default_rng(1))
    ds = SCTrajectoryDataset(
        latents_path=str(latents_path),
        coupling_cache_dir=str(coupling_dir),
        train_tps=train_tps, test_tps=test_tps,
        T=T, N=12, time_key="tp",
        mask_sampler=ms, length=100, seed=0,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0,
                                         shuffle=False)
    model = SCDiT(D_latent=D, D_model=32, num_blocks=2, num_heads=4, T_max=16).to(device)
    sched = FlowMatchingScheduler()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step, batch in enumerate(loader):
        if step >= 3:
            break
        x0 = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        t_real = batch["t_real"][0].to(device)
        B = x0.shape[0]
        tau = sched.sample_timesteps(B, device)
        noise = torch.randn_like(x0)
        xt = sched.add_noise(x0, noise, tau)
        v_t = sched.compute_target(x0, noise, tau)
        v_p = model(xt, tau, obs_mask, t_real)
        ld = diffusion_loss(v_p, v_t, loss_mask)
        loss = ld["total"]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_masked_diffusion_inpaint_flow(setup):
    latents_path, coupling_dir, train_tps, test_tps, T, D = setup
    device = torch.device("cpu")
    model = SCDiT(D_latent=D, D_model=32, num_blocks=2, num_heads=4, T_max=16).to(device)
    model.eval()
    sched = FlowMatchingScheduler()
    B, N = 1, 10
    x_obs = torch.randn(B, T, N, D)
    obs_mask = torch.zeros(B, T, dtype=torch.bool)
    for t in train_tps:
        obs_mask[0, t] = True
    t_real = torch.arange(T, dtype=torch.float32)
    out = masked_diffusion_inpaint(
        model=model, x_obs=x_obs, obs_mask=obs_mask, t_real=t_real,
        scheduler=sched, num_steps=5, cfg_weight=1.0,
        device=device,
    )
    assert out.shape == (B, T, N, D)
    for t in train_tps:
        assert torch.allclose(out[0, t], x_obs[0, t], atol=1e-5)
