"""Inference entry point: repeated masked inpainting + concat, yielding the latent predictions at test_tps."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..data.splits import t_real_from_obs
from ..diffusion.sampling import masked_diffusion_inpaint


@torch.no_grad()
def predict_test_tps_from_adata(
    model: nn.Module,
    vae,
    latents_path: str | Path,
    h5ad_path: str | Path,
    train_tps: list[int],
    test_tps: list[int],
    T: int,
    time_key: str,
    scheduler,
    real_time_key: Optional[str] = None,
    N: int = 256,
    num_repeats: int = 12,
    cfg_weight: float = 3.0,
    num_steps: int = 20,
    device: torch.device = torch.device("cuda"),
    eval_seed: int = 0,
    n_eval_cells: int = 2000,
    predict_all_tps: bool = False,
) -> dict:
    """Given a model, return the predictions (gene space) + the real data (gene space).

    predict_all_tps=False (default): produce predictions for test_tps only (used for validation, saves compute).
    predict_all_tps=True: produce predictions for *all* time points (range(T), train + test),
        with exactly n_eval_cells cells per tp (down-sampled without replacement if there are more,
        up-sampled with replacement if there are fewer).

    n_eval_cells defaults to the 2000 of the evaluation protocol: num_repeats draws
    of N cells are pooled and reduced to that many per stage, which is the number
    every baseline delivers as well (supplementary material, Training Details ->
    "Couplings and Sampling"). At the paper's setting, 12 x 256 = 3072 > 2000, so
    no cell is ever drawn twice.

    Returns dict:
        'predictions_latent': {t: [n_eval_cells, D_latent]}
        'predictions_gene':   {t: [n_eval_cells, G]}
        'true_gene_list':     {t: [N_t, G]} -- read from the original gene space of h5ad_path
        'true_gene_concat':   [N_total, G] -- concat of the real gene space over all time points
    """
    model.eval()
    adata_lat = ad.read_h5ad(str(latents_path))
    if "X_latent" in adata_lat.obsm:
        Z_all = np.asarray(adata_lat.obsm["X_latent"], dtype=np.float32)
    else:
        Z_all = np.asarray(adata_lat.X, dtype=np.float32)
    tps_col_lat = np.asarray(adata_lat.obs[time_key].values).astype(float)
    Z_by_tp: dict[int, np.ndarray] = {}
    for t in range(T):
        mask = np.isclose(tps_col_lat, float(t))
        idx = np.where(mask)[0]
        Z_by_tp[t] = Z_all[idx] if len(idx) > 0 else Z_all[:0]

    adata_gene = ad.read_h5ad(str(h5ad_path))
    X = adata_gene.X
    if sp.issparse(X):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X, dtype=np.float32)
    tps_col_gene = np.asarray(adata_gene.obs[time_key].values).astype(float)

    true_gene_list: dict[int, np.ndarray] = {}
    for t in range(T):
        mask = np.isclose(tps_col_gene, float(t))
        idx = np.where(mask)[0]
        true_gene_list[t] = X_dense[idx]

    # Must match training exactly: the same column selects the values fed to
    # PE(t_k) there (src/data/splits.py::t_real_from_obs).
    t_real_vals = np.asarray(
        t_real_from_obs(adata_lat.obs, time_key=time_key,
                        real_time_key=real_time_key), dtype=np.float32)
    if len(t_real_vals) != T:
        t_real_vals = t_real_vals[:T]
    t_real = torch.from_numpy(t_real_vals).to(device)

    # --- obs_mask ---
    obs_mask_full = torch.zeros(1, T, dtype=torch.bool, device=device)
    for t in train_tps:
        obs_mask_full[0, t] = True

    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed)
    rng_np = np.random.default_rng(eval_seed)

    target_tps = list(range(T)) if predict_all_tps else list(test_tps)

    pred_latent_per_tp: dict[int, list[np.ndarray]] = {t: [] for t in target_tps}

    for rep in range(num_repeats):
        x_obs = torch.zeros(1, T, N, Z_all.shape[1],
                            dtype=torch.float32, device=device)
        for t in train_tps:
            Zt = Z_by_tp.get(t, None)
            if Zt is None or len(Zt) == 0:
                continue
            replace = N > len(Zt)
            sel = rng_np.choice(len(Zt), N, replace=replace)
            x_obs[0, t] = torch.from_numpy(Zt[sel]).to(device)

        # Inpaint
        x_final = masked_diffusion_inpaint(
            model=model,
            x_obs=x_obs,
            obs_mask=obs_mask_full,
            t_real=t_real,
            scheduler=scheduler,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            device=device,
            rng_generator=gen,
        )  # [1, T, N, D_latent]

        for t in target_tps:
            pred_latent_per_tp[t].append(x_final[0, t].cpu().numpy())

    predictions_latent: dict[int, np.ndarray] = {}
    for t in target_tps:
        arr = np.concatenate(pred_latent_per_tp[t], axis=0)
        if arr.shape[0] != n_eval_cells:
            replace = arr.shape[0] < n_eval_cells
            sel = rng_np.choice(arr.shape[0], n_eval_cells, replace=replace)
            arr = arr[sel]
        predictions_latent[t] = arr.astype(np.float32)

    # --- decode ---
    predictions_gene: dict[int, np.ndarray] = {}
    if vae is not None:
        from ..data.vae import decode_all
        for t in target_tps:
            predictions_gene[t] = decode_all(vae, predictions_latent[t],
                                             device=str(device))
    else:
        for t in target_tps:
            predictions_gene[t] = predictions_latent[t]

    true_gene_concat = np.concatenate(
        [true_gene_list[t] for t in range(T) if len(true_gene_list[t]) > 0],
        axis=0)

    return {
        "predictions_latent": predictions_latent,
        "predictions_gene": predictions_gene,
        "true_gene_list": true_gene_list,
        "true_gene_concat": true_gene_concat,
    }
