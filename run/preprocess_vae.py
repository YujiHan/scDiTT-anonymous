#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import TIME_KEY, cache_paths, get_tp_split, train_cell_mask
from src.data.vae import encode_all, save_vae, train_vae
from src.utils.logging import get_logger
from src.utils.seeds import set_all_seeds


def load_config_for_dataset(ds: str) -> dict:
    import yaml
    with open(Path(__file__).resolve().parent.parent /
              "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    with open(Path(__file__).resolve().parent.parent /
              "configs" / "dataset" / f"{ds.lower()}.yaml") as f:
        ds_cfg = yaml.safe_load(f) or {}
    # shallow merge on top level is enough here
    cfg["dataset"] = {**(cfg.get("dataset") or {}), **(ds_cfg.get("dataset") or {})}
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=["ZB", "DR", "SC", "GA", "MA",
                            "zb", "dr", "sc", "ga", "ma"])
    p.add_argument("--task", type=str, required=True,
                   choices=["interpolation", "extrapolation",
                            "extrapolation_steps", "all"],
                   help="determines train_tps (the VAE is pre-trained only on those) and the cache naming")
    p.add_argument("--ex_steps", type=int, default=None,
                   help="only for task=extrapolation_steps: number of trailing steps held out as the test set")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_root", type=str, default="results",
                   help="output/cache root directory, used to isolate experiments (e.g. results/task1_ts_pred); "
                        "VAE ckpt/latents land under {results_root}/vae_checkpoints/")
    args = p.parse_args()

    ds_up = args.dataset.upper()
    if args.task == "extrapolation_steps" and args.ex_steps is None:
        p.error("task 'extrapolation_steps' requires --ex_steps N (integer, 0..T-1).")
    cfg = load_config_for_dataset(ds_up)
    h5ad_path = cfg["dataset"]["h5ad_path"]
    time_key = TIME_KEY[ds_up]
    latent_dim = args.latent_dim or cfg["vae"]["latent_dim"]
    epochs = args.epochs or cfg["vae"]["epochs"]
    lr = args.lr or cfg["vae"]["lr"]
    batch_size = args.batch_size or cfg["vae"]["batch_size"]
    beta = args.beta or cfg["vae"]["beta"]
    hidden = tuple(cfg["vae"]["hidden"])

    _paths = cache_paths(ds_up, args.task, args.ex_steps, args.seed,
                         results_root=args.results_root)
    vae_ckpt_path = _paths["vae_ckpt_path"]
    latents_path = _paths["latents_path"]

    set_all_seeds(args.seed)
    log = get_logger("scdit.preprocess_vae",
                     Path(vae_ckpt_path).with_suffix(".log"))
    log.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    log.info(f"X shape: {X.shape}, time_key={time_key}")
    log.info(f"tps unique: {np.sort(np.unique(adata.obs[time_key].values))}")

    train_tps, test_tps = get_tp_split(ds_up, args.task, n_steps=args.ex_steps)
    fit_mask = train_cell_mask(adata.obs[time_key].values, train_tps)
    n_fit = int(fit_mask.sum())
    if n_fit == 0:
        raise ValueError(
            f"No cells fall on train_tps={train_tps} for {ds_up}/{args.task}; "
            f"check obs['{time_key}'] against src/data/splits.py: ALL_TPS.")
    fit_idx = np.flatnonzero(fit_mask)
    log.info(f"train_tps={train_tps} test_tps={test_tps} (held out from VAE)")
    log.info(f"VAE fit on {n_fit}/{X.shape[0]} cells "
             f"({100.0 * n_fit / X.shape[0]:.1f}%); "
             f"{X.shape[0] - n_fit} test-tp cells are encode-only")

    model, hist = train_vae(
        X=X,
        fit_indices=fit_idx,
        gene_dim=X.shape[1],
        latent_dim=latent_dim,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        device=args.device,
        seed=args.seed,
        logger=log,
    )

    save_vae(model, vae_ckpt_path, meta={
        "dataset": ds_up,
        "task": args.task,
        "ex_steps": -1 if args.ex_steps is None else int(args.ex_steps),
        "seed": int(args.seed),
        "train_tps": [int(t) for t in train_tps],
        "n_train_cells": n_fit,
        "n_total_cells": int(X.shape[0]),
    })
    log.info(f"VAE saved to {vae_ckpt_path}")

    Z = encode_all(model, X, device=args.device)
    log.info(f"Encoded latent shape: {Z.shape}")

    obs_out = adata.obs.copy()
    obs_out["vae_train_cell"] = fit_mask
    out = ad.AnnData(
        X=Z.astype(np.float32),
        obs=obs_out,
        obsm={"X_latent": Z.astype(np.float32)},
        uns={"vae_train_tps": [int(t) for t in train_tps]},
    )
    Path(latents_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(latents_path)
    log.info(f"Latents h5ad saved to {latents_path}")


if __name__ == "__main__":
    main()
