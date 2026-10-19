from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE_ROOT, REPO_ROOT, build_cfg, run_dir

os.chdir(REPO_ROOT)
from src.data.splits import get_tp_split
from src.diffusion.schedulers import build_scheduler
from src.diffusion.sampling import masked_diffusion_inpaint

CELLTYPE_KEYS = ["celltype", "cell_type", "lineage"]


def load_ctx(ds: str, task: str, seed: int, arm: str = "full",
             cache_root: str | None = None,
             device: str = "cuda") -> dict:
    cfg = build_cfg(ds, task, seed, arm, cache_root=cache_root or CACHE_ROOT)
    T = cfg["dataset"]["T"]
    time_key = cfg["dataset"]["time_key"]
    train_tps, test_tps = get_tp_split(ds, task)

    adata = ad.read_h5ad(cfg["dataset"]["latents_path"])
    Z = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    tp_col = np.asarray(adata.obs[time_key].values).astype(float)

    ct_key = next((k for k in CELLTYPE_KEYS if k in adata.obs.columns), None)
    ct_all = (np.asarray(adata.obs[ct_key].values).astype(str)
              if ct_key else None)

    Z_by_tp, ct_by_tp, rows_by_tp = {}, {}, {}
    for t in range(T):
        idx = np.where(np.isclose(tp_col, float(t)))[0]
        rows_by_tp[t] = idx
        Z_by_tp[t] = Z[idx]
        ct_by_tp[t] = ct_all[idx] if ct_all is not None else None

    t_real = torch.from_numpy(
        np.sort(np.unique(tp_col))[:T].astype(np.float32)).to(device)

    return {
        "cfg": cfg, "ds": ds.upper(), "task": task, "seed": int(seed),
        "T": T, "train_tps": list(train_tps), "test_tps": list(test_tps),
        "Z_by_tp": Z_by_tp, "ct_by_tp": ct_by_tp, "rows_by_tp": rows_by_tp,
        "celltype_key": ct_key, "t_real": t_real,
        "N": int(cfg["data"]["N"]), "D": int(Z.shape[1]),
        "device": torch.device(device),
        "scheduler": build_scheduler(cfg["diffusion"]),
        "num_steps": int(cfg["inference"]["num_steps"]),
        "cfg_weight": float(cfg["inference"]["cfg_weight"]),
    }


def sample_condition(ctx: dict, rng: np.random.Generator) -> tuple[torch.Tensor, dict]:
    T, N, D = ctx["T"], ctx["N"], ctx["D"]
    x_obs = torch.zeros(1, T, N, D, dtype=torch.float32, device=ctx["device"])
    sel_rows: dict[int, np.ndarray] = {}
    for t in ctx["train_tps"]:
        Zt = ctx["Z_by_tp"][t]
        if len(Zt) == 0:
            continue
        sel = rng.choice(len(Zt), N, replace=N > len(Zt))
        sel_rows[t] = sel
        x_obs[0, t] = torch.from_numpy(Zt[sel]).to(ctx["device"])
    return x_obs, sel_rows


def obs_mask_of(ctx: dict) -> torch.Tensor:
    m = torch.zeros(1, ctx["T"], dtype=torch.bool, device=ctx["device"])
    for t in ctx["train_tps"]:
        m[0, t] = True
    return m


@torch.no_grad()
def inpaint(ctx: dict, model, x_obs: torch.Tensor, noise_seed: int,
            cfg_weight: float | None = None) -> torch.Tensor:
    gen = torch.Generator(device=ctx["device"])
    gen.manual_seed(int(noise_seed))
    return masked_diffusion_inpaint(
        model=model, x_obs=x_obs, obs_mask=obs_mask_of(ctx),
        t_real=ctx["t_real"], scheduler=ctx["scheduler"],
        num_steps=ctx["num_steps"],
        cfg_weight=ctx["cfg_weight"] if cfg_weight is None else cfg_weight,
        device=ctx["device"], rng_generator=gen,
    )


def load_model(ctx: dict, arm: str, use_ema: bool = True):
    from blocks_ablation import load_arm_model
    ckpt = run_dir(ctx["ds"], ctx["task"], ctx["seed"], arm) / "best.pt"
    if not ckpt.exists():
        return None
    cfg = build_cfg(ctx["ds"], ctx["task"], ctx["seed"], arm)
    return load_arm_model(arm, cfg, ckpt, device=str(ctx["device"]),
                          use_ema=use_ema)


def nearest_observed_tp(ctx: dict, t: int) -> int:
    return min(ctx["train_tps"], key=lambda o: (abs(o - t), o > t))
