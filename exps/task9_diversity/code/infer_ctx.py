from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch

import common as C  # noqa: E402
from src.data.splits import get_tp_split
from src.data.vae import decode_all, load_vae
from src.diffusion.sampling import masked_diffusion_inpaint
from src.diffusion.schedulers import build_scheduler
from src.model.dit import SCDiT
from src.model.model_utils import load_checkpoint
from src.training.ema import EMA


class InferCtx:

    def __init__(self, ds: str, task: str, train_seed: int,
                 device: str = "cuda", use_ema: bool = True):
        self.ds, self.task, self.train_seed = ds, task, train_seed
        self.device = torch.device(device)

        ck_path = C.ckpt_path(ds, task, train_seed)
        ckpt = load_checkpoint(ck_path, map_location=self.device)
        cfg = C.fix_config(ckpt["config"], ds, task, train_seed)
        self.cfg = cfg

        self.model = SCDiT(
            D_latent=cfg["model"]["D_latent"],
            D_model=cfg["model"]["D_model"],
            num_blocks=cfg["model"]["num_blocks"],
            num_heads=cfg["model"]["num_heads"],
            T_max=max(32, cfg["dataset"]["T"]),
        ).to(self.device)
        if use_ema and ckpt.get("ema") is not None:
            ema = EMA(self.model)
            ema.load_state_dict(ckpt["ema"])
            ema.apply_to(self.model)
            self.weights = "ema"
        else:
            self.model.load_state_dict(ckpt["model"])
            self.weights = "raw"
        self.model.eval()
        del ckpt

        vae_ck = cfg["dataset"]["vae_ckpt_path"]
        if not Path(vae_ck).exists():
            raise FileNotFoundError(f"VAE ckpt missing: {vae_ck}")
        self.vae = load_vae(vae_ck, device=str(self.device))

        self.T = int(cfg["dataset"]["T"])
        self.time_key = cfg["dataset"]["time_key"]
        self.train_tps, self.test_tps = get_tp_split(ds, task)
        self.scheduler = build_scheduler(cfg["diffusion"])

        adata_lat = ad.read_h5ad(str(cfg["dataset"]["latents_path"]))
        if "X_latent" in adata_lat.obsm:
            Z_all = np.asarray(adata_lat.obsm["X_latent"], dtype=np.float32)
        else:
            Z_all = np.asarray(adata_lat.X, dtype=np.float32)
        tps_col_lat = np.asarray(adata_lat.obs[self.time_key].values).astype(float)
        self.D_latent = Z_all.shape[1]
        self.Z_by_tp = {}
        for t in range(self.T):
            idx = np.where(np.isclose(tps_col_lat, float(t)))[0]
            self.Z_by_tp[t] = Z_all[idx] if len(idx) > 0 else Z_all[:0]

        # t_real (inference.py:78-82)
        t_real_vals = np.sort(np.unique(tps_col_lat))
        if len(t_real_vals) != self.T:
            t_real_vals = t_real_vals[:self.T]
        self.t_real = torch.from_numpy(t_real_vals.astype(np.float32)).to(self.device)

        # obs_mask (inference.py:84-87)
        self.obs_mask = torch.zeros(1, self.T, dtype=torch.bool, device=self.device)
        for t in self.train_tps:
            self.obs_mask[0, t] = True

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, cfg_weight: float, eval_seed: int,
                 N: int = C.N_EVAL, num_repeats: int = C.NUM_REPEATS,
                 num_steps: int = C.NUM_STEPS,
                 n_eval_cells: int = C.N_PRED_CELLS,
                 return_latent: bool = False):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(eval_seed)
        rng_np = np.random.default_rng(eval_seed)

        target_tps = list(range(self.T))
        pred_latent_per_tp: dict[int, list[np.ndarray]] = {t: [] for t in target_tps}

        for _ in range(num_repeats):
            x_obs = torch.zeros(1, self.T, N, self.D_latent,
                                dtype=torch.float32, device=self.device)
            for t in self.train_tps:
                Zt = self.Z_by_tp.get(t, None)
                if Zt is None or len(Zt) == 0:
                    continue
                replace = N > len(Zt)
                sel = rng_np.choice(len(Zt), N, replace=replace)
                x_obs[0, t] = torch.from_numpy(Zt[sel]).to(self.device)

            x_final = masked_diffusion_inpaint(
                model=self.model, x_obs=x_obs, obs_mask=self.obs_mask,
                t_real=self.t_real, scheduler=self.scheduler,
                num_steps=num_steps, cfg_weight=cfg_weight,
                device=self.device, rng_generator=gen,
            )
            for t in target_tps:
                pred_latent_per_tp[t].append(x_final[0, t].cpu().numpy())

        predictions_gene: dict[int, np.ndarray] = {}
        predictions_latent: dict[int, np.ndarray] = {}
        for t in target_tps:
            arr = np.concatenate(pred_latent_per_tp[t], axis=0)
            if arr.shape[0] != n_eval_cells:
                replace = arr.shape[0] < n_eval_cells
                sel = rng_np.choice(arr.shape[0], n_eval_cells, replace=replace)
                arr = arr[sel]
            arr = arr.astype(np.float32)
            predictions_latent[t] = arr
            predictions_gene[t] = decode_all(self.vae, arr, device=str(self.device))
        if return_latent:
            return predictions_gene, predictions_latent
        return predictions_gene


def save_preds(out_dir: Path, preds: dict[int, np.ndarray],
               test_tps: list[int]) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack([preds[int(t)] for t in test_tps], axis=0).astype(np.float32)
    assert arr.shape[-1] > 1000, (
        f"pred gene dim = {arr.shape[-1]} <= 1000 -- the VAE was not loaded, "
        f"what was written is latent space, not gene space (pitfall 8)")
    np.savez_compressed(out_dir / "preds.npz",
                        pred=arr, tps=np.asarray(test_tps, dtype=np.int64))
    return arr
