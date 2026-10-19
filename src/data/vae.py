"""VAE model + pre-training pipeline.

Compresses gene-space expression into a D_latent latent space. The encoder/decoder are frozen for all
downstream diffusion training.

Paper reference
---------------
Methodology -> "Latent population encoding"; supplementary material, Training
Details -> "Autoencoder".

The encoder E and decoder D of the paper. Following latent diffusion, all
subsequent modelling happens on latents: this keeps the diffusion model away
from the sparsity of raw counts and makes attention over hundreds of cells
tractable. Defaults match the paper -- hidden widths 512 and 256, a
64-dimensional latent, 150 epochs of Adam at 1e-3, batch size 256, KL weight
beta = 0.1. The autoencoder is fitted on the cells of the *training* stages
only and then frozen, and is refitted for every split and every seed rather
than shared across them, which is why the reported spread over five seeds
reflects the full pipeline rather than diffusion training alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellVAE(nn.Module):
    """
    Input:  x in [B, G]   (HVG expression, already normalize_total + log1p)
    Output: z in [B, D_latent], mu in [B, D_latent], logvar in [B, D_latent]
    """

    def __init__(self, gene_dim: int, latent_dim: int = 64,
                 hidden: list[int] | tuple[int, ...] = (512, 256)):
        super().__init__()
        self.gene_dim = gene_dim
        self.latent_dim = latent_dim

        enc_layers: list[nn.Module] = []
        prev = gene_dim
        for h in hidden:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.SiLU())
            prev = h
        self.encoder_trunk = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(nn.SiLU())
            prev = h
        dec_layers.append(nn.Linear(prev, gene_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.1) -> dict:
    """
    Recon: MSE (input already log1p normalized)
    KL: standard Gaussian KL
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_per_element.mean()
    total = recon + beta * kl
    return {"total": total, "recon": recon, "kl": kl}


def _iter_minibatches(X: np.ndarray, batch_size: int, rng: np.random.Generator,
                      pool: Optional[np.ndarray] = None):
    idx_pool = np.arange(X.shape[0]) if pool is None else np.asarray(pool)
    idx = idx_pool[rng.permutation(idx_pool.shape[0])]
    for i in range(0, idx.shape[0], batch_size):
        yield idx[i:i + batch_size]


def train_vae(
    X: np.ndarray,
    gene_dim: int,
    latent_dim: int = 64,
    hidden: tuple[int, ...] = (512, 256),
    epochs: int = 150,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 0.1,
    device: str = "cuda",
    seed: int = 0,
    logger=None,
    fit_indices: Optional[np.ndarray] = None,
) -> tuple[CellVAE, dict]:
    device_t = torch.device(device)
    model = CellVAE(gene_dim, latent_dim, hidden).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    history = {"total": [], "recon": [], "kl": []}

    model.train()
    for ep in range(epochs):
        agg = {"total": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        for batch_idx in _iter_minibatches(X, batch_size, rng, pool=fit_indices):
            x = torch.from_numpy(X[batch_idx]).float().to(device_t)
            x_hat, mu, logvar = model(x)
            ld = vae_loss(x_hat, x, mu, logvar, beta=beta)
            opt.zero_grad()
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.shape[0]
            agg["total"] += ld["total"].item() * bs
            agg["recon"] += ld["recon"].item() * bs
            agg["kl"] += ld["kl"].item() * bs
            agg["n"] += bs

        avg = {k: agg[k] / max(agg["n"], 1) for k in ("total", "recon", "kl")}
        for k in ("total", "recon", "kl"):
            history[k].append(avg[k])

        if logger is not None and (ep % max(epochs // 20, 1) == 0 or ep == epochs - 1):
            logger.info(f"[VAE] epoch {ep:4d}/{epochs} "
                        f"total={avg['total']:.4f} recon={avg['recon']:.4f} "
                        f"kl={avg['kl']:.4f}")

    model.eval()
    return model, history


@torch.no_grad()
def encode_all(model: CellVAE, X: np.ndarray,
               batch_size: int = 1024, device: str = "cuda") -> np.ndarray:
    model.eval()
    device_t = torch.device(device)
    out = np.empty((X.shape[0], model.latent_dim), dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).float().to(device_t)
        mu, _ = model.encode(x)
        out[i:i + batch_size] = mu.cpu().numpy()
    return out


@torch.no_grad()
def decode_all(model: CellVAE, Z: np.ndarray,
               batch_size: int = 1024, device: str = "cuda") -> np.ndarray:
    """z -> x_hat, shape [N, G]."""
    model.eval()
    device_t = torch.device(device)
    out = np.empty((Z.shape[0], model.gene_dim), dtype=np.float32)
    for i in range(0, Z.shape[0], batch_size):
        z = torch.from_numpy(Z[i:i + batch_size]).float().to(device_t)
        x_hat = model.decode(z)
        out[i:i + batch_size] = x_hat.cpu().numpy()
    return out


def save_vae(model: CellVAE, path: str | Path,
             meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "gene_dim": model.gene_dim,
        "latent_dim": model.latent_dim,
        "meta": dict(meta or {}),
    }, str(path))


def load_vae(path: str | Path, hidden: tuple[int, ...] = (512, 256),
             device: str = "cuda") -> CellVAE:
    path = Path(path)
    ckpt = torch.load(str(path), map_location=device)
    model = CellVAE(gene_dim=ckpt["gene_dim"], latent_dim=ckpt["latent_dim"],
                    hidden=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
