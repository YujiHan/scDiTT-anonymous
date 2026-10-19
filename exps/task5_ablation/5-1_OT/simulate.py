from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd

from common import (T, K, D, TIME_KEY, REGIMES, SIGMAS, DATA_DIR,
                    tag, sim_h5ad_path)

MU_U = 1.0
C_U = 0.3
K_V = 1.5
B_SEP = 2.5
T_SPLIT = 4.0
THETA_W = 0.1
C_W = 1.0
D_W = 3
D_INT = 2 + D_W
W_INIT = 2.0
N_SUB = 20
DT = 1.0 / N_SUB

EMBED_SEED = 12345
N_FEAT = 32
ALPHA_NL = 0.3
SIGMA_AMB = 0.03


def _make_embedding(rng: np.random.Generator, d_int: int = D_INT):
    A = rng.standard_normal((D, d_int))
    Q, _ = np.linalg.qr(A)
    Q = Q[:, :d_int]
    Wf = rng.standard_normal((d_int, N_FEAT)) * 0.8
    bf = rng.uniform(0, 2 * np.pi, size=N_FEAT)
    Vf = rng.standard_normal((N_FEAT, D)) / np.sqrt(N_FEAT)
    return Q, Wf, bf, Vf


TRI_PROPS = np.array([0.6, 0.3, 0.1])
TRI_B = B_SEP
TRI_ANGLES = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
TRI_CENTERS = TRI_B * np.stack([np.cos(TRI_ANGLES), np.sin(TRI_ANGLES)], axis=1)  # [3,2]
D_INT_TRI = 3 + D_W


def _simulate_trifurcation(sigma, seed, sigma_amb, theta_w, w_init):
    rng = np.random.default_rng(seed)
    emb = _make_embedding(np.random.default_rng(EMBED_SEED), D_INT_TRI)

    S = np.zeros((K, D_INT_TRI), dtype=np.float64)
    S[:, 0] = 0.0                                        # u
    S[:, 1:3] = 0.05 * rng.standard_normal((K, 2))
    S[:, 3:] = w_init * rng.standard_normal((K, D_W))

    branch = rng.choice(3, size=K, p=TRI_PROPS).astype(int)
    target_v = TRI_CENTERS[branch]                       # [K, 2]

    sqrt_dt = np.sqrt(DT)
    S_obs = np.zeros((T, K, D_INT_TRI), dtype=np.float64)
    S_obs[0] = S
    cur_time = 0.0
    for t_idx in range(1, T):
        for _ in range(N_SUB):
            cur_time += DT
            S[:, 0] += MU_U * DT + sigma * C_U * sqrt_dt * rng.standard_normal(K)
            tgt = target_v if cur_time >= T_SPLIT else 0.0
            S[:, 1:3] += -K_V * (S[:, 1:3] - tgt) * DT \
                + sigma * sqrt_dt * rng.standard_normal((K, 2))
            S[:, 3:] += -theta_w * S[:, 3:] * DT \
                + sigma * C_W * sqrt_dt * rng.standard_normal((K, D_W))
        S_obs[t_idx] = S

    amb_rng = np.random.default_rng(seed + 777)
    X_obs = np.zeros((T, K, D), dtype=np.float32)
    for t_idx in range(T):
        X_obs[t_idx] = _embed(S_obs[t_idx], emb, amb_rng, sigma_amb)

    flatX = X_obs.reshape(T * K, D)
    mu = flatX.mean(0, keepdims=True)
    sd = flatX.std(0, keepdims=True) + 1e-6
    flatX = (flatX - mu) / sd

    tp = np.repeat(np.arange(T), K).astype(int)
    particle = np.tile(np.arange(K), T).astype(int)
    branch_flat = np.tile(branch, T).astype(int)
    S_flat = S_obs.reshape(T * K, D_INT_TRI).astype(np.float32)
    return flatX.astype(np.float32), tp, particle, branch_flat, S_flat


def _embed(S: np.ndarray, emb, ambient_rng: np.random.Generator,
           sigma_amb: float) -> np.ndarray:
    Q, Wf, bf, Vf = emb
    base = S @ Q.T
    nl = np.sin(S @ Wf + bf[None, :]) @ Vf
    X = base + ALPHA_NL * nl
    X = X + sigma_amb * ambient_rng.standard_normal(X.shape)
    return X.astype(np.float32)


def simulate(regime: str, sigma: float, seed: int,
             sigma_amb: float = SIGMA_AMB, theta_w: float = THETA_W,
             w_init: float = W_INIT):
    if regime == "trifurcation":
        return _simulate_trifurcation(sigma, seed, sigma_amb, theta_w, w_init)
    if regime != "bifurcation":
        raise ValueError(f"unknown regime: {regime} (expected one of {REGIMES})")
    rng = np.random.default_rng(seed)
    emb = _make_embedding(np.random.default_rng(EMBED_SEED))

    S = np.zeros((K, D_INT), dtype=np.float64)
    S[:, 0] = 0.0                                       # u
    S[:, 1] = 0.05 * rng.standard_normal(K)
    S[:, 2:] = w_init * rng.standard_normal((K, D_W))

    # Each particle is assigned to one of two branches with equal probability,
    # giving fate targets at +/- B_SEP.
    b_sign = rng.choice([-1.0, 1.0], size=K)
    branch = ((b_sign + 1) / 2).astype(int)

    sqrt_dt = np.sqrt(DT)
    S_obs = np.zeros((T, K, D_INT), dtype=np.float64)
    S_obs[0] = S
    cur_time = 0.0
    for t_idx in range(1, T):
        for _ in range(N_SUB):
            cur_time += DT
            S[:, 0] += MU_U * DT + sigma * C_U * sqrt_dt * rng.standard_normal(K)
            # The fate coordinate mean-reverts towards 0 before the branch point
            # and towards the particle's own branch centre after it.
            target = b_sign * B_SEP if cur_time >= T_SPLIT else 0.0
            S[:, 1] += -K_V * (S[:, 1] - target) * DT \
                + sigma * sqrt_dt * rng.standard_normal(K)
            S[:, 2:] += -theta_w * S[:, 2:] * DT \
                + sigma * C_W * sqrt_dt * rng.standard_normal((K, D_W))
        S_obs[t_idx] = S

    amb_rng = np.random.default_rng(seed + 777)
    X_obs = np.zeros((T, K, D), dtype=np.float32)
    for t_idx in range(T):
        X_obs[t_idx] = _embed(S_obs[t_idx], emb, amb_rng, sigma_amb)

    flatX = X_obs.reshape(T * K, D)
    mu = flatX.mean(0, keepdims=True)
    sd = flatX.std(0, keepdims=True) + 1e-6
    flatX = (flatX - mu) / sd

    tp = np.repeat(np.arange(T), K).astype(int)
    particle = np.tile(np.arange(K), T).astype(int)
    branch_flat = np.tile(branch, T).astype(int)
    S_flat = S_obs.reshape(T * K, D_INT).astype(np.float32)
    return flatX.astype(np.float32), tp, particle, branch_flat, S_flat


def save_sim(regime: str, sigma: float, seed: int, **kw) -> str:
    X, tp, particle, branch, S = simulate(regime, sigma, seed, **kw)
    obs = pd.DataFrame({
        TIME_KEY: tp.astype(int),
        "particle_id": particle.astype(int),
        "branch": branch.astype(int),
    })
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_latent"] = X
    adata.obsm["X_intrinsic"] = S
    path = sim_h5ad_path(regime, sigma, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(path))
    b_last = branch[tp == T - 1]
    counts = np.bincount(b_last, minlength=int(b_last.max()) + 1)
    print(f"[sim] {tag(regime, sigma)} seed{seed}: X={X.shape} "
          f"branch@lastTP={counts.tolist()}/{K} -> {path.name}")
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all",
                    help="linear / bifurcation / all")
    ap.add_argument("--sigma", default="all",
                    help="a float, or 'all' to sweep SIGMAS")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seed list; the paper uses the five seeds 42,43,44,45,46")
    ap.add_argument("--sigma_amb", type=float, default=SIGMA_AMB)
    ap.add_argument("--theta_w", type=float, default=THETA_W)
    ap.add_argument("--w_init", type=float, default=W_INIT)
    args = ap.parse_args()

    regimes = REGIMES if args.regime == "all" else [args.regime]
    sigmas = SIGMAS if args.sigma == "all" else [float(args.sigma)]
    seeds = [int(s) for s in args.seeds.split(",")]
    kw = dict(sigma_amb=args.sigma_amb, theta_w=args.theta_w, w_init=args.w_init)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for regime in regimes:
        for sigma in sigmas:
            for seed in seeds:
                save_sim(regime, sigma, seed, **kw)


if __name__ == "__main__":
    main()
