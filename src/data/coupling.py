"""OT coupling precomputation -- entropic OT in latent space.

Only couplings between adjacent pairs inside train_tps are computed. Each (dataset, task_type) gets its
own set.

Paper reference
---------------
Eq. (2), Methodology -> "Stochastic temporal alignment"; supplementary material,
Training Details -> "Couplings and Sampling".

This module solves the balanced, entropically regularised optimal-transport
problem between the latent point clouds of adjacent stages and row-normalises
the plan into the stochastic kernel pi_bar. That kernel is a *prior over
ancestries*, not a lineage assignment: it is what lets unpaired snapshots be
assembled into trainable temporal sequences without any observed cell-level
correspondence. `data/dataset.py` draws trajectories from it.

Defaults match the paper: epsilon = 0.05, squared-Euclidean cost median-
normalised so that one epsilon transfers across datasets and stage pairs, and
stages with more than 2000 cells subsampled to 2000 per side before solving.

Replacing this construction with an alternative kernel -- hard OT (the row-wise
argmax of the same plan), a uniform kernel, or k-NN matching -- is the
trajectory-construction ablation. `exps/task5_ablation/5-1_OT/` runs it against
the oracle coupling on the simulated branching courses (Table S4), where the
particle identities are known.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np

try:
    import ot as pot  # POT
    _HAS_POT = True
except ImportError:
    _HAS_POT = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _sinkhorn_pot(cost: np.ndarray, epsilon: float,
                  num_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
    """POT's sinkhorn; uniform marginals; returns an unnormalised pi.

    Paper: solves Eq. (2), argmin_pi <pi, C-tilde> - epsilon * H(pi) over the
    transport polytope Pi(a, b). The marginals a and b are uniform, i.e. the
    problem is *balanced*: proliferation and death are left for the generative
    model to learn rather than encoded in the plan.
    """
    N_a, N_b = cost.shape
    a = np.ones(N_a) / N_a
    b = np.ones(N_b) / N_b
    pi = pot.sinkhorn(a, b, cost, reg=epsilon, numItermax=num_iter,
                      stopThr=tol, method="sinkhorn_stabilized")
    return pi


def _sinkhorn_manual(cost: np.ndarray, epsilon: float,
                     num_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
    N_a, N_b = cost.shape
    log_a = -np.log(N_a)
    log_b = -np.log(N_b)
    K = -cost / epsilon
    log_u = np.full(N_a, 0.0)
    log_v = np.full(N_b, 0.0)
    for _ in range(num_iter):
        # u update
        new_log_u = log_a - _logsumexp(K + log_v[None, :], axis=1)
        # v update
        new_log_v = log_b - _logsumexp(K + new_log_u[:, None], axis=0)
        if (np.max(np.abs(new_log_u - log_u)) < tol
                and np.max(np.abs(new_log_v - log_v)) < tol):
            log_u, log_v = new_log_u, new_log_v
            break
        log_u, log_v = new_log_u, new_log_v
    log_pi = log_u[:, None] + K + log_v[None, :]
    pi = np.exp(log_pi)
    return pi


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def _squared_cdist(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    a = z_a.astype(np.float64, copy=False)
    b = z_b.astype(np.float64, copy=False)
    a_sq = np.einsum("ij,ij->i", a, a)          # [N_a]
    b_sq = np.einsum("ij,ij->i", b, b)          # [N_b]
    cost = a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
    np.maximum(cost, 0.0, out=cost)
    return cost


def _squared_cdist_torch(z_a, z_b):
    a_sq = (z_a * z_a).sum(dim=1)               # [N_a]
    b_sq = (z_b * z_b).sum(dim=1)               # [N_b]
    cost = a_sq[:, None] + b_sq[None, :] - 2.0 * (z_a @ z_b.T)
    return torch.clamp(cost, min=0.0)


def _sinkhorn_torch(cost, epsilon: float, num_iter: int = 1000,
                    tol: float = 1e-7):
    N_a, N_b = cost.shape
    log_a = float(-np.log(N_a))
    log_b = float(-np.log(N_b))
    K = -cost / epsilon
    log_u = torch.zeros(N_a, dtype=cost.dtype, device=cost.device)
    log_v = torch.zeros(N_b, dtype=cost.dtype, device=cost.device)
    for _ in range(num_iter):
        new_log_u = log_a - torch.logsumexp(K + log_v[None, :], dim=1)
        new_log_v = log_b - torch.logsumexp(K + new_log_u[:, None], dim=0)
        if (torch.max(torch.abs(new_log_u - log_u)) < tol
                and torch.max(torch.abs(new_log_v - log_v)) < tol):
            log_u, log_v = new_log_u, new_log_v
            break
        log_u, log_v = new_log_u, new_log_v
    log_pi = log_u[:, None] + K + log_v[None, :]
    return torch.exp(log_pi)


KERNELS = ("entropic", "hard_ot", "random", "knn")


def _kernel_from_cost(cost: np.ndarray, pi: Optional[np.ndarray],
                      kernel: str) -> np.ndarray:
    """Turn a cost matrix (and, where needed, a solved plan) into the row-
    stochastic sampling kernel pi_bar that `data/dataset.py` draws successors from.

    The four variants are the rows of the trajectory-construction ablation:

    ``entropic``  the row-normalised entropic-OT plan -- scDiTT itself. A
                  stochastic ancestry prior that respects the stage marginals.
    ``hard_ot``   the row-wise argmax of that same plan, i.e. a deterministic
                  transport map. Same solve, all mass on one successor.
    ``random``    a uniform kernel: every next-stage cell equally likely, so the
                  coupling carries no temporal information at all, while still
                  reproducing the next-stage marginal exactly.
    ``knn``       each cell paired with its nearest neighbour at the next stage.
                  Deterministic like ``hard_ot``, but with no marginal
                  constraint whatsoever: several cells may claim the same
                  successor and others go unclaimed.
    """
    n_a, n_b = cost.shape
    if kernel == "random":
        return np.full((n_a, n_b), 1.0 / n_b, dtype=np.float64)
    if kernel == "knn":
        tgt = np.argmin(cost, axis=1)
    elif kernel == "hard_ot":
        if pi is None:
            raise ValueError("hard_ot needs the solved plan")
        # argmax is invariant to the row normalisation below, so it does not
        # matter whether the raw or the normalised plan is passed in.
        tgt = np.argmax(pi, axis=1)
    else:
        raise ValueError(f"unknown kernel: {kernel} (expected one of {KERNELS})")
    out = np.zeros((n_a, n_b), dtype=np.float64)
    out[np.arange(n_a), tgt] = 1.0
    return out


def compute_coupling(
    z_a: np.ndarray,
    z_b: np.ndarray,
    epsilon: float = 0.05,
    normalize_cost: bool = True,
    max_cells: Optional[int] = 2000,
    rng: Optional[np.random.Generator] = None,
    ot_type: str = "CPU",
    device: str = "cuda:0",
    kernel: str = "entropic",
) -> dict:
    """Build the sampling kernel between two adjacent stages.

    ``kernel`` selects the trajectory construction; see :func:`_kernel_from_cost`.
    Subsampling and the cost matrix are computed identically for every variant
    and consume the rng in the same order, so the arms of the ablation differ
    only in how the cost is turned into a kernel.
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel: {kernel} (expected one of {KERNELS})")
    if rng is None:
        rng = np.random.default_rng(0)

    N_a, N_b = z_a.shape[0], z_b.shape[0]
    if max_cells is not None and N_a > max_cells:
        idx_a = rng.choice(N_a, max_cells, replace=False)
    else:
        idx_a = np.arange(N_a)
    if max_cells is not None and N_b > max_cells:
        idx_b = rng.choice(N_b, max_cells, replace=False)
    else:
        idx_b = np.arange(N_b)

    za_s = z_a[idx_a]
    zb_s = z_b[idx_b]

    ot_type = (ot_type or "CPU").upper()
    if ot_type not in ("CPU", "CPU_COST", "CUDA"):
        raise ValueError(
            f"unknown ot_type={ot_type!r}, expected 'CPU' / 'CPU_COST' / 'CUDA'")

    # The entropic plan is only solved for the two arms that need it; `random`
    # and `knn` are defined by the cost (or by nothing at all).
    need_plan = kernel in ("entropic", "hard_ot")
    pi = None

    if ot_type == "CUDA":
        if not _HAS_TORCH:
            raise RuntimeError("ot_type=CUDA but torch could not be imported")
        if not torch.cuda.is_available():
            raise RuntimeError("ot_type=CUDA but torch.cuda.is_available()=False")
        dev_str = device if (isinstance(device, str)
                             and device.startswith("cuda")) else "cuda:0"
        dev = torch.device(dev_str)
        za_t = torch.as_tensor(np.ascontiguousarray(za_s),
                               dtype=torch.float32, device=dev)
        zb_t = torch.as_tensor(np.ascontiguousarray(zb_s),
                               dtype=torch.float32, device=dev)
        cost_t = _squared_cdist_torch(za_t, zb_t)
        if normalize_cost:
            med = torch.median(cost_t)
            if med > 0:
                cost_t = cost_t / med
        if need_plan:
            pi = _sinkhorn_torch(cost_t, epsilon=epsilon).double().cpu().numpy()
        cost = cost_t.double().cpu().numpy()
    else:
        if ot_type == "CPU":
            cost = ((za_s[:, None, :] - zb_s[None, :, :]) ** 2).sum(axis=-1).astype(np.float64)
        else:
            cost = _squared_cdist(za_s, zb_s)
        if normalize_cost:
            # Paper: the squared-Euclidean cost is median-normalised so that one
            # value of epsilon transfers across datasets and stage pairs
            # (supplementary material -> "Couplings and Sampling"); this is the
            # C-tilde of Eq. (2).
            med = np.median(cost)
            if med > 0:
                cost = cost / med
        if need_plan:
            if _HAS_POT:
                try:
                    pi = _sinkhorn_pot(cost, epsilon=epsilon)
                except Exception:
                    pi = _sinkhorn_manual(cost, epsilon=epsilon)
            else:
                pi = _sinkhorn_manual(cost, epsilon=epsilon)

    if kernel != "entropic":
        pi = _kernel_from_cost(cost, pi, kernel)

    # row-normalize: pi[i, :] / sum_j pi[i, j]
    # Paper: this turns the plan pi of Eq. (2) into the stochastic kernel
    # pi_bar, whose i-th row is a distribution over the cells at the next stage
    # that cell i may plausibly correspond to. Because the problem is balanced
    # with uniform marginals, row-normalising cannot bias the population the
    # model sees at any one stage -- see the marginal-preservation argument in
    # the supplementary material, Training Details -> "Couplings and Sampling".
    # (The three alternative kernels are already row-stochastic; this is a no-op
    # for them, and is what makes them drop-in replacements for pi_bar.)
    row_sum = pi.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-30, 1.0, row_sum)
    pi = pi / row_sum

    return {
        "pi": pi.astype(np.float32),
        "idx_a": idx_a.astype(np.int64),
        "idx_b": idx_b.astype(np.int64),
        "epsilon": float(epsilon),
        "kernel": kernel,
    }


def precompute_all_couplings(
    latents_path: str | Path,
    train_tps: list[int],
    output_dir: str | Path,
    time_key: str = "tp",
    epsilon: float = 0.05,
    normalize_cost: bool = True,
    max_cells: Optional[int] = 2000,
    seed: int = 0,
    ot_type: str = "CPU",
    device: str = "cuda:0",
    kernel: str = "entropic",
    logger=None,
) -> None:
    """Compute and save the coupling for every (t_a, t_b) in zip(train_tps[:-1], train_tps[1:]).

    ot_type: cost matrix computation, "CPU" / "CPU_COST" / "CUDA" (see compute_coupling).
    device:  only used when ot_type=="CUDA", selecting the GPU ("cuda:N").
    kernel:  trajectory construction, one of KERNELS (see compute_coupling). The
             default "entropic" is scDiTT; the other three are the ablation arms.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(str(latents_path))
    if "X_latent" in adata.obsm:
        Z_all = adata.obsm["X_latent"]
    else:
        Z_all = adata.X

    Z_all = np.asarray(Z_all, dtype=np.float32)
    tps = np.asarray(adata.obs[time_key].values).astype(float)

    rng = np.random.default_rng(seed)

    saved_pairs = []
    for (t_a, t_b) in zip(train_tps[:-1], train_tps[1:]):
        mask_a = np.isclose(tps, float(t_a))
        mask_b = np.isclose(tps, float(t_b))
        idx_a_orig = np.where(mask_a)[0]
        idx_b_orig = np.where(mask_b)[0]

        z_a = Z_all[idx_a_orig]
        z_b = Z_all[idx_b_orig]

        res = compute_coupling(
            z_a, z_b,
            epsilon=epsilon,
            normalize_cost=normalize_cost,
            max_cells=max_cells,
            rng=rng,
            ot_type=ot_type,
            device=device,
            kernel=kernel,
        )

        idx_a_global = idx_a_orig[res["idx_a"]]
        idx_b_global = idx_b_orig[res["idx_b"]]

        file_path = out_dir / f"pi_{t_a}_to_{t_b}.npz"
        np.savez(
            str(file_path),
            pi=res["pi"],
            idx_a_global=idx_a_global,
            idx_b_global=idx_b_global,
            t_a=np.int64(t_a),
            t_b=np.int64(t_b),
            epsilon=np.float32(res["epsilon"]),
        )
        saved_pairs.append((int(t_a), int(t_b), str(file_path.name)))

        if logger is not None:
            logger.info(f"[OT] ({t_a}->{t_b}) pi shape={res['pi'].shape} "
                        f"saved to {file_path.name}")

    meta = {
        "train_tps": [int(t) for t in train_tps],
        "pairs": [{"t_a": a, "t_b": b, "file": f} for a, b, f in saved_pairs],
        "epsilon": float(epsilon),
        "max_cells": (None if max_cells is None else int(max_cells)),
        "normalize_cost": bool(normalize_cost),
        "ot_type": str(ot_type).upper(),
        "kernel": str(kernel),
        "time_key": time_key,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_coupling(npz_path: str | Path) -> dict:
    d = np.load(str(npz_path))
    return {
        "pi": d["pi"],
        "idx_a_global": d["idx_a_global"],
        "idx_b_global": d["idx_b_global"],
        "t_a": int(d["t_a"]),
        "t_b": int(d["t_b"]),
        "epsilon": float(d["epsilon"]),
    }
