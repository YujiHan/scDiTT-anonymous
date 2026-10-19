from __future__ import annotations

import itertools

import numpy as np

import common as C
from src.evaluation import unified_metrics as M


def get_device(gpu=None):
    return M.get_device(None if gpu is None else str(gpu))


def to_pca(X: np.ndarray, pca) -> np.ndarray:
    return pca.transform(X) if pca is not None else X


def w2_pca(a_pca: np.ndarray, b_pca: np.ndarray, device) -> float:
    import torch
    try:
        return M.sinkhorn_w2(a_pca, b_pca, device=device)
    except RuntimeError as e:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type == "cuda" and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return M.sinkhorn_w2(a_pca, b_pca, device=torch.device("cpu"))
        raise


def w2_safe(a: np.ndarray, b: np.ndarray, pca, device) -> float:
    return w2_pca(to_pca(a, pca), to_pca(b, pca), device)


def subsample(X: np.ndarray, n: int, seed) -> np.ndarray:
    if len(X) == n:
        return X
    if len(X) < n:
        raise ValueError(f"cannot subsample {n} from {len(X)} cells without replacement")
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n, replace=False)]


def block_floor(X: np.ndarray, n: int, pca, device,
                R: int = C.FLOOR_R_DEFAULT, base_seed: int = 0,
                max_blocks: int | None = None) -> dict:
    N = len(X)
    Mblk = N // n
    if Mblk < 2:
        raise ValueError(f"N={N}, n={n} -> M={Mblk} < 2, cannot split into two disjoint blocks")
    Muse = Mblk if max_blocks is None else min(Mblk, int(max_blocks))
    dists = []
    for r in range(R):
        rng = np.random.default_rng([base_seed, n, r])
        perm = rng.permutation(N)[: Mblk * n]
        blocks = [to_pca(X[perm[i * n:(i + 1) * n]], pca) for i in range(Muse)]
        for i, j in itertools.combinations(range(Muse), 2):
            dists.append(w2_pca(blocks[i], blocks[j], device))
    d = np.asarray(dists, dtype=np.float64)
    return {
        "F": float(d.mean()),
        "F_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "F_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
        "n": int(n), "N": int(N), "M_blocks": int(Mblk),
        "M_used": int(Muse), "R": int(R), "n_pairs": int(len(d)),
    }


def sampling_floor(real_t: np.ndarray, n: int, pca, device,
                   R: int = C.FLOOR_R_DEFAULT, base_seed: int = 0) -> dict:
    return block_floor(real_t, n, pca, device, R=R, base_seed=base_seed)


def floor_R_for(N_t: int) -> int:
    return C.FLOOR_R_SMALL if N_t < C.SMALL_STAGE_THRESHOLD else C.FLOOR_R_DEFAULT
