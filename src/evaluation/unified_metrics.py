#!/usr/bin/env python
"""Self-contained implementations of the five scDiTT evaluation metrics.

    W2            -- PCA-Sinkhorn Wasserstein-2       (cell-distribution distance)
    MMD           -- multi-scale RBF MMD^2            (cell-distribution distance)
    PearsonMean   -- per-gene mean-expression Pearson (gene-marginal fidelity)
    SampledPearson-- sampled per-gene mean Pearson    (gene-marginal fidelity)
    perGeneW1     -- per-gene Wasserstein-1           (gene-marginal fidelity)

Every method in the paper -- scDiTT and all six published baselines -- is scored
by the functions below rather than by its own implementation of the metric
(supplementary material, Evaluation -> "Evaluation Metrics"). The module is
self-contained: it takes plain NumPy arrays, so the same code path serves the
baseline reruns, whose data loading and run-naming logic lives outside this
repository.

Design
------
* No dependency on any experiment directory layout: you pass in plain NumPy
  arrays of shape ``[n_cells, n_genes]``.
* Two interfaces for every metric:
    - a single-pair function  f(pred, real, ...)               -> one condition
    - a per-timepoint wrapper f(pred_by_tp, real_by_tp, tps)   -> averaged
  where ``*_by_tp`` is a ``dict[int -> np.ndarray[n_cells, n_genes]]``.
* All heavy imports (torch, geomloss, sklearn, scipy) are LAZY -- importing this
  module costs nothing, and you only need a dependency for the metric you call.
* Every stochastic step (PCA solver, subsampling, sampled-Pearson draws) is
  seeded (``EVAL_SEED``) so results are fully reproducible and paired across
  models compared on the same data.

Dependencies (only for the metric you use)
    numpy               : always
    scikit-learn        : W2 (PCA), PearsonMean (r2/pearson helpers)
    scipy               : MMD, PearsonMean, perGeneW1
    torch + geomloss    : W2 (Sinkhorn solver)

Quick start
-----------
    import numpy as np
    import metrics as M

    pred = np.load("pred.npy")   # [n_cells, n_genes]
    real = np.load("real.npy")   # [n_cells, n_genes]

    # gene-marginal metrics (no GPU, no torch needed)
    print(M.pearson_mean(pred, real))       # PearsonMean
    print(M.per_gene_w1_mean(pred, real))   # perGeneW1 (mean over genes)
    print(M.mmd(pred, real))                # MMD
    print(M.sampled_pearson(pred, real))    # SampledPearson

    # W2 needs a PCA fitted on the reference (real) data + torch/geomloss
    pca    = M.fit_pca([real])              # fit PCA-50 on real cells
    device = M.get_device()                 # cuda if available else cpu
    print(M.w2(pred, real, pca=pca, device=device))
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# =====================================================================
# Consistency knobs -- keep these fixed to reproduce the paper's numbers
# =====================================================================
EVAL_SEED = 0          # seed for PCA solver / subsampling / sampled-Pearson draws
PCA_DIM = 50           # PCA components for the W2 protocol
SINKHORN_BLUR = 0.05   # geomloss Sinkhorn blur
SAMPLED_PEARSON_SIZE = 2000     # cells sampled from the prediction each repeat
SAMPLED_PEARSON_REPEATS = 100   # repeats, averaged to remove sampling noise
W1_CELLS_PER_TP = 2000          # cells kept per tp before per-gene W1 (speed)


# =====================================================================
# Device helper (only needed for W2)
# =====================================================================
def get_device(prefer: Optional[str] = None):
    """Return a torch.device -- ``prefer`` (e.g. 'cuda:0'/'cpu') or auto-select."""
    import torch
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. W2  --  PCA-Sinkhorn Wasserstein-2  (the paper's main "WD")
# =====================================================================
def fit_pca(real_arrays: Sequence[np.ndarray], n_components: int = PCA_DIM,
            seed: int = EVAL_SEED):
    """Fit ONE PCA on the concatenation of all reference (real) arrays.

    The W2 protocol projects both prediction and reference into this shared,
    seeded PCA space before computing the Sinkhorn distance.  Fit it once on the
    full real reference (e.g. all timepoints) and reuse it for every prediction.

    Parameters
    ----------
    real_arrays : sequence of ``[n_cells, n_genes]`` arrays (real cells).  A
                  single array may be passed as ``[real]`` or ``fit_pca([real])``.
    n_components: number of PCA components (default 50).
    seed        : ``random_state`` of the PCA solver (reproducibility).

    Returns
    -------
    A fitted ``sklearn.decomposition.PCA``.
    """
    from sklearn.decomposition import PCA
    parts = [np.asarray(a, dtype=np.float32) for a in real_arrays
             if a is not None and len(a)]
    concat = np.concatenate(parts, axis=0)
    pca = PCA(n_components=n_components, svd_solver="arpack", random_state=seed)
    pca.fit(concat)
    return pca


def fit_pca_on_all(real_by_tp: dict[int, np.ndarray], all_tps: Iterable[int],
                   n_components: int = PCA_DIM, seed: int = EVAL_SEED):
    """Convenience: fit the shared PCA on all timepoints of a ``real_by_tp`` dict."""
    return fit_pca([real_by_tp[t] for t in all_tps if t in real_by_tp],
                   n_components=n_components, seed=seed)


def sinkhorn_w2(pred: np.ndarray, real: np.ndarray, device,
                blur: float = SINKHORN_BLUR, p: int = 2,
                scaling: float = 0.5) -> float:
    """Debiased Sinkhorn W2 between two point clouds (each ``[n, d]``).

    Matches the reference solver exactly:
    ``SamplesLoss('sinkhorn', p=2, blur=0.05, scaling=0.5, debias=True,
    backend='tensorized')``.  Operates in whatever space the inputs are already
    in -- pass PCA-projected arrays for the paper protocol (see ``w2``).
    """
    import torch
    from geomloss import SamplesLoss
    solver = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling,
                         debias=True, backend="tensorized")
    tp = torch.as_tensor(pred, dtype=torch.float32, device=device)
    tt = torch.as_tensor(real, dtype=torch.float32, device=device)
    return float(solver(tp, tt).item())


def w2(pred: np.ndarray, real: np.ndarray, pca=None, device=None,
       blur: float = SINKHORN_BLUR) -> float:
    """W2 for ONE condition: project to the shared PCA space, then Sinkhorn.

    Parameters
    ----------
    pred, real : ``[n_cells, n_genes]`` arrays.
    pca        : a PCA fitted with :func:`fit_pca` on the real reference.  If
                 ``None`` the distance is computed directly in the input space
                 (NOT the paper protocol -- only do this if you pre-projected).
    device     : torch device from :func:`get_device`; auto-selected if ``None``.

    Returns
    -------
    float -- the (lower is better) Sinkhorn W2.
    """
    if device is None:
        device = get_device()
    if pca is not None:
        pred = pca.transform(pred)
        real = pca.transform(real)
    return sinkhorn_w2(pred, real, device=device, blur=blur)


def w2_per_tp(pred_by_tp: dict[int, np.ndarray], real_by_tp: dict[int, np.ndarray],
              tps: Sequence[int], pca, device) -> dict:
    """PCA-Sinkhorn W2 for every tp in ``tps`` + the average.

    Returns ``{"per_tp": {t: w2_t}, "avg": mean_over_tps}``.
    """
    per_tp, totals = {}, []
    for t in tps:
        wd = w2(pred_by_tp[t], real_by_tp[t], pca=pca, device=device)
        per_tp[int(t)] = wd
        totals.append(wd)
    return {"per_tp": per_tp,
            "avg": float(np.mean(totals)) if totals else float("nan")}


# =====================================================================
# 2. MMD  --  multi-scale RBF MMD^2
# =====================================================================
def _median_heuristic(x, max_n: int = 1000, seed: int = EVAL_SEED) -> float:
    """Median pairwise Euclidean distance (subsampled) -- the RBF length scale."""
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    d = cdist(x, x)
    iu = np.triu_indices_from(d, k=1)
    return float(np.median(d[iu])) + 1e-8


def mmd(X: np.ndarray, Y: np.ndarray, sigmas=None, max_n: int = 1500,
        seed: int = EVAL_SEED) -> float:
    """Multi-scale (5-bandwidth) RBF MMD^2 between two samples (order-agnostic).

    Each group is subsampled to ``max_n`` cells for speed.  When ``sigmas`` is
    ``None`` the bandwidths are ``[m/4, m/2, m, 2m, 4m]`` where ``m`` is the
    median pairwise distance of the pooled sample.  Lower is better; 0 means the
    two empirical distributions are indistinguishable at every scale.
    """
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    if sigmas is None:
        med = _median_heuristic(np.concatenate([X, Y], 0), max_n=1000, seed=seed)
        sigmas = [med / 4, med / 2, med, med * 2, med * 4]
    Dxx = cdist(X, X, "sqeuclidean")
    Dyy = cdist(Y, Y, "sqeuclidean")
    Dxy = cdist(X, Y, "sqeuclidean")
    mmd2 = 0.0
    for s in sigmas:
        g = 1.0 / (2 * s * s + 1e-12)
        mmd2 += (np.exp(-g * Dxx).mean() + np.exp(-g * Dyy).mean()
                 - 2 * np.exp(-g * Dxy).mean())
    return float(mmd2)


def mmd_per_tp(pred_by_tp: dict[int, np.ndarray], real_by_tp: dict[int, np.ndarray],
               tps: Sequence[int], seed: int = EVAL_SEED) -> dict:
    """MMD^2 for every tp in ``tps`` + the average.

    Returns ``{"per_tp": {t: mmd_t}, "avg": mean_over_tps}``.
    """
    per_tp = {int(t): mmd(real_by_tp[t], pred_by_tp[t], seed=seed) for t in tps}
    vals = list(per_tp.values())
    return {"per_tp": per_tp,
            "avg": float(np.mean(vals)) if vals else float("nan")}


# =====================================================================
# 3. PearsonMean  --  per-gene mean-expression Pearson correlation
# =====================================================================
def pearson_mean(pred: np.ndarray, real: np.ndarray) -> float:
    """Pearson correlation between the per-gene MEAN expression of real vs pred.

    Both arrays are averaged over cells (axis 0) to give a length-``n_genes``
    mean-expression vector each; the returned scalar is the Pearson r between
    those two vectors (higher is better, in ``[-1, 1]``).
    """
    from scipy.stats import pearsonr
    # Keep the input dtype for the per-gene mean so the result is byte-identical
    # to the reference (which averages the float32 expression matrix directly).
    real = np.asarray(real)
    pred = np.asarray(pred)
    return float(pearsonr(real.mean(0), pred.mean(0)).statistic)


def pearson_mean_over_tps(pred_by_tp: dict[int, np.ndarray],
                          real_by_tp: dict[int, np.ndarray],
                          tps: Sequence[int]) -> float:
    """PearsonMean on cells POOLED across ``tps`` (the paper's convention).

    All cells from every tp are concatenated first, then the per-gene means are
    taken over that pool -- NOT a per-tp average.
    """
    real_full = np.concatenate([real_by_tp[t] for t in tps], axis=0)
    pred_full = np.concatenate([pred_by_tp[t] for t in tps], axis=0)
    return pearson_mean(pred_full, real_full)


# =====================================================================
# 4. SampledPearson  --  sampled per-gene mean Pearson
# =====================================================================
def sampled_pearson(pred: np.ndarray, real: np.ndarray,
                    sample_size: int = SAMPLED_PEARSON_SIZE,
                    n_repeats: int = SAMPLED_PEARSON_REPEATS,
                    seed: int = EVAL_SEED, eps: float = 1e-8) -> float:
    """Sampled Pearson correlation for ONE condition (paper-faithful).

    * reference ``mu_true`` = mean over ALL real cells (computed once; the real
      set is NEVER sub-sampled);
    * for each of ``n_repeats`` repeats, draw ``sample_size`` predicted cells
      (without replacement when enough are available, else with replacement),
      take their per-gene mean ``mu_pred``, and compute the Pearson correlation
      between ``mu_true`` and ``mu_pred`` ALONG THE GENE DIMENSION;
    * return the mean correlation over the repeats (higher is better).

    Reproducible via the seeded RNG.  ``eps`` guards the degenerate constant
    vector case.
    """
    pred = np.asarray(pred, dtype=np.float32)
    real = np.asarray(real, dtype=np.float32)
    rng = np.random.default_rng(seed)
    mu_true = real.mean(0)
    b = mu_true - mu_true.mean()
    b_norm = float(np.sqrt((b ** 2).sum()))
    n_pred = pred.shape[0]
    replace = n_pred < sample_size
    corrs = np.empty(n_repeats, dtype=np.float64)
    done, CH = 0, 25
    while done < n_repeats:
        k = min(CH, n_repeats - done)
        if replace:
            idx = rng.integers(0, n_pred, size=(k, sample_size))
        else:
            idx = np.stack([rng.choice(n_pred, sample_size, replace=False)
                            for _ in range(k)])
        mu = pred[idx].mean(axis=1)                         # [k, n_genes]
        a = mu - mu.mean(axis=1, keepdims=True)
        num = (a * b).sum(1)
        den = np.sqrt((a ** 2).sum(1)) * b_norm + eps
        corrs[done:done + k] = num / den
        done += k
    return float(corrs.mean())


def sampled_pearson_over_tps(pred_by_tp: dict[int, np.ndarray],
                             real_by_tp: dict[int, np.ndarray],
                             tps: Sequence[int], seed: int = EVAL_SEED) -> float:
    """Per-tp SampledPearson averaged over ``tps`` (equal weight).

    Each tp gets an independent, reproducible RNG stream (``seed + tp``), so the
    random sub-samples are identical across models compared at a given tp (paired)
    yet independent across timepoints.
    """
    vals = [sampled_pearson(pred_by_tp[t], real_by_tp[t], seed=seed + int(t))
            for t in tps]
    return float(np.mean(vals)) if vals else float("nan")


# =====================================================================
# 5. perGeneW1  --  per-gene Wasserstein-1 distance
# =====================================================================
def per_gene_wasserstein(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """1-D Wasserstein-1 distance between real & pred, computed per gene.

    Returns a length-``n_genes`` array; reduce it (mean/median/quantile) to a
    scalar.  Each gene's two 1-D expression distributions are compared directly.
    """
    from scipy.stats import wasserstein_distance
    real = np.asarray(real, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    G = real.shape[1]
    out = np.empty(G, dtype=np.float64)
    for g in range(G):
        out[g] = wasserstein_distance(real[:, g], pred[:, g])
    return out


def _subsample(X, n, rng):
    return X[rng.choice(len(X), n, replace=False)] if len(X) > n else X


def per_gene_w1_mean(pred: np.ndarray, real: np.ndarray,
                     subsample: Optional[int] = W1_CELLS_PER_TP,
                     seed: int = EVAL_SEED) -> float:
    """perGeneW1 for ONE condition: mean over genes of the per-gene W1 (lower is
    better).

    If ``subsample`` is given, each side is randomly reduced to that many cells
    first (default 2000, matching the paper) -- set ``subsample=None`` to use all
    cells.
    """
    if subsample is not None:
        rng = np.random.default_rng(seed)
        real = _subsample(np.asarray(real), subsample, rng)
        pred = _subsample(np.asarray(pred), subsample, rng)
    return float(per_gene_wasserstein(pred, real).mean())


def per_gene_w1_over_tps(pred_by_tp: dict[int, np.ndarray],
                         real_by_tp: dict[int, np.ndarray],
                         tps: Sequence[int], subsample: int = W1_CELLS_PER_TP,
                         seed: int = EVAL_SEED) -> dict:
    """perGeneW1 on cells POOLED across ``tps`` (the paper's convention).

    Each tp is sub-sampled to ``subsample`` cells, the tps are concatenated, and
    the per-gene W1 is taken once over the pool.  Returns
    ``{"mean": ..., "median": ..., "p90": ...}`` (reductions over genes).
    """
    rng = np.random.default_rng(seed)
    real_sub = np.concatenate([_subsample(real_by_tp[t], subsample, rng) for t in tps], 0)
    pred_sub = np.concatenate([_subsample(pred_by_tp[t], subsample, rng) for t in tps], 0)
    w1 = per_gene_wasserstein(pred_sub, real_sub)
    return {"mean": float(w1.mean()),
            "median": float(np.median(w1)),
            "p90": float(np.quantile(w1, 0.9))}


# =====================================================================
# Convenience: compute all five metrics at once
# =====================================================================
def compute_all_metrics(pred_by_tp: dict[int, np.ndarray],
                        real_by_tp: dict[int, np.ndarray],
                        tps: Sequence[int],
                        all_tps: Optional[Sequence[int]] = None,
                        pca=None, device=None,
                        include_w2: bool = True) -> dict:
    """Compute all five metrics for a per-timepoint prediction in one call.

    Parameters
    ----------
    pred_by_tp, real_by_tp : ``dict[int -> [n_cells, n_genes]]``.
    tps       : timepoints to score (must be keys of both dicts).
    all_tps   : timepoints to fit the shared W2-PCA on (defaults to ``tps``).
                Pass the FULL set of real timepoints to match the paper.
    pca       : pre-fitted W2 PCA; if ``None`` (and ``include_w2``) one is fitted
                on ``real_by_tp`` over ``all_tps``.
    device    : torch device for W2; auto-selected if ``None``.
    include_w2: set ``False`` to skip W2 (avoids the torch/geomloss dependency).

    Returns
    -------
    ``dict`` with keys ``w2_avg`` (or NaN if skipped), ``mmd_avg``,
    ``pearson_mean``, ``sampled_pearson``, ``pg_w1_mean``.
    """
    out = {}
    if include_w2:
        if pca is None:
            pca = fit_pca_on_all(real_by_tp, all_tps if all_tps is not None else tps)
        if device is None:
            device = get_device()
        out["w2_avg"] = w2_per_tp(pred_by_tp, real_by_tp, tps, pca, device)["avg"]
    else:
        out["w2_avg"] = float("nan")
    out["mmd_avg"] = mmd_per_tp(pred_by_tp, real_by_tp, tps)["avg"]
    out["pearson_mean"] = pearson_mean_over_tps(pred_by_tp, real_by_tp, tps)
    out["sampled_pearson"] = sampled_pearson_over_tps(pred_by_tp, real_by_tp, tps)
    out["pg_w1_mean"] = per_gene_w1_over_tps(pred_by_tp, real_by_tp, tps)["mean"]
    return out
