from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import controls, data_io as io, sampling, stats  # noqa: E402

import panels as P  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════
def symbol_index(ds: str, adata) -> Dict[str, int]:
    ds = ds.upper()
    space = P.SYMBOL_SPACE[ds]
    syms = (adata.var["gene_symbols"].astype(str).values if space == "gene_symbols"
            else np.asarray(adata.var_names, dtype=str))
    out: Dict[str, int] = {}
    for i, s in enumerate(syms):
        out.setdefault(s.lower(), i)
    return out


@dataclass
class Resolved:
    ds: str
    rows: List[dict]
    idx: np.ndarray
    names: List[str]


def resolve_panel(ds: str, adata) -> Resolved:
    ds = ds.upper()
    sidx = symbol_index(ds, adata)
    lin = P.gene_lineage(ds)
    alias = P.ALIASES.get(ds, {})
    rev_alias = {v: k for k, v in alias.items()}

    rows, idx, names = [], [], []
    for g in P.panel_genes(ds):
        col = sidx.get(g.lower())
        if col is None:
            rows.append({"marker": g, "dataset": ds, "status": "panel_miss",
                         "resolved_name": "", "alias_of": rev_alias.get(g, ""),
                         "lineage": lin.get(g, ""), "symbol_space": P.SYMBOL_SPACE[ds],
                         "panel_type": "knowledge", "source": "analysis_plan"})
            continue
        is_alias = g in rev_alias
        rows.append({"marker": rev_alias.get(g, g), "dataset": ds,
                     "status": "alias_hit" if is_alias else "hit",
                     "resolved_name": g, "alias_of": rev_alias.get(g, ""),
                     "lineage": lin.get(g, ""), "symbol_space": P.SYMBOL_SPACE[ds],
                     "panel_type": "knowledge", "source": "analysis_plan"})
        idx.append(col)
        names.append(g)

    for g in P.MISSES[ds]:
        if g in alias:
            continue
        present = g.lower() in sidx
        rows.append({"marker": g, "dataset": ds,
                     "status": "miss_UNEXPECTED_PRESENT" if present else "miss",
                     "resolved_name": "", "alias_of": "", "lineage": "",
                     "symbol_space": P.SYMBOL_SPACE[ds], "panel_type": "knowledge",
                     "source": "analysis_plan"})
    return Resolved(ds, rows, np.asarray(idx, dtype=int), names)


# ═════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════
@dataclass
class RealSide:
    ds: str
    task: str
    tps: List[int]
    n_star: Dict[int, int]
    n_real: Dict[int, int]
    names: List[str]
    mu: np.ndarray
    se_bar: np.ndarray
    rng_span: np.ndarray      # [n_marker] max-min
    informative: np.ndarray   # [n_marker] bool
    shape: List[str]          # [n_marker] up/down/nonmono/flat


def real_side(ds: str, task: str, marker_idx: np.ndarray, names: List[str],
              adata=None, cell_mask=None) -> RealSide:
    import scipy.sparse as sp
    ds = ds.upper()
    a = io.load_real(ds) if adata is None else adata
    tps = io.held_out_tps(ds, task)
    tp_vals = np.asarray(a.obs[io.TIME_KEY].values).astype(int)

    n_m, n_t = len(marker_idx), len(tps)
    mu = np.zeros((n_m, n_t), dtype=np.float64)
    se = np.zeros((n_m, n_t), dtype=np.float64)
    nstar, nreal = {}, {}
    for j, t in enumerate(tps):
        sel = tp_vals == t
        if cell_mask is not None:
            sel = sel & np.asarray(cell_mask, bool)
        X = a.X[sel][:, marker_idx]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        ns = sampling.n_star(n)
        nreal[t], nstar[t] = int(n), int(ns)
        mu[:, j] = X.mean(axis=0)
        se[:, j] = X.std(axis=0, ddof=0) / np.sqrt(max(ns, 1))

    se_bar = se.mean(axis=1)
    span = mu.max(axis=1) - mu.min(axis=1)
    # informative <=> range >= max(0.02, 3 * SE_bar)
    thr = np.maximum(P.GATE_ABS_FLOOR, P.GATE_K_SE * se_bar)
    informative = span >= thr

    shape = []
    d = np.diff(mu, axis=1)                       # [n_m, n_t-1]
    tol = (P.SHAPE_TOL_SE * se_bar)[:, None]
    s = np.where(np.abs(d) < tol, 0, np.sign(d))
    for i in range(n_m):
        si = s[i]
        nz = si[si != 0]
        if nz.size == 0:
            shape.append("flat")
        elif np.all(nz > 0):
            shape.append("up")
        elif np.all(nz < 0):
            shape.append("down")
        else:
            shape.append("nonmono")
    return RealSide(ds, task, tps, nstar, nreal, list(names),
                    mu, se_bar, span, informative, shape)


# ═════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════
def _train_tps(ds: str, task: str, adata) -> List[int]:
    ho = set(io.held_out_tps(ds, task))
    all_tp = sorted(np.unique(np.asarray(adata.obs[io.TIME_KEY].values).astype(int)).tolist())
    return [t for t in all_tp if t not in ho]


def real_marker_by_tp(ds: str, task: str, marker_idx: np.ndarray, adata,
                      ) -> Dict[int, np.ndarray]:
    import scipy.sparse as sp
    tp_vals = np.asarray(adata.obs[io.TIME_KEY].values).astype(int)
    out = {}
    for t in io.held_out_tps(ds.upper(), task):
        X = adata.X[tp_vals == t][:, marker_idx]
        out[t] = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
    return out


def train_marker_pool(ds: str, task: str, marker_idx: np.ndarray, adata) -> np.ndarray:
    import scipy.sparse as sp
    tp_vals = np.asarray(adata.obs[io.TIME_KEY].values).astype(int)
    tr = _train_tps(ds.upper(), task, adata)
    X = adata.X[np.isin(tp_vals, tr)][:, marker_idx]
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def build_arm_cells(ds: str, task: str, seed: int, marker_idx: np.ndarray,
                    adata, arms: Sequence[str],
                    real_mk: Optional[Dict[int, np.ndarray]] = None,
                    train_pool: Optional[np.ndarray] = None,
                    ) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, np.ndarray], Dict[int, int]]:
    ds = ds.upper()
    tps = io.held_out_tps(ds, task)
    rng = np.random.default_rng(seed)
    if real_mk is None:
        real_mk = real_marker_by_tp(ds, task, marker_idx, adata)

    ref: Dict[int, np.ndarray] = {}
    halfB: Dict[int, np.ndarray] = {}
    nstar: Dict[int, int] = {}
    for t in tps:
        X = real_mk[t]
        ns = sampling.n_star(len(X))
        nstar[t] = ns
        a, b = controls.positive_split_half(X, ns, rng)
        ref[t], halfB[t] = a, b

    out: Dict[str, Dict[int, np.ndarray]] = {}

    if "pos_split_half" in arms:
        out["pos_split_half"] = dict(halfB)

    if "neg_shuffle" in arms:
        out["neg_shuffle"] = controls.negative_shuffle_tp(halfB, rng)

    if "neg_mean_noise" in arms:
        Xtr = (train_marker_pool(ds, task, marker_idx, adata)
               if train_pool is None else train_pool)
        out["neg_mean_noise"] = {t: controls.negative_mean_noise(Xtr, nstar[t], rng)
                                 for t in tps}

    for m in arms:
        if m in ("pos_split_half", "neg_shuffle", "neg_mean_noise"):
            continue
        b = (io.load_scditt(ds, task, seed) if m == "scDiTT"
             else io.load_baseline(m, ds, task, seed))
        pred = b.pred_by_tp()
        out[m] = {t: sampling.subsample(
            np.asarray(pred[t][:, marker_idx], dtype=np.float32), nstar[t],
            np.random.default_rng(seed * 1000 + t)) for t in tps}
    return out, ref, nstar


# ═════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════
def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    na = np.sqrt((Ac ** 2).sum(axis=1))
    nb = np.sqrt((Bc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Ac * Bc).sum(axis=1) / (na * nb)
    r[(na == 0) | (nb == 0)] = np.nan
    return np.clip(r, -1.0, 1.0)


def _rank_rows(A: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 1, np.asarray(A, float))


def curve_metrics(gen_mu: np.ndarray, ref_mu: np.ndarray,
                  se_bar: np.ndarray) -> Dict[str, np.ndarray]:
    r = _pearson_rows(gen_mu, ref_mu)
    rho = _pearson_rows(_rank_rows(gen_mu), _rank_rows(ref_mu))

    sd = ref_mu.std(axis=1, ddof=0)
    rmse = np.sqrt(((gen_mu - ref_mu) ** 2).mean(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_n = np.where(sd > 0, rmse / sd, np.nan)

    dg, dr = np.diff(gen_mu, axis=1), np.diff(ref_mu, axis=1)
    tol = (P.SHAPE_TOL_SE * se_bar)[:, None]
    sg = np.where(np.abs(dg) < tol, 0, np.sign(dg))
    sr = np.where(np.abs(dr) < tol, 0, np.sign(dr))
    dir_agree = (sg == sr).mean(axis=1) if dg.shape[1] else np.full(len(gen_mu), np.nan)
    return {"pearson_r": r, "spearman_rho": rho,
            "rmse_norm_by_time": rmse_n, "dir_agree": dir_agree, "rmse_raw": rmse}


def fisher_z_mean(r: Sequence[float]) -> float:
    v = np.asarray([x for x in np.asarray(r, float) if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan")
    v = np.clip(v, -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(v).mean()))


def arm_curves(cells: Dict[int, np.ndarray], tps: Sequence[int]) -> np.ndarray:
    return np.stack([np.asarray(cells[t], float).mean(axis=0) for t in tps], axis=1)


__all__ = ["symbol_index", "resolve_panel", "Resolved", "real_side", "RealSide",
           "real_marker_by_tp", "train_marker_pool",
           "build_arm_cells", "curve_metrics", "fisher_z_mean", "arm_curves"]
