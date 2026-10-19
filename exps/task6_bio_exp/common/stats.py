from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    ps = p[ok]
    n = ps.size
    order = np.argsort(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def ks_stat(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import ks_2samp
    r = ks_2samp(np.asarray(a, float), np.asarray(b, float))
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def paired_t(ctrl: Sequence[float], prtb: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import ttest_rel
    c, p_ = np.asarray(ctrl, float), np.asarray(prtb, float)
    if c.shape != p_.shape:
        raise ValueError(f"a paired test requires equal lengths: {c.shape} vs {p_.shape}")
    r = ttest_rel(p_, c)
    d = p_ - c
    return {"t": float(r.statistic), "p": float(r.pvalue),
            "mean_diff": float(d.mean()),
            "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
            "n_pairs": int(len(d))}


def spearman_monotonic(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


def permutation_test(observed: float, null_samples: Sequence[float],
                     alternative: str = "greater") -> float:
    null = np.asarray(null_samples, float)
    n = len(null)
    if alternative == "greater":
        k = int((null >= observed).sum())
    elif alternative == "less":
        k = int((null <= observed).sum())
    else:
        k = int((np.abs(null) >= abs(observed)).sum())
    return (k + 1) / (n + 1)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def macro_mean(per_tp: Dict[int, float]) -> float:
    v = np.array([x for x in per_tp.values() if not (x is None or np.isnan(x))], float)
    return float(v.mean()) if v.size else float("nan")


def aggregate_seeds(per_seed: Dict[int, float]) -> Dict[str, float]:
    v = np.array([x for x in per_seed.values() if not (x is None or np.isnan(x))], float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n": int(v.size)}


def normalized_by_positive_control(value: float, pos_control: float) -> float:
    if pos_control is None or np.isnan(pos_control) or abs(pos_control) < 1e-12:
        return float("nan")
    return float(value / pos_control)


__all__ = ["bh_fdr", "cohens_d", "ks_stat", "paired_t", "spearman_monotonic",
           "permutation_test", "macro_mean", "aggregate_seeds",
           "normalized_by_positive_control"]
