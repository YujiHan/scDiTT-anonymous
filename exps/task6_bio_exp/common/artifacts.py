from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional

import numpy as np

EPS_ABS = 1e-3
EPS_REL = 1e-2

ZERO_SENSITIVE = frozenset({
    "dropout_rate", "ks", "ks_stat", "zero_inflation", "p_zero",
    "gene_gene_corr", "spearman_pairwise", "grn_edge",
})


def exact_zero_rate(X: np.ndarray) -> float:
    return float((np.asarray(X) == 0).mean())


def negative_rate(X: np.ndarray) -> float:
    return float((np.asarray(X) < 0).mean())


def near_zero_rate(X: np.ndarray, eps_abs: float = EPS_ABS,
                   eps_rel: Optional[float] = EPS_REL) -> float:
    A = np.asarray(X, dtype=np.float64)
    thr = np.full(A.shape[1], eps_abs, dtype=np.float64)
    if eps_rel is not None:
        with np.errstate(all="ignore"):
            pos = np.where(A > 0, A, np.nan)
            med = np.nanmedian(pos, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        thr = np.maximum(thr, eps_rel * med)
    return float((np.abs(A) <= thr[None, :]).mean())


def profile(X: np.ndarray, name: str = "") -> Dict[str, float]:
    A = np.asarray(X)
    return {
        "name": name,
        "n_cells": int(A.shape[0]),
        "n_genes": int(A.shape[1]),
        "exact_zero_rate": exact_zero_rate(A),
        "negative_rate": negative_rate(A),
        "near_zero_rate": near_zero_rate(A),
        "mean": float(A.mean()),
        "std": float(A.std()),
    }


def compare_profiles(real: np.ndarray, gen: np.ndarray,
                     gen_name: str = "generated") -> Dict[str, Dict[str, float]]:
    return {"real": profile(real, "real"), gen_name: profile(gen, gen_name)}


def warn_if_zero_sensitive(metric_name: str) -> None:
    key = metric_name.lower()
    if any(k in key for k in ZERO_SENSITIVE):
        warnings.warn(
            f"metric '{metric_name}' is sensitive to zero structure: real data is ~92-95% exact zeros, "
            "while scDiTT and all 6 baselines have 0% exact zeros."
            "this metric is dominated by the decoder densification artefact; either (1) use a proxy such as near_zero_rate, "
            "or (2) also report the cross-method ranking (all methods share the same artefact, so the ranking stays fair), "
            "and state in the caption that the positive control preserves zero mass and is an architecturally unreachable upper bound.",
            RuntimeWarning, stacklevel=2)


def clip_nonneg(X: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(X), 0.0, None)


__all__ = ["EPS_ABS", "EPS_REL", "ZERO_SENSITIVE",
           "exact_zero_rate", "negative_rate", "near_zero_rate",
           "profile", "compare_profiles", "warn_if_zero_sensitive", "clip_nonneg"]
