from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import unified_metrics as M

METRIC_KEYS = ["w2_avg", "mmd_avg", "pearson_mean", "sampled_pearson", "pg_w1_mean"]
PER_TP_KEYS = ["w2", "mmd", "pearson_mean", "sampled_pearson", "pg_w1"]


def eval_tps(task: str, n_tps: int, test_tps: Sequence[int],
             scheme: Optional[str] = None) -> List[int]:
    """The time points the Evaluator scores.

    * interpolation / extrapolation / all -> the held-out ``test_tps``.
    * extrapolation_steps -> fixed sets *independent* of ex_step:
        ``scheme="alltp"``  -> all tps ``0..n_tps-1``
        ``scheme="lasttp"`` -> only ``[n_tps-1]``
    """
    if task in ("interpolation", "extrapolation", "all"):
        return [int(t) for t in test_tps]
    if task == "extrapolation_steps":
        if scheme == "alltp":
            return list(range(n_tps))
        if scheme == "lasttp":
            return [n_tps - 1]
        raise ValueError("extrapolation_steps needs scheme in {'alltp','lasttp'}")
    raise ValueError(f"unknown task: {task}")


def _w2_oom_safe(pred: np.ndarray, real: np.ndarray, pca, device_obj) -> float:
    pred_pca = pca.transform(pred) if pca is not None else pred
    real_pca = pca.transform(real) if pca is not None else real
    try:
        return M.sinkhorn_w2(pred_pca, real_pca, device=device_obj)
    except RuntimeError as e:
        dev = (device_obj if isinstance(device_obj, torch.device)
               else torch.device(device_obj))
        if dev.type == "cuda" and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return M.sinkhorn_w2(pred_pca, real_pca, device=torch.device("cpu"))
        raise


def score(pred_by_tp: Dict[int, np.ndarray],
          real_by_tp: Dict[int, np.ndarray],
          tps: Sequence[int],
          all_tps: Sequence[int],
          include_w2: bool = True,
          device=None) -> dict:
    """Compute ``{"avg": {...5...}, "per_tp": {t: {...5...}}}`` for one prediction.

    Aligned verbatim with ``common/evaluator.py::evaluate``:
      * the PCA used for W2 is fitted once on ``all_tps`` (the complete real trajectory) and reused for every tp;
      * ``avg`` = ``compute_all_metrics`` (W2/MMD/SampledPearson are per-tp means,
        PearsonMean/perGeneW1 are pooled across tps);
      * ``per_tp[t]`` = the 5 single-pair metrics, where ``sampled_pearson`` uses
        ``seed=EVAL_SEED + t`` and ``mmd`` uses the ``(real, pred)`` order (matching the baseline).

    Parameters
    ----------
    pred_by_tp : ``{t: [n_cells, n_genes]}`` predictions (gene space), indexed by absolute tp.
    real_by_tp : ``{t: [n_cells, n_genes]}`` real data (gene space).
    tps        : the absolute tps to score (see :func:`eval_tps`).
    all_tps    : the complete real trajectory -- the PCA for W2 is fitted on these tps.
    include_w2 : set False when torch/geomloss is missing; W2 is then recorded as NaN.
    """
    tps = [int(t) for t in tps]
    pred = {t: np.ascontiguousarray(pred_by_tp[t]) for t in tps}

    pca = device_obj = None
    if include_w2:
        pca = M.fit_pca_on_all(real_by_tp, all_tps)
        device_obj = M.get_device(device if device is None else str(device))

    avg = M.compute_all_metrics(
        pred, real_by_tp, tps, all_tps=list(all_tps),
        pca=None, device=None, include_w2=False,
    )
    if include_w2:
        w2s = [_w2_oom_safe(pred[t], real_by_tp[t], pca, device_obj) for t in tps]
        avg["w2_avg"] = float(np.mean(w2s)) if w2s else float("nan")

    per_tp: Dict[int, dict] = {}
    for t in tps:
        p, r = pred[t], real_by_tp[t]
        per_tp[t] = {
            "w2": (_w2_oom_safe(p, r, pca, device_obj)
                   if include_w2 else float("nan")),
            "mmd": M.mmd(r, p),
            "pearson_mean": M.pearson_mean(p, r),
            "sampled_pearson": M.sampled_pearson(p, r, seed=M.EVAL_SEED + t),
            "pg_w1": M.per_gene_w1_mean(p, r),
        }
    return {"avg": avg, "per_tp": per_tp}


def w2_only(pred_by_tp: Dict[int, np.ndarray],
            real_by_tp: Dict[int, np.ndarray],
            tps: Sequence[int],
            all_tps: Sequence[int],
            device=None) -> dict:
    tps = [int(t) for t in tps]
    if not tps:
        return {"avg_wd": float("nan"), "per_tp_wd": {}}
    pca = M.fit_pca_on_all(real_by_tp, all_tps)
    dev = M.get_device(device if device is None else str(device))
    per_tp = {t: _w2_oom_safe(pred_by_tp[t], real_by_tp[t], pca, dev) for t in tps}
    return {"avg_wd": float(np.mean(list(per_tp.values()))),
            "per_tp_wd": per_tp}


def aggregate(metric_dicts: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)) if len(vals) else float("nan"),
                  "std": float(np.nanstd(vals)) if len(vals) else float("nan")}
    return out
