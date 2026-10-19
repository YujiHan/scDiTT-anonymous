"""[compatibility shim] The old single-metric PCA-Sinkhorn W2 evaluation interface.

Now that the evaluation protocol is aligned with the published baselines, the mathematics of every metric
is provided uniformly by :mod:`unified_metrics`, and the per-tp organisation by :mod:`scoring`. This
module only keeps the backward-compatible function names (``compute_wd`` / ``compute_sinkhorn_w2`` /
``fit_pca_on_all`` / ``w2_only``); ``compute_wd`` is the one still called, by
``exps/task5_ablation/5-1_OT/`` (``traj_eval.py`` / ``train_eval.py``).
The W2 values agree with the baseline protocol bit for bit (the Sinkhorn divergence is symmetric, and
distances in PCA space are invariant to the sign of the basis vectors).

New code should use :mod:`src.evaluation.scoring` (the baseline 5 metrics + the alltp/lasttp schemes)
and :mod:`src.evaluation.evaluate` directly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from . import unified_metrics as M
from .scoring import w2_only  # noqa: F401  (re-exported for backward compatibility)


def compute_wd(
    pred_pca: np.ndarray,
    true_pca: np.ndarray,
    device: torch.device | str,
    blur: float = 0.05,
    p: int = 2,
    scaling: float = 0.5,
) -> float:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    pred = np.asarray(pred_pca)
    true = np.asarray(true_pca)
    try:
        return M.sinkhorn_w2(pred, true, device=dev, blur=blur, p=p, scaling=scaling)
    except RuntimeError as e:
        if dev.type == "cuda" and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"[metrics] CUDA OOM on {dev} "
                  f"(true={true.shape[0]}, pred={pred.shape[0]} cells); "
                  f"falling back to CPU recomputation (numerically identical, just slower).")
            return M.sinkhorn_w2(pred, true, device=torch.device("cpu"),
                                 blur=blur, p=p, scaling=scaling)
        raise


compute_sinkhorn_w2 = compute_wd


def fit_pca_on_all(
    true_gene_by_tp: dict[int, np.ndarray],
    all_tps: Iterable[int],
    n_components: int = 50,
):
    """Concatenate the real gene-space data of all tps and fit PCA-50 on it (arpack, seed=0).

    Delegates to ``unified_metrics.fit_pca_on_all`` (matching the baseline).
    """
    return M.fit_pca_on_all(true_gene_by_tp, list(all_tps),
                            n_components=n_components)
