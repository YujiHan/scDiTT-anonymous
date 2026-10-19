from __future__ import annotations

import numpy as np
import torch

from src.data.splits import train_cell_mask
from src.data.vae import encode_all, train_vae


def _toy(n_per_tp: int = 30, G: int = 6, tps=(0, 1, 2)):
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(float(t), 0.3, size=(n_per_tp, G))
                        for t in tps]).astype(np.float32)
    tp_col = np.repeat(np.asarray(tps), n_per_tp)
    return X, tp_col


def _fit(X, fit_indices, seed=0):
    torch.manual_seed(seed)
    model, hist = train_vae(
        X=X, gene_dim=X.shape[1], latent_dim=3, hidden=(8,), epochs=3,
        batch_size=16, lr=1e-3, beta=0.1, device="cpu", seed=seed,
        fit_indices=fit_indices)
    return model, hist


def test_non_fit_rows_never_touch_the_optimizer():
    X, tp_col = _toy()
    mask = train_cell_mask(tp_col, [0, 2])
    X[~mask] = np.nan
    model, hist = _fit(X, np.flatnonzero(mask))

    assert all(np.isfinite(v) for v in hist["total"]), hist["total"]
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} was contaminated into NaN/Inf by held-out rows"


def test_fit_indices_matches_pre_sliced_matrix():
    X, tp_col = _toy()
    mask = train_cell_mask(tp_col, [0, 2])
    m_idx, _ = _fit(X.copy(), np.flatnonzero(mask))
    m_sliced, _ = _fit(X[mask].copy(), None)
    for (n_a, p_a), (_, p_b) in zip(m_idx.named_parameters(),
                                    m_sliced.named_parameters()):
        assert torch.equal(p_a, p_b), f"{n_a} mismatch"


def test_encode_covers_all_cells_including_held_out():
    X, tp_col = _toy()
    mask = train_cell_mask(tp_col, [0, 2])
    model, _ = _fit(X, np.flatnonzero(mask))
    Z = encode_all(model, X, device="cpu")
    assert Z.shape == (X.shape[0], 3)
    assert np.isfinite(Z).all()
