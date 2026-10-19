from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class RealFittedEmbedding:
    """A dimensionality reducer fitted once on real data and shared by every arm.

    Examples
    --------
    >>> emb = RealFittedEmbedding.fit(real_X, n_pcs=50, seed=0)
    >>> Z_real = emb.transform(real_X)
    >>> Z_gen  = emb.transform(gen_X)      # same projection, comparable
    """

    n_pcs: int
    seed: int
    _pca: object = field(repr=False, default=None)
    _umap: object = field(repr=False, default=None)
    n_genes: int = 0

    @classmethod
    def fit(cls, real: np.ndarray, n_pcs: int = 50, seed: int = 0
            ) -> "RealFittedEmbedding":
        from sklearn.decomposition import PCA
        real = np.asarray(real, dtype=np.float32)
        k = int(min(n_pcs, real.shape[0] - 1, real.shape[1]))
        pca = PCA(n_components=k, random_state=seed).fit(real)
        return cls(n_pcs=k, seed=seed, _pca=pca, n_genes=int(real.shape[1]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_genes:
            raise ValueError(f"gene dimension mismatch: {X.shape[1]} vs the value seen at fit time {self.n_genes}")
        return np.ascontiguousarray(self._pca.transform(X), dtype=np.float32)

    def transform_all(self, arms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: self.transform(v) for k, v in arms.items()}

    def fit_umap(self, real: np.ndarray, n_neighbors: int = 15,
                 min_dist: float = 0.5) -> "RealFittedEmbedding":
        import umap
        Z = self.transform(real)
        self._umap = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                               random_state=self.seed).fit(Z)
        return self

    def umap_transform(self, X: np.ndarray) -> np.ndarray:
        if self._umap is None:
            raise RuntimeError("call fit_umap(real) first")
        return np.ascontiguousarray(self._umap.transform(self.transform(X)))

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_


def fit_on_real(real_by_tp: Dict[int, np.ndarray], tps: Optional[Sequence[int]] = None,
                n_pcs: int = 50, seed: int = 0,
                max_cells: int = 20000) -> RealFittedEmbedding:
    use = sorted(real_by_tp) if tps is None else [int(t) for t in tps]
    X = np.concatenate([real_by_tp[t] for t in use], axis=0)
    if len(X) > max_cells:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_cells, replace=False)]
    return RealFittedEmbedding.fit(X, n_pcs=n_pcs, seed=seed)


__all__ = ["RealFittedEmbedding", "fit_on_real"]
