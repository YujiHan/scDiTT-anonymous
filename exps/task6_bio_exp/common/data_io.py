from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
#
#
#     export BASELINE_ROOT_A=/path/to/baselines/A         # PI-SDE / scIMF / scNODE
#     export BASELINE_ROOT_B=/path/to/baselines/B         # MIOFlow
#     export BASELINE_ROOT_C=/path/to/baselines/C         # PRESCIENT / scDiffEq
#
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]      # .../scDiTT
_RUNS_DEFAULT = _REPO_ROOT.parent / "scDiTT_runs"
_BL_DEFAULT = _REPO_ROOT.parent / "baseline_results"


def _env_path(var: str, default: Path) -> Path:
    import os
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


RUNS_ROOT = _env_path("SCDITT_RUNS_ROOT", _RUNS_DEFAULT)
SCDITT_RESULTS = RUNS_ROOT / "scditt_results"
REAL_DATA_ROOT = RUNS_ROOT / "Data"

BASELINE_REDONE = _env_path("BASELINE_REDONE_ROOT", _BL_DEFAULT / "redone" / "baseline_results")
_BL_A = _env_path("BASELINE_ROOT_A", _BL_DEFAULT / "A" / "baseline_results")
_BL_B = _env_path("BASELINE_ROOT_B", _BL_DEFAULT / "B" / "baseline_results")
_BL_C = _env_path("BASELINE_ROOT_C", _BL_DEFAULT / "C" / "baseline_results")

BASELINE_OWN_PARENT: Dict[str, Path] = {
    "PI-SDE": _BL_A, "scIMF": _BL_A, "scNODE": _BL_A,
    "MIOFlow": _BL_B,
    "PRESCIENT": _BL_C, "scDiffEq": _BL_C,
}
ALL_METHODS: List[str] = ["PI-SDE", "scIMF", "scNODE", "MIOFlow", "PRESCIENT", "scDiffEq"]
ALL_DATASETS: List[str] = ["ZB", "DR", "GA", "MA", "SC"]
ALL_SEEDS: List[int] = [42, 43, 44, 45, 46]

_REAL_FILE = {
    "ZB": "zebrafish_hvg_annotated.h5ad",
    "DR": "DR_Calderon_2022_hvg_annotated_subsampled.h5ad",
    "GA": "garcia_alonso.h5ad",
    "MA": "ma_forecast.h5ad",
    "SC": "schiebinger_2019_preprocess_subset.h5ad",
}
TIME_KEY = "tp"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _import_splits():
    try:
        from src.data import splits  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "cannot import src.data.splits -- run from the scDiTT_code/ root directory "
            "with PYTHONPATH=. (see the environment notes)"
        ) from e
    return splits


def n_tps(dataset: str) -> int:
    return len(_import_splits().ALL_TPS[dataset.upper()])


def held_out_tps(dataset: str, task: str, ex_step: Optional[int] = None) -> List[int]:
    splits = _import_splits()
    _, test = splits.get_tp_split(dataset.upper(), task, n_steps=ex_step)
    return [int(t) for t in test]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class PredBundle:

    source: str
    dataset: str
    task: str
    seed: int
    array: np.ndarray
    held_out_tps: List[int]
    scoring_eval_tps: List[int]
    metrics: Optional[dict]
    path: Path
    provenance: str = "scDiTT"
    ex_step: Optional[int] = None

    def pred_by_tp(self, tps: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
        want = list(self.held_out_tps) if tps is None else [int(t) for t in tps]
        T = self.array.shape[0]
        bad = [t for t in want if not (0 <= t < T)]
        if bad:
            raise IndexError(f"tp out of range {bad}; this run has T={T}")
        return {t: np.ascontiguousarray(self.array[t]) for t in want}

    @property
    def n_genes(self) -> int:
        return int(self.array.shape[2])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<PredBundle {self.source} {self.dataset}/{self.task}"
                f"{'' if self.ex_step is None else f'/ex{self.ex_step}'}"
                f"/seed{self.seed} shape={tuple(self.array.shape)} "
                f"held_out={self.held_out_tps} src={self.provenance}>")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def scditt_run_dir(dataset: str, task: str, seed: int,
                   ex_step: Optional[int] = None) -> Path:
    ds = dataset.upper()
    if task == "extrapolation_steps":
        if ex_step is None:
            raise ValueError("extrapolation_steps requires ex_step")
        return SCDITT_RESULTS / ds / task / f"ex{ex_step}" / f"seed{seed}"
    return SCDITT_RESULTS / ds / task / f"seed{seed}"


def _read_metrics(run_dir: Path, scheme: Optional[str]) -> Optional[dict]:
    """Read ``metrics.json``; ``extrapolation_steps`` reads ``metrics_{scheme}.json``."""
    cands = ([run_dir / f"metrics_{scheme}.json"] if scheme
             else [run_dir / "metrics.json"])
    for p in cands:
        if p.exists():
            return json.loads(p.read_text())
    return None


def load_scditt(dataset: str, task: str, seed: int,
                ex_step: Optional[int] = None,
                scheme: Optional[str] = None,
                mmap: bool = True) -> PredBundle:
    """Load scDiTT's ``pred_genes.npy`` + metrics.

    Parameters
    ----------
    scheme : only needed for ``extrapolation_steps``; ``"lasttp"`` or ``"alltp"``,
             selecting which metrics file to read (**does not affect** the held-out decision).
    """
    ds = dataset.upper()
    d = scditt_run_dir(ds, task, seed, ex_step)
    pred_path = d / "pred_genes.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing pred_genes.npy: {pred_path}")

    if task == "extrapolation_steps" and scheme is None:
        scheme = "lasttp"
    m = _read_metrics(d, scheme if task == "extrapolation_steps" else None)

    arr = np.load(pred_path, mmap_mode="r" if mmap else None)
    ho = held_out_tps(ds, task, ex_step)
    _assert_pred_shape(arr, ds, source="scDiTT", path=pred_path)

    return PredBundle(
        source="scDiTT", dataset=ds, task=task, seed=seed, array=arr,
        held_out_tps=ho,
        scoring_eval_tps=[int(t) for t in (m or {}).get("eval_tps", ho)],
        metrics=m, path=d, provenance="scDiTT", ex_step=ex_step,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _dir_ok(d: Path) -> bool:
    """Matches task1 ``baseline_umap_visualize.py::_dir_ok``:
    ``prediction.npy`` exists **and** ``meta.json`` has ``status == "ok"``."""
    mp, pp = d / "meta.json", d / "prediction.npy"
    if not mp.exists() or not pp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "ok"
    except Exception:
        return False


def resolve_baseline_dir(method: str, dataset: str, task: str, seed: int):
    if method not in BASELINE_OWN_PARENT:
        raise KeyError(f"unknown baseline: {method}; available {ALL_METHODS}")
    ds = dataset.upper()
    red = BASELINE_REDONE / method / ds / task / f"seed{seed}"
    own = BASELINE_OWN_PARENT[method] / method / ds / task / f"seed{seed}"
    if _dir_ok(red):
        return red, "REDONE"
    if _dir_ok(own):
        return own, "own"
    return None, None


def load_baseline(method: str, dataset: str, task: str, seed: int,
                  mmap: bool = True) -> PredBundle:
    ds = dataset.upper()
    d, prov = resolve_baseline_dir(method, ds, task, seed)
    if d is None:
        raise FileNotFoundError(
            f"baseline artefact unavailable: {method}/{ds}/{task}/seed{seed}"
            " (neither the REDONE nor the own root has a status==ok result)")
    arr = np.load(d / "prediction.npy", mmap_mode="r" if mmap else None)
    ho = held_out_tps(ds, task, None)
    _assert_pred_shape(arr, ds, source=method, path=d / "prediction.npy")

    m = None
    if (d / "metrics.json").exists():
        try:
            m = json.loads((d / "metrics.json").read_text())
        except Exception:
            m = None
    return PredBundle(
        source=method, dataset=ds, task=task, seed=seed, array=arr,
        held_out_tps=ho,
        scoring_eval_tps=[int(t) for t in (m or {}).get("eval_tps", ho)],
        metrics=m, path=d, provenance=prov,
    )


def baseline_coverage(datasets: Optional[Iterable[str]] = None,
                      tasks: Sequence[str] = ("interpolation", "extrapolation"),
                      seeds: Optional[Iterable[int]] = None) -> "list[dict]":
    datasets = list(datasets or ALL_DATASETS)
    seeds = list(seeds or ALL_SEEDS)
    rows = []
    for m in ALL_METHODS:
        for ds in datasets:
            for t in tasks:
                for s in seeds:
                    d, prov = resolve_baseline_dir(m, ds, t, s)
                    rows.append({"method": m, "dataset": ds, "task": t, "seed": s,
                                 "ok": d is not None, "provenance": prov,
                                 "path": str(d) if d else ""})
    return rows


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def real_h5ad_path(dataset: str) -> Path:
    ds = dataset.upper()
    if ds not in _REAL_FILE:
        raise KeyError(f"unknown dataset {ds}")
    p = REAL_DATA_ROOT / ds / _REAL_FILE[ds]
    if not p.exists():
        raise FileNotFoundError(
            f"real data not found: {p}\n"
            "note: the training-source Data tree must be used, not the workspace Data/")
    return p


def load_real(dataset: str, backed: Optional[str] = None):
    import anndata as ad
    return ad.read_h5ad(real_h5ad_path(dataset), backed=backed)


def real_by_tp(dataset: str, tps: Optional[Sequence[int]] = None,
               adata=None) -> Dict[int, np.ndarray]:
    import scipy.sparse as sp
    a = load_real(dataset) if adata is None else adata
    tp_vals = np.asarray(a.obs[TIME_KEY].values).astype(int)
    want = sorted(np.unique(tp_vals).tolist()) if tps is None else [int(t) for t in tps]
    out: Dict[int, np.ndarray] = {}
    for t in want:
        sel = tp_vals == t
        if not sel.any():
            raise ValueError(f"{dataset} tp={t} has no real cells")
        X = a.X[sel]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        out[t] = np.ascontiguousarray(X, dtype=np.float32)
    return out


def latents_path(dataset: str, task: str, seed: int,
                 ex_step: Optional[int] = None) -> Path:
    ds = dataset.upper()
    d = scditt_run_dir(ds, task, seed, ex_step) / "vae_checkpoints"
    tag = task if ex_step is None else f"{task}_{ex_step}"
    p = d / f"{ds}_{tag}_seed{seed}_latents.h5ad"
    if p.exists():
        return p
    cands = sorted(d.glob("*_latents.h5ad"))
    if not cands:
        raise FileNotFoundError(f"latents.h5ad not found: {d}")
    return cands[0]


def load_labels(dataset: str, task: str, seed: int,
                ex_step: Optional[int] = None):
    import anndata as ad
    lat = ad.read_h5ad(latents_path(dataset, task, seed, ex_step))
    return lat.obs.copy()


def vae_ckpt_path(dataset: str, task: str, seed: int,
                  ex_step: Optional[int] = None) -> Path:
    ds = dataset.upper()
    d = scditt_run_dir(ds, task, seed, ex_step) / "vae_checkpoints"
    tag = task if ex_step is None else f"{task}_{ex_step}"
    p = d / f"{ds}_{tag}_seed{seed}.pt"
    if p.exists():
        return p
    cands = [q for q in sorted(d.glob("*.pt"))]
    if not cands:
        raise FileNotFoundError(f"VAE weights not found: {d}")
    return cands[0]


def load_vae(dataset: str, task: str, seed: int,
             ex_step: Optional[int] = None, device: str = "cpu"):
    import torch
    from src.data.vae import CellVAE  # type: ignore
    ck = torch.load(vae_ckpt_path(dataset, task, seed, ex_step),
                    map_location=device, weights_only=False)
    m = CellVAE(gene_dim=ck["gene_dim"], latent_dim=ck["latent_dim"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m.to(device)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _assert_pred_shape(arr: np.ndarray, dataset: str, source: str, path: Path) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{source} prediction must be 3-D [T, cells, genes], "
                         f"got {arr.shape}; {path}")
    T_expect = n_tps(dataset)
    if arr.shape[0] != T_expect:
        raise ValueError(
            f"{source}/{dataset} has T={arr.shape[0]}, which differs from the value declared by splits {T_expect}; "
            f"most likely the wrong data version was used; {path}")


def assert_gene_dim_match(dataset: str, *arrays: np.ndarray) -> None:
    import anndata as ad
    g = ad.read_h5ad(real_h5ad_path(dataset), backed="r").shape[1]
    for a in arrays:
        if a.shape[-1] != g:
            raise ValueError(
                f"{dataset} gene dimension mismatch: prediction {a.shape[-1]} vs real {g}; "
                "please confirm the real data comes from the training-source Data tree")


__all__ = [
    "RUNS_ROOT", "SCDITT_RESULTS", "REAL_DATA_ROOT", "BASELINE_REDONE",
    "ALL_METHODS", "ALL_DATASETS", "ALL_SEEDS", "TIME_KEY",
    "PredBundle", "n_tps", "held_out_tps",
    "scditt_run_dir", "load_scditt",
    "resolve_baseline_dir", "load_baseline", "baseline_coverage",
    "real_h5ad_path", "load_real", "real_by_tp",
    "latents_path", "load_labels", "vae_ckpt_path", "load_vae",
    "assert_gene_dim_match",
]
