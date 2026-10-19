from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parent.parent          # exps/task9_diversity
REPO_ROOT = EXP_ROOT.parent.parent                          # repository root
ARCHIVE_ROOT = Path(os.environ.get("SCDITT_ARCHIVE_ROOT", ""))
CKPT_ROOT = ARCHIVE_ROOT / "results" / "model_checkpoints"
DATA_ROOT = ARCHIVE_ROOT / "Data"
BASELINE_ROOT = Path(os.environ.get("BASELINE_RESULTS_ROOT", ""))

RUNS_DIR = EXP_ROOT / "runs"
FLOOR_DIR = EXP_ROOT / "floor"
METRICS_DIR = EXP_ROOT / "metrics"
FIGS_DIR = EXP_ROOT / "figs"
CACHE_DIR = EXP_ROOT / "cache"
LOGS_DIR = EXP_ROOT / "logs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ["ZB", "DR", "SC", "GA", "MA"]
TASKS = ["interpolation", "extrapolation"]
CFG_WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
CFG_MAIN = 3.0
K = 12
EVAL_SEEDS = list(range(K))
TRAIN_SEEDS = [42]

N_PRED_CELLS = 2000
N_EVAL = 256
NUM_REPEATS = 12
NUM_STEPS = 20

H5AD_NAME = {
    "ZB": "zebrafish_hvg_annotated.h5ad",
    "DR": "DR_Calderon_2022_hvg_annotated_subsampled.h5ad",
    "SC": "schiebinger_2019_preprocess_subset.h5ad",
    "GA": "garcia_alonso.h5ad",
    "MA": "ma_forecast.h5ad",
}

FLOOR_R_DEFAULT = 20
FLOOR_R_SMALL = 50
SMALL_STAGE_THRESHOLD = 1000


def h5ad_path(ds: str) -> Path:
    return DATA_ROOT / ds / H5AD_NAME[ds]


def run_dir(ds: str, task: str, seed: int) -> Path:
    return CKPT_ROOT / f"{ds}_{task}_seed{seed}"


def ckpt_path(ds: str, task: str, seed: int) -> Path:
    return run_dir(ds, task, seed) / "best.pt"


def vae_paths(ds: str, task: str, seed: int) -> tuple[Path, Path]:
    d = run_dir(ds, task, seed) / "vae_checkpoints"
    tag = f"{ds}_{task}_seed{seed}"
    return d / f"{tag}.pt", d / f"{tag}_latents.h5ad"


def fix_config(cfg: dict, ds: str, task: str, seed: int) -> dict:
    cfg = dict(cfg)
    cfg["dataset"] = dict(cfg["dataset"])
    vae_ck, lat = vae_paths(ds, task, seed)
    for p, key in ((vae_ck, "vae_ckpt_path"), (lat, "latents_path")):
        if not p.exists():
            raise FileNotFoundError(f"missing {key}: {p}")
    cfg["dataset"]["vae_ckpt_path"] = str(vae_ck)
    cfg["dataset"]["latents_path"] = str(lat)
    hp = h5ad_path(ds)
    if not hp.exists():
        raise FileNotFoundError(f"missing h5ad: {hp}")
    cfg["dataset"]["h5ad_path"] = str(hp)
    return cfg


def pred_dir(ds: str, task: str, cfg_w: float, train_seed: int,
             eval_seed: int) -> Path:
    return (RUNS_DIR / "scDiTT" / ds / task / f"cfg{cfg_w:g}" /
            f"trainseed{train_seed}" / f"evalseed{eval_seed}")


_REAL_CACHE: dict[str, dict] = {}


def load_real_by_tp(ds: str) -> dict[int, np.ndarray]:
    """{tp: [N_t, G]} gene-space (log-norm) real cells.

    The read logic matches ``src/evaluation/inference.py:63-76`` (np.isclose on obs['tp']).
    """
    if ds in _REAL_CACHE:
        return _REAL_CACHE[ds]
    import anndata as ad
    import scipy.sparse as sp
    from src.data.splits import ALL_TPS, TIME_KEY

    adata = ad.read_h5ad(str(h5ad_path(ds)))
    X = adata.X
    X = X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    col = np.asarray(adata.obs[TIME_KEY[ds]].values).astype(float)
    out = {}
    for t in ALL_TPS[ds]:
        out[int(t)] = X[np.where(np.isclose(col, float(t)))[0]]
    _REAL_CACHE[ds] = out
    return out


def n_t_for(N_t: int) -> int:
    return int(min(N_PRED_CELLS, N_t // 2))


def get_pca(ds: str, verbose: bool = True):
    import joblib
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"pca_{ds}.joblib"
    if p.exists():
        return joblib.load(p)
    from src.data.splits import ALL_TPS
    from src.evaluation import unified_metrics as M
    real = load_real_by_tp(ds)
    if verbose:
        print(f"[pca] fitting PCA-50 for {ds} ...", flush=True)
    pca = M.fit_pca_on_all(real, ALL_TPS[ds])
    joblib.dump(pca, p)
    return pca


def set_gpu(gpu: int | str | None) -> str:
    if gpu is None:
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0"
