from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                 # .../exps/task5_ablation/5-1_OT
REPO_ROOT = HERE.parents[2]
OUT_ROOT = REPO_ROOT / "results" / "task5_ablation" / "5-1_OT"
DATA_DIR = OUT_ROOT / "data"
COUPLING_DIR = OUT_ROOT / "couplings"
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"
METRIC_DIR = OUT_ROOT / "metrics"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

T = 10
K = 2000
D = 64
TIME_KEY = "tp"

SPLITS: dict[str, tuple[list[int], list[int]]] = {
    "interp": ([0, 1, 2, 4, 5, 7, 8, 9], [3, 6]),
    "extrap": ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
}


def get_split(task: str = "interp") -> tuple[list[int], list[int], list]:
    """task -> (train_tps, test_tps, train_pairs)."""
    tr, te = SPLITS[task]
    return list(tr), list(te), list(zip(tr[:-1], tr[1:]))


TRAIN_TPS, TEST_TPS, TRAIN_PAIRS = get_split("interp")

# The two synthetic branching courses of Table S4: "Two branches" and
# "Three branches". Both are generated directly in the 64-d latent space, so the
# autoencoder is bypassed and only the coupling is under test.
REGIMES = ["bifurcation", "trifurcation"]
SIGMAS = [0.10, 0.25, 0.50]
SIGMA_MAIN = 0.25

ARMS = ["true", "ot", "random"]

OT_EPSILON = 0.05


def tag(regime: str, sigma: float) -> str:
    return f"{regime}_s{sigma:g}"


def sim_h5ad_path(regime: str, sigma: float, seed: int) -> Path:
    return DATA_DIR / tag(regime, sigma) / f"sim_latents_seed{seed}.h5ad"


def _arm_sub(arm: str, task: str) -> str:
    return arm if task == "interp" else f"{task}_{arm}"


def coupling_dir(regime: str, sigma: float, seed: int, arm: str,
                 task: str = "interp") -> Path:
    return COUPLING_DIR / tag(regime, sigma) / f"seed{seed}" / _arm_sub(arm, task)


def ckpt_dir(regime: str, sigma: float, seed: int, arm: str,
             task: str = "interp") -> Path:
    return CKPT_DIR / tag(regime, sigma) / f"seed{seed}" / _arm_sub(arm, task)


def global_row(t: int, p: int) -> int:
    return t * K + p


def tp_row_range(t: int) -> np.ndarray:
    return np.arange(t * K, (t + 1) * K, dtype=np.int64)


def load_sim(path: str | Path) -> dict:
    import anndata as ad
    adata = ad.read_h5ad(str(path))
    X = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    tp = np.asarray(adata.obs[TIME_KEY].values).astype(int)
    particle = np.asarray(adata.obs["particle_id"].values).astype(int)
    branch = np.asarray(adata.obs["branch"].values).astype(int)
    return {"X": X, "tp": tp, "particle_id": particle, "branch": branch}
