from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ALL_TPS: dict[str, list[int]] = {
    "ZB": list(range(12)),  # zebrafish:      tp 0..11
    "DR": list(range(11)),  # drosophila:     tp 0..10
    "SC": list(range(19)),  # schiebinger:    tp 0..18
    "GA": list(range(11)),  # garcia_alonso:  PCW 9..21 -> tp 0..10
    "MA": list(range(8)),   # ma_forecast:    PCW 4..11 -> tp 0..7
}

TIME_KEY: dict[str, str] = {
    "ZB": "tp",
    "DR": "tp",
    "SC": "tp",
    "GA": "tp",
    "MA": "tp",
}

# obs column carrying the *physical* acquisition time of each cell.
#
# obs['tp'] is always the contiguous stage index 0..T-1, so consecutive stages
# are one unit apart whatever the experiment actually did. The columns below
# keep the original units and are therefore unevenly spaced wherever the
# sampling was uneven -- ZB, DR and GA are all irregular, SC and MA are not.
# They are what `real_time_key` in configs/dataset/*.yaml selects; see
# `t_real_from_obs`.
REAL_TIME_KEY: dict[str, str] = {
    "ZB": "HPF",       # hours post fertilisation: 3.3 3.8 4.3 4.7 5.3 6 7 ... 12
    "DR": "hrs_real",  # midpoint of the collection window, in hours: 1 2 3 5 6 8 ... 18
    "SC": "day",       # reprogramming day: 0 .. 18
    "GA": "pcw_real",  # post-conception week, the 11 weeks occurring between 9 and 21
    "MA": "pcw_real",  # post-conception week: 4 .. 11
}


def t_real_from_obs(obs, time_key: str = "tp",
                    real_time_key: str | None = None) -> list[float]:
    """One time value per stage, ordered by stage index.

    Naming a column -- which every shipped ``configs/dataset/*.yaml`` does --
    returns the physical acquisition time of each stage in its original units, so
    the spacing handed to ``PE(t_k)`` reflects the real sampling intervals rather
    than a uniform grid. ``real_time_key=None`` falls back on the stage index
    itself, ``0, 1, ..., T-1``.

    The column must carry exactly one value per stage and increase strictly
    along the stage axis; both are checked here rather than left to fail
    silently downstream.
    """
    tp = np.asarray(obs[time_key].values).astype(float)
    stages = np.sort(np.unique(tp))
    if real_time_key is None or real_time_key == time_key:
        return [float(s) for s in stages]

    if real_time_key not in obs.columns:
        raise KeyError(
            f"real_time_key={real_time_key!r} is not a column of obs. "
            f"Available columns: {sorted(map(str, obs.columns))}")
    raw = pd.to_numeric(pd.Series(np.asarray(obs[real_time_key].values)),
                        errors="coerce").to_numpy(dtype=float)

    out: list[float] = []
    for s in stages:
        vals = np.unique(raw[np.isclose(tp, s)])
        vals = vals[~np.isnan(vals)]
        if len(vals) != 1:
            raise ValueError(
                f"obs['{real_time_key}'] must take exactly one value per stage, "
                f"but stage {s:g} carries {len(vals)}: {vals[:5]}")
        out.append(float(vals[0]))

    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(
            f"obs['{real_time_key}'] is not strictly increasing across stages: {out}")
    return out

SPLITS: dict[str, dict[str, tuple[list[int], list[int]]]] = {
    "ZB": {
        "interpolation": ([0, 1, 2, 3, 5, 7, 9, 10, 11], [4, 6, 8]),
        "extrapolation": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11]),
    },
    "DR": {
        "interpolation": ([0, 1, 2, 3, 5, 7, 9, 10], [4, 6, 8]),
        "extrapolation": ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    },
    "SC": {
        "interpolation": ([0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18], [5, 7, 9, 11]),
        "extrapolation": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18]),
    },
    "GA": {
        "interpolation": ([0, 1, 2, 3, 5, 7, 9, 10], [4, 6, 8]),
        "extrapolation": ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    },
    "MA": {
        "interpolation": ([0, 1, 3, 4, 5, 7], [2, 6]),
        "extrapolation": ([0, 1, 2, 3, 4, 5, 6], [7]),
    },
}


def get_tp_split(dataset: str, task_type: str,
                 n_steps: int | None = None) -> tuple[list[int], list[int]]:
    """(dataset, task_type[, n_steps]) -> (train_tps, test_tps).

    Args:
        dataset:   dataset abbreviation (case-insensitive).
        task_type: interpolation / extrapolation / extrapolation_steps / all.
        n_steps:   required only when task_type=='extrapolation_steps'; the last n_steps
                   time points become the test set.
    """
    ds = dataset.upper()
    if ds not in ALL_TPS:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Available: {sorted(ALL_TPS.keys())}")
    all_tps = list(ALL_TPS[ds])

    if task_type == "all":
        return list(all_tps), list(all_tps)

    if task_type == "extrapolation_steps":
        T = len(all_tps)
        if n_steps is None:
            raise ValueError(
                "task 'extrapolation_steps' requires n_steps (e.g. --ex_steps 3).")
        n_steps = int(n_steps)
        if not (0 <= n_steps <= T - 1):
            raise ValueError(
                f"n_steps must be in [0, {T - 1}] for dataset {ds} (T={T}), "
                f"got {n_steps}.")
        if n_steps == 0:
            train_tps, test_tps = list(all_tps), []
        else:
            train_tps = all_tps[:-n_steps]
            test_tps = all_tps[-n_steps:]
        return list(train_tps), list(test_tps)

    splits = SPLITS.get(ds, {})
    if task_type not in splits:
        raise ValueError(
            f"Split '{task_type}' not defined for dataset '{dataset}'. "
            f"Available: {list(splits.keys()) + ['extrapolation_steps', 'all']}")
    train_tps, test_tps = splits[task_type]
    return list(train_tps), list(test_tps)


def train_cell_mask(tp_values, train_tps) -> np.ndarray:
    col = np.asarray(tp_values).astype(float)
    allowed = np.asarray([float(t) for t in train_tps], dtype=float)
    if allowed.size == 0:
        return np.zeros(col.shape[0], dtype=bool)
    return np.isin(col, allowed)


def task_tag(task_type: str, n_steps: int | None = None) -> str:
    if task_type == "extrapolation_steps":
        if n_steps is None:
            raise ValueError("task 'extrapolation_steps' requires n_steps for task_tag.")
        return f"extrapolation_steps_{int(n_steps)}"
    return task_type


def cache_paths(dataset: str, task_type: str,
                n_steps: int | None, seed: int,
                ot_type: str = "CPU",
                results_root: str = "results") -> dict[str, str]:
    ds_up = dataset.upper()
    tag = task_tag(task_type, n_steps)
    root = str(results_root).rstrip("/")
    base = f"{root}/vae_checkpoints/{ds_up}_{tag}_seed{int(seed)}"
    ot_tag = str(ot_type).upper()
    return {
        "vae_ckpt_path": f"{base}.pt",
        "latents_path": f"{base}_latents.h5ad",
        "coupling_cache_dir":
            f"{root}/coupling_cache/{ds_up}_{tag}_seed{int(seed)}_{ot_tag}",
    }


def coupling_complete(dataset: str, task_type: str,
                      n_steps: int | None, seed: int,
                      ot_type: str = "CPU",
                      results_root: str = "results",
                      coupling_cache_dir: str | Path | None = None,
                      latents_path: str | Path | None = None) -> bool:
    if coupling_cache_dir is None or latents_path is None:
        _p = cache_paths(dataset, task_type, n_steps, seed,
                         ot_type=ot_type, results_root=results_root)
        coupling_cache_dir = coupling_cache_dir or _p["coupling_cache_dir"]
        latents_path = latents_path or _p["latents_path"]
    coup = Path(coupling_cache_dir)
    if not coup.is_dir():
        return False
    train_tps, _ = get_tp_split(dataset, task_type, n_steps=n_steps)
    if len(train_tps) < 2:
        return False
    pi_files = [coup / f"pi_{t_a}_to_{t_b}.npz"
                for t_a, t_b in zip(train_tps[:-1], train_tps[1:])]
    if not all(f.exists() for f in pi_files):
        return False
    lat = Path(latents_path)
    if lat.is_file():
        lat_mtime = lat.stat().st_mtime
        if any(f.stat().st_mtime < lat_mtime for f in pi_files):
            return False
    return True


def vae_cache_valid(dataset: str, task_type: str,
                    n_steps: int | None, seed: int,
                    results_root: str = "results",
                    vae_ckpt_path: str | Path | None = None,
                    latents_path: str | Path | None = None) -> bool:
    if vae_ckpt_path is None or latents_path is None:
        _p = cache_paths(dataset, task_type, n_steps, seed,
                         results_root=results_root)
        vae_ckpt_path = vae_ckpt_path or _p["vae_ckpt_path"]
        latents_path = latents_path or _p["latents_path"]
    if not (Path(vae_ckpt_path).is_file() and Path(latents_path).is_file()):
        return False

    import torch
    try:
        try:
            ckpt = torch.load(str(vae_ckpt_path), map_location="cpu",
                              weights_only=True)
        except TypeError:
            ckpt = torch.load(str(vae_ckpt_path), map_location="cpu")
    except Exception:
        return False
    recorded = (ckpt.get("meta") or {}).get("train_tps")
    if recorded is None:
        return False
    expected, _ = get_tp_split(dataset, task_type, n_steps=n_steps)
    return [int(t) for t in recorded] == [int(t) for t in expected]


def load_t_real(latents_path: str | Path,
                time_key: str = "tp",
                real_time_key: str | None = None) -> list[float]:
    """Derive t_real from {DATASET}_latents.h5ad; see `t_real_from_obs`.

    run/preprocess_vae.py copies the whole obs frame of the source h5ad into the
    latents file, so any real-time column of the original dataset is available
    here too.
    """
    adata = ad.read_h5ad(str(latents_path))
    return t_real_from_obs(adata.obs, time_key=time_key,
                           real_time_key=real_time_key)


def load_t_real_from_raw(h5ad_path: str | Path, time_key: str,
                         real_time_key: str | None = None) -> list[float]:
    adata = ad.read_h5ad(str(h5ad_path))
    return t_real_from_obs(adata.obs, time_key=time_key,
                           real_time_key=real_time_key)
