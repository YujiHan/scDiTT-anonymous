"""Verifies the train/test split protocol after the 2026-06 refactor.

Datasets: ZB / DR / SC / GA / MA
Tasks:    interpolation / extrapolation / extrapolation_steps / all
"""
import numpy as np
import pytest

from src.data.splits import (ALL_TPS, SPLITS, cache_paths, coupling_complete,
                             get_tp_split, task_tag, train_cell_mask,
                             vae_cache_valid)


DATASETS = ["ZB", "DR", "SC", "GA", "MA"]


def test_zb_interpolation():
    train, test = get_tp_split("ZB", "interpolation")
    assert train == [0, 1, 2, 3, 5, 7, 9, 10, 11]
    assert test == [4, 6, 8]


def test_zb_extrapolation():
    train, test = get_tp_split("ZB", "extrapolation")
    assert train == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert test == [10, 11]


def test_dr_interpolation():
    train, test = get_tp_split("DR", "interpolation")
    assert train == [0, 1, 2, 3, 5, 7, 9, 10]
    assert test == [4, 6, 8]


def test_sc_interpolation():
    train, test = get_tp_split("SC", "interpolation")
    assert train == [0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18]
    assert test == [5, 7, 9, 11]


def test_ga_extrapolation():
    train, test = get_tp_split("GA", "extrapolation")
    assert train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert test == [8, 9, 10]


def test_ma_interpolation():
    train, test = get_tp_split("MA", "interpolation")
    assert train == [0, 1, 3, 4, 5, 7]
    assert test == [2, 6]


def test_extrapolation_steps_basic():
    train, test = get_tp_split("ZB", "extrapolation_steps", n_steps=2)
    assert test == [10, 11]
    assert train == list(range(10))


def test_extrapolation_steps_example_from_spec():
    train, test = get_tp_split("MA", "extrapolation_steps", n_steps=3)
    assert test == [5, 6, 7]
    assert train == [0, 1, 2, 3, 4]


def test_extrapolation_steps_requires_n_steps():
    with pytest.raises(ValueError):
        get_tp_split("ZB", "extrapolation_steps")


def test_extrapolation_steps_out_of_range():
    with pytest.raises(ValueError):
        get_tp_split("MA", "extrapolation_steps", n_steps=8)   # T=8, max=7
    with pytest.raises(ValueError):
        get_tp_split("ZB", "extrapolation_steps", n_steps=12)  # T=12, max=11
    with pytest.raises(ValueError):
        get_tp_split("ZB", "extrapolation_steps", n_steps=-1)


def test_extrapolation_steps_zero():
    for ds in DATASETS:
        train, test = get_tp_split(ds, "extrapolation_steps", n_steps=0)
        assert train == list(ALL_TPS[ds]), ds
        assert test == [], ds


def test_all_task_returns_full_tps():
    for ds in DATASETS:
        train, test = get_tp_split(ds, "all")
        assert train == list(ALL_TPS[ds])
        assert test == list(ALL_TPS[ds])


def test_union_and_disjoint():
    for ds in DATASETS:
        T = len(ALL_TPS[ds])
        cases = [("interpolation", None), ("extrapolation", None)]
        cases += [("extrapolation_steps", n) for n in range(0, T)]
        for task, n in cases:
            train, test = get_tp_split(ds, task, n_steps=n)
            assert set(train) & set(test) == set(), f"{ds}/{task}/{n} overlap"
            assert set(train) | set(test) == set(ALL_TPS[ds]), \
                f"{ds}/{task}/{n} union != all_tps"
            assert len(train) > 0, f"{ds}/{task}/{n} empty train"
            if not (task == "extrapolation_steps" and n == 0):
                assert len(test) > 0, f"{ds}/{task}/{n} empty test"


def test_task_tag():
    assert task_tag("interpolation") == "interpolation"
    assert task_tag("extrapolation") == "extrapolation"
    assert task_tag("all") == "all"
    assert task_tag("extrapolation_steps", 3) == "extrapolation_steps_3"
    with pytest.raises(ValueError):
        task_tag("extrapolation_steps")


def test_removed_datasets_rejected():
    for ds in ["OE", "SU", "VE", "CO"]:
        assert ds not in ALL_TPS
        with pytest.raises(ValueError):
            get_tp_split(ds, "interpolation")


def test_removed_task_rejected():
    for ds in DATASETS:
        assert "hard_task" not in SPLITS.get(ds, {})
    with pytest.raises(ValueError):
        get_tp_split("ZB", "hard_task")


def test_coupling_complete(tmp_path):
    ds, task = "ZB", "interpolation"
    train, _ = get_tp_split(ds, task)                       # [0,1,2,3,5,7,9,10,11]
    pairs = list(zip(train[:-1], train[1:]))
    coup = tmp_path / "coupling_cache" / f"{ds}_{task}_seed42_CPU"
    lat = tmp_path / "no_such_latents.h5ad"

    assert coupling_complete(ds, task, None, 42, coupling_cache_dir=coup,
                             latents_path=lat) is False

    coup.mkdir(parents=True)
    for a, b in pairs[:-1]:
        (coup / f"pi_{a}_to_{b}.npz").write_bytes(b"")
    (coup / "preprocess_coupling.log").write_text("x")
    assert coupling_complete(ds, task, None, 42, coupling_cache_dir=coup,
                             latents_path=lat) is False

    a, b = pairs[-1]
    (coup / f"pi_{a}_to_{b}.npz").write_bytes(b"")
    assert coupling_complete(ds, task, None, 42, coupling_cache_dir=coup,
                             latents_path=lat) is True


def test_coupling_stale_against_latents(tmp_path):
    import os

    ds, task = "MA", "interpolation"
    train, _ = get_tp_split(ds, task)                       # [0, 1, 3, 4, 5, 7]
    coup = tmp_path / "coup"
    coup.mkdir()
    for a, b in zip(train[:-1], train[1:]):
        f = coup / f"pi_{a}_to_{b}.npz"
        f.write_bytes(b"")
        os.utime(f, (2000, 2000))
    lat = tmp_path / "lat.h5ad"
    lat.write_bytes(b"")

    os.utime(lat, (1000, 1000))
    assert coupling_complete(ds, task, None, 42, coupling_cache_dir=coup,
                             latents_path=lat) is True
    os.utime(lat, (3000, 3000))
    assert coupling_complete(ds, task, None, 42, coupling_cache_dir=coup,
                             latents_path=lat) is False


def test_train_cell_mask():
    tps = np.array([0, 1, 2, 2, 1, 0])
    assert train_cell_mask(tps, [0, 2]).tolist() == [True, False, True,
                                                     True, False, True]
    import pandas as pd
    cat = pd.Categorical([0, 1, 2, 1], categories=[0, 1, 2])
    assert train_cell_mask(cat, [1]).tolist() == [False, True, False, True]
    assert train_cell_mask(tps, []).tolist() == [False] * 6


def test_vae_cache_valid(tmp_path):
    torch = pytest.importorskip("torch")
    ds, task, seed = "MA", "interpolation", 42
    root = str(tmp_path)
    p = cache_paths(ds, task, None, seed, results_root=root)
    from pathlib import Path
    Path(p["vae_ckpt_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(p["latents_path"]).write_bytes(b"")

    base = {"state_dict": {}, "gene_dim": 5, "latent_dim": 2}

    torch.save(base, p["vae_ckpt_path"])
    assert vae_cache_valid(ds, task, None, seed, results_root=root) is False

    torch.save({**base, "meta": {"train_tps": [0, 1]}}, p["vae_ckpt_path"])
    assert vae_cache_valid(ds, task, None, seed, results_root=root) is False

    train, _ = get_tp_split(ds, task)                       # [0, 2]
    torch.save({**base, "meta": {"train_tps": train}}, p["vae_ckpt_path"])
    assert vae_cache_valid(ds, task, None, seed, results_root=root) is True

    Path(p["latents_path"]).unlink()
    assert vae_cache_valid(ds, task, None, seed, results_root=root) is False
