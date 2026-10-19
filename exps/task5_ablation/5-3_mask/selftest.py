#!/usr/bin/env python
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

import common  # noqa: F401
from common import (ARMS, CONFIG_ROOT, DATASETS, OT_TYPE, REPO_ROOT,
                    RESULTS_ROOT_REL, SEEDS, TASKS, get_split, get_T)
from mask_arms import ARM_WEIGHTS, sampler_factory

from src.utils.config import build_config

os.chdir(REPO_ROOT)
ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    print(f"  {'✓' if cond else '✗ failed:'} {msg}")
    ok = ok and cond


print("\n[1] cache path: depends only on (dataset, task, seed), independent of the arm -> shared by all four arms")
for ds in DATASETS:
    for task in TASKS:
        paths = set()
        for arm in ARMS:
            ns = SimpleNamespace(dataset=ds.lower(), task=task, seed=42,
                                 output_dir=None, results_root=RESULTS_ROOT_REL,
                                 ot_type=OT_TYPE, ex_steps=None,
                                 resume_from=None, smoke_test=False)
            cfg = build_config(ns, extras=[], cfg_root=CONFIG_ROOT)
            paths.add((cfg["dataset"]["latents_path"],
                       cfg["dataset"]["coupling_cache_dir"]))
        check(len(paths) == 1, f"{ds}/{task}: the four arms derive the same cache path")
        lat, coup = paths.pop()
        check(lat.startswith(RESULTS_ROOT_REL) and coup.startswith(RESULTS_ROOT_REL),
              f"{ds}/{task}: the cache lands in {RESULTS_ROOT_REL}/ ({lat})")

print("\n[2] monkeypatch factory signature (matching the call shape at src/training/train.py:74)")
for arm in ARMS:
    train_tps, _ = get_split("ZB", "interpolation")
    factory = sampler_factory(arm)
    s = factory(train_tps=train_tps, T=get_T("ZB"),
                rng=np.random.default_rng(42 + 12345))
    check(hasattr(s, "sample") and hasattr(s, "rng"),
          f"`{arm}`: the object returned by the factory has .sample() and .rng (dataset rewrites .rng per worker)")

print("\n[3][4] mask validity + CFG null-condition budget")
for ds in DATASETS:
    for task in TASKS:
        train_tps, test_tps = get_split(ds, task)
        T = get_T(ds)
        obs_base = np.zeros(T, dtype=bool)
        obs_base[train_tps] = True
        for arm in ARMS:
            s = sampler_factory(arm)(train_tps=train_tps, T=T,
                                     rng=np.random.default_rng(0))
            n, n_full, bad = 20000, 0, 0
            for _ in range(n):
                m = s.sample()
                eff_obs = obs_base & ~m
                loss = obs_base & m
                if (eff_obs & loss).any():
                    bad += 1
                if m[test_tps].any() and not m[train_tps].all():
                    pass
                if not eff_obs.any():
                    n_full += 1
            check(bad == 0, f"{ds}/{task} `{arm}`: no invalid samples (obs and loss do not overlap)")
            check(0.03 <= n_full / n <= 0.10,
                  f"{ds}/{task} `{arm}`: full rate {n_full/n:.3f} ∈ [0.03, 0.10] "
                  f"(CFG null-condition budget)")

print("\n[5] per-dataset hyper-parameters match the main experiment (mix arm)")
expect = {"ZB": {"N": 128, "peak_lr": 1e-4}, "GA": {"N": 256, "peak_lr": 1e-4}}
for ds in DATASETS:
    ns = SimpleNamespace(dataset=ds.lower(), task="interpolation", seed=42,
                         output_dir=None, results_root=RESULTS_ROOT_REL,
                         ot_type=OT_TYPE, ex_steps=None, resume_from=None,
                         smoke_test=False)
    cfg = build_config(ns, extras=[], cfg_root=CONFIG_ROOT)
    check(cfg["data"]["N"] == expect[ds]["N"],
          f"{ds}: data.N = {cfg['data']['N']} (expected {expect[ds]['N']})")
    check(abs(cfg["training"]["peak_lr"] - expect[ds]["peak_lr"]) < 1e-12,
          f"{ds}: training.peak_lr = {cfg['training']['peak_lr']}")
    check(cfg["inference"]["cfg_weight"] == 3.0,
          f"{ds}: inference.cfg_weight = 3.0 (so full / CFG null-condition must be kept)")

print("\n" + ("all checks passed ✅ -- the wiring is correct, ready to run as soon as a GPU frees up."
              if ok else "some checks failed ❌ -- fix them before running."))
raise SystemExit(0 if ok else 1)
