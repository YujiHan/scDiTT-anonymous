#!/usr/bin/env python
"""Check the AR arms against the construction described in Appendix J.

Runs without any dataset: the training pattern is checked combinatorially and the
sampling loop against a counting stub model.

    PYTHONPATH=. python exps/task2_extra_steps/ar_variants/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from ar_patch import UNCOND_PROB, make_ar_sampler, make_pattern_sampler  # noqa: E402

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL += 1


def test_pattern(arm: str, T: int = 12, n: int = 20000) -> None:
    print(f"\n--- training pattern: AR-{arm} ---")
    train_tps = list(range(9))          # a horizon-3 extrapolation split on T=12
    sampler = make_pattern_sampler(arm)
    rng = np.random.default_rng(0)

    n_uncond = 0
    single_target = True
    disjoint = True
    target_is_next = True
    shape_ok = True
    never_test = True

    for _ in range(n):
        obs, loss = sampler(rng, train_tps, T)
        disjoint &= not bool((obs & loss).any())
        single_target &= int(loss.sum()) == 1
        k1 = int(np.flatnonzero(loss)[0])
        never_test &= k1 in train_tps and not obs[len(train_tps):].any()
        if not obs.any():
            n_uncond += 1
            continue
        idx = [train_tps.index(t) for t in np.flatnonzero(obs)]
        i = max(idx)
        target_is_next &= (k1 == train_tps[i + 1])
        if arm == "prefix":
            shape_ok &= idx == list(range(i + 1))       # a contiguous prefix
        else:
            shape_ok &= idx == [i]                       # exactly one stage

    check("obs and loss masks are disjoint", disjoint)
    check("the loss supervises exactly one stage", single_target)
    check("the supervised stage is k+1 (the next observed stage)", target_is_next)
    check("nothing outside train_tps is ever touched", never_test)
    check(f"conditioning shape is {'a prefix j<=k' if arm == 'prefix' else 'stage k alone'}",
          shape_ok)
    rate = n_uncond / n
    check(f"mask emptied with p~{UNCOND_PROB} for the CFG branch",
          abs(rate - UNCOND_PROB) < 0.005, f"measured {rate:.4f}")


class _CountingModel(torch.nn.Module):
    """Records every forward call and the obs_mask it was given."""

    def __init__(self, D: int):
        super().__init__()
        self.calls = 0
        self.masks: list[list[int]] = []
        self.D = D

    def forward(self, x, tau, obs_mask, t_real):
        self.calls += 1
        self.masks.append([int(v) for v in obs_mask[0].tolist()])
        return torch.ones_like(x)


def test_sampler(arm: str, T: int = 10, horizon: int = 4,
                 N: int = 3, D: int = 5, steps: int = 20) -> None:
    print(f"\n--- sampling loop: AR-{arm} (T={T}, horizon={horizon}) ---")
    obs_mask = torch.zeros(1, T, dtype=torch.bool)
    obs_mask[0, : T - horizon] = True
    x_obs = torch.zeros(1, T, N, D)
    for t in range(T - horizon):
        x_obs[0, t] = float(t + 1)              # a recognisable evidence value

    model = _CountingModel(D)
    gen = torch.Generator().manual_seed(0)
    out = make_ar_sampler(arm)(
        model=model, x_obs=x_obs, obs_mask=obs_mask, t_real=torch.arange(T).float(),
        scheduler=None, num_steps=steps, cfg_weight=3.0,
        device=torch.device("cpu"), rng_generator=gen)

    check("output shape", tuple(out.shape) == (1, T, N, D), str(tuple(out.shape)))
    check("observed stages come back untouched",
          all(torch.allclose(out[0, t], x_obs[0, t]) for t in range(T - horizon)))
    check("every withheld stage was filled",
          all(not torch.allclose(out[0, t], torch.zeros(N, D))
              for t in range(T - horizon, T)))

    expect = 2 * steps * horizon
    check(f"network evaluations = 2 x {steps} x horizon = {expect}",
          model.calls == expect, f"measured {model.calls}")

    # One conditional and one unconditional call per step; look at the
    # conditional masks only, one per stage (they repeat within a stage).
    cond_masks = [m for m in model.masks if any(m)]
    per_stage = [cond_masks[i * steps] for i in range(horizon)]
    ok = True
    for j, m in enumerate(per_stage):
        t = T - horizon + j
        on = [i for i, v in enumerate(m) if v]
        if arm == "prefix":
            ok &= on == list(range(t))          # all earlier stages, incl. generated
        else:
            ok &= on == [t - 1]                 # the previous stage alone
    check("conditioning set grows as " +
          ("a prefix over real + generated stages" if arm == "prefix"
           else "the previous stage alone"), ok,
          f"stage {T - 1} conditioned on {[i for i, v in enumerate(per_stage[-1]) if v]}")


def main() -> int:
    print(f"AR self-test  (arms: {', '.join(C.ARMS)})")
    for arm in C.ARMS:
        test_pattern(arm)
        test_sampler(arm)
    print(f"\n{'all checks passed' if FAIL == 0 else f'{FAIL} check(s) FAILED'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
