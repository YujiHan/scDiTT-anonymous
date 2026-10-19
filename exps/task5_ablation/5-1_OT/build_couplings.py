from __future__ import annotations

import argparse
import json

import numpy as np

from common import (K, OT_EPSILON, ARMS, REGIMES, SIGMAS, tag,
                    tp_row_range, load_sim, sim_h5ad_path, coupling_dir,
                    get_split)
from src.data.coupling import compute_coupling


def _save_pi(out_dir, t_a, t_b, pi, idx_a_global, idx_b_global):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / f"pi_{t_a}_to_{t_b}.npz"),
             pi=pi.astype(np.float32),
             idx_a_global=idx_a_global.astype(np.int64),
             idx_b_global=idx_b_global.astype(np.int64),
             t_a=np.int64(t_a), t_b=np.int64(t_b),
             epsilon=np.float32(OT_EPSILON))


def _write_meta(out_dir, arm, train_tps, train_pairs):
    meta = {
        "train_tps": [int(t) for t in train_tps],
        "pairs": [{"t_a": int(a), "t_b": int(b), "file": f"pi_{a}_to_{b}.npz"}
                  for a, b in train_pairs],
        "epsilon": float(OT_EPSILON),
        "arm": arm,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_for_sim(regime, sigma, seed, task="interp", arms=None,
                  ot_type="CUDA", device="cuda:0"):
    arms = list(arms) if arms is not None else list(ARMS)
    train_tps, _, train_pairs = get_split(task)
    sim = load_sim(sim_h5ad_path(regime, sigma, seed))
    Z = sim["X"]
    rng = np.random.default_rng(seed)

    need_ot = "ot" in arms
    uniform_row = (np.full((K, K), 1.0 / K, dtype=np.float32)
                   if ("random" in arms or need_ot) else None)
    eye = np.eye(K, dtype=np.float32) if "true" in arms else None

    for (t_a, t_b) in train_pairs:
        rows_a = tp_row_range(t_a)
        rows_b = tp_row_range(t_b)
        za, zb = Z[rows_a], Z[rows_b]
        idx_a, idx_b = rows_a, rows_b
        arm_pi = {}
        if "true" in arms:
            arm_pi["true"] = eye
        if "random" in arms:
            arm_pi["random"] = uniform_row
        if need_ot:
            res = compute_coupling(za, zb, epsilon=OT_EPSILON,
                                   normalize_cost=True, max_cells=None, rng=rng,
                                   ot_type=ot_type, device=device)
            pi_ot = res["pi"].astype(np.float32)
            idx_a = rows_a[res["idx_a"]]
            idx_b = rows_b[res["idx_b"]]
            if "ot" in arms:
                arm_pi["ot"] = pi_ot

        for arm in arms:
            _save_pi(coupling_dir(regime, sigma, seed, arm, task), t_a, t_b,
                     arm_pi[arm], idx_a, idx_b)
    for arm in arms:
        _write_meta(coupling_dir(regime, sigma, seed, arm, task), arm,
                    train_tps, train_pairs)
    print(f"[coupling/{task}] {tag(regime, sigma)} seed{seed}: built "
          f"{arms} × {len(train_pairs)} pairs ({ot_type})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all")
    ap.add_argument("--sigma", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--task", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"comma-separated; available {ARMS}")
    ap.add_argument("--ot_type", default="CUDA",
                    help="CUDA / CPU_COST / CPU")
    ap.add_argument("--device", default="cuda:0",
                    help="only used when ot_type=CUDA (CUDA_VISIBLE_DEVICES already selects the physical device)")
    args = ap.parse_args()

    regimes = REGIMES if args.regime == "all" else [args.regime]
    sigmas = SIGMAS if args.sigma == "all" else [float(args.sigma)]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    for regime in regimes:
        for sigma in sigmas:
            for seed in seeds:
                build_for_sim(regime, sigma, seed, task=args.task, arms=arms,
                              ot_type=args.ot_type, device=args.device)


if __name__ == "__main__":
    main()
