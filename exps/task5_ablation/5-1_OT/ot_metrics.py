from __future__ import annotations

import argparse
import json

import numpy as np

from common import (K, ARMS, REGIMES, SIGMAS, METRIC_DIR, tag, load_sim,
                    sim_h5ad_path, coupling_dir, get_split)


def _nn_scale(z_b: np.ndarray, rng: np.random.Generator, n_sample: int = 512) -> float:
    n = z_b.shape[0]
    n_sample = min(n_sample, n)
    sel = rng.choice(n, n_sample, replace=False)
    zs = z_b[sel]                                   # [s, D]
    a2 = np.einsum("ij,ij->i", zs, zs)[:, None]
    b2 = np.einsum("ij,ij->i", z_b, z_b)[None, :]
    d2 = a2 + b2 - 2.0 * (zs @ z_b.T)
    d2[np.arange(n_sample), sel] = np.inf
    nn = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return float(np.median(nn))


def eval_pair(pi, idx_a, idx_b, sim, rng) -> dict:
    particle, branch, X = sim["particle_id"], sim["branch"], sim["X"]
    pa = particle[idx_a]
    pb = particle[idx_b]
    ba = branch[idx_a]
    bb = branch[idx_b]
    z_b = X[idx_b]
    Na = pi.shape[0]

    col_of = np.empty(int(pb.max()) + 1, dtype=np.int64)
    col_of[pb] = np.arange(len(pb))
    j_true = col_of[pa]                              # [Na]

    # top-1 / top-5
    top1 = float((pi.argmax(axis=1) == j_true).mean())
    k5 = min(5, pi.shape[1])
    topk_idx = np.argpartition(-pi, k5 - 1, axis=1)[:, :k5]
    top5 = float((topk_idx == j_true[:, None]).any(axis=1).mean())

    mass_truth = float(pi[np.arange(Na), j_true].mean())

    xhat = pi @ z_b                                  # [Na, D]
    x_true = z_b[j_true]
    err = np.linalg.norm(xhat - x_true, axis=1)
    scale = _nn_scale(z_b, rng)
    transport = float(err.mean() / (scale + 1e-8))

    mass_same = np.zeros(Na, dtype=np.float64)
    for g in np.unique(bb):
        mg = pi @ (bb == g).astype(np.float32)
        mass_same += np.where(ba == g, mg, 0.0)
    cross_branch = float((1.0 - mass_same).mean())

    counts = np.bincount(ba)
    present = np.where(counts > 0)[0]
    rare = int(present[np.argmin(counts[present])])
    sel = (ba == rare)
    rare_keep = float(mass_same[sel].mean()) if sel.any() else float("nan")

    Nb = pi.shape[1]
    vals = pi[np.arange(Na), j_true]
    gt = (pi > vals[:, None]).sum(axis=1)
    eq = (pi == vals[:, None]).sum(axis=1)
    rank = gt + (eq + 1.0) / 2.0
    mrr = float(np.mean(1.0 / rank))
    median_rank = float(np.median(rank))

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(pi > 0, np.log(pi), 0.0)
    H = -(pi * logp).sum(axis=1)                      # [Na] nats
    coupling_entropy = float(np.mean(H) / np.log(Nb)) # random→≈1, true→0
    eff_partners = float(np.mean(np.exp(H)))          # random→≈K, true→1

    z_a = X[idx_a]
    pred_disp = xhat - z_a
    true_disp = x_true - z_a
    num = np.einsum("ij,ij->i", pred_disp, true_disp)
    den = np.linalg.norm(pred_disp, axis=1) * np.linalg.norm(true_disp, axis=1)
    m = den > 1e-8
    dir_cos = float(np.mean(num[m] / den[m])) if m.any() else float("nan")

    return {"top1": top1, "top5": top5, "mass_truth": mass_truth,
            "transport": transport, "cross_branch": cross_branch,
            "rare_keep": rare_keep,
            "mrr": mrr, "median_rank": median_rank,
            "coupling_entropy": coupling_entropy, "eff_partners": eff_partners,
            "dir_cos": dir_cos}


def eval_arm(regime, sigma, seed, arm, task="interp") -> dict:
    sim = load_sim(sim_h5ad_path(regime, sigma, seed))
    cdir = coupling_dir(regime, sigma, seed, arm, task)
    _, _, train_pairs = get_split(task)
    rng = np.random.default_rng(123)
    per_pair = []
    for (t_a, t_b) in train_pairs:
        d = np.load(str(cdir / f"pi_{t_a}_to_{t_b}.npz"))
        m = eval_pair(d["pi"], d["idx_a_global"], d["idx_b_global"], sim, rng)
        m["pair"] = f"{t_a}->{t_b}"
        m["t_a"] = int(t_a)
        per_pair.append(m)
    keys = ["top1", "top5", "mass_truth", "transport", "cross_branch",
            "rare_keep", "mrr", "median_rank", "coupling_entropy",
            "eff_partners", "dir_cos"]
    agg = {k: float(np.nanmean([p[k] for p in per_pair])) for k in keys}
    post = [p for p in per_pair if p["t_a"] >= 5]
    agg["cross_ps"] = float(np.mean([p["cross_branch"] for p in post])) if post else 0.0
    agg["rare_keep_ps"] = float(np.nanmean([p["rare_keep"] for p in post])) if post else 0.0
    return {"agg": agg, "per_pair": per_pair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all")
    ap.add_argument("--sigma", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--task", default="interp", choices=["interp", "extrap"])
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    regimes = REGIMES if args.regime == "all" else [args.regime]
    sigmas = SIGMAS if args.sigma == "all" else [float(args.sigma)]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    print(f"{'regime':<13}{'sigma':>6}{'seed':>5}{'arm':>8}"
          f"{'top1':>7}{'mrr':>7}{'medRk':>7}{'entH':>7}{'effP':>8}"
          f"{'dirCos':>8}{'transp':>8}{'xbr_ps':>8}{'rareKp_ps':>10}")
    for regime in regimes:
        for sigma in sigmas:
            for seed in seeds:
                for arm in arms:
                    r = eval_arm(regime, sigma, seed, arm, task=args.task)
                    a = r["agg"]
                    key = f"{tag(regime, sigma)}|seed{seed}|{arm}"
                    all_results[key] = r
                    print(f"{regime:<13}{sigma:>6.2f}{seed:>5}{arm:>8}"
                          f"{a['top1']:>7.3f}{a['mrr']:>7.3f}"
                          f"{a['median_rank']:>7.0f}{a['coupling_entropy']:>7.3f}"
                          f"{a['eff_partners']:>8.1f}{a['dir_cos']:>8.3f}"
                          f"{a['transport']:>8.3f}"
                          f"{a['cross_ps']:>8.3f}{a['rare_keep_ps']:>10.3f}")
    fname = "ot_quality.json" if args.task == "interp" \
        else f"ot_quality_{args.task}.json"
    out = METRIC_DIR / fname
    merged = {}
    if out.exists():
        merged = json.load(open(out))
    merged.update(all_results)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[ot_metrics/{args.task}] saved -> {out} (total {len(merged)} rows)")


if __name__ == "__main__":
    main()
