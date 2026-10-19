from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import (T, K, D, TIME_KEY, ARMS, REGIMES, METRIC_DIR, tag,
                    get_split, sim_h5ad_path, ckpt_dir, load_sim, global_row)
from src.model.dit import SCDiT
from src.diffusion.schedulers import build_scheduler as build_diff_scheduler
from src.diffusion.sampling import masked_diffusion_inpaint
from src.evaluation.metrics import compute_wd
from src.evaluation.unified_metrics import mmd as compute_mmd


def load_model(regime, sigma, seed, arm, task, device):
    ck = ckpt_dir(regime, sigma, seed, arm, task) / "best.pt"
    if not ck.exists():
        return None
    sd = torch.load(str(ck), map_location=device)
    model = SCDiT(D_latent=D, D_model=384, num_blocks=12, num_heads=6,
                  T_max=max(32, T)).to(device)
    model.load_state_dict(sd["ema"])
    model.eval()
    return model


@torch.no_grad()
def anchored_generate(model, Z_all, particle_ids, train_tps, test_tps,
                      diff_sched, device, N=256, num_repeats=8,
                      num_steps=20, cfg_weight=1.0, seed=0):
    """Particle-anchored generation: slot n = particle particle_ids[n].
    Returns gen_mean[test_tp] = [N, D] (conditional mean prediction across repeats),
            gen_pool[test_tp] = [N*num_repeats, D] (used for W2).
    """
    t_real = torch.arange(T, dtype=torch.float32, device=device)
    obs_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    for t in train_tps:
        obs_mask[0, t] = True

    x_obs = torch.zeros(1, T, N, D, dtype=torch.float32, device=device)
    for t in train_tps:
        rows = np.array([global_row(t, int(p)) for p in particle_ids])
        x_obs[0, t] = torch.from_numpy(Z_all[rows]).to(device)

    g = torch.Generator(device=device)
    reps = {t: [] for t in test_tps}
    for r in range(num_repeats):
        g.manual_seed(seed * 1000 + r)
        x_final = masked_diffusion_inpaint(
            model=model, x_obs=x_obs, obs_mask=obs_mask, t_real=t_real,
            scheduler=diff_sched, num_steps=num_steps, cfg_weight=cfg_weight,
            device=device, rng_generator=g)
        for t in test_tps:
            reps[t].append(x_final[0, t].cpu().numpy())  # [N, D]
    gen_mean = {t: np.mean(reps[t], axis=0) for t in test_tps}          # [N,D]
    gen_pool = {t: np.concatenate(reps[t], axis=0) for t in test_tps}   # [N*R,D]
    gen_reps = {t: np.stack(reps[t]) for t in test_tps}                 # [R,N,D]
    return gen_mean, gen_pool, gen_reps


def pearson(gen, true):
    cs = []
    for d in range(gen.shape[1]):
        a, b = gen[:, d], true[:, d]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs)) if cs else 0.0


def eval_model(regime, sigma, seed, arm, task, device, N=256, num_repeats=8):
    sim = load_sim(sim_h5ad_path(regime, sigma, seed))
    Z_all, tp_col = sim["X"], sim["tp"]
    branch = sim["branch"]
    train_tps, test_tps, _ = get_split(task)
    model = load_model(regime, sigma, seed, arm, task, device)
    if model is None:
        return None
    diff_sched = build_diff_scheduler({"num_train_timesteps": 1000})

    rng = np.random.default_rng(0)
    part_ids = rng.choice(K, N, replace=False)

    gen_mean, gen_pool, gen_reps = anchored_generate(
        model, Z_all, part_ids, train_tps, test_tps, diff_sched, device,
        N=N, num_repeats=num_repeats)

    out = {}
    for t in test_tps:
        rows = np.array([global_row(t, int(p)) for p in part_ids])
        true = Z_all[rows]
        gm = gen_mean[t]
        mse = float(np.mean((gm - true) ** 2))
        pr = pearson(gm, true)
        spr = float(np.mean([pearson(gen_reps[t][r], true)
                             for r in range(gen_reps[t].shape[0])]))
        true_full = Z_all[tp_col == t]
        w2 = float(compute_wd(gen_pool[t], true_full, device=device))
        mmd_val = float(compute_mmd(gen_pool[t], true_full))
        bacc = None
        rare_recall = None
        if regime in ("bifurcation", "trifurcation"):
            b_true_particle = branch[rows]                     # [N]
            uniq = np.unique(branch[tp_col == t])
            cents = np.stack([Z_all[(tp_col == t) & (branch == b)].mean(0)
                              for b in uniq])                  # [C, D]
            dists = np.linalg.norm(gm[:, None, :] - cents[None, :, :], axis=2)
            gen_branch = uniq[dists.argmin(axis=1)]
            bacc = float((gen_branch == b_true_particle).mean())
            cnts = np.bincount(branch[tp_col == t])
            pres = np.where(cnts > 0)[0]
            rare = int(pres[np.argmin(cnts[pres])])
            sel = (b_true_particle == rare)
            if sel.any():
                rare_recall = float((gen_branch[sel] == rare).mean())
        out[int(t)] = {"mse": mse, "pearson": pr, "sampled_pearson": spr,
                       "w2": w2, "mmd": mmd_val,
                       "branch_acc": bacc, "rare_recall": rare_recall}
    agg = {k: float(np.mean([v[k] for v in out.values()]))
           for k in ("mse", "pearson", "sampled_pearson", "w2", "mmd")}
    baccs = [v["branch_acc"] for v in out.values() if v["branch_acc"] is not None]
    agg["branch_acc"] = float(np.mean(baccs)) if baccs else None
    rrs = [v["rare_recall"] for v in out.values() if v["rare_recall"] is not None]
    agg["rare_recall"] = float(np.mean(rrs)) if rrs else None
    return {"agg": agg, "per_tp": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--tasks", default="interp,extrap")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--num_repeats", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    regimes = REGIMES if args.regime == "all" else [args.regime]
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")
    arms = args.arms.split(",")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    hdr = (f"{'task':<8}{'regime':<13}{'arm':>8}{'MSE↓':>8}{'Pearson↑':>9}"
           f"{'brAcc↑':>8}{'rareRec↑':>9}{'W2':>7}{'MMD':>8}")
    print(hdr)
    for task in tasks:
        for regime in regimes:
            for seed in seeds:
                for arm in arms:
                    r = eval_model(regime, args.sigma, seed, arm, task, device,
                                   N=args.N, num_repeats=args.num_repeats)
                    if r is None:
                        continue
                    a = r["agg"]
                    key = f"{task}|{tag(regime, args.sigma)}|seed{seed}|{arm}"
                    results[key] = r
                    ba = f"{a['branch_acc']:.3f}" if a['branch_acc'] is not None else "-"
                    rr = f"{a['rare_recall']:.3f}" if a.get('rare_recall') is not None else "-"
                    print(f"{task:<8}{regime:<13}{arm:>8}{a['mse']:>8.3f}"
                          f"{a['pearson']:>9.3f}{ba:>8}{rr:>9}"
                          f"{a['w2']:>7.2f}{a['mmd']:>8.4f}")
        print()
    out = METRIC_DIR / "traj_metrics.json"
    merged = json.load(open(out)) if out.exists() else {}
    merged.update(results)
    json.dump(merged, open(out, "w"), indent=2)
    print(f"[traj_eval] saved -> {out} (total {len(merged)} rows)")


if __name__ == "__main__":
    main()
