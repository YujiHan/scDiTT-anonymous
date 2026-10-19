from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from common import (REGIMES, ARMS, METRIC_DIR, tag, ckpt_dir)

ARM_ORDER = ["true", "ot", "random"]


def load_ot_quality(task="interp"):
    p = METRIC_DIR / ("ot_quality.json" if task == "interp"
                      else f"ot_quality_{task}.json")
    if not p.exists():
        return {}
    d = json.load(open(p))
    out = {}
    for key, r in d.items():
        tagk, seed, arm = key.split("|")
        out[(tagk, seed, arm)] = r["agg"]
    return out


def load_downstream(sigma, seeds, task="interp"):
    recs = []
    for regime in REGIMES:
        tagk = tag(regime, sigma)
        for seed in seeds:
            for arm in ARMS:
                mp = ckpt_dir(regime, sigma, seed, arm, task) / "metrics.json"
                if not mp.exists():
                    continue
                m = json.load(open(mp))
                recs.append({
                    "regime": regime, "tag": tagk, "seed": seed, "arm": arm,
                    "w2": m.get("best_avg_w2", float("nan")),
                    "w2_per_tp": m.get("best_w2", {}),
                })
    return recs


def main_table(recs):
    """(regime, arm) -> mean/std of w2 across seeds."""
    grp = defaultdict(list)
    for r in recs:
        grp[(r["regime"], r["arm"])].append(r["w2"])
    rows = []
    print(f"\n{'regime':<12}{'arm':>10}{'n':>4}{'W2_mean':>10}{'W2_std':>9}")
    for regime in REGIMES:
        for arm in ARM_ORDER:
            vals = grp.get((regime, arm))
            if not vals:
                continue
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            rows.append({"regime": regime, "arm": arm, "n": len(vals),
                         "w2_mean": mu, "w2_std": sd})
            print(f"{regime:<12}{arm:>10}{len(vals):>4}{mu:>10.4f}{sd:>9.4f}")
        print()
    return rows


def save_csv(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[agg] -> {path}")


def make_plots(recs, otq, sigma, sfx=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[agg] matplotlib unavailable, skipping plots: {e}")
        return
    try:
        from scipy.stats import spearmanr
    except Exception:
        spearmanr = None

    grp = defaultdict(list)
    for r in recs:
        grp[(r["regime"], r["arm"])].append(r["w2"])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs, ys = [], []
    for r in recs:
        q = otq.get((r["tag"], f"seed{r['seed']}", r["arm"]))
        if q is None:
            continue
        x = q["transport"]; y = r["w2"]
        xs.append(x); ys.append(y)
        ax.scatter(x, y, s=28, alpha=0.75)
        ax.annotate(f"{r['arm']}", (x, y), fontsize=6, alpha=0.6)
    if len(xs) >= 3:
        xs_a, ys_a = np.array(xs), np.array(ys)
        b, a = np.polyfit(xs_a, ys_a, 1)
        xx = np.linspace(xs_a.min(), xs_a.max(), 50)
        ax.plot(xx, b * xx + a, "k--", alpha=0.6)
        r_txt = ""
        if spearmanr is not None:
            rho, pval = spearmanr(xs_a, ys_a)
            r_txt = f"  Spearman ρ={rho:.2f}"
        ax.set_title(f"Money plot: pairing quality → prediction{r_txt}")
    ax.set_xlabel("OT transport error (worse pairing →)")
    ax.set_ylabel("downstream latent W2 (↓)")
    fig.tight_layout()
    fig.savefig(METRIC_DIR / f"money_plot{sfx}.png", dpi=140)
    plt.close(fig)
    print(f"[agg] -> {METRIC_DIR/f'money_plot{sfx}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--task", default="interp", choices=["interp", "extrap"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sfx = "" if args.task == "interp" else f"_{args.task}"

    recs = load_downstream(args.sigma, seeds, task=args.task)
    if not recs:
        print(f"[agg/{args.task}] no downstream metrics.json yet (run train first).")
        return
    otq = load_ot_quality(task=args.task)

    print(f"\n===== downstream main table (task={args.task}, σ={args.sigma}) =====")
    rows = main_table(recs)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(rows, METRIC_DIR / f"main_table{sfx}.csv",
             ["regime", "arm", "n", "w2_mean", "w2_std"])
    save_csv(recs, METRIC_DIR / f"downstream_runs{sfx}.csv",
             ["regime", "seed", "arm", "w2"])
    json.dump({"main_table": rows, "runs": recs},
              open(METRIC_DIR / f"downstream_summary{sfx}.json", "w"), indent=2)
    make_plots(recs, otq, args.sigma, sfx=sfx)


if __name__ == "__main__":
    main()
