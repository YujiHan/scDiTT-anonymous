"""Multi-seed trajectory-metric aggregation + error-bar figure (the three core arms true/ot/random).

Usage: python report_traj_multiseed.py [--regime bifurcation|trifurcation]
Reads metrics/traj_metrics.json (run traj_eval --seeds 42,43,44,45,46 first) and, for the given regime,
computes mean ± std across seeds by (task, arm). trifurcation additionally includes a "rare branch recall" panel.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import METRIC_DIR

ARMS = ["true", "ot", "random"]
COL = {"true": "#2ca02c", "ot": "#1f77b4", "random": "#7f7f7f"}
TASKS = ["interp", "extrap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="bifurcation")
    ap.add_argument("--sigma", default="0.25")
    args = ap.parse_args()
    tg_want = f"{args.regime}_s{args.sigma}"
    sfx = "" if args.regime == "bifurcation" else f"_{args.regime}"

    d = json.load(open(METRIC_DIR / "traj_metrics.json"))
    metrics = ["mse", "pearson", "branch_acc", "rare_recall"]
    vals = defaultdict(lambda: defaultdict(list))   # (task,arm) -> metric -> [seeds]
    seeds_seen = set()
    for k, r in d.items():
        task, tg, sd, arm = k.split("|")
        if tg != tg_want:
            continue
        seeds_seen.add(sd)
        for m in metrics:
            v = r["agg"].get(m)
            if v is not None:
                vals[(task, arm)][m].append(v)

    nseed = len(seeds_seen)
    has_rare = any(vals[(t, a)].get("rare_recall") for t in TASKS for a in ARMS)
    print(f"{args.regime}, across {nseed} seed ({sorted(seeds_seen)}) -- mean ± std\n")

    rows = []
    show = ["mse", "pearson", "branch_acc"] + (["rare_recall"] if has_rare else [])
    for task in TASKS:
        print(f"=== {task} ===")
        print(f"{'arm':>8}" + "".join(f"{m:>18}" for m in show))
        for arm in ARMS:
            v = vals[(task, arm)]
            if not v:
                continue
            row = {"task": task, "arm": arm, "n": len(v.get("mse", []))}
            cells = []
            for m in show:
                a = v.get(m, [])
                mu, sd = (np.mean(a), np.std(a)) if a else (np.nan, 0)
                row[f"{m}_mean"], row[f"{m}_std"] = mu, sd
                cells.append(f"{mu:>9.3f}±{sd:<6.3f}")
            rows.append(row)
            print(f"{arm:>8}" + "".join(cells))
        print()

    csv_path = METRIC_DIR / f"traj_multiseed{sfx}_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[report] -> {csv_path}")

    panels = [("pearson", "(a) Pearson vs true traj (UP)\ntrue~ot >> random(~0)", 0),
              ("mse", "(b) trajectory MSE (DOWN)", None),
              ("branch_acc", "(c) branch accuracy (UP)", 0.5)]
    if has_rare:
        panels.append(("rare_recall",
                       "(d) rare-branch recall (UP)\nOT keeps rare branch, random drowns it",
                       None))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.4))
    x = np.arange(len(TASKS)); bw = 0.22
    for ax, (metric, title, hline) in zip(axes, panels):
        for j, arm in enumerate(ARMS):
            mu = [np.mean(vals[(t, arm)].get(metric, [np.nan])) for t in TASKS]
            sd = [np.std(vals[(t, arm)].get(metric, [0])) for t in TASKS]
            xp = x + (j - 1) * bw
            ax.bar(xp, mu, bw, yerr=sd, capsize=2.5, label=arm, color=COL[arm])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{args.regime[:4]} {t}" for t in TASKS])
        ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, ls="--", color="gray", alpha=0.6)
    axes[0].legend(fontsize=8, ncol=3, loc="lower center")
    fig.suptitle(f"OT ablation downstream — {args.regime} "
                 f"(trajectory metrics, mean±std over {nseed} seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = METRIC_DIR / f"traj_multiseed{sfx}.png"
    fig.savefig(png, dpi=140)
    print(f"[report] -> {png}")


if __name__ == "__main__":
    main()
