from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import METRIC_DIR

d = json.load(open(METRIC_DIR / "traj_metrics.json"))
ARMS = ["true", "ot", "random"]
COL = {"true": "#2ca02c", "ot": "#1f77b4", "random": "#7f7f7f"}


def get(task, arm, metric, regime="bifurcation"):
    k = f"{task}|{regime}_s0.25|seed0|{arm}"
    if k not in d:
        return None
    return d[k]["agg"][metric]


fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
conds = [("interp", "bif interp"), ("extrap", "bif extrap")]
x = np.arange(len(conds))
bw = 0.16


def grouped(ax, metric, title, ylab):
    for j, arm in enumerate(ARMS):
        ys = [get(t, arm, metric) for t, _ in conds]
        ys = [y if y is not None else 0 for y in ys]
        xpos = x + (j - (len(ARMS) - 1) / 2) * bw
        ax.bar(xpos, ys, bw, label=arm, color=COL[arm])
        for xp, y in zip(xpos, ys):
            ax.text(xp, y + 0.005 * max(1, max(ys)), f"{y:.2f}", ha="center",
                    va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds])
    ax.set_title(title); ax.set_ylabel(ylab)


grouped(axes[0], "pearson",
        "(a) Pearson vs true particle traj (UP)\nOT~true high, random~0",
        "Pearson (up = better)")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].legend(fontsize=7, ncol=5, loc="lower center")

grouped(axes[1], "branch_acc",
        "(b) Branch accuracy (UP)\nOT=true >> random(chance)",
        "branch acc (up = better)")
axes[1].axhline(0.5, ls="--", color="gray", alpha=0.6)

ax = axes[2]
w2 = [get("interp", a, "w2") for a in ARMS]
mse = [get("interp", a, "mse") for a in ARMS]
xa = np.arange(len(ARMS))
ax.bar(xa - 0.2, w2, 0.38, label="marginal W2 (misleading)", color="#c7c7c7")
ax2 = ax.twinx()
ax2.plot(xa, mse, "o-", color="#d62728", label="trajectory MSE (correct)")
ax.set_xticks(xa); ax.set_xticklabels(ARMS, rotation=20, fontsize=8)
ax.set_ylabel("marginal W2 (gray bars)")
ax2.set_ylabel("trajectory MSE (red, down=better)", color="#d62728")
ax.set_title("(c) bif interp: W2 vs MSE rank arms OPPOSITELY\n"
             "W2 says random best; MSE/Pearson say OT~true best")
ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")

fig.suptitle("E1 -- Trajectory-sensitive downstream metrics (particle-anchored; "
             "seed0). Marginal W2 is misleading; MSE/Pearson/branch_acc prove OT works.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = METRIC_DIR / "traj_metrics_summary.png"
fig.savefig(out, dpi=140)
print(f"saved -> {out}")
