from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CKPT_DIR, METRIC_DIR, tag

ARMS = ["true", "ot", "random"]
COND = [("bifurcation", "interp"), ("linear", "interp"),
        ("bifurcation", "extrap"), ("linear", "extrap")]
COLORS = {"true": "#2ca02c", "ot": "#1f77b4", "random": "#7f7f7f",
          }


def w2(regime, arm, task, sigma=0.25, seed=0):
    sub = arm if task == "interp" else f"extrap_{arm}"
    mj = os.path.join(str(CKPT_DIR), f"{tag(regime, sigma)}", f"seed{seed}",
                      sub, "metrics.json")
    if not os.path.exists(mj):
        return None
    return json.load(open(mj))["best_avg_w2"]


fig, ax = plt.subplots(figsize=(11, 4.6))
n_cond = len(COND)
n_arm = len(ARMS)
bw = 0.16
x = np.arange(n_cond)
for j, arm in enumerate(ARMS):
    ys = [w2(reg, arm, task) for reg, task in COND]
    xpos = x + (j - (n_arm - 1) / 2) * bw
    ys_plot = [y if y is not None else 0 for y in ys]
    bars = ax.bar(xpos, ys_plot, bw, label=arm, color=COLORS[arm])
    for xp, y in zip(xpos, ys):
        if y is not None:
            ax.text(xp, y + 0.1, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([f"{reg}\n{task}" for reg, task in COND])
ax.set_ylabel("held-out latent W2 (down = better)")
ax.set_title("Downstream prediction vs pairing arm (seed0, sigma=0.25)\n"
             "OT best/near-best everywhere; but true(oracle) often worst "
             "(overfits), noise arms erratic -> marginal W2 weakly pairing-sensitive")
ax.legend(title="pairing arm", fontsize=8, ncol=5, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = METRIC_DIR / "downstream_summary.png"
fig.savefig(out, dpi=140)
print(f"saved -> {out}")
