from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import METRIC_DIR, SIGMAS

d = json.load(open(METRIC_DIR / "ot_quality.json"))
agg = defaultdict(lambda: defaultdict(list))
for key, r in d.items():
    tagk, seed, arm = key.split("|")
    for m, v in r["agg"].items():
        agg[(tagk, arm)][m].append(v)


def curve(regime, arm, metric):
    mu = [np.mean(agg[(f"{regime}_s{s:g}", arm)][metric]) for s in SIGMAS]
    sd = [np.std(agg[(f"{regime}_s{s:g}", arm)][metric]) for s in SIGMAS]
    return np.array(mu), np.array(sd)


fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
colors = {"linear": "#1f77b4", "bifurcation": "#d62728"}

ax = axes[0]
for regime in ("linear", "bifurcation"):
    mu, sd = curve(regime, "ot", "top1")
    ax.errorbar(SIGMAS, mu, yerr=sd, marker="o", color=colors[regime],
                label=f"{regime} OT top-1", capsize=3)
    mu5, _ = curve(regime, "ot", "top5")
    ax.plot(SIGMAS, mu5, marker="^", ls=":", color=colors[regime], alpha=0.7,
            label=f"{regime} OT top-5")
ax.axhline(0.001, ls="--", color="gray", label="random (~1/K)")
ax.set_xlabel("SDE diffusion sigma (harder ->)")
ax.set_ylabel("identity recovery accuracy")
ax.set_title("(a) Identity recovery: OT degrades gracefully")
ax.legend(fontsize=7)

ax = axes[1]
for regime in ("linear", "bifurcation"):
    mu, sd = curve(regime, "ot", "transport")
    ax.errorbar(SIGMAS, mu, yerr=sd, marker="o", color=colors[regime],
                label=f"{regime} OT", capsize=3)
    mr, _ = curve(regime, "random", "transport")
    ax.plot(SIGMAS, mr, marker="x", ls="--", color=colors[regime], alpha=0.6,
            label=f"{regime} random")
ax.axhline(0.0, ls="--", color="green", alpha=0.6, label="true (=0)")
ax.set_xlabel("SDE diffusion sigma (harder ->)")
ax.set_ylabel("barycentric transport error (norm, down)")
ax.set_title("(b) Transport error: OT << random")
ax.legend(fontsize=7)

ax = axes[2]
arms = ["true", "ot", "random"]
xs = [0.0, 0.0, 0.3, 0.6, 1.0]
ys = [np.mean(agg[("bifurcation_s0.25", a)]["cross_ps"]) for a in arms]
bars = ax.bar(range(len(arms)), ys,
              color=["#2ca02c", "#2ca02c", "#ff7f0e", "#ff7f0e", "#7f7f7f"])
ax.set_xticks(range(len(arms)))
ax.set_xticklabels(arms, rotation=20, fontsize=8)
for i, y in enumerate(ys):
    ax.text(i, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
ax.set_ylabel("post-split cross-branch leak (down)")
ax.set_title("(c) Branch purity (bifurcation sigma=0.25): OT=true=0")
ax.set_ylim(0, 0.58)

ax = axes[3]
for regime in ("linear", "bifurcation"):
    mu, sd = curve(regime, "ot", "mrr")
    ax.errorbar(SIGMAS, mu, yerr=sd, marker="o", color=colors[regime],
                label=f"{regime} OT MRR", capsize=3)
    mc, _ = curve(regime, "ot", "dir_cos")
    ax.plot(SIGMAS, mc, marker="s", ls=":", color=colors[regime], alpha=0.7,
            label=f"{regime} OT dir-cos")
rc = np.mean([np.mean(agg[(f"bifurcation_s{s:g}", "random")]["dir_cos"])
              for s in SIGMAS])
ax.axhline(rc, ls="--", color="gray", alpha=0.7, label="random dir-cos (~0.42)")
ax.axhline(0.001, ls="--", color="black", alpha=0.4, label="random MRR (~0)")
ax.set_xlabel("SDE diffusion sigma (harder ->)")
ax.set_ylabel("MRR / direction cosine (up)")
ax.set_title("(d) Ranking & motion direction: OT >> random")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7)

fig.suptitle("Stage C -- Direct OT pairing-quality diagnostic "
             "(simulated data, ground-truth pairing; mean over 5 seeds)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = METRIC_DIR / "ot_quality_diagnostic.png"
fig.savefig(out, dpi=140)
print(f"saved -> {out}")
