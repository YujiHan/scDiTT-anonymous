#!/usr/bin/env python3
"""Line plots of the extrapolation-steps experiment: W2 vs number of steps.

Two figures (alltp / lasttp), each with one panel per dataset. One colour +
marker per method; error bars = std across seeds at every node.

READ-ONLY on source results (reuses build_metrics_tables collection logic).
Writes only PNG/PDF into ./figures/.
Run with the project env:
  python plot_extrap_steps.py
"""
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_metrics_tables import METHOD_ORDER, DATASETS, collect

STEPS = list(range(1, 7))  # ex1..ex6 -> 1..6 extrapolation steps (ex0=0 excluded)
KINDS = {
    "alltp": ("metrics_alltp.json", "all timepoints averaged"),
    "lasttp": ("metrics_lasttp.json", "last (furthest) timepoint only"),
}
METRIC = "w2_avg"

# datasets to display (GA excluded per request), in requested order
PLOT_DATASETS = ["DR", "MA", "SC", "ZB"]

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

# scDiTT = black (emphasised: thick line + big marker, stands out clearly);
# baselines take tab10 hues. Distinct markers per method give secondary
# (non-colour) encoding.
STYLE = {
    "scDiTT":    dict(color="#000000", marker="o"),  # black (our method)
    "MIOFlow":   dict(color="#1f77b4", marker="s"),  # tab10 blue
    "PI-SDE":    dict(color="#2ca02c", marker="^"),  # tab10 green
    "PRESCIENT": dict(color="#d62728", marker="D"),  # tab10 red
    "scDiffEq":  dict(color="#9467bd", marker="v"),  # tab10 purple
    "scIMF":     dict(color="#8c564b", marker="P"),  # tab10 brown
    "scNODE":    dict(color="#e377c2", marker="X"),  # tab10 pink
}
LABELS = {"scDiTT": "scDiTT (ours)"}


def series(method, ds, filename):
    """Return (xs, means, stds) over STEPS for one method/dataset/kind."""
    xs, means, stds = [], [], []
    for k in STEPS:
        tmpl = "%s/extrapolation_steps/ex%d/seed{s}" % (ds, k)
        avgs = collect(method, ds, tmpl, filename)
        vals = [a[METRIC] for a in avgs.values() if METRIC in a]
        if not vals:
            continue
        xs.append(k)
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) >= 2 else 0.0)
    return xs, means, stds


def draw(kind, datasets, nrows, ncols, figsize, suffix, legend_ncol):
    filename, subtitle = KINDS[kind]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()

    for idx, ax in enumerate(axes):
        if idx >= len(datasets):
            ax.axis("off")
            continue
        ds = datasets[idx]
        row, col = idx // ncols, idx % ncols
        bottom_row = row == nrows - 1
        left_col = col == 0
        for method in METHOD_ORDER:
            xs, means, stds = series(method, ds, filename)
            if not xs:
                continue
            # log y-axis: clip the lower whisker so mean - lower stays positive
            lower = [min(s, m * 0.999) for m, s in zip(means, stds)]
            yerr = [lower, stds]
            st = STYLE[method]
            emph = method == "scDiTT"
            ax.errorbar(
                xs, means, yerr=yerr,
                color=st["color"], marker=st["marker"],
                markersize=8 if emph else 6.5,
                linewidth=2.6 if emph else 1.8,
                capsize=3, elinewidth=1.2, capthick=1.2,
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=6 if emph else 3,
                label=LABELS.get(method, method),
            )
        ax.set_title(ds, fontsize=14, fontweight="bold", pad=6)
        ax.set_yscale("log")
        ax.set_xticks(STEPS)
        ax.grid(axis="y", which="both", color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.06)
        if bottom_row:
            ax.set_xlabel("Extrapolation steps", fontsize=12)
        if left_col:
            ax.set_ylabel("W2  (log scale, lower is better)", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
               fontsize=12, frameon=False, columnspacing=1.6,
               handlelength=2.2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Extrapolation-steps: W2 vs number of steps  ({subtitle})",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, f"extrap_steps_{kind}_{suffix}.png")
    pdf = os.path.join(FIG_DIR, f"extrap_steps_{kind}_{suffix}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main():
    # scDiTT = black. 4 datasets (GA excluded), both alltp & lasttp, two layouts.
    # New "_black" suffix keeps the earlier orange-scDiTT figures un-overwritten.
    layouts = [
        (1, 4, (19.0, 4.9), "row1x4"),
        (2, 2, (10.5, 9.0), "grid2x2"),
    ]
    for kind in ["lasttp", "alltp"]:
        for nrows, ncols, figsize, lay in layouts:
            draw(kind, PLOT_DATASETS, nrows, ncols, figsize,
                 suffix=f"{lay}_black", legend_ncol=7)


if __name__ == "__main__":
    main()
