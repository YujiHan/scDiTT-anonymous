from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from matplotlib.lines import Line2D  # noqa: E402
from common import plotting as plot  # noqa: E402
from step6b_allmarker_figures import (  # noqa: E402
    OUT, PANEL_ARMS, Combo, _r_table, _title_for)

VARIANT = "combined"
GENES = {
    "DR": ["wg", "srp", "Antp", "sr", "ct", "Dl", "eya", "Zasp52"],
    "SC": ["Dppa5a", "Krt7", "Hand1", "Cdx2", "Cdkn2a", "Gata3", "Sox2", "Nanog"],
    "ZB": ["sox2", "mespaa", "eve1", "klf17", "cdx4", "aldh1a2", "pax2a", "foxd3"],
}
NROW, NCOL = 2, 4

REAL_LW = 2.0
FS_TITLE, FS_TICK, FS_LAB, FS_LEG = 13, 11, 13, 12


def render(ds: str, rt) -> None:
    cb = Combo(ds, VARIANT)
    low = {n.lower(): n for n in cb.names}
    resolved, missing = [], []
    for g in GENES[ds]:
        real = low.get(g.lower())
        (resolved if real else missing).append(real if real else g)
    if missing:
        raise SystemExit(f"{ds} the following genes are not in the panel: {missing}")

    labels = cb.tps
    x = np.arange(len(labels))

    fig, axes = plot.figure(width=12.5, height=7.8, nrows=NROW, ncols=NCOL,
                            squeeze=False, layout="constrained")
    for k, gene in enumerate(resolved):
        ax = axes[k // NCOL][k % NCOL]
        gi = cb.names.index(gene)
        plot.line_by_arm(ax, x,
                         {a: cb.mu[a][gi] for a in PANEL_ARMS},
                         {a: cb.sd[a][gi] for a in PANEL_ARMS})
        ax.plot(x, cb.ref[gi], color="black", lw=REAL_LW, zorder=6)   # Real
        ttl, gated = _title_for(rt, cb, gene)
        ax.set_title(ttl, fontsize=FS_TITLE,
                     color="#8A8A8A" if gated else "black")
        ax.set_box_aspect(1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        if k % NCOL == 0:
            ax.set_ylabel("mean expr.", fontsize=FS_LAB)
        if k // NCOL == NROW - 1:
            ax.set_xlabel("held-out tp", fontsize=FS_LAB)

    h, l = plot.legend_handles(PANEL_ARMS)
    h = [Line2D([], [], color="black", lw=REAL_LW)] + h
    l = ["Real (held-out)"] + l
    fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=FS_LEG)

    stem = f"6-1-1f_panel_{ds}_{VARIANT}_8genes"
    plot.save_figure(fig, OUT, stem, formats=("png",))
    print(f"{ds} 2x4 panel: {resolved}  tps={labels}  → {OUT / (stem + '.png')}")


def main() -> None:
    plot.set_style("paper")
    OUT.mkdir(parents=True, exist_ok=True)
    rt = _r_table()
    for ds in ("SC", "ZB", "DR"):
        render(ds, rt)


if __name__ == "__main__":
    main()
