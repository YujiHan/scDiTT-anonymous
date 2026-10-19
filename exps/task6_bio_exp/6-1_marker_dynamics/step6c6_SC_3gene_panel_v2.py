from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from matplotlib.lines import Line2D  # noqa: E402
from common import plotting as plot  # noqa: E402
from step6b_allmarker_figures import OUT, PANEL_ARMS, Combo  # noqa: E402

DS, VARIANT = "SC", "combined"
GENES = ["Dppa5a", "Cdkn2a", "Krt7"]
STEM = "6-1-1e_panel_SC_combined_3genes"
BOX_ASPECT = 3 / 4

LINE_SCALE, MS_SCALE, REAL_LW, LEG_LW = 1.9, 1.25, 3.6, 3.2
FS_TITLE, FS_TICK, FS_YLAB, FS_XLAB, FS_LEG = 28, 22, 24, 24, 20


def main() -> None:
    plot.set_style("paper")
    OUT.mkdir(parents=True, exist_ok=True)

    cb = Combo(DS, VARIANT)
    name_lower = {n.lower(): n for n in cb.names}
    resolved = []
    for g in GENES:
        real = name_lower.get(g.lower())
        if real is None:
            raise SystemExit(f"gene {g} is not in the SC panel; please check the spelling.")
        resolved.append(real)

    labels = cb.tps
    x = np.arange(len(labels))

    n = len(resolved)
    fig, axes = plot.figure(width=7.4, height=5.7 * n + 1.4,
                            nrows=n, ncols=1, squeeze=False,
                            layout="constrained")
    axes = [row[0] for row in axes]
    for ax, gene in zip(axes, resolved):
        gi = cb.names.index(gene)
        mu = {a: cb.mu[a][gi] for a in PANEL_ARMS}
        sd = {a: cb.sd[a][gi] for a in PANEL_ARMS}
        for a in plot.check_arms(mu):
            y = np.asarray(mu[a], dtype=float)
            st = plot.style_for(a)
            ax.plot(x, y, **plot.plot_kwargs(
                a, linewidth=st.linewidth * LINE_SCALE,
                markersize=st.markersize * MS_SCALE, label=st.label))
            e = np.asarray(sd[a], dtype=float)
            ax.fill_between(x, y - e, y + e, color=st.color, alpha=0.15,
                            linewidth=0, zorder=st.zorder - 0.5)
        ax.plot(x, cb.ref[gi], color="black", lw=REAL_LW, zorder=6)   # Real
        gated = not bool(cb.inf[gi])
        ax.set_title(gene, fontsize=FS_TITLE,
                     color="#8A8A8A" if gated else "black")
        ax.set_box_aspect(BOX_ASPECT)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_ylabel("mean expr.", fontsize=FS_YLAB)
    axes[-1].set_xlabel("held-out tp", fontsize=FS_XLAB)

    h, l = plot.legend_handles(PANEL_ARMS)
    h = [Line2D([], [], color="black", lw=REAL_LW)] + h
    l = ["Real (held-out)"] + l
    l = ["Pos. ctrl" if s.startswith("Pos. ctrl") else s for s in l]
    leg = fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=FS_LEG)
    for ln in leg.get_lines():
        ln.set_linewidth(LEG_LW)
    plot.save_figure(fig, OUT, STEM, formats=("png",))
    print(f"SC combined 3-gene panel v2 (3x1, 3:4 subplot aspect, gene-name titles only): {resolved}")
    print(f"figure -> {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
