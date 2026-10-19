from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import plotting as plot  # noqa: E402
from step6b_allmarker_figures import OUT, HEAT_ARMS, Combo  # noqa: E402

DS = "SC"
STEM = "6-1-4e_heatmap_SC_combined_35genes"

GENES = ["Hand1", "Cdx2", "Cdkn2a", "Top2a", "Cdkn1a", "Gata3", "Peg3", "Peg10",
         "Esx1", "Elavl3", "Apoe", "Lama1", "Dab2", "Sparc", "Esrrb", "Dppa5a",
         "Lin28a", "Fbxo15", "Tfcp2l1", "Morc1", "Krt7", "Tfap2c", "Plac1",
         "Eln", "Pdgfrb", "Krt8", "Krt18", "Cldn3", "Cldn6", "Sox2", "Nanog",
         "Fn1", "Snai1", "Serpine1", "Mmp2"]

FS_COLTITLE, FS_GENE, FS_TP, FS_XLAB, FS_CBAR = 13, 10.5, 11, 12, 10


def main() -> None:
    plot.set_style("paper")
    OUT.mkdir(parents=True, exist_ok=True)

    cb = Combo(DS, "combined")
    low = {n.lower(): n for n in cb.names}
    idx, lab, missing = [], [], []
    for g in GENES:
        real = low.get(g.lower())
        if real is None:
            missing.append(g); continue
        idx.append(cb.names.index(real)); lab.append(real)
    if missing:
        raise SystemExit(f"the following genes are not in the SC panel: {missing} (please check the spelling).")
    idx = np.asarray(idx, dtype=int)
    gated = ~cb.inf[idx]
    tps = cb.tps

    mats = [("Real (held-out)", cb.ref[idx])]
    for a in HEAT_ARMS:
        mats.append((plot.style_for(a).label, cb.heat[a][idx]))
    vmax = float(np.nanmax([np.nanmax(np.abs(M)) for _, M in mats]))

    fig, axes = plot.figure(width=1.95 * len(mats),
                            height=max(5.5, 0.34 * len(lab)),
                            nrows=1, ncols=len(mats), squeeze=False)
    axes = axes[0]
    im = None
    for ax, (ttl, M) in zip(axes, mats):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.grid(False)
        ax.set_title(ttl, loc="left", fontsize=FS_COLTITLE)
        ax.set_xticks(range(len(tps)))
        ax.set_xticklabels(tps, fontsize=FS_TP)
        ax.set_xlabel("held-out tp", fontsize=FS_XLAB)
    axes[0].set_yticks(range(len(lab)))
    axes[0].set_yticklabels(lab, fontsize=FS_GENE)
    for t, g in zip(axes[0].get_yticklabels(), gated):
        if g:
            t.set_color("#A0A0A0")
    for a in axes[1:]:
        a.set_yticks([])
    cb_ = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb_.ax.tick_params(labelsize=FS_CBAR)

    plot.save_figure(fig, OUT, STEM, formats=("png",))
    ni = int((~gated).sum())
    print(f"SC combined selected-gene heatmap: {len(mats)} columns (Real + {len(HEAT_ARMS)} models), "
          f"tps={tps}, {len(lab)} genes (informative {ni}, grey-labelled {len(lab)-ni}), no title / large font")
    print(f"figure -> {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
