from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import plotting as plot  # noqa: E402
from step6b_allmarker_figures import OUT, Combo  # noqa: E402
from step6c7_SC_combined_genelist_heatmap import GENES

DS = "SC"
STEM = "6-1-4f_heatmap_SC_combined_35genes_3arms_sorted"
DISPLAY_ARMS = ["scDiTT", "scIMF"]

FS_COLTITLE, FS_GENE, FS_TP, FS_XLAB, FS_CBAR = 14, 11, 12, 12.5, 10.5
W_PER_COL, H_PER_ROW = 2.75, 0.42


def render(drop_bottom: int = 0, stem: str = STEM, display_arms=None,
           font_scale: float = 1.0, w_per_col: float | None = None) -> None:
    arms = DISPLAY_ARMS if display_arms is None else list(display_arms)
    wpc = W_PER_COL if w_per_col is None else float(w_per_col)
    fc = float(font_scale)
    fs_coltitle, fs_gene, fs_tp = FS_COLTITLE * fc, FS_GENE * fc, FS_TP * fc
    fs_xlab, fs_cbar = FS_XLAB * fc, FS_CBAR * fc
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
        raise SystemExit(f"the following genes are not in the SC panel: {missing}")
    idx = np.asarray(idx, dtype=int)

    expr_level = cb.ref[idx].mean(axis=1)
    order = np.argsort(-expr_level)
    idx = idx[order]
    lab = [lab[i] for i in order]
    expr_sorted = expr_level[order]

    dropped = []
    if drop_bottom > 0:
        keep = len(lab) - drop_bottom
        dropped = list(zip(lab[keep:], expr_sorted[keep:]))
        idx, lab, expr_sorted = idx[:keep], lab[:keep], expr_sorted[:keep]

    gated = ~cb.inf[idx]
    tps = cb.tps

    mats = [("Real (held-out)", cb.ref[idx])]
    for a in arms:
        mats.append((plot.style_for(a).label, cb.heat[a][idx]))
    vmax = float(np.nanmax([np.nanmax(np.abs(M)) for _, M in mats]))

    fig, axes = plot.figure(width=wpc * len(mats) + 0.6,
                            height=max(6.0, H_PER_ROW * len(lab)),
                            nrows=1, ncols=len(mats), squeeze=False)
    axes = axes[0]
    im = None
    for ax, (ttl, M) in zip(axes, mats):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.grid(False)
        ax.set_title(ttl, loc="left", fontsize=fs_coltitle)
        ax.set_xticks(range(len(tps)))
        ax.set_xticklabels(tps, fontsize=fs_tp)
        ax.set_xlabel("held-out tp", fontsize=fs_xlab)
    axes[0].set_yticks(range(len(lab)))
    axes[0].set_yticklabels(lab, fontsize=fs_gene)
    for t, g in zip(axes[0].get_yticklabels(), gated):
        if g:
            t.set_color("#A0A0A0")
    for a in axes[1:]:
        a.set_yticks([])
    cb_ = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    cb_.ax.tick_params(labelsize=fs_cbar)

    plot.save_figure(fig, OUT, stem, formats=("png",))
    ni = int((~gated).sum())
    print(f"SC combined sorted heatmap: Real + {arms}, tps={tps}, "
          f"{len(lab)} genes (informative {ni}, grey-labelled {len(lab)-ni}) sorted by expression, descending"
          + (f"; dropped the bottom {drop_bottom} lowest-expression rows" if drop_bottom else ""))
    print("row order (top->bottom, high->low expression):")
    for r, (g, e) in enumerate(zip(lab, expr_sorted), 1):
        print(f"  {r:2d}. {g:10s} mean_expr={e:.3f}")
    if dropped:
        print("dropped (bottom, lowest expression):",
              ", ".join(f"{g}({e:.3f})" for g, e in dropped))
    print(f"figure -> {OUT / (stem + '.png')}")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
