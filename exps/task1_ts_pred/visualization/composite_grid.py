#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umap_visualize import (  # noqa: E402
    GRAY, PROJ_ROOT, DEFAULT_RESULTS_ROOT, DEFAULT_OUT_DIR,
    build_or_load_embedding, project_pred_test_cells, tp_color_list, read_eval_tps,
)
from baseline_umap_visualize import resolve_seed_dir, PRED_FILE, H5AD  # noqa: E402

SEED = 46
ALL_ROWS = ["ZB", "SC", "DR", "MA"]
ALL_COLUMNS = [
    ("Ground Truth", "GT"),
    ("scDiTT", "scDiTT"),
    ("scIMF", "scIMF"),
    ("PI-SDE", "PI-SDE"),
    ("scNODE", "scNODE"),
    ("scDiffEq", "scDiffEq"),
    ("PRESCIENT", "PRESCIENT"),
    ("MIOFlow", "MIOFlow"),
]
DROP_ROWS = {"MA"}
DROP_COLS = {"scDiffEq", "MIOFlow"}
ROWS = [r for r in ALL_ROWS if r not in DROP_ROWS]
COLUMNS = [c for c in ALL_COLUMNS if c[1] not in DROP_COLS]

PANEL_W = 2.5
COL_GAP = 0.05
ROW_GAP = 0.16
LEFT_M = 0.5
RIGHT_M = 1.25
TOP_M = 0.5
BOT_M = 0.12
GRAY_S = 1.5
COLOR_S = 3.0
DPI = 160


def dataset_bundle(ds: str):
    rel, tkey = H5AD[ds]
    embed = build_or_load_embedding(
        ds, PROJ_ROOT / rel, tkey,
        cache_dir=DEFAULT_OUT_DIR / "_embedding_cache", refit=False)
    true_umap = embed["true_umap"]
    true_cell_tps = embed["true_cell_tps"]
    n_genes = embed["n_genes"]

    i_dir = DEFAULT_RESULTS_ROOT / ds / "interpolation" / f"seed{SEED}"
    e_dir = DEFAULT_RESULTS_ROOT / ds / "extrapolation" / f"seed{SEED}"
    i_tps = read_eval_tps(i_dir / "metrics.json")
    e_tps = read_eval_tps(e_dir / "metrics.json")
    extrap_only = [t for t in e_tps if t not in set(i_tps)]
    gt_test_tps = sorted(set(i_tps) | set(e_tps))
    colors = tp_color_list(embed["all_tps"])

    def pred_paths(model):
        if model == "scDiTT":
            return (i_dir / "pred_genes.npy", e_dir / "pred_genes.npy")
        idir, _ = resolve_seed_dir(model, ds, "interpolation", SEED)
        edir, _ = resolve_seed_dir(model, ds, "extrapolation", SEED)
        return (idir / PRED_FILE, edir / PRED_FILE)

    col_points = {}
    for _, key in COLUMNS:
        if key == "GT":
            col_points[key] = {t: true_umap[true_cell_tps == t] for t in gt_test_tps}
        else:
            ip, ep = pred_paths(key)
            pp = {}
            pp.update(project_pred_test_cells(embed, ip, i_tps, n_genes))
            pp.update(project_pred_test_cells(embed, ep, extrap_only, n_genes))
            col_points[key] = pp
    print(f"[{ds}] test tps={gt_test_tps} (extrap_only={extrap_only}) projection complete", flush=True)

    xmin, xmax = true_umap[:, 0].min(), true_umap[:, 0].max()
    ymin, ymax = true_umap[:, 1].min(), true_umap[:, 1].max()
    mx, my = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
    xlim = (xmin - mx, xmax + mx)
    ylim = (ymin - my, ymax + my)
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    return dict(true_umap=true_umap, col_points=col_points, gt_test_tps=gt_test_tps,
                colors=colors, xlim=xlim, ylim=ylim, aspect=aspect)


def main():
    bundles = {ds: dataset_bundle(ds) for ds in ROWS}

    panel_h = {ds: PANEL_W * bundles[ds]["aspect"] for ds in ROWS}
    fig_w = LEFT_M + len(COLUMNS) * PANEL_W + (len(COLUMNS) - 1) * COL_GAP + RIGHT_M
    fig_h = TOP_M + sum(panel_h.values()) + (len(ROWS) - 1) * ROW_GAP + BOT_M

    fig = plt.figure(figsize=(fig_w, fig_h))

    row_top_in = {}
    cum = 0.0
    for r, ds in enumerate(ROWS):
        row_top_in[ds] = TOP_M + cum + r * ROW_GAP
        cum += panel_h[ds]

    for ds in ROWS:
        B = bundles[ds]
        ph = panel_h[ds]
        pb_norm = (fig_h - row_top_in[ds] - ph) / fig_h
        yc_norm = (fig_h - row_top_in[ds] - ph / 2) / fig_h

        for c, (title, key) in enumerate(COLUMNS):
            x_in = LEFT_M + c * (PANEL_W + COL_GAP)
            ax = fig.add_axes([x_in / fig_w, pb_norm, PANEL_W / fig_w, ph / fig_h])
            ax.scatter(B["true_umap"][:, 0], B["true_umap"][:, 1], color=GRAY,
                       s=GRAY_S, alpha=0.38, linewidths=0, rasterized=True)
            for t in B["gt_test_tps"]:
                coords = B["col_points"][key].get(t)
                if coords is None or len(coords) == 0:
                    continue
                ax.scatter(coords[:, 0], coords[:, 1], color=B["colors"][t],
                           s=COLOR_S, alpha=0.95, linewidths=0, rasterized=True)
            ax.set_xlim(*B["xlim"]); ax.set_ylim(*B["ylim"])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ds == ROWS[0]:
                ax.set_title(title, fontsize=14, fontfamily="serif", pad=8)

        fig.text((LEFT_M * 0.42) / fig_w, yc_norm, ds, rotation=90,
                 ha="center", va="center", fontsize=22, fontweight="bold",
                 fontfamily="serif", color="black")

        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=B["colors"][t], markeredgecolor="none",
                          label=f"{t}") for t in B["gt_test_tps"]]
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                              markerfacecolor=GRAY, markeredgecolor="none",
                              label="others"))
        x_leg = (LEFT_M + len(COLUMNS) * PANEL_W
                 + (len(COLUMNS) - 1) * COL_GAP + 0.15) / fig_w
        leg = fig.legend(handles=handles, loc="center left",
                         bbox_to_anchor=(x_leg, yc_norm), fontsize=8.5,
                         frameon=True, fancybox=True, handletextpad=0.3,
                         labelspacing=0.3, borderpad=0.4,
                         title="test TP", title_fontsize=8.5)
        leg.get_frame().set_boxstyle("round,pad=0.35,rounding_size=0.18")
        leg.get_frame().set_edgecolor("0.75")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_facecolor("white")

    out_dir = DEFAULT_OUT_DIR / "_composite"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not DROP_ROWS and not DROP_COLS:
        out_path = out_dir / f"grid_all_methods_seed{SEED}.png"
    else:
        out_path = out_dir / f"grid_{len(ROWS)}x{len(COLUMNS)}_seed{SEED}.png"
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[composite] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
