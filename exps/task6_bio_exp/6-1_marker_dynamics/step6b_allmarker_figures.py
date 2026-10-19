from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from matplotlib.lines import Line2D  # noqa: E402
from common import data_io as io, plotting as plot  # noqa: E402
import panels as pan  # noqa: E402

ROOT = HERE.parent / "results" / "6-1_marker_dynamics"
FIGDATA, MET = ROOT / "figdata", ROOT / "metrics"
OUT = ROOT / "figs_allmarkers"

DATASETS = ("ZB", "DR", "SC")
TASKS = ("interpolation", "extrapolation")
VARIANTS = ("interpolation", "extrapolation", "combined")

PANEL_ARMS = [a for a in pan.ARMS_10 if not a.startswith("neg_")]
HEAT_ARMS = ["scDiTT", "PI-SDE", "scIMF", "scNODE",
             "MIOFlow", "PRESCIENT", "scDiffEq"]

NCOL, NROW = 4, 5
PER_PAGE = NCOL * NROW
FMT = ("png",)


def _load(ds: str, task: str):
    return np.load(FIGDATA / f"curves_{ds}_{task}.npz", allow_pickle=True)


def _mean_over_seeds(z, arm: str) -> np.ndarray:
    return np.stack([z[f"{arm}_seed{s}"] for s in io.ALL_SEEDS]).mean(0)


def _std_over_seeds(z, arm: str) -> np.ndarray:
    return np.stack([z[f"{arm}_seed{s}"] for s in io.ALL_SEEDS]).std(0, ddof=1)


class Combo:

    def __init__(self, ds: str, variant: str):
        self.ds, self.variant = ds, variant
        tasks = TASKS if variant == "combined" else (variant,)
        self.segments = []
        names_ref = None
        for tk in tasks:
            z = _load(ds, tk)
            names = list(z["names"])
            if names_ref is None:
                names_ref = names
            elif names != names_ref:
                raise RuntimeError(
                    f"{ds} has mismatched interpolation / extrapolation marker lists, "
                    f"cannot concatenate (the same row would map to different genes).")
            self.segments.append(dict(
                task=tk,
                tps=[int(t) for t in z["tps"]],
                inf=np.asarray(z["informative"], dtype=bool),
                ref=_mean_over_seeds(z, "ref"),
                mu={a: _mean_over_seeds(z, a) for a in PANEL_ARMS},
                sd={a: _std_over_seeds(z, a) for a in PANEL_ARMS},
                heat={a: _mean_over_seeds(z, a) for a in HEAT_ARMS},
                n_star=[int(v) for v in z["n_star"]],
            ))
        self.names = names_ref
        self.inf = np.logical_or.reduce([s["inf"] for s in self.segments])
        self.order = sorted(range(len(self.names)),
                            key=lambda i: (not bool(self.inf[i]), i))
        self.n_pages = int(np.ceil(len(self.order) / PER_PAGE))
        self._merge()

    def _merge(self) -> None:
        cols, self.tps, self.src, self.dropped = [], [], [], []
        seen = {}
        for si, s in enumerate(self.segments):
            for j, t in enumerate(s["tps"]):
                if t in seen:
                    self.dropped.append((t, s["task"], seen[t]))
                    continue
                seen[t] = s["task"]
                cols.append((si, j))
                self.tps.append(t)
                self.src.append(s["task"])
        if self.tps != sorted(self.tps):
            raise RuntimeError(
                f"{self.tag} merged tps are not increasing: {self.tps} —— "
                f"a continuous line would fold back on the time axis; sort first or change the convention.")

        def pick(getter):
            return np.stack([getter(self.segments[si])[:, j] for si, j in cols],
                            axis=1)

        self.ref = pick(lambda s: s["ref"])
        self.mu = {a: pick(lambda s, a=a: s["mu"][a]) for a in PANEL_ARMS}
        self.sd = {a: pick(lambda s, a=a: s["sd"][a]) for a in PANEL_ARMS}
        self.heat = {a: pick(lambda s, a=a: s["heat"][a]) for a in HEAT_ARMS}

    @property
    def tag(self) -> str:
        return f"{self.ds}_{self.variant}"

    def title_txt(self) -> str:
        parts = []
        for s in self.segments:
            kept = [t for t, src in zip(self.tps, self.src) if src == s["task"]]
            parts.append(f"{plot.TASK_LABEL[s['task']]} {kept}")
        txt = " + ".join(parts)
        if self.dropped:
            txt += "  [" + "; ".join(
                f"tp{t} overlaps, kept from {plot.TASK_LABEL[keep]}"
                for t, _, keep in self.dropped) + "]"
        return txt


def _r_table() -> pd.DataFrame:
    t = pd.read_csv(MET / "T4_per_marker_r.csv")
    return t.set_index(["dataset", "task", "marker"])


def _title_for(rt: pd.DataFrame, cb: Combo, gene: str) -> tuple:
    parts = []
    for s in cb.segments:
        sfx = {"interpolation": "_i", "extrapolation": "_e"}[s["task"]]
        sfx = sfx if len(cb.segments) > 1 else ""
        if len(s["tps"]) < 3:
            parts.append(f"[{sfx.lstrip('_') or 'r'}: n_tp=2]")
            continue
        key = (cb.ds, s["task"], gene)
        r = rt.loc[key, "scDiTT"] if key in rt.index else np.nan
        parts.append(f"$r${sfx}={r:.2f}")
    gated = not bool(cb.inf[cb.names.index(gene)])
    return f"{gene}  " + " ".join(parts) + ("  (gated)" if gated else ""), gated


def panels_for(cb: Combo, rt: pd.DataFrame) -> None:
    labels = cb.tps
    x_all = np.arange(len(labels))

    for page in range(cb.n_pages):
        chunk = cb.order[page * PER_PAGE:(page + 1) * PER_PAGE]
        fig, axes = plot.figure(width=plot.WIDTH_DOUBLE, height=1.55 * NROW,
                                nrows=NROW, ncols=NCOL, squeeze=False)
        for k, gi in enumerate(chunk):
            ax = axes[k // NCOL][k % NCOL]
            gene = cb.names[gi]
            plot.line_by_arm(ax, x_all,
                             {a: cb.mu[a][gi] for a in PANEL_ARMS},
                             {a: cb.sd[a][gi] for a in PANEL_ARMS})
            ax.plot(x_all, cb.ref[gi], color="black", lw=1.8, zorder=6)
            ttl, gated = _title_for(rt, cb, gene)
            ax.set_title(ttl, fontsize=6.0,
                         color="#8A8A8A" if gated else "black")
            ax.set_xticks(x_all)
            ax.set_xticklabels(labels, fontsize=5.5)
            ax.tick_params(axis="y", labelsize=5.5)
            if k % NCOL == 0:
                ax.set_ylabel("mean expr.", fontsize=6.5)
            if k // NCOL == NROW - 1 or k >= len(chunk) - NCOL:
                ax.set_xlabel("held-out tp", fontsize=6.5)
        for k in range(len(chunk), NROW * NCOL):
            axes[k // NCOL][k % NCOL].axis("off")

        h, l = plot.legend_handles(PANEL_ARMS)
        h = [Line2D([], [], color="black", lw=1.8)] + h
        l = ["Real (held-out)"] + l
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=6,
                   bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(f"{cb.ds} — {cb.title_txt()}   "
                     f"(page {page + 1}/{cb.n_pages}, "
                     f"marker {page * PER_PAGE + 1}–"
                     f"{page * PER_PAGE + len(chunk)} / {len(cb.order)})",
                     y=1.0, fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 0.975))
        plot.save_figure(fig, OUT, f"6-1-1b_panel_{cb.tag}_p{page + 1}",
                         formats=FMT)


def heatmap_for(cb: Combo) -> None:
    labels = cb.tps
    order = np.argsort(cb.ref[:, -1] - cb.ref[:, 0])
    lab = [cb.names[i] for i in order]
    gated = ~cb.inf[order]

    mats = [("Real (held-out)", cb.ref[order])]
    for a in HEAT_ARMS:
        mats.append((plot.style_for(a).label, cb.heat[a][order]))

    vmax = float(np.nanmax([np.nanmax(np.abs(M)) for _, M in mats]))
    fig, axes = plot.figure(width=1.30 * len(mats),
                            height=max(3.0, 0.105 * len(lab)),
                            nrows=1, ncols=len(mats), squeeze=False)
    axes = axes[0]
    for ax, (ttl, M) in zip(axes, mats):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.grid(False)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_xlabel("held-out tp", fontsize=6.5)
    axes[0].set_yticks(range(len(lab)))
    axes[0].set_yticklabels(lab, fontsize=3.8)
    for t, g in zip(axes[0].get_yticklabels(), gated):
        if g:
            t.set_color("#A0A0A0")
    for a in axes[1:]:
        a.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    fig.suptitle(f"{cb.ds} — {cb.title_txt()}   (all {len(lab)} markers; "
                 f"grey label = failed informativeness gate)", y=1.0, fontsize=8)
    plot.save_figure(fig, OUT, f"6-1-4b_heatmap_{cb.tag}", formats=FMT)


# ─────────────────────────────────────────────────────────────────── main
def main() -> None:
    plot.set_style("paper")
    OUT.mkdir(parents=True, exist_ok=True)
    rt = _r_table()
    n_panel = n_heat = 0
    for ds in DATASETS:
        for variant in VARIANTS:
            cb = Combo(ds, variant)
            panels_for(cb, rt)
            heatmap_for(cb)
            n_panel += cb.n_pages
            n_heat += 1
            drop = ("  dropping overlapping column " + ", ".join(
                f"tp{t}@{tk} (keeping {keep})" for t, tk, keep in cb.dropped)
            ) if cb.dropped else ""
            print(f"  {cb.tag:22s} tps={cb.tps}  "
                  f"markers={len(cb.order)} (informative {int(cb.inf.sum())})  "
                  f"→ {cb.n_pages} panel pages + 1 heatmap{drop}")
    print(f"\nwrote {n_panel} panel PNGs + {n_heat} heatmap PNGs to {OUT}")
    print(f"actual PNG file count: {len(list(OUT.glob('*.png')))}")


if __name__ == "__main__":
    main()
