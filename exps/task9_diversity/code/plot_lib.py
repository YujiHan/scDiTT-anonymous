from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

TASK_TITLE = {"interpolation": "Interpolation", "extrapolation": "Extrapolation"}
SCDITT_C = "#c1272d"
BASE_C = {"PRESCIENT": "#1f77b4", "PI-SDE": "#2ca02c", "scDiffEq": "#9467bd",
          "MIOFlow": "#ff7f0e", "scNODE": "#17becf", "scIMF": "#8c564b"}
BASE_M = {"PRESCIENT": "o", "PI-SDE": "s", "scDiffEq": "^", "MIOFlow": "D",
          "scNODE": "v", "scIMF": "P"}
DS_M = {"ZB": "o", "DR": "s", "SC": "^", "GA": "D", "MA": "v"}

WATERMARK: str | None = None


def stamp(fig) -> None:
    if not WATERMARK:
        return
    n = max(len(fig.axes), 1)
    size = 9.0 if n > 2 else 7.5
    for ax in fig.axes:
        ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes, fontsize=size,
                color="#c1272d", alpha=0.20, ha="center", va="center",
                rotation=20, zorder=100, fontweight="bold", clip_on=True,
                wrap=True)


_BASE = {"font": 10.0, "label": 10.5, "title": 12.0, "tick": 9.0, "legend": 8.5,
         "axlw": 1.0, "dot_s": 22.0, "ms_bl": 8.0, "ms_sc": 9.0, "elw": 1.4,
         "cap": 3.0, "capw": 1.4, "mew": 0.8, "refline": 1.1, "ann": 7.5,
         "note": 7.0}
SZ = dict(_BASE)


def set_scale(k: float = 1.0) -> None:
    for key, v in _BASE.items():
        SZ[key] = v * (k * k if key == "dot_s" else k)
    plt.rcParams.update({
        "font.size": SZ["font"], "axes.labelsize": SZ["label"],
        "axes.titlesize": SZ["title"], "xtick.labelsize": SZ["tick"],
        "ytick.labelsize": SZ["tick"], "legend.fontsize": SZ["legend"],
        "axes.linewidth": SZ["axlw"], "xtick.major.width": SZ["axlw"],
        "ytick.major.width": SZ["axlw"],
    })


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight"})
set_scale(1.0)


def decorate(ax, xlim, ylim, xlabel=True, ylabel=True, label_lines=True,
             ref_lines: bool = True, xlabel_text: str | None = None,
             ylabel_text: str | None = None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if ref_lines:
        ax.axvline(1.0, ls="--", lw=SZ["refline"], color="0.45", zorder=1)
        ax.axhline(1.0, ls="--", lw=SZ["refline"], color="0.45", zorder=1)
        ax.add_patch(plt.Rectangle((1.0, ylim[0]), xlim[1] - 1.0, 1.0 - ylim[0],
                                   facecolor="#4daf4a", alpha=0.06, zorder=0, lw=0))
    if ref_lines and label_lines:
        ax.annotate("sampling floor", xy=(1.0, ylim[1]), xytext=(3, -3),
                    textcoords="offset points", rotation=90, va="top",
                    ha="left", fontsize=SZ["ann"], color="0.35")
        ax.annotate("real subsample", xy=(xlim[0], 1.0), xytext=(3, -3),
                    textcoords="offset points", va="top", ha="left",
                    fontsize=SZ["ann"], color="0.35")
    if xlabel:
        ax.set_xlabel(xlabel_text or
                      r"normalised diversity  $\widetilde{\mathrm{Div}}$  ($\uparrow$)")
    if ylabel:
        ax.set_ylabel(ylabel_text or
                      r"normalised fidelity  $\widetilde{\mathrm{Fid}}$  ($\downarrow$)")
    ax.tick_params(length=3.5, pad=2.0)


def limits(xs, ys, pad=0.08, include_one: bool = True):
    anchor = [1.0] if include_one else []
    xs = np.asarray([v for v in xs if np.isfinite(v)] + anchor)
    ys = np.asarray([v for v in ys if np.isfinite(v)] + anchor)
    def rng(v):
        lo, hi = float(v.min()), float(v.max())
        m = max(hi - lo, 1e-3) * pad
        return lo - m, hi + m
    return rng(xs), rng(ys)


# ---------------------------------------------------------------- version A
def fig_version_a(df: pd.DataFrame, div_col: str, out: Path,
                  note: str | None = None, cfg: float = C.CFG_MAIN,
                  xlabel_text: str | None = None,
                  ylabel_text: str | None = None,
                  hline_one: bool = False) -> None:
    sc = df[(df.method == "scDiTT") & np.isclose(df.cfg_weight, cfg)]
    bl = df[df.method != "scDiTT"]

    sc_agg = (sc.groupby(["task", "cfg_weight"], as_index=False)
                [["fid_norm", div_col]].agg(["mean", "std"]))
    sc_agg.columns = ["_".join(c).rstrip("_") for c in sc_agg.columns.to_flat_index()]
    bl_agg = (bl.groupby(["task", "method"], as_index=False)
                [["fid_norm", div_col]].agg(["mean", "std"]))
    bl_agg.columns = ["_".join(c).rstrip("_") for c in bl_agg.columns.to_flat_index()]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, task in zip(axes, C.TASKS):
        s = sc[sc.task == task]
        bt = bl[bl.task == task]
        xlim, ylim = limits(list(s[div_col]) + list(bt[div_col]),
                            list(s.fid_norm) + list(bt.fid_norm),
                            include_one=False)
        ax.scatter(s[div_col], s.fid_norm, s=SZ["dot_s"], c=SCDITT_C, lw=0,
                   alpha=0.85, zorder=2)
        b = bl[bl.task == task]
        if len(b):
            for m in b.method.unique():
                bm = b[b.method == m]
                ax.scatter(bm[div_col], bm.fid_norm, s=SZ["dot_s"],
                           c=BASE_C.get(m, "0.4"), lw=0, alpha=0.85, zorder=2)
        ba = bl_agg[bl_agg.task == task] if len(bl_agg) else bl_agg
        for _, r in ba.iterrows():
            m = r["method"]
            if not np.isfinite(r[f"{div_col}_mean"]):
                ax.axhline(r["fid_norm_mean"], color=BASE_C.get(m, "0.4"),
                           lw=1.0, ls=":", alpha=0.85, zorder=3)
                ax.text(xlim[1], r["fid_norm_mean"], f"{m} ", fontsize=SZ["note"],
                        color=BASE_C.get(m, "0.4"), va="center", ha="right", zorder=6)
                continue
            ax.errorbar(r[f"{div_col}_mean"], r["fid_norm_mean"],
                        xerr=r.get(f"{div_col}_std", np.nan),
                        yerr=r.get("fid_norm_std", np.nan),
                        fmt=BASE_M.get(m, "o"), ms=SZ["ms_bl"], lw=SZ["elw"],
                        capsize=SZ["cap"], capthick=SZ["capw"],
                        color=BASE_C.get(m, "0.4"), mec="white",
                        mew=SZ["mew"], zorder=4)
        a = sc_agg[sc_agg.task == task]
        xa, ya = a[f"{div_col}_mean"], a["fid_norm_mean"]
        ax.errorbar(xa, ya, xerr=a[f"{div_col}_std"], yerr=a["fid_norm_std"],
                    fmt="o", ms=SZ["ms_sc"], lw=SZ["elw"], elinewidth=SZ["elw"],
                    capsize=SZ["cap"], capthick=SZ["capw"], color=SCDITT_C,
                    mec="white", mew=SZ["mew"], zorder=5)
        if False:
            span_x = float(xa.max() - xa.min()) / max(xlim[1] - xlim[0], 1e-9)
            span_y = float(ya.max() - ya.min()) / max(ylim[1] - ylim[0], 1e-9)
            if max(span_x, span_y) < 0.04:
                ax.annotate(r"$w$=1$-$7 (no effect)",
                            (xa.iloc[-1], ya.iloc[-1]),
                            textcoords="offset points", xytext=(7, 7),
                            ha="left", va="bottom", fontsize=7.5, color=SCDITT_C)
            else:
                for _, r in a.iloc[[0, -1]].iterrows():
                    ax.annotate(f"$w${r.cfg_weight:g}",
                                (r[f"{div_col}_mean"], r["fid_norm_mean"]),
                                textcoords="offset points", xytext=(0, -11),
                                ha="center", fontsize=7.5, color=SCDITT_C)
        ax.set_title(TASK_TITLE[task])
        decorate(ax, xlim, ylim, ylabel=(ax is axes[0]), ref_lines=False,
                 xlabel_text=xlabel_text, ylabel_text=ylabel_text)
        if hline_one:
            ax.axhline(1.0, ls="--", lw=SZ["refline"], color="0.45", zorder=1)
            ax.annotate("no better than copy-last", xy=(xlim[0], 1.0),
                        xytext=(4, 3), textcoords="offset points",
                        va="bottom", ha="left", fontsize=SZ["ann"], color="0.35")

    handles = [Line2D([], [], color=SCDITT_C, marker="o", ms=SZ["ms_bl"],
                      ls="None", label="scDiTT")]
    for m in [m for m in BASE_M if m in set(bl.method)]:
        has_div = np.isfinite(bl[bl.method == m][div_col]).any()
        handles.append(Line2D([], [], color=BASE_C[m],
                              marker=BASE_M[m] if has_div else "None",
                              ls="None" if has_div else ":", ms=SZ["ms_bl"],
                              lw=SZ["elw"],
                              label=m + ("" if has_div else " (fidelity only)")))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=len(handles), frameon=False, handlelength=1.6,
               borderpad=0.3, columnspacing=1.4, handletextpad=0.5)
    if note:
        axes[0].text(0.02, 0.02, note, transform=axes[0].transAxes,
                     fontsize=SZ["note"], color="0.45", va="bottom",
                     ha="left", linespacing=1.4)
    fig.tight_layout(pad=0.4, w_pad=1.4, rect=(0, 0.055, 1, 1))
    stamp(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[fig] {out}")


# ---------------------------------------------------------------- version B
