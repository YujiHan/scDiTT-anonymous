from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .controls import ARM_ORDER, sort_arms

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
WIDTH_SINGLE = 3.50
WIDTH_HALF15 = 4.92
WIDTH_DOUBLE = 7.20

DATASET_ORDER: Sequence[str] = ("ZB", "DR", "GA", "MA", "SC")

TASK_LABEL: Dict[str, str] = {
    "interpolation": "Interpolation",
    "extrapolation": "Extrapolation",
}


@dataclass(frozen=True)
class ArmStyle:
    color: str
    label: str
    label_zh: str
    group: str                      # 'negative' | 'baseline' | 'ours' | 'positive'
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.2
    markersize: float = 4.0
    alpha: float = 0.95
    zorder: float = 2.0


ARM_STYLE: Dict[str, ArmStyle] = {
    "neg_mean_noise": ArmStyle(
        color="#C9C9C9", label="Neg. ctrl (mean+noise)", label_zh="Neg. ctrl 1 mean+noise",
        group="negative", marker="x", linestyle=":", linewidth=1.0,
        markersize=4.0, alpha=0.90, zorder=1.0),
    "neg_shuffle": ArmStyle(
        color="#8A8A8A", label="Neg. ctrl (shuffled)", label_zh="Neg. ctrl 2 shuffled",
        group="negative", marker="+", linestyle=":", linewidth=1.0,
        markersize=4.5, alpha=0.90, zorder=1.1),

    "PI-SDE": ArmStyle(color="#0072B2", label="PI-SDE", label_zh="PI-SDE",
                       group="baseline", marker="s", linewidth=1.0,
                       markersize=3.4, alpha=0.85, zorder=2.0),
    "scIMF": ArmStyle(color="#56B4E9", label="scIMF", label_zh="scIMF",
                      group="baseline", marker="^", linewidth=1.0,
                      markersize=3.6, alpha=0.85, zorder=2.0),
    "scNODE": ArmStyle(color="#009E73", label="scNODE", label_zh="scNODE",
                       group="baseline", marker="v", linewidth=1.0,
                       markersize=3.6, alpha=0.85, zorder=2.0),
    "MIOFlow": ArmStyle(color="#E69F00", label="MIOFlow", label_zh="MIOFlow",
                        group="baseline", marker="D", linewidth=1.0,
                        markersize=3.0, alpha=0.85, zorder=2.0),
    "PRESCIENT": ArmStyle(color="#CC79A7", label="PRESCIENT", label_zh="PRESCIENT",
                          group="baseline", marker="P", linewidth=1.0,
                          markersize=3.8, alpha=0.85, zorder=2.0),
    "scDiffEq": ArmStyle(color="#A6761D", label="scDiffEq", label_zh="scDiffEq",
                         group="baseline", marker="*", linewidth=1.0,
                         markersize=5.0, alpha=0.85, zorder=2.0),

    "scDiTT": ArmStyle(color="#D62728", label="scDiTT (ours)", label_zh="scDiTT (ours)",
                       group="ours", marker="o", linewidth=2.0,
                       markersize=5.0, alpha=1.0, zorder=5.0),

    "vae_recon": ArmStyle(
        color="#6A3D9A", label="VAE-recon (decoder(μ))", label_zh="VAE reconstruction only",
        group="reference", marker="", linestyle="-.", linewidth=1.3,
        markersize=0.0, alpha=0.90, zorder=3.5),

    "pos_split_half": ArmStyle(
        color="#000000", label="Pos. ctrl (real split-half)", label_zh="Pos. ctrl real split-half",
        group="positive", marker="", linestyle="--", linewidth=1.3,
        markersize=0.0, alpha=0.90, zorder=4.0),
}

GROUP_BAND: Dict[str, str] = {
    "negative": "#F5F5F5",
    "baseline": "#FFFFFF",
    "ours": "#FDECEC",
    "reference": "#F3EEF7",
    "positive": "#F0F0F0",
}


# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
def set_style(context: str = "paper", grid: bool = True) -> None:
    base = {"paper": 8.0, "talk": 11.0}.get(context)
    if base is None:
        raise ValueError(f"context must be 'paper' or 'talk', got {context!r}")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "axes.axisbelow": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.autolayout": False,
    })


# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
_UNKNOWN_WARNED: set = set()


def style_for(name: str) -> ArmStyle:
    st = ARM_STYLE.get(name)
    if st is None:
        if name not in _UNKNOWN_WARNED:
            _UNKNOWN_WARNED.add(name)
            warnings.warn(
                f"method name {name!r} is not registered in ARM_STYLE; falling back to neutral grey."
                f"Register new methods in common/plotting.py; fix the spelling otherwise."
                f"registered: {sorted(ARM_STYLE)}",
                stacklevel=2)
        return ArmStyle(color="#B0B0B0", label=name, label_zh=name,
                        group="baseline", marker="o")
    return st


def register_arm(name: str, style: ArmStyle, after: Optional[str] = None,
                 overwrite: bool = False) -> None:
    global _ARM_ORDER_RUNTIME
    if name in ARM_STYLE and not overwrite:
        return
    ARM_STYLE[name] = style
    order = list(_ARM_ORDER_RUNTIME)
    if name in order:
        order.remove(name)
    anchor = after if after in order else "scDiTT"
    idx = order.index(anchor) + (1 if after in order else 0)
    order.insert(idx, name)
    _ARM_ORDER_RUNTIME = tuple(order)


_ARM_ORDER_RUNTIME: Sequence[str] = tuple(ARM_ORDER)


def runtime_arm_order() -> Sequence[str]:
    return _ARM_ORDER_RUNTIME


def plot_kwargs(name: str, **override) -> dict:
    st = style_for(name)
    kw = dict(color=st.color, marker=st.marker or None, linestyle=st.linestyle,
              linewidth=st.linewidth, markersize=st.markersize,
              alpha=st.alpha, zorder=st.zorder, label=st.label)
    kw.update(override)
    return kw


def bar_kwargs(name: str, **override) -> dict:
    st = style_for(name)
    kw = dict(color=st.color, alpha=st.alpha, zorder=st.zorder, label=st.label,
              edgecolor="black" if st.group in ("ours", "reference", "positive") else "none",
              linewidth=0.8 if st.group in ("ours", "reference", "positive") else 0.0)
    kw.update(override)
    return kw


def colors_for(names: Sequence[str]) -> list:
    return [style_for(n).color for n in names]


def labels_for(names: Sequence[str], zh: bool = False) -> list:
    return [(style_for(n).label_zh if zh else style_for(n).label) for n in names]


def check_arms(names: Iterable[str]) -> list:
    names = list(names)
    unknown = [n for n in names if n not in ARM_STYLE]
    if unknown:
        warnings.warn(f"the following arm names are not registered: {unknown} (use register_arm() to register a local arm)",
                      stacklevel=2)
    idx = {k: i for i, k in enumerate(_ARM_ORDER_RUNTIME)}
    return sorted(names, key=lambda x: (idx.get(x, len(idx)), x))


# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
def legend_handles(names: Sequence[str], kind: str = "line", zh: bool = False):
    """Build legend handles **sorted by the canonical order**.

    Parameters
    ----------
    kind
        ``'line'`` uses a line + marker, ``'patch'`` uses a colour patch (for bar / box plots).

    Examples
    --------
    ::

        h, l = legend_handles(arms)
        fig.legend(h, l, loc="lower center", ncol=5)
    """
    ordered = sort_arms(names)
    handles, labels = [], []
    for n in ordered:
        st = style_for(n)
        if kind == "patch":
            handles.append(Patch(facecolor=st.color, alpha=st.alpha,
                                 edgecolor="black" if st.group in ("ours", "reference", "positive") else "none",
                                 linewidth=0.8))
        else:
            handles.append(Line2D([], [], color=st.color, marker=st.marker or None,
                                  linestyle=st.linestyle, linewidth=st.linewidth,
                                  markersize=st.markersize, alpha=st.alpha))
        labels.append(st.label_zh if zh else st.label)
    return handles, labels


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.06,
                size: Optional[float] = None) -> None:
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size or (mpl.rcParams["font.size"] + 2),
            fontweight="bold", va="bottom", ha="right")


def positive_control_line(ax, value: float, orient: str = "h",
                          annotate: bool = True, label: str = "Pos. ctrl") -> None:
    st = style_for("pos_split_half")
    fn = ax.axhline if orient == "h" else ax.axvline
    fn(value, color=st.color, linestyle=st.linestyle,
       linewidth=st.linewidth, alpha=st.alpha, zorder=st.zorder)
    if annotate:
        if orient == "h":
            ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"),
                        xytext=(2, 1), textcoords="offset points",
                        fontsize=mpl.rcParams["font.size"] - 1.5,
                        color=st.color, va="bottom", ha="left")
        else:
            ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(1, 2), textcoords="offset points",
                        fontsize=mpl.rcParams["font.size"] - 1.5,
                        color=st.color, va="bottom", ha="left", rotation=90)


def bar_by_arm(ax, values: Dict[str, float], errs: Optional[Dict[str, float]] = None,
               zh: bool = False, rotate: float = 45.0,
               band: bool = True) -> list:
    names = check_arms(values)
    pos_val = values.pop("pos_split_half", None) if "pos_split_half" in values else None
    names = [n for n in names if n != "pos_split_half"]

    xs = np.arange(len(names))
    for x, n in zip(xs, names):
        e = (errs or {}).get(n)
        ax.bar(x, values[n], width=0.72, yerr=e, capsize=2.0,
               error_kw=dict(elinewidth=0.7, capthick=0.7),
               **bar_kwargs(n, label=None))
        if band and style_for(n).group == "ours":
            ax.axvspan(x - 0.5, x + 0.5, color=GROUP_BAND["ours"], zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_for(names, zh=zh), rotation=rotate,
                       ha="right" if rotate else "center")
    if pos_val is not None:
        positive_control_line(ax, pos_val)
    return names


def line_by_arm(ax, x, series: Dict[str, Sequence[float]],
                errs: Optional[Dict[str, Sequence[float]]] = None,
                zh: bool = False) -> None:
    for n in check_arms(series):
        y = np.asarray(series[n], dtype=float)
        ax.plot(x, y, **plot_kwargs(n, label=(style_for(n).label_zh if zh
                                              else style_for(n).label)))
        if errs and n in errs:
            e = np.asarray(errs[n], dtype=float)
            ax.fill_between(x, y - e, y + e, color=style_for(n).color,
                            alpha=0.15, linewidth=0, zorder=style_for(n).zorder - 0.5)


# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
def save_figure(fig, out_dir, stem: str, formats: Sequence[str] = ("pdf", "png"),
                dpi: int = 400, close: bool = True) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, dpi=dpi)
        written.append(p)
    if close:
        plt.close(fig)
    return written


def figure(width: float = WIDTH_SINGLE, height: Optional[float] = None,
           nrows: int = 1, ncols: int = 1, **kw):
    if height is None:
        height = width / 1.618 * nrows
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


__all__ = [
    "WIDTH_SINGLE", "WIDTH_HALF15", "WIDTH_DOUBLE",
    "DATASET_ORDER", "TASK_LABEL", "ArmStyle", "ARM_STYLE", "GROUP_BAND",
    "set_style", "style_for", "plot_kwargs", "bar_kwargs", "colors_for",
    "labels_for", "check_arms", "legend_handles", "panel_label",
    "positive_control_line", "bar_by_arm", "line_by_arm",
    "save_figure", "figure",
]
