#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import common
from common import (ARMS, DATASETS, LOWER_IS_BETTER, METRIC_DIR, METRIC_KEYS,
                    PRIMARY_METRIC, REFERENCE_CSV_DIR, SEEDS, TASKS, run_dir)
from mask_arms import ARM_DESC, ARM_MATCH


def load_local(arm: str, ds: str, task: str, seed: int) -> dict | None:
    d = run_dir(arm, ds, task, seed)
    f = d / "metrics.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    avg = rec.get("avg", {})
    out = {k: float(avg[k]) for k in METRIC_KEYS if k in avg}
    if len(out) != len(METRIC_KEYS):
        return None
    stop = d / "stop_step.json"
    out["stop_step"] = (json.loads(stop.read_text()).get("stop_step")
                        if stop.exists() else None)
    out["source"] = "local"
    return out


def load_reference_csv() -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    if not REFERENCE_CSV_DIR.is_dir():
        return out
    for ds in DATASETS:
        files = sorted(REFERENCE_CSV_DIR.glob(f"progress_{ds}_*.csv"))
        if not files:
            continue
        with files[-1].open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                task = (row.get("task") or "").strip()
                if task not in TASKS:
                    continue
                if (row.get("ex_step") or "").strip():
                    continue
                try:
                    seed = int(row["n"])
                    m = {k: float(row[k]) for k in METRIC_KEYS}
                except (KeyError, TypeError, ValueError):
                    continue
                m["stop_step"] = None
                m["source"] = "reference_csv"
                out[(ds.upper(), task, seed)] = m
    return out


def collect(arms: list[str], fallback: bool) -> list[dict]:
    csv_mix = load_reference_csv() if fallback else {}
    rows: list[dict] = []
    for arm in arms:
        for ds in DATASETS:
            for task in TASKS:
                for seed in SEEDS:
                    m = load_local(arm, ds, task, seed)
                    if m is None and arm == "mix":
                        m = csv_mix.get((ds, task, seed))
                    if m is None:
                        continue
                    rows.append({"arm": arm, "dataset": ds, "task": task,
                                 "seed": seed, **m})
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """(arm, ds, task) -> {metric: (mean, std, n), sources}"""
    out: dict[tuple[str, str, str], dict] = {}
    for arm in ARMS:
        for ds in DATASETS:
            for task in TASKS:
                sub = [r for r in rows if r["arm"] == arm
                       and r["dataset"] == ds and r["task"] == task]
                if not sub:
                    continue
                rec: dict = {"n_seeds": len(sub),
                             "seeds": sorted(r["seed"] for r in sub),
                             "sources": sorted({r["source"] for r in sub})}
                for k in METRIC_KEYS:
                    v = np.array([r[k] for r in sub], dtype=float)
                    rec[k] = (float(v.mean()),
                              float(v.std(ddof=1)) if len(v) > 1 else 0.0)
                out[(arm, ds, task)] = rec
    return out


def _fmt(rec: dict, k: str) -> str:
    m, s = rec[k]
    return f"{m:.2f} ± {s:.2f}" if abs(m) >= 1 else f"{m:.4f} ± {s:.4f}"


def _md(rows: list[dict], summ: dict) -> str:
    L: list[str] = [
        "# 5-3_mask -- mask ablation main table",
        "",
        "The four arms differ only in the mask distribution; **everything else is identical** (same VAE/latents/OT cache, "
        "same per-dataset hyper-parameters, same evaluation protocol). All keep 5% `full` to fix the CFG "
        "null-condition budget -- see the top of `mask_arms.py`.",
        "",
        "| Arm | Definition |", "|---|---|",
    ]
    L += [f"| `{a}` | {ARM_DESC[a]} |" for a in ARMS]
    L += ["", f"Primary metric **{PRIMARY_METRIC}** (lower is better), mean ± std over {len(SEEDS)} seeds.", ""]

    for ds in DATASETS:
        L += [f"## {ds}", "",
              "| Arm | Task | Pattern match | " + " | ".join(METRIC_KEYS) + " | Source |",
              "|---|---|---|" + "---|" * (len(METRIC_KEYS) + 1)]
        for task in TASKS:
            base = summ.get(("mix", ds, task))
            for arm in ARMS:
                rec = summ.get((arm, ds, task))
                if rec is None:
                    L.append(f"| `{arm}` | {task} | {ARM_MATCH[arm][task]} | "
                             + " | ".join(["—"] * len(METRIC_KEYS)) + " | (not run) |")
                    continue
                cells = []
                for k in METRIC_KEYS:
                    s = _fmt(rec, k)
                    if base is not None and arm != "mix":
                        d = rec[k][0] - base[k][0]
                        worse = (d > 0) if LOWER_IS_BETTER[k] else (d < 0)
                        s += f"<br><sub>{'+' if d >= 0 else ''}{d:.2f} " \
                             f"{'✗' if worse else '✓'}</sub>"
                    cells.append(s)
                L.append(f"| `{arm}` | {task} | {ARM_MATCH[arm][task]} | "
                         + " | ".join(cells) + " | " + ",".join(rec["sources"]) + " |")
        L.append("")

    # --- matched / mismatched 2x2 ---
    L += ["## matched / mismatched 2x2 for the task-specific arms",
          "",
          f"A single task-specific mask may suffice on its **matched** task but collapses on the other -- "
          f"whereas one `mix` model comes close to both matched cells. That is the real selling point of the mixed mask.",
          f"(the table below reports {PRIMARY_METRIC}, lower is better; parentheses = difference against `mix` in the same cell)", ""]
    for ds in DATASETS:
        L += [f"### {ds}", "",
              "| Training mask | interpolation | extrapolation |", "|---|---|---|"]
        for arm in ["mix", "random", "scatter_only", "tail_only"]:
            cells = []
            for task in TASKS:
                rec = summ.get((arm, ds, task))
                base = summ.get(("mix", ds, task))
                if rec is None:
                    cells.append("—")
                    continue
                s = f"{rec[PRIMARY_METRIC][0]:.2f}"
                if base is not None and arm != "mix":
                    s += f" ({rec[PRIMARY_METRIC][0] - base[PRIMARY_METRIC][0]:+.2f})"
                tag = ARM_MATCH[arm][task]
                if tag == "matched":
                    s = f"**{s}** ✅matched"
                elif tag == "mismatched":
                    s = f"{s} ⚠️mismatched"
                cells.append(s)
            L.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        L.append("")

    loc = {(r["dataset"], r["task"], r["seed"]): r[PRIMARY_METRIC]
           for r in rows if r["arm"] == "mix" and r["source"] == "local"}
    csv_mix = load_reference_csv()
    common_cells = [k for k in loc if k in csv_mix]
    if common_cells:
        L += ["## Calibration: local mix vs reference CSV mix", "",
              "Difference for the same (dataset, task, seed) across two machines / two VAE+OT caches."
              "If |Δ| stays within seed noise the CSV mix numbers can go straight into the main table; otherwise mix must be fully retrained locally.",
              "", f"| Cell | local mix | CSV mix | Δ ({PRIMARY_METRIC}) |", "|---|---|---|---|"]
        for (ds, task, seed) in sorted(common_cells):
            a, b = loc[(ds, task, seed)], csv_mix[(ds, task, seed)][PRIMARY_METRIC]
            L.append(f"| {ds}/{task}/seed{seed} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
        L.append("")
        for ds in DATASETS:
            for task in TASKS:
                v = [csv_mix[(ds, task, s)][PRIMARY_METRIC]
                     for s in SEEDS if (ds, task, s) in csv_mix]
                if len(v) > 1:
                    L.append(f"- {ds}/{task} seed noise (CSV mix, {len(v)} seeds): "
                             f"std = {np.std(v, ddof=1):.2f}")
        L.append("")

    missing = [(a, ds, t, s) for a in ARMS for ds in DATASETS for t in TASKS
               for s in SEEDS
               if not any(r["arm"] == a and r["dataset"] == ds and r["task"] == t
                          and r["seed"] == s for r in rows)]
    if missing:
        L += ["## Cells not yet finished", "",
              *[f"- `{a}` {ds}/{t}/seed{s}" for a, ds, t, s in missing], ""]
    return "\n".join(L)


def _figure(summ: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mix": "#d62728", "random": "#7f7f7f",
              "tail_only": "#1f77b4", "scatter_only": "#2ca02c"}
    cells = [(ds, t) for ds in DATASETS for t in TASKS]
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.4))
    if len(cells) == 1:
        axes = [axes]
    for ax, (ds, task) in zip(axes, cells):
        xs, ys, es, cs, hs = [], [], [], [], []
        for arm in ARMS:
            rec = summ.get((arm, ds, task))
            if rec is None:
                continue
            xs.append(arm)
            ys.append(rec[PRIMARY_METRIC][0])
            es.append(rec[PRIMARY_METRIC][1])
            cs.append(colors.get(arm, "#999"))
            hs.append("//" if ARM_MATCH[arm][task] == "mismatched" else None)
        if not xs:
            continue
        b = ax.bar(range(len(xs)), ys, yerr=es, capsize=4, color=cs,
                   edgecolor="k", linewidth=.6)
        for bar, h in zip(b, hs):
            if h:
                bar.set_hatch(h)
        base = summ.get(("mix", ds, task))
        if base is not None:
            ax.axhline(base[PRIMARY_METRIC][0], color="#d62728", ls="--", lw=1,
                       alpha=.7, label="mix (ours)")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=20, fontsize=9)
        ax.set_ylabel(f"{PRIMARY_METRIC} (↓)")
        ax.set_title(f"{ds} / {task}\n(hatched = shape-mismatched arm)", fontsize=10)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="mask ablation result aggregation (stage D)")
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--no-fallback", action="store_true",
                   help="do not fall back to the reference CSV when the mix arm is missing")
    args = p.parse_args()

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect(args.arms, fallback=not args.no_fallback)
    if not rows:
        print("nothing to aggregate (run run_all.sh train first).")
        return

    summ = summarize(rows)

    with (METRIC_DIR / "runs_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "dataset", "task", "seed",
                                           *METRIC_KEYS, "stop_step", "source"])
        w.writeheader()
        w.writerows(rows)

    with (METRIC_DIR / "main_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "dataset", "task", "match", "n_seeds", "sources",
                    *[f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]])
        for (arm, ds, task), rec in sorted(summ.items()):
            vals = []
            for k in METRIC_KEYS:
                vals += [f"{rec[k][0]:.6f}", f"{rec[k][1]:.6f}"]
            w.writerow([arm, ds, task, ARM_MATCH[arm][task], rec["n_seeds"],
                        "|".join(rec["sources"]), *vals])

    (METRIC_DIR / "main_table.md").write_text(_md(rows, summ))
    try:
        _figure(summ, METRIC_DIR / "fig_main.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    print(f"total {len(rows)} runs.")
    for (arm, ds, task), rec in sorted(summ.items()):
        m, s = rec[PRIMARY_METRIC]
        print(f"  {arm:13s} {ds}/{task:15s} {PRIMARY_METRIC}={m:7.2f} ± {s:5.2f} "
              f"({rec['n_seeds']} seeds, {','.join(rec['sources'])})")
    print(f"\n-> {METRIC_DIR/'main_table.md'}")
    print(f"-> {METRIC_DIR/'main_table.csv'} / runs_long.csv / fig_main.png")


if __name__ == "__main__":
    main()
