#!/usr/bin/env python
"""Chain A aggregation -- main table / delta table / figure.

Reads runs/<DS>_<task>_seed<seed>/<arm>/metrics.json (the unified 5 metrics written by the official evaluate_model),
and produces:
  metrics/raw.csv          one row per run (including stop_step / parameter count)
  metrics/main_table.csv   (dataset, task, arm) x 5 metrics, mean ± std across seeds
  metrics/delta_table.csv  delta and delta% of each arm against full (paired per seed, then averaged)
  metrics/fig_main.png     w2_avg bar chart (4 combinations x 4 arms, error bars = std across seeds)

Metric direction: lower is better for w2_avg / mmd_avg / pg_w1_mean; higher is better for pearson_mean / sampled_pearson.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (ARMS, ARM_DESC, DATASETS, METRIC_DIR, REPO_ROOT,  # noqa: E402
                    SEEDS, TASKS, run_dir)
from src.evaluation.scoring import METRIC_KEYS  # noqa: E402

LOWER_BETTER = {"w2_avg": True, "mmd_avg": True, "pg_w1_mean": True,
                "pearson_mean": False, "sampled_pearson": False}

REF_DIR = REPO_ROOT / "results" / "reference_csv"


def load_reference() -> dict[tuple[str, str, int], dict]:
    import csv as _csv
    ref: dict[tuple[str, str, int], dict] = {}
    for p in sorted(REF_DIR.glob("progress_*.csv")):
        with open(p, encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                if row.get("ex_step"):
                    continue
                ds, task = row["dataset"], row["task"]
                if ds not in DATASETS or task not in TASKS:
                    continue
                try:
                    seed = int(row["seed"])
                except (KeyError, ValueError):
                    continue
                if seed not in SEEDS:
                    continue
                ref[(ds, task, seed)] = {k: float(row[k]) for k in METRIC_KEYS
                                         if row.get(k)}
    return ref


def load_runs() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for task in TASKS:
            for seed in SEEDS:
                for arm in ARMS:
                    d = run_dir(ds, task, seed, arm)
                    mp = d / "metrics.json"
                    if not mp.exists():
                        continue
                    m = json.loads(mp.read_text())
                    row = {"dataset": ds, "task": task, "seed": seed, "arm": arm}
                    row.update({k: float(m["avg"][k]) for k in METRIC_KEYS
                                if k in m.get("avg", {})})
                    sp = d / "stop_step.json"
                    if sp.exists():
                        row["stop_step"] = json.loads(sp.read_text()).get("stop_step")
                    ap = d / "arm.json"
                    if ap.exists():
                        a = json.loads(ap.read_text())
                        row["params_M"] = round(a.get("params_total", 0) / 1e6, 2)
                        row["wall_min"] = round(a.get("wall_sec", 0) / 60, 1)
                        if a.get("best_step") is not None:
                            row["best_step"] = a["best_step"]
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    from math import comb
    d = [x for x in deltas if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for x in d if x > 0)
    tail = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return k, n, float(min(1.0, 2 * tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(METRIC_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs()
    if not rows:
        print("no metrics.json yet -- run training first (bash run_all.sh train).")
        return
    write_csv(out_dir / "raw.csv", rows,
              ["dataset", "task", "seed", "arm", *METRIC_KEYS,
               "best_step", "stop_step", "params_M", "wall_min"])
    print(f"total {len(rows)}/{len(DATASETS)*len(TASKS)*len(SEEDS)*len(ARMS)} runs completed\n")

    main_rows = []
    for ds in DATASETS:
        for task in TASKS:
            for arm in ARMS:
                rs = [r for r in rows if (r["dataset"], r["task"], r["arm"])
                      == (ds, task, arm)]
                if not rs:
                    continue
                row = {"dataset": ds, "task": task, "arm": arm, "n_seeds": len(rs)}
                for k in METRIC_KEYS:
                    v = [r[k] for r in rs if k in r]
                    if v:
                        row[f"{k}_mean"] = round(float(np.mean(v)), 4)
                        row[f"{k}_std"] = round(float(np.std(v, ddof=1)) if len(v) > 1
                                                else 0.0, 4)
                main_rows.append(row)
    cols = ["dataset", "task", "arm", "n_seeds"] + \
           [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]
    write_csv(out_dir / "main_table.csv", main_rows, cols)

    print("=== main table (mean ± std over seeds) ===")
    hdr = f"{'ds':4s} {'task':14s} {'arm':8s} {'w2_avg':>16s} {'mmd_avg':>16s} {'pearson':>16s}"
    print(hdr)
    for r in main_rows:
        def f(k):
            return (f"{r.get(k+'_mean', float('nan')):8.3f}±{r.get(k+'_std', 0):<6.3f}")
        print(f"{r['dataset']:4s} {r['task']:14s} {r['arm']:8s} "
              f"{f('w2_avg'):>16s} {f('mmd_avg'):>16s} {f('pearson_mean'):>16s}")

    delta_rows = []
    for ds in DATASETS:
        for task in TASKS:
            for arm in ARMS:
                if arm == "full":
                    continue
                row = {"dataset": ds, "task": task, "arm": arm}
                for k in METRIC_KEYS:
                    ds_ = []
                    for seed in SEEDS:
                        a = next((r[k] for r in rows if (r["dataset"], r["task"],
                                  r["seed"], r["arm"]) == (ds, task, seed, arm)
                                  and k in r), None)
                        b = next((r[k] for r in rows if (r["dataset"], r["task"],
                                  r["seed"], r["arm"]) == (ds, task, seed, "full")
                                  and k in r), None)
                        if a is not None and b is not None:
                            ds_.append((a - b) if LOWER_BETTER[k] else (b - a))
                    if ds_:
                        row[f"{k}_delta"] = round(float(np.mean(ds_)), 4)
                        row[f"{k}_delta_std"] = round(
                            float(np.std(ds_, ddof=1)) if len(ds_) > 1 else 0.0, 4)
                        base = [r[k] for r in rows if (r["dataset"], r["task"],
                                r["arm"]) == (ds, task, "full") and k in r]
                        if base and abs(np.mean(base)) > 1e-9:
                            row[f"{k}_delta_pct"] = round(
                                100 * float(np.mean(ds_)) / abs(float(np.mean(base))), 2)
                delta_rows.append(row)
    cols = ["dataset", "task", "arm"] + \
           [f"{k}_{s}" for k in METRIC_KEYS for s in ("delta", "delta_std", "delta_pct")]
    write_csv(out_dir / "delta_table.csv", delta_rows, cols)

    print("\n=== Δ vs full (positive = worse than full; paired per seed, then averaged) ===")
    print(f"{'ds':4s} {'task':14s} {'arm':8s} {'Δw2':>10s} {'Δw2%':>8s} {'Δmmd':>9s} {'Δpearson':>10s}")
    for r in delta_rows:
        print(f"{r['dataset']:4s} {r['task']:14s} {r['arm']:8s} "
              f"{r.get('w2_avg_delta', float('nan')):10.3f} "
              f"{r.get('w2_avg_delta_pct', float('nan')):7.1f}% "
              f"{r.get('mmd_avg_delta', float('nan')):9.4f} "
              f"{r.get('pearson_mean_delta', float('nan')):10.4f}")

    n_pairs = len(DATASETS) * len(TASKS) * len(SEEDS)
    combo = " × ".join(filter(None, [
        f"{len(DATASETS)} ds" if len(DATASETS) > 1 else DATASETS[0],
        f"{len(TASKS)} task", f"{len(SEEDS)} seed"]))
    print(f"\n=== summary test: each arm vs full, {n_pairs} pairs ({combo}) on w2_avg ===")
    for arm in ARMS:
        if arm == "full":
            continue
        deltas = []
        for ds in DATASETS:
            for task in TASKS:
                for seed in SEEDS:
                    a = next((r["w2_avg"] for r in rows if (r["dataset"], r["task"],
                              r["seed"], r["arm"]) == (ds, task, seed, arm)), None)
                    b = next((r["w2_avg"] for r in rows if (r["dataset"], r["task"],
                              r["seed"], r["arm"]) == (ds, task, seed, "full")), None)
                    if a is not None and b is not None:
                        deltas.append(a - b)
        if not deltas:
            continue
        k, n, p = sign_test(deltas)
        print(f"  {arm:8s}: Δw2 = {np.mean(deltas):+7.3f} ± {np.std(deltas):5.3f} "
              f"| worse in {k}/{n} pairs | sign test p={p:.4f}  ({ARM_DESC[arm]})")
    print(f"  note: only {n_pairs} pairs; the sign test is weak here and is indicative only.")

    imported = [r for r in rows if r["arm"] == "full"
                and (run_dir(r["dataset"], r["task"], r["seed"], "full")
                     / "arm.json").exists()
                and "imported_from" in json.loads(
                    (run_dir(r["dataset"], r["task"], r["seed"], "full")
                     / "arm.json").read_text())]
    if imported:
        print(f"\n=== provenance of the full arm: {len(imported)}/{len(DATASETS)*len(TASKS)*len(SEEDS)} "
              f" runs imported from the main experiment (not retrained here) ===")
        print(f"{'ds':4s} {'task':14s} {'seed':>5s} {'best_step':>10s} "
              f"{'stop_step':>10s} {'w2_avg':>8s}")
        early = []
        for r in sorted(imported, key=lambda r: (r["dataset"], r["task"], r["seed"])):
            bs = r.get("best_step", -1)
            flag = "  ⚠️ barely trained" if 0 <= bs <= 2000 else ""
            if 0 <= bs <= 2000:
                early.append(r)
            print(f"{r['dataset']:4s} {r['task']:14s} {r['seed']:5d} {str(bs):>10s} "
                  f"{str(r.get('stop_step', '?')):>10s} "
                  f"{r.get('w2_avg', float('nan')):8.3f}{flag}")
        if early:
            combos = sorted({(r["dataset"], r["task"]) for r in early})
            print(f"  note: {len(early)} full runs have best.pt at <=2000 steps "
                  f"({', '.join(f'{d}/{t}' for d, t in combos)}): the probe loss on the "
                  f"training distribution reaches its minimum very early on these "
                  f"combinations and does not improve over the remaining steps. Every arm "
                  f"is selected by that same probe -- held-out data enters neither training "
                  f"nor model selection -- and every arm runs the full step budget, so the "
                  f"comparison is like for like; but on these combinations it compares "
                  f"checkpoints taken early in training, which is worth stating alongside "
                  f"the numbers.")

    ref = load_reference()
    if ref:
        rep_rows = []
        print("\n=== consistency with the main-experiment reference CSV (Δ = 0 if the import is correct) ===")
        print(f"{'ds':4s} {'task':14s} {'seed':>5s} {'ours':>8s} {'ref':>8s} {'Δ':>8s}")
        for ds in DATASETS:
            for task in TASKS:
                for seed in SEEDS:
                    ours = next((r["w2_avg"] for r in rows if (r["dataset"], r["task"],
                                 r["seed"], r["arm"]) == (ds, task, seed, "full")
                                 and "w2_avg" in r), None)
                    rf = ref.get((ds, task, seed), {}).get("w2_avg")
                    if ours is None or rf is None:
                        continue
                    rep_rows.append({"dataset": ds, "task": task, "seed": seed,
                                     "ours_w2": round(ours, 4), "ref_w2": round(rf, 4),
                                     "delta": round(ours - rf, 4),
                                     "delta_pct": round(100 * (ours - rf) / abs(rf), 2)})
                    print(f"{ds:4s} {task:14s} {seed:5d} {ours:8.3f} {rf:8.3f} "
                          f"{ours-rf:+8.3f}")
        if rep_rows:
            write_csv(out_dir / "reproduction_check.csv", rep_rows,
                      ["dataset", "task", "seed", "ours_w2", "ref_w2", "delta", "delta_pct"])
            mx = max(abs(r["delta"]) for r in rep_rows)
            print(f"  max |Δ| = {mx:.4f} "
                  + ("-> identical to the main experiment (the full arm is a direct import)." if mx < 1e-3 else
                     "⚠️ non-zero! The full arm should be an import; check import_full_runs.py."))

    fig, axes = plt.subplots(1, len(DATASETS) * len(TASKS),
                             figsize=(4.0 * len(DATASETS) * len(TASKS), 4))
    axes = np.atleast_1d(axes)
    i = 0
    colors = ["#2c7fb8", "#7fcdbb", "#fdae61", "#d7191c"]
    for ds in DATASETS:
        for task in TASKS:
            ax = axes[i]; i += 1
            means, stds, labs = [], [], []
            for j, arm in enumerate(ARMS):
                r = next((r for r in main_rows if (r["dataset"], r["task"], r["arm"])
                          == (ds, task, arm)), None)
                if r is None or "w2_avg_mean" not in r:
                    continue
                means.append(r["w2_avg_mean"]); stds.append(r["w2_avg_std"]); labs.append(arm)
            ax.bar(labs, means, yerr=stds, capsize=3, color=colors[:len(labs)])
            ax.set_title(f"{ds} / {task}", fontsize=10)
            ax.set_ylabel("w2_avg (lower is better)")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle(f"Attention ablation: held-out W2 (error bars = std over {len(SEEDS)} seeds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_main.png", dpi=150)
    plt.close(fig)

    print(f"\nWritten: {out_dir}/raw.csv, main_table.csv, delta_table.csv, fig_main.png")


if __name__ == "__main__":
    main()
