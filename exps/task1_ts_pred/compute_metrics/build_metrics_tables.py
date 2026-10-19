#!/usr/bin/env python3
"""Consolidate task1 time-series prediction metrics into per-dataset CSVs.

READ-ONLY on all source result directories. Writes only to metrics_results/.

Metrics reported: w2_avg  and  sampled_pearson  (from the `avg` block).
Aggregation: mean +/- sample std (ddof=1) over available random seeds.

Layout of each <DATASET>.csv:
  - one header row (experiment columns)
  - a "[w2_avg]" section, one row per method
  - blank rows
  - a "[sampled_pearson]" section, one row per method

Experiment columns (16):
  interpolation, extrapolation,
  allTP_ex0..allTP_ex6   (extrapolation_steps, metrics_alltp.json),
  lastTP_ex0..lastTP_ex6 (extrapolation_steps, metrics_lasttp.json)

Baseline source policy: A/B/C dir is primary; baseline_standard fills
missing seeds (dedupe by seed number, A/B/C preferred). Variant = "best"
(metrics.json / metrics_alltp.json / metrics_lasttp.json, never *_final).
"""
import csv
import json
import os
import statistics

# ---------------------------------------------------------------- config
DATASETS = ["DR", "GA", "MA", "SC", "ZB"]
SEEDS = [42, 43, 44, 45, 46]
EX_LEVELS = list(range(7))  # ex0..ex6

SCDITT_ROOT = ""

# method -> ordered list of candidate roots (each contains <DS>/<task>/...)
METHOD_ROOTS = {
    "scDiTT": [SCDITT_ROOT],
    "MIOFlow": [""],
    "PI-SDE": [""],
    "PRESCIENT": [""],
    "scDiffEq": [""],
    "scIMF": [""],
    "scNODE": [""],
}
METHOD_ORDER = ["scDiTT", "MIOFlow", "PI-SDE", "PRESCIENT", "scDiffEq", "scIMF", "scNODE"]

METRICS = ["w2_avg", "sampled_pearson"]
DECIMALS = {"w2_avg": 2, "sampled_pearson": 4}

OUT_DIR = ""

# columns: (col_id, display, relpath_template, filename)
COLUMNS = [
    ("interpolation", "interpolation", "{ds}/interpolation/seed{s}", "metrics.json"),
    ("extrapolation", "extrapolation", "{ds}/extrapolation/seed{s}", "metrics.json"),
]
for k in EX_LEVELS:
    COLUMNS.append((f"alltp_ex{k}", f"allTP_ex{k}",
                    f"{{ds}}/extrapolation_steps/ex{k}/seed{{s}}", "metrics_alltp.json"))
for k in EX_LEVELS:
    COLUMNS.append((f"lasttp_ex{k}", f"lastTP_ex{k}",
                    f"{{ds}}/extrapolation_steps/ex{k}/seed{{s}}", "metrics_lasttp.json"))


# ---------------------------------------------------------------- helpers
def load_avg(path):
    """Return the avg dict if the metrics file is valid & ok, else None."""
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if d.get("status") not in ("ok", None):
        return None
    avg = d.get("avg")
    if not isinstance(avg, dict) or "w2_avg" not in avg:
        return None
    return avg


def collect(method, ds, relpath_tmpl, filename):
    """Return {seed: avg_dict} deduped across candidate roots (first root wins)."""
    out = {}
    for root in METHOD_ROOTS[method]:
        for s in SEEDS:
            if s in out:
                continue
            p = os.path.join(root, relpath_tmpl.format(ds=ds, s=s), filename)
            avg = load_avg(p)
            if avg is not None:
                out[s] = avg
    return out


def fmt(values, metric):
    """Format a list of floats as 'mean ± std' at the metric's precision."""
    dec = DECIMALS[metric]
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) >= 2:
        std = statistics.stdev(values)
        return f"{mean:.{dec}f} ± {std:.{dec}f}"
    return f"{mean:.{dec}f}"  # n == 1: no std


# ---------------------------------------------------------------- build
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coverage_rows = []  # dataset, method, column, n_seeds, seeds

    for ds in DATASETS:
        # cache collected avg-dicts per (method, col)
        collected = {}
        for method in METHOD_ORDER:
            for col_id, _disp, tmpl, fn in COLUMNS:
                avgs = collect(method, ds, tmpl, fn)
                collected[(method, col_id)] = avgs
                coverage_rows.append([
                    ds, method, col_id, len(avgs),
                    ";".join(str(s) for s in sorted(avgs)),
                ])

        out_path = os.path.join(OUT_DIR, f"{ds}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["method / experiment"] + [disp for _id, disp, _t, _fn in COLUMNS]
            w.writerow(header)
            for metric in METRICS:
                w.writerow([f"[{metric}]"] + ["" for _ in COLUMNS])
                for method in METHOD_ORDER:
                    row = [method]
                    for col_id, _disp, _t, _fn in COLUMNS:
                        avgs = collected[(method, col_id)]
                        vals = [a[metric] for a in avgs.values() if metric in a]
                        row.append(fmt(vals, metric))
                    w.writerow(row)
                w.writerow([])  # blank separator
                w.writerow([])
        print(f"wrote {out_path}")

    # coverage report
    cov_path = os.path.join(OUT_DIR, "_seed_coverage.csv")
    with open(cov_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "column", "n_seeds", "seeds"])
        w.writerows(coverage_rows)
    print(f"wrote {cov_path}")


if __name__ == "__main__":
    main()
