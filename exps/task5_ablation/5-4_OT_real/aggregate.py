#!/usr/bin/env python
"""Aggregate the 5-4 arms into Table S3.

Reads runs/<DS>_<task>_seed<seed>/<arm>/metrics.json -- the unified 5 metrics
written by the standard evaluator -- and reports mean +/- sample std (ddof=1)
over seeds, one row per arm, for the two columns the table prints.

Writes metrics/main_table.csv, metrics/main_table.md and metrics/runs_long.csv.
Read-only on runs/.
"""
from __future__ import annotations

import csv
import json
import statistics as st

import common as C  # noqa: E402  (inserts REPO_ROOT into sys.path)


def load_avg(ds: str, task: str, seed: int, arm: str) -> dict | None:
    p = C.run_dir(ds, task, seed, arm) / "metrics.json"
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if rec.get("status") not in ("ok", None):
        return None
    avg = rec.get("avg")
    return avg if isinstance(avg, dict) and "w2_avg" in avg else None


def collect() -> list[dict]:
    rows = []
    for ds, task, seed, arm in C.all_jobs():
        avg = load_avg(ds, task, seed, arm)
        if avg is None:
            continue
        rows.append({"dataset": ds, "task": task, "seed": seed, "arm": arm,
                     **{k: float(avg[k]) for k in C.REPORT_KEYS if k in avg}})
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(dataset, arm) -> {metric: (mean, std, n)}."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["dataset"], r["arm"]), []).append(r)
    out: dict[tuple[str, str], dict] = {}
    for key, rs in buckets.items():
        rec = {"n": len(rs)}
        for m in C.REPORT_KEYS:
            vals = [r[m] for r in rs if m in r]
            if not vals:
                continue
            rec[m] = (st.mean(vals),
                      st.stdev(vals) if len(vals) > 1 else 0.0,
                      len(vals))
        out[key] = rec
    return out


def _fmt(rec: dict, m: str) -> str:
    if m not in rec:
        return "--"
    mu, sd, _ = rec[m]
    dec = 2 if m == "w2_avg" else 4
    return f"{mu:.{dec}f} ± {sd:.{dec}f}"


def main() -> int:
    C.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        print("no metrics.json found under "
              f"{C.RUNS_DIR}\nRun run_all.sh first.")
        return 1

    summ = summarize(rows)
    expected = len(C.SEEDS)

    with (C.METRIC_DIR / "runs_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "task", "seed", "arm"] + C.REPORT_KEYS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["dataset"], r["arm"], r["seed"])):
            w.writerow(r)

    header = ["arm"] + [f"{ds}_{m}" for ds in C.DATASETS for m in C.REPORT_KEYS]
    with (C.METRIC_DIR / "main_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for arm in C.ARMS:
            row = [C.ARM_LABEL[arm]]
            for ds in C.DATASETS:
                rec = summ.get((ds, arm), {})
                for m in C.REPORT_KEYS:
                    row.append(f"{rec[m][0]:.6f}" if m in rec else "")
            w.writerow(row)

    md = ["# Table S3 -- trajectory construction on real data",
          "",
          f"Task: {C.TASKS[0]}. Mean ± sample std over seeds {C.SEEDS}.",
          "`entropic OT (ours)` is scDiTT itself and reproduces its entries in the",
          "extrapolation table of the main paper.",
          "",
          "| Trajectory construction | " +
          " | ".join(f"{ds} W2 ↓ | {ds} Pearson ↑" for ds in C.DATASETS) + " |",
          "|---|" + "---|" * (2 * len(C.DATASETS))]
    incomplete = []
    for arm in C.ARMS:
        cells = []
        for ds in C.DATASETS:
            rec = summ.get((ds, arm), {})
            n = rec.get("n", 0)
            if n != expected:
                incomplete.append(f"{ds}/{arm} ({n}/{expected} seeds)")
            cells += [_fmt(rec, m) for m in C.REPORT_KEYS]
        md.append(f"| {C.ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    md.append("")
    if incomplete:
        md.append("> **Incomplete:** " + ", ".join(incomplete) +
                  ". The means above are taken over the seeds that finished, "
                  "so they are not comparable with the complete rows.")
        md.append("")
    (C.METRIC_DIR / "main_table.md").write_text("\n".join(md))

    print("\n".join(md))
    if incomplete:
        print(f"\n[warn] incomplete cells: {', '.join(incomplete)}")
    print(f"\n-> {C.METRIC_DIR / 'main_table.md'}")
    print(f"-> {C.METRIC_DIR / 'main_table.csv'} / runs_long.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
