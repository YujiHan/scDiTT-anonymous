#!/usr/bin/env python
"""Aggregate the AR arms into the rows they occupy in the horizon tables.

Reads runs/<DS>_ex<l>_seed<seed>/<arm>/metrics_lasttp.json -- W2 at the furthest
withheld stage, which is what the horizon tables report -- and gives mean +/-
sample std (ddof=1) over seeds, one row per arm and one column per horizon.

Writes metrics/ar_horizons.csv and metrics/ar_horizons.md. Read-only on runs/.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def load_w2(ds: str, horizon: int, seed: int, arm: str) -> float | None:
    p = C.run_dir(ds, horizon, seed, arm) / C.METRICS_FILE
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if rec.get("status") not in ("ok", None):
        return None
    avg = rec.get("avg")
    if not isinstance(avg, dict) or C.REPORT_KEY not in avg:
        return None
    return float(avg[C.REPORT_KEY])


def main() -> int:
    C.METRIC_DIR.mkdir(parents=True, exist_ok=True)

    long_rows, cells, missing = [], {}, []
    for ds, h, seed, arm in C.all_jobs():
        v = load_w2(ds, h, seed, arm)
        if v is None:
            missing.append((ds, h, seed, arm))
            continue
        long_rows.append({"dataset": ds, "horizon": h, "seed": seed,
                          "arm": arm, C.REPORT_KEY: v})
        cells.setdefault((ds, arm, h), []).append(v)

    if not long_rows:
        print(f"no {C.METRICS_FILE} found under {C.RUNS_DIR}\nRun run_all.sh first.")
        return 1

    with (C.METRIC_DIR / "runs_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "horizon", "seed", "arm", C.REPORT_KEY])
        w.writeheader()
        for r in sorted(long_rows, key=lambda r: (r["dataset"], r["arm"], r["horizon"], r["seed"])):
            w.writerow(r)

    with (C.METRIC_DIR / "ar_horizons.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm"]
                   + [f"ex{h}_mean" for h in C.HORIZONS]
                   + [f"ex{h}_std" for h in C.HORIZONS])
        for ds in C.DATASETS:
            for arm in C.ARMS:
                mus, sds = [], []
                for h in C.HORIZONS:
                    v = cells.get((ds, arm, h), [])
                    mus.append(f"{st.mean(v):.4f}" if v else "")
                    sds.append(f"{st.stdev(v):.4f}" if len(v) > 1 else "")
                w.writerow([ds, C.ARM_LABEL[arm]] + mus + sds)

    md = ["# Autoregressive variants -- W2 at the furthest withheld stage",
          "",
          f"Mean ± sample std over seeds {C.SEEDS}. These are the AR rows of the",
          "extrapolation-horizon tables; the scDiTT and published-baseline rows come",
          "from `exps/task2_extra_steps/` and the baseline reruns respectively.",
          ""]
    for ds in C.DATASETS:
        md += [f"## {ds}", "",
               "| Method | " + " | ".join(f"ℓ={h}" for h in C.HORIZONS) + " |",
               "|---|" + "---|" * len(C.HORIZONS)]
        for arm in C.ARMS:
            row = []
            for h in C.HORIZONS:
                v = cells.get((ds, arm, h), [])
                if not v:
                    row.append("--")
                elif len(v) == 1:
                    row.append(f"{v[0]:.2f}")
                else:
                    row.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
            md.append(f"| {C.ARM_LABEL[arm]} | " + " | ".join(row) + " |")
        md.append("")

    n_exp = len(C.all_jobs())
    if missing:
        md += [f"> **Incomplete:** {len(missing)} of {n_exp} runs are missing; "
               "cells above average only the seeds that finished.", ""]
    (C.METRIC_DIR / "ar_horizons.md").write_text("\n".join(md))

    print("\n".join(md))
    print(f"[runs] {len(long_rows)}/{n_exp} present")
    if missing:
        head = ", ".join(f"{d}/ex{h}/seed{s}/{a}" for d, h, s, a in missing[:8])
        print(f"[warn] missing: {head}{' ...' if len(missing) > 8 else ''}")
    print(f"-> {C.METRIC_DIR / 'ar_horizons.md'}")
    print(f"-> {C.METRIC_DIR / 'ar_horizons.csv'} / runs_long.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
