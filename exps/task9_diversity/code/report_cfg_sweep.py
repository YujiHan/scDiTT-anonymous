from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

DIV_COLS = ["div_norm_self", "div_norm_real"]


def main() -> int:
    src = C.METRICS_DIR / "aggregated.csv"
    if not src.exists():
        raise FileNotFoundError(f"run aggregate.py first to produce {src}")
    df = pd.read_csv(src)
    df = df[df.method == "scDiTT"].copy()
    if df.empty:
        raise SystemExit("no scDiTT rows in aggregated.csv")

    for col in DIV_COLS:
        if col not in df.columns or df[col].isna().all():
            print(f"[skip] {col} unavailable")
            continue
        print(f"\n{'='*70}\n=== {col} as a function of w ===")
        for task in C.TASKS:
            sub = df[df.task == task]
            per = (sub.groupby(["dataset", "cfg_weight"], as_index=False)[col]
                      .mean())
            rows = []
            for ds, g in per.groupby("dataset"):
                g = g.sort_values("cfg_weight")
                lo, hi, mu = g[col].min(), g[col].max(), g[col].mean()
                rel = (hi - lo) / mu if mu else float("nan")
                best_w = g.loc[g[col].idxmax(), "cfg_weight"]
                rows.append((ds, len(g), rel * 100, best_w))
            if not rows:
                continue
            print(f"  {task}")
            print(f"    {'dataset':8s}{'n_w':>5s}{'rel_range %':>13s}{'argmax w':>10s}")
            for ds, n, rel, bw in sorted(rows):
                print(f"    {ds:8s}{n:>5d}{rel:>13.2f}{bw:>10g}")
            print(f"    -> largest relative range across the five datasets: {max(r[2] for r in rows):.2f}%"
                  f"   set of argmax w values: {sorted({r[3] for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
