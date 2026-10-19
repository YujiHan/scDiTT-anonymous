from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ORDER = ["scDiTT", "scIMF", "scNODE", "PI-SDE", "PRESCIENT", "scDiffEq", "MIOFlow"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=float, default=C.CFG_MAIN,
                    help="which guidance weight to use for scDiTT (keep only this operating point)")
    args = ap.parse_args()

    raw = pd.read_csv(C.METRICS_DIR / "aggregated.csv")
    raw = raw[raw.div_raw.notna() & raw.fid_raw.notna()].copy()
    raw = raw[(raw.method != "scDiTT") | np.isclose(raw.cfg_weight, args.cfg)]

    keys = ["dataset", "task", "stage"]
    g = raw.groupby(keys)
    out = raw[keys + ["method"]].copy()
    for col, tag in [("fid_raw", "fid"), ("div_raw", "div")]:
        out[f"{tag}_rank"] = g[col].transform(
            lambda s: s.rank(method="average").sub(1) / max(len(s) - 1, 1))

    dst = C.METRICS_DIR / "norm_compare.csv"
    out.to_csv(dst, index=False)

    dsl = out.groupby(["dataset", "task", "method"], as_index=False).mean(
        numeric_only=True)
    ov = dsl.groupby(["task", "method"]).mean(numeric_only=True)
    for task in C.TASKS:
        t = ov.loc[task]
        idx = [m for m in ORDER if m in t.index]
        print(f"\n=== {task} (rank; lower is better for fidelity, higher is better for diversity) ===")
        print(f"{'method':10s}{'fidelity':>12s}{'diversity':>12s}")
        for m in idx:
            print(f"{m:10s}{t.loc[m, 'fid_rank']:12.3f}{t.loc[m, 'div_rank']:12.3f}")
    print(f"\n[rank] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
