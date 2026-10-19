from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import plot_lib as PL  # noqa: E402

XLABEL = r"rank-normalised diversity  ($\uparrow$)"
YLABEL = r"rank-normalised fidelity  ($\downarrow$)"
ORDER = ["scDiTT", "scIMF", "scNODE", "PI-SDE", "PRESCIENT", "scDiffEq", "MIOFlow"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=float, default=C.CFG_MAIN)
    ap.add_argument("--figs_dir", default=None)
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="scale fonts and markers up globally (the canvas is unchanged, so text looks larger)")
    args = ap.parse_args()

    PL.set_scale(args.font_scale)

    src = C.METRICS_DIR / "norm_compare.csv"
    if not src.exists():
        raise FileNotFoundError(f"run rank_normalize.py first to produce {src}")
    figs = Path(args.figs_dir) if args.figs_dir else C.EXP_ROOT / "figs_norm"
    figs.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(src)
    dsl = raw.groupby(["dataset", "task", "method"], as_index=False).mean(
        numeric_only=True)
    dsl["cfg_weight"] = np.where(dsl.method == "scDiTT", args.cfg, np.nan)
    dsl["train_seed"] = 42

    d = dsl.rename(columns={"fid_rank": "fid_norm"}).copy()
    out = figs / "fig4_rank.pdf"
    PL.fig_version_a(d, "div_rank", out, cfg=args.cfg,
                     xlabel_text=XLABEL, ylabel_text=YLABEL, hline_one=False)
    print(f"[fig4] {out}")

    ov = dsl.groupby(["task", "method"]).mean(numeric_only=True)
    for task in C.TASKS:
        t = ov.loc[task]
        idx = [m for m in ORDER if m in t.index]
        print(f"\n=== {task} ===")
        print(f"{'method':10s}{'fidelity':>12s}{'diversity':>12s}")
        for m in idx:
            print(f"{m:10s}{t.loc[m, 'fid_rank']:12.3f}{t.loc[m, 'div_rank']:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
