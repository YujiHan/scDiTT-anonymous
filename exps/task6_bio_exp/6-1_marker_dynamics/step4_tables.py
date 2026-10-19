"""6-1 Step 6 (tables) -- T1-T4 + significance tests.

Primary metric = **mean marker Pearson r on the informative subset (Fisher-z averaged)**.
KS is **not in the main table** (a frozen decision).

Outputs
----
``metrics/T1_main_{interpolation,extrapolation}.csv``  main table (with Mean / Mean(norm))
``metrics/T1_long.csv``                                 long form of the main table (for plotting)
``metrics/T2_rmse_direction.csv``                       all 10 combinations
``metrics/T3_shape_strata.csv``                         shape strata, per combination
``metrics/T4_per_marker_r.csv``                         r per marker x arm
``metrics/significance_scditt_vs_baseline.csv``         Wilcoxon + Cohen's d
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import data_io as io, stats  # noqa: E402
import core  # noqa: E402
import panels as P  # noqa: E402

OUT = HERE.parent / "results" / "6-1_marker_dynamics" / "metrics"
COMBO = lambda ds, tk: f"{ds}/{'i' if tk == 'interpolation' else 'e'}"  # noqa: E731


def per_seed_r(df: pd.DataFrame, col: str = "pearson_r") -> pd.DataFrame:
    return (df[df.informative]
            .groupby(["dataset", "task", "arm", "seed"])[col]
            .apply(core.fisher_z_mean).reset_index(name="r"))


def main() -> None:
    d = pd.read_csv(OUT / "per_marker.csv")
    arms = P.ARMS_10

    ps = per_seed_r(d)
    ps.to_csv(OUT / "T1_long.csv", index=False)
    agg = (ps.groupby(["dataset", "task", "arm"])["r"]
             .agg(["mean", "std", "count"]).reset_index())

    pos = agg[agg.arm == "pos_split_half"].set_index(["dataset", "task"])["mean"]
    agg["r_norm"] = agg.apply(
        lambda r: stats.normalized_by_positive_control(
            r["mean"], pos.get((r["dataset"], r["task"]), np.nan)), axis=1)
    agg["combo"] = agg.apply(lambda r: COMBO(r["dataset"], r["task"]), axis=1)
    agg.to_csv(OUT / "T1_agg.csv", index=False)

    for task in ("interpolation", "extrapolation"):
        combos = [(ds, tk) for ds, tk in P.USABLE_COMBOS if tk == task]
        sub = agg[(agg.task == task) &
                  agg.set_index(["dataset", "task"]).index.isin(combos)]
        rows = []
        for arm in arms:
            s = sub[sub.arm == arm].set_index("dataset")
            row = {"arm": arm}
            for ds, tk in combos:
                if ds in s.index:
                    row[COMBO(ds, tk)] = f"{s.loc[ds, 'mean']:.3f} ± {s.loc[ds, 'std']:.3f}"
                else:
                    row[COMBO(ds, tk)] = ""
            row["Mean"] = f"{s['mean'].mean():.3f}" if len(s) else ""
            row["Mean(norm)"] = f"{s['r_norm'].mean():.3f}" if len(s) else ""
            rows.append(row)
        t1 = pd.DataFrame(rows)
        t1.to_csv(OUT / f"T1_main_{task}.csv", index=False)
        print(f"\n=== T1 main table ({task}) -- marker mean Pearson r, mean ± std (5 seeds) ===")
        print(t1.to_string(index=False))

    t2rows = []
    for (ds, tk), g in d.groupby(["dataset", "task"]):
        n_tp = int(g.n_tp.iloc[0])
        use = g if n_tp == 1 else g[g.informative]
        for arm in arms:
            a = use[use.arm == arm]
            if a.empty:
                continue
            rm = a.groupby("seed")["rmse_norm_by_time"].mean()
            da = a.groupby("seed")["dir_agree"].mean()
            t2rows.append({
                "dataset": ds, "task": tk, "combo": COMBO(ds, tk), "n_tp": n_tp,
                "arm": arm, "n_marker": int(a.marker.nunique()),
                "a1_usable": (ds, tk) in P.USABLE_COMBOS,
                "rmse_norm_mean": rm.mean(), "rmse_norm_std": rm.std(ddof=1),
                "dir_agree_mean": (np.nan if n_tp < 2 else da.mean()),
                "dir_agree_std": (np.nan if n_tp < 2 else da.std(ddof=1)),
                "rmse_denominator": ("by_gene(single tp, normalised across genes)" if n_tp == 1
                                     else "by_time(std across tps)"),
            })
    pd.DataFrame(t2rows).to_csv(OUT / "T2_rmse_direction.csv", index=False)

    t3rows = []
    di = d[d.informative]
    for (ds, tk), g in di.groupby(["dataset", "task"]):
        if (ds, tk) not in P.USABLE_COMBOS:
            continue
        for arm in arms:
            row = {"dataset": ds, "task": tk, "combo": COMBO(ds, tk), "arm": arm}
            for sh in ("up", "down", "nonmono"):
                s = g[(g.arm == arm) & (g["shape"] == sh)]
                if s.empty:
                    row[f"r_{sh}"], row[f"n_{sh}"] = np.nan, 0
                    continue
                v = s.groupby("seed")["pearson_r"].apply(core.fisher_z_mean)
                row[f"r_{sh}"] = v.mean()
                row[f"r_{sh}_std"] = v.std(ddof=1)
                row[f"n_{sh}"] = int(s.marker.nunique())
            row["confound"] = ("HIGH: condition bifurcation (2i/serum ratio 52-70% per tp)"
                               if (ds, tk) == ("SC", "extrapolation") else
                               "HIGH: n_star 251-326, low cell count"
                               if (ds, tk) == ("GA", "extrapolation") else
                               "MED: tp7->tp9 protocol switch" if (ds, tk) == ("SC", "interpolation")
                               else "LOW")
            t3rows.append(row)
    pd.DataFrame(t3rows).to_csv(OUT / "T3_shape_strata.csv", index=False)

    d4 = d.copy()
    d4["shape"] = d4["shape"].fillna("non_informative")
    t4 = (d4.groupby(["dataset", "task", "marker", "lineage", "informative",
                      "shape", "arm"])["pearson_r"].mean().reset_index())
    t4 = t4.pivot_table(index=["dataset", "task", "marker", "lineage",
                               "informative", "shape"],
                        columns="arm", values="pearson_r").reset_index()
    t4.to_csv(OUT / "T4_per_marker_r.csv", index=False)

    from scipy.stats import wilcoxon
    srows = []
    for (ds, tk), g in ps.groupby(["dataset", "task"]):
        if (ds, tk) not in P.USABLE_COMBOS:
            continue
        piv = g.pivot(index="seed", columns="arm", values="r")
        if "scDiTT" not in piv:
            continue
        for m in io.ALL_METHODS:
            if m not in piv:
                continue
            a, b = piv["scDiTT"].values, piv[m].values
            try:
                p = float(wilcoxon(a, b).pvalue)
            except ValueError:
                p = np.nan
            srows.append({
                "dataset": ds, "task": tk, "combo": COMBO(ds, tk),
                "baseline": m, "r_scDiTT": a.mean(), "r_baseline": b.mean(),
                "delta_r": a.mean() - b.mean(),
                "cohens_d": stats.cohens_d(a, b),
                "wilcoxon_p": p, "n_pairs": len(a),
                "scDiTT_wins": bool(a.mean() > b.mean()),
            })
    sig = pd.DataFrame(srows)
    sig.to_csv(OUT / "significance_scditt_vs_baseline.csv", index=False)
    print("\n=== scDiTT vs baseline (7 A1 combinations x 6 baselines = %d comparisons) ===" % len(sig))
    print("comparisons won by scDiTT: %d / %d" % (int(sig.scDiTT_wins.sum()), len(sig)))
    print("\ndelta_r aggregated by combination (scDiTT - baseline, positive = ours is better):")
    print(sig.pivot_table(index="combo", columns="baseline",
                          values="delta_r").round(3).to_string())
    print("\nNote: with n=5 the smallest possible Wilcoxon p = 0.0625, so **p<0.05 is unreachable**"
          "(a power ceiling); effect size is therefore the primary readout.")


if __name__ == "__main__":
    main()
