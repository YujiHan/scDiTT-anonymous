from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import data_io as io  # noqa: E402
import core  # noqa: E402
import panels as P  # noqa: E402

OUT = HERE.parent / "results" / "6-1_marker_dynamics" / "metrics"


def main() -> None:
    P.self_check()
    OUT.mkdir(parents=True, exist_ok=True)

    gate_rows, cmp_rows, sc_rows = [], [], []
    for ds in io.ALL_DATASETS:
        a = io.load_real(ds)
        res = core.resolve_panel(ds, a)
        df = pd.DataFrame(res.rows)

        per_combo_inf = {}
        for d2, task in P.ALL_COMBOS:
            if d2 != ds:
                continue
            rs = core.real_side(ds, task, res.idx, res.names, adata=a)
            n_tp = len(rs.tps)
            for i, nm in enumerate(rs.names):
                gate_rows.append({
                    "dataset": ds, "task": task, "marker": nm,
                    "lineage": P.gene_lineage(ds).get(nm, ""),
                    "n_tp": n_tp, "held_out_tps": ",".join(map(str, rs.tps)),
                    **{f"mu_tp{t}": rs.mu[i, j] for j, t in enumerate(rs.tps)},
                    "se_bar": rs.se_bar[i], "range": rs.rng_span[i],
                    "gate_thr": max(P.GATE_ABS_FLOOR, P.GATE_K_SE * rs.se_bar[i]),
                    "informative": bool(rs.informative[i]),
                    "shape": rs.shape[i] if rs.informative[i] else "",
                })
            per_combo_inf[task] = {nm: bool(v) for nm, v in
                                   zip(rs.names, rs.informative)}

            n_inf = int(rs.informative.sum())
            sh = [s for s, ok in zip(rs.shape, rs.informative) if ok]
            got = (sh.count("up"), sh.count("down"), sh.count("nonmono"), sh.count("flat"))
            want_inf = P.FROZEN_INFORMATIVE.get((ds, task))
            want_sh = P.FROZEN_SHAPES.get((ds, task))
            cmp_rows.append({
                "dataset": ds, "task": task, "n_tp": n_tp,
                "panel_n": len(rs.names),
                "informative": n_inf, "informative_frozen": want_inf,
                "informative_match": (want_inf is None or n_inf == want_inf),
                "up": got[0], "down": got[1], "nonmono": got[2], "flat": got[3],
                "shapes_frozen": str(want_sh),
                "shapes_match": (want_sh is None or got == want_sh),
                "n_star": ",".join(str(rs.n_star[t]) for t in rs.tps),
                "n_real": ",".join(str(rs.n_real[t]) for t in rs.tps),
            })

        def _status(row):
            if row["status"] not in ("hit", "alias_hit"):
                return row["status"]
            nm = row["resolved_name"]
            flags = [per_combo_inf[t].get(nm, False) for t in per_combo_inf]
            return row["status"] if any(flags) else "non_informative"

        df["status_final"] = df.apply(_status, axis=1)
        for task, m in per_combo_inf.items():
            df[f"informative_{task}"] = df["resolved_name"].map(m)
        df.to_csv(OUT / f"marker_resolved_{ds}.csv", index=False)
        print(f"[{ds}] hit={int((df.status=='hit').sum())} "
              f"alias_hit={int((df.status=='alias_hit').sum())} "
              f"miss={int((df.status=='miss').sum())} "
              f"unexpected_present={int((df.status=='miss_UNEXPECTED_PRESENT').sum())} "
              f"panel_miss={int((df.status=='panel_miss').sum())}")

        if ds == "SC":
            tpv = np.asarray(a.obs[io.TIME_KEY].values).astype(int)
            cond = np.asarray(a.obs["condition"].astype(str).values)
            for task in ("interpolation", "extrapolation"):
                for t in io.held_out_tps("SC", task):
                    sel = tpv == t
                    row = {"task": task, "tp": int(t), "n": int(sel.sum())}
                    for c in np.unique(cond):
                        row[c] = int((cond[sel] == c).sum())
                    sc_rows.append(row)

    pd.DataFrame(gate_rows).to_csv(OUT / "real_side_gate.csv", index=False)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT / "gate_vs_frozen.csv", index=False)
    pd.DataFrame(sc_rows).to_csv(OUT / "sc_condition_composition.csv", index=False)

    print("\n=== recomputation vs pre-registered frozen values ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(cmp_df[["dataset", "task", "n_tp", "panel_n", "informative",
                      "informative_frozen", "informative_match",
                      "up", "down", "nonmono", "flat", "shapes_frozen",
                      "shapes_match"]].to_string(index=False))
    bad = cmp_df[~(cmp_df.informative_match & cmp_df.shapes_match)]
    print(f"\nnumber of combinations disagreeing with the frozen values: {len(bad)}")


if __name__ == "__main__":
    main()
