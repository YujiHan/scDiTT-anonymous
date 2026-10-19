"""6-1 Steps 3-5 -- control gate + full combination sweep (10 combinations x 5 seeds x 10 arms).

Following the pre-registered analysis plan:
* Step 3 control gate (``neg_shuffle`` |r| < 0.2, ``pos_split_half`` r >= 0.95);
* Step 5 sweep: parallel over (DS, task, seed).

Outputs (``results/6-1_marker_dynamics/``)
------------------------------------
``metrics/per_marker.csv``      r / rho / RMSE / direction agreement per (combination, seed, arm, marker)
``metrics/ks_diagnostic.csv``   KS per (combination, seed, arm, marker, tp) -- **diagnostic only, not in the main table**
``metrics/provenance.json``     resolution of the two baseline roots (H3)
``metrics/zero_structure.csv``  zero-structure profile per arm (``common.artifacts``)
``figdata/curves_{DS}_{task}.npz`` mean curves per arm (for plotting)
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import artifacts, data_io as io  # noqa: E402
import core  # noqa: E402
import panels as P  # noqa: E402

ROOT = HERE.parent / "results" / "6-1_marker_dynamics"
OUT, FIGDATA = ROOT / "metrics", ROOT / "figdata"
SEEDS = io.ALL_SEEDS


def _ks_rows(arm_cells, ref_cells, tps, names, meta) -> list:
    from scipy.stats import ks_2samp
    rows = []
    for t in tps:
        R = ref_cells[t]
        G = arm_cells[t]
        for i, nm in enumerate(names):
            r_, g_ = R[:, i], G[:, i]
            raw = ks_2samp(r_, g_)
            rp, gp = r_[r_ > 0], g_[g_ > 0]
            if rp.size >= 10 and gp.size >= 10:
                pos = ks_2samp(rp, gp)
                ks_pos, p_pos = float(pos.statistic), float(pos.pvalue)
            else:
                ks_pos, p_pos = np.nan, np.nan
            rows.append({**meta, "marker": nm, "tp": int(t),
                         "ks_raw": float(raw.statistic), "p_raw": float(raw.pvalue),
                         "ks_pos": ks_pos, "p_pos": p_pos,
                         "n_pos_real": int(rp.size), "n_pos_gen": int(gp.size)})
    return rows


def run_combo(ds: str, task: str) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a = io.load_real(ds)
    res = core.resolve_panel(ds, a)
    rs = core.real_side(ds, task, res.idx, res.names, adata=a)
    tps, names = rs.tps, res.names

    real_mk = core.real_marker_by_tp(ds, task, res.idx, a)
    train_pool = core.train_marker_pool(ds, task, res.idx, a)

    per_marker, ks_rows, prov, zrows = [], [], [], []
    curves = {}
    for seed in SEEDS:
        arm_cells, ref_cells, nstar = core.build_arm_cells(
            ds, task, seed, res.idx, a, P.ARMS_10,
            real_mk=real_mk, train_pool=train_pool)
        ref_mu = core.arm_curves(ref_cells, tps)
        curves[f"ref_seed{seed}"] = ref_mu
        full_mu = rs.mu

        for arm in P.ARMS_10:
            gen_mu = core.arm_curves(arm_cells[arm], tps)
            curves[f"{arm}_seed{seed}"] = gen_mu
            m = core.curve_metrics(gen_mu, ref_mu, rs.se_bar)
            m_full = core.curve_metrics(gen_mu, full_mu, rs.se_bar)
            for i, nm in enumerate(names):
                per_marker.append({
                    "dataset": ds, "task": task, "seed": seed, "arm": arm,
                    "marker": nm, "lineage": P.gene_lineage(ds).get(nm, ""),
                    "informative": bool(rs.informative[i]),
                    "shape": rs.shape[i] if rs.informative[i] else "",
                    "n_tp": len(tps),
                    "pearson_r": m["pearson_r"][i],
                    "spearman_rho": m["spearman_rho"][i],
                    "rmse_norm_by_time": m["rmse_norm_by_time"][i],
                    "rmse_raw": m["rmse_raw"][i],
                    "dir_agree": m["dir_agree"][i],
                    "pearson_r_fullref": m_full["pearson_r"][i],
                })
            if seed == SEEDS[0]:
                ks_rows += _ks_rows(arm_cells[arm], ref_cells, tps, names,
                                    {"dataset": ds, "task": task, "seed": seed,
                                     "arm": arm})
                allc = np.concatenate([arm_cells[arm][t] for t in tps], axis=0)
                zrows.append({"dataset": ds, "task": task, "arm": arm,
                              **artifacts.profile(allc, arm)})
            if arm not in ("neg_mean_noise", "neg_shuffle", "pos_split_half",
                           "scDiTT"):
                d, pv = io.resolve_baseline_dir(arm, ds, task, seed)
                prov.append({"dataset": ds, "task": task, "seed": seed,
                             "method": arm, "ok": d is not None,
                             "provenance": pv, "path": str(d) if d else ""})
        if seed == SEEDS[0]:
            allr = np.concatenate([ref_cells[t] for t in tps], axis=0)
            zrows.append({"dataset": ds, "task": task, "arm": "real_reference",
                          **artifacts.profile(allr, "real_reference")})

    FIGDATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIGDATA / f"curves_{ds}_{task}.npz",
                        tps=np.asarray(tps), names=np.asarray(names, dtype=object),
                        informative=rs.informative,
                        shape=np.asarray(rs.shape, dtype=object),
                        full_mu=rs.mu, se_bar=rs.se_bar,
                        n_star=np.asarray([rs.n_star[t] for t in tps]),
                        **curves)
    return {"per_marker": per_marker, "ks": ks_rows,
            "prov": prov, "zero": zrows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = list(P.ALL_COMBOS)
    pm, ks, prov, zero = [], [], [], []
    with ProcessPoolExecutor(max_workers=min(10, len(jobs))) as ex:
        fut = {ex.submit(run_combo, ds, tk): (ds, tk) for ds, tk in jobs}
        for f in as_completed(fut):
            ds, tk = fut[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {ds}/{tk}: {type(e).__name__}: {e}")
                raise
            pm += r["per_marker"]; ks += r["ks"]
            prov += r["prov"]; zero += r["zero"]
            print(f"[done] {ds}/{tk}  rows={len(r['per_marker'])}")

    pd.DataFrame(pm).to_csv(OUT / "per_marker.csv", index=False)
    pd.DataFrame(ks).to_csv(OUT / "ks_diagnostic.csv", index=False)
    pd.DataFrame(zero).to_csv(OUT / "zero_structure.csv", index=False)
    pdf = pd.DataFrame(prov)
    (OUT / "provenance.json").write_text(json.dumps({
        "n_baseline_runs": int(len(pdf)),
        "n_ok": int(pdf.ok.sum()),
        "by_source": pdf.provenance.value_counts(dropna=False).to_dict(),
        "missing": pdf[~pdf.ok][["dataset", "task", "seed", "method"]]
                   .to_dict("records"),
    }, indent=2, ensure_ascii=False))
    print(f"\nbaseline runs: {len(pdf)}  ok={int(pdf.ok.sum())}  "
          f"source={pdf.provenance.value_counts().to_dict()}")

    df = pd.DataFrame(pm)
    g = df[df.informative].groupby(["dataset", "task", "arm", "seed"])["pearson_r"] \
                          .apply(core.fisher_z_mean).reset_index()
    gate = g.groupby(["dataset", "task", "arm"])["pearson_r"].mean().unstack()
    keep = [c for c in ("neg_shuffle", "neg_mean_noise", "pos_split_half")
            if c in gate.columns]
    print("\n=== Step 3 control gate (marker mean r, averaged across seeds) ===")
    print(gate[keep].round(4).to_string())
    ok = True
    for (ds, tk), row in gate.iterrows():
        if (ds, tk) not in P.USABLE_COMBOS:
            continue
        if abs(row.get("neg_shuffle", np.nan)) >= 0.2:
            print(f"  [GATE FAIL] {ds}/{tk} neg_shuffle |r|={row['neg_shuffle']:.3f} >= 0.2")
            ok = False
        if row.get("pos_split_half", np.nan) < 0.90:
            print(f"  [GATE WARN] {ds}/{tk} pos_split_half r={row['pos_split_half']:.3f} < 0.90")
            ok = False
    print("GATE:", "PASS" if ok else "see the failed items above (recorded in FAILURE_LOG)")
    gate.to_csv(OUT / "control_gate.csv")


if __name__ == "__main__":
    main()
