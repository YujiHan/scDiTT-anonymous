"""6-2 Step 5: metric computation (JSD / L1 / macro-L1 / rare recall / detection / false positives).

**Prerequisites**: ``run_gate.py`` -> ``run_labeling.py``.

Frozen conventions (from the pre-registered analysis plan)::

    n_eff = min(n_real, 2000)      both sides aligned
    B     = 100                    bootstrap, reporting mean ± 95% CI
    JSD   base 2 -> [0, 1]
    Unknown kept as a normal class (primary convention) + removed and renormalised (secondary), both reported
    equal-weight macro average across tps, mean +- std across seeds

Usage::

    cd scDiTT_code
    PYTHONPATH=. python exps/task6_bio_exp/6-2_celltype_composition/run_metrics.py

Artefacts::

    metrics/composition_per_tp.csv
    metrics/rare_fate_recall.csv
    metrics/main_table_interpolation.csv
    metrics/main_table_extrapolation.csv
    metrics/scditt_vs_arms_tests.csv     paired tests + Cohen's d + BH-FDR
    metrics/failure_list.csv             appendix table 3: failure list (guards against selective reporting)
"""
from __future__ import annotations

import json
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config62 as C                                            # noqa: E402
from exps.task6_bio_exp.common import data_io as io             # noqa: E402
from exps.task6_bio_exp.common import stats as st               # noqa: E402


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _kl2(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(A > 0, A * np.log2(np.where(M > 0, A / M, 1.0)), 0.0)
    return t.sum(axis=-1)


def jsd2(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    M = 0.5 * (P + Q)
    return np.clip(0.5 * _kl2(P, M) + 0.5 * _kl2(Q, M), 0.0, 1.0)


def _summ(v: np.ndarray, name: str) -> dict:
    v = np.asarray(v, dtype=float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return {f"{name}_{s}": float("nan") for s in ("mean", "std", "lo", "hi")}
    lo, hi = np.percentile(ok, [100 * C.CI_ALPHA / 2, 100 * (1 - C.CI_ALPHA / 2)])
    return {f"{name}_mean": float(ok.mean()), f"{name}_std": float(ok.std(ddof=1))
            if ok.size > 1 else 0.0, f"{name}_lo": float(lo), f"{name}_hi": float(hi)}


def _drop_unknown(P: np.ndarray, unk: Optional[int]) -> np.ndarray:
    if unk is None:
        return P
    Q = np.delete(P, unk, axis=-1)
    s = Q.sum(axis=-1, keepdims=True)
    return np.divide(Q, s, out=np.full_like(Q, np.nan), where=s > 0)


# ---------------------------------------------------------------------------
def _boot(labels: np.ndarray, proba: Optional[np.ndarray], K: int, n_eff: int,
          B: int, rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(labels)
    idx = rng.integers(0, n, size=(B, n_eff))
    lab = np.asarray(labels, dtype=np.int64)[idx]
    hard = np.stack([np.bincount(r, minlength=K) for r in lab]).astype(np.float64) / n_eff
    soft = proba[idx].mean(axis=1).astype(np.float64) if proba is not None else None
    return hard, soft


def _rng_seed(*parts) -> int:
    return int(zlib.crc32("|".join(map(str, parts)).encode("utf-8")))


def _load_arm(dskey: str, task: str, seed: int, arm: str):
    p = C.labels_npz_path(dskey, task, seed, arm)
    return np.load(p, allow_pickle=True) if p.exists() else None


# ---------------------------------------------------------------------------
def run_one(dskey: str, task: str) -> dict:
    ds, label_key = C.DATASET_SPECS[dskey]
    gate = json.loads((C.DIR_CLF / f"gate_report_{dskey}.json").read_text())
    classes: List[str] = list(gate["classes"])
    K = len(classes)
    unk = classes.index(C.UNKNOWN_LABEL) if C.UNKNOWN_LABEL in classes else None
    clf_recall = {c: gate["per_class"][c]["recall"] for c in classes}
    gate_pass = bool(gate["gate_pass"])

    rp = np.load(C.DIR_CLF / f"{dskey}_real_pred.npz", allow_pickle=True)
    y_true, tp_all = rp["y_true"].astype(np.int64), rp["tp"]
    eval_tps = io.held_out_tps(ds, task)
    low_power = dskey in C.LOW_POWER_DSKEYS

    comp_rows: List[dict] = []
    rare_rows: List[dict] = []

    for tp in eval_tps:
        ref_all = y_true[tp_all == tp]
        n_real = len(ref_all)
        n_eff = C.n_eff(n_real)
        cnt_ref = np.bincount(ref_all, minlength=K).astype(float)
        p_full = cnt_ref / n_real

        frozen = C.frozen_rare_for(dskey, task, tp)
        implied = C.rule_implied_rare({c: int(cnt_ref[i]) for i, c in enumerate(classes)},
                                      n_real)
        rare_spec: Dict[str, bool] = {c: True for c in frozen if c in classes}
        for c in implied:
            rare_spec.setdefault(c, False)
        has_list = C.has_frozen_list(dskey, task)

        for seed in C.SEEDS:
            for arm in C.arm_order():
                z = _load_arm(dskey, task, seed, arm)
                if z is None:
                    continue
                lab = z[f"tp{tp}_pred"].astype(np.int64)
                pro = z[f"tp{tp}_proba"] if f"tp{tp}_proba" in z.files else None
                ref = (z[f"tp{tp}_ref"].astype(np.int64)
                       if f"tp{tp}_ref" in z.files else ref_all)

                rng = np.random.default_rng(_rng_seed(dskey, task, tp, seed, arm))
                P, _ = _boot(ref, None, K, n_eff, C.BOOTSTRAP_B, rng)
                Q, Qs = _boot(lab, pro, K, n_eff, C.BOOTSTRAP_B, rng)

                row = dict(dataset_key=dskey, dataset=ds, label_key=label_key, task=task,
                           tp=int(tp), seed=int(seed), arm=arm,
                           n_real=int(n_real), n_eff=int(n_eff), n_arm_cells=int(len(lab)),
                           gate_pass=gate_pass, low_power=low_power,
                           has_unknown=unk is not None)
                row.update(_summ(jsd2(P, Q), "jsd"))
                row.update(_summ(np.abs(P - Q).sum(-1), "l1"))
                row.update(_summ(np.abs(P - Q).mean(-1), "macro_l1"))
                Pu, Qu = _drop_unknown(P, unk), _drop_unknown(Q, unk)
                row.update(_summ(jsd2(Pu, Qu), "jsd_noUnk"))
                row.update(_summ(np.abs(Pu - Qu).sum(-1), "l1_noUnk"))
                row.update(_summ(np.abs(Pu - Qu).mean(-1), "macro_l1_noUnk"))
                if Qs is not None:
                    row.update(_summ(jsd2(P, Qs), "jsd_soft"))
                    row.update(_summ(np.abs(P - Qs).sum(-1), "l1_soft"))
                comp_rows.append(row)

                q_full = np.bincount(lab, minlength=K).astype(float) / len(lab)
                for cls_name, is_frozen in rare_spec.items():
                    i = classes.index(cls_name)
                    n_k, p_k, q_k = int(cnt_ref[i]), float(p_full[i]), float(q_full[i])
                    cat = C.rare_category(n_k, p_k)
                    qb = Q[:, i]
                    rec = q_k / p_k if p_k > 0 else float("nan")
                    recb = qb / p_k if p_k > 0 else np.full(C.BOOTSTRAP_B, np.nan)
                    cr = clf_recall.get(cls_name, float("nan"))
                    r = dict(dataset_key=dskey, dataset=ds, label_key=label_key, task=task,
                             tp=int(tp), seed=int(seed), arm=arm, cls=cls_name,
                             category=cat, frozen_listed=bool(is_frozen),
                             use_in_main=bool(is_frozen or not has_list),
                             boundary=bool(p_k >= C.RARE_P_THRESHOLD),
                             n_k=n_k, n_real=int(n_real), n_eff=int(n_eff),
                             p_ref=p_k, q=q_k, recall=rec,
                             detected_full=int(q_k > 0),
                             detect_rate_boot=float((qb > 0).mean()),
                             fp_rate=(q_k if cat == "false_positive" else float("nan")),
                             clf_recall=cr,
                             normalized_recall=(rec / cr if (cr and cr > 0
                                                             and np.isfinite(rec))
                                                else float("nan")),
                             gate_class_pass=bool(np.isfinite(cr)
                                                  and cr > C.GATE_RARE_RECALL_MIN),
                             gate_pass=gate_pass,
                             power_note=(C.LOW_POWER_NOTE if low_power else ""))
                    r.update(_summ(recb, "recall_boot"))
                    r.update(_summ(qb, "q_boot"))
                    rare_rows.append(r)

    return {"comp": comp_rows, "rare": rare_rows}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _agg_arm_dataset(comp: pd.DataFrame, rare: pd.DataFrame, task: str) -> pd.DataFrame:
    order = C.arm_order()
    if not len(comp):
        return pd.DataFrame({"arm": order})
    if not len(rare):
        rare = pd.DataFrame(columns=["arm", "dataset_key", "task", "category",
                                     "use_in_main", "gate_class_pass", "seed",
                                     "recall", "cls"])
    out = []
    for arm in order:
        row = {"arm": arm}
        for dskey in C.ALL_DSKEYS:
            sub = comp[(comp.arm == arm) & (comp.dataset_key == dskey) & (comp.task == task)]
            if len(sub):
                per_seed = {int(s): st.macro_mean(
                    dict(zip(g.tp, g.jsd_mean))) for s, g in sub.groupby("seed")}
                a = st.aggregate_seeds(per_seed)
                row[f"{dskey}_jsd"] = f"{a['mean']:.4f} ± {a['std']:.4f}"
                row[f"{dskey}_jsd_mean"] = a["mean"]
                row[f"{dskey}_gate_pass"] = bool(sub.gate_pass.iloc[0])
            else:
                row[f"{dskey}_jsd"] = ""
                row[f"{dskey}_jsd_mean"] = np.nan
                row[f"{dskey}_gate_pass"] = None
            rr = rare[(rare.arm == arm) & (rare.dataset_key == dskey) & (rare.task == task)
                      & (rare.category == "recall") & rare.use_in_main
                      & rare.gate_class_pass]
            if len(rr):
                import warnings as _w
                with np.errstate(divide="ignore", invalid="ignore"), \
                        _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    per_seed = {int(s): float(np.nanmean(g.recall.values))
                                for s, g in rr.groupby("seed")}
                    per_seed_dev = {
                        int(s): float(np.nanmean(np.abs(np.log2(
                            np.where(g.recall.values > 0, g.recall.values, np.nan)))))
                        for s, g in rr.groupby("seed")}
                a = st.aggregate_seeds(per_seed)
                b = st.aggregate_seeds(per_seed_dev)
                row[f"{dskey}_rare_recall"] = f"{a['mean']:.3f} ± {a['std']:.3f}"
                row[f"{dskey}_rare_recall_mean"] = a["mean"]
                row[f"{dskey}_rare_log2dev"] = f"{b['mean']:.3f} ± {b['std']:.3f}"
                row[f"{dskey}_rare_log2dev_mean"] = b["mean"]
                row[f"{dskey}_rare_zero_frac"] = float((rr.recall.values == 0).mean())
                row[f"{dskey}_rare_n_classes"] = int(rr.cls.nunique())
            else:
                row[f"{dskey}_rare_recall"] = "n/a"
                row[f"{dskey}_rare_recall_mean"] = np.nan
                row[f"{dskey}_rare_log2dev"] = "n/a"
                row[f"{dskey}_rare_log2dev_mean"] = np.nan
                row[f"{dskey}_rare_zero_frac"] = np.nan
                row[f"{dskey}_rare_n_classes"] = 0
            if dskey in C.LOW_POWER_DSKEYS:
                row[f"{dskey}_rare_note"] = C.LOW_POWER_NOTE
        out.append(row)
    return pd.DataFrame(out)


def _paired_tests(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(comp):
        return pd.DataFrame()
    for (dskey, task), g in comp.groupby(["dataset_key", "task"]):
        piv = g.pivot_table(index=["tp", "seed"], columns="arm", values="jsd_mean")
        if C.OUR_ARM not in piv.columns:
            continue
        base = piv[C.OUR_ARM]
        for arm in piv.columns:
            if arm == C.OUR_ARM:
                continue
            ok = base.notna() & piv[arm].notna()
            if ok.sum() < 3:
                continue
            try:
                r = st.paired_t(base[ok].values, piv[arm][ok].values)
            except Exception:
                continue
            rows.append(dict(dataset_key=dskey, task=task, arm=arm,
                             scditt_jsd=float(base[ok].mean()),
                             arm_jsd=float(piv[arm][ok].mean()),
                             mean_diff=r["mean_diff"], t=r["t"], p=r["p"],
                             cohens_dz=r["cohens_dz"], n_pairs=r["n_pairs"],
                             cohens_d=st.cohens_d(piv[arm][ok].values, base[ok].values),
                             gate_pass=bool(g.gate_pass.iloc[0])))
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = st.bh_fdr(df["p"].values)
        # paired_t(ctrl=scDiTT, prtb=arm) ⇒ mean_diff = JSD(arm) − JSD(scDiTT)
        df["scditt_better"] = df["mean_diff"] > 0
    return df


def _failure_list(comp: pd.DataFrame, rare: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for dskey in C.ALL_DSKEYS:
        p = C.DIR_CLF / f"gate_report_{dskey}.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        if not g["gate_pass"]:
            rows.append(dict(kind="gate_fail_dataset", dataset_key=dskey, task="*", tp="*",
                             cls="*", arm="*",
                             detail=f"accuracy={g['accuracy']:.4f} "
                                    f"macro_f1={g['macro_f1']:.4f} -> the whole dataset is not evaluable"))
        for c, d in g["per_class"].items():
            if d["support"] and d["recall"] <= C.GATE_RARE_RECALL_MIN:
                rows.append(dict(kind="gate_fail_class", dataset_key=dskey, task="*", tp="*",
                                 cls=c, arm="*",
                                 detail=f"classifier recall={d['recall']:.3f} "
                                        f"(support={d['support']}, "
                                        f"→{d['top_confusion_class']} "
                                        f"{d['top_confusion_frac']:.3f})"))
    if len(rare):
        low = rare[(rare.category == "recall") & (rare.recall < 0.1)]
        for (dk, tk, tp, cl, arm), g in low.groupby(
                ["dataset_key", "task", "tp", "cls", "arm"]):
            rows.append(dict(kind="rare_recall_below_0.1", dataset_key=dk, task=tk, tp=tp,
                             cls=cl, arm=arm,
                             detail=f"recall={g.recall.mean():.4f} (mean over "
                                    f"{len(g)} seeds), p_ref={g.p_ref.iloc[0]:.5f}"))
        fp = rare[(rare.category == "false_positive") & (rare.q > 0)]
        for (dk, tk, tp, cl, arm), g in fp.groupby(
                ["dataset_key", "task", "tp", "cls", "arm"]):
            rows.append(dict(kind="false_positive", dataset_key=dk, task=tk, tp=tp,
                             cls=cl, arm=arm,
                             detail=f"p_ref=0 but generated proportion={g.q.mean():.5f} "
                                    f"({int((g.q > 0).sum())}/{len(g)} seeds)"))
        nd = rare[(rare.category == "detection") & (rare.detected_full == 0)]
        for (dk, tk, tp, cl, arm), g in nd.groupby(
                ["dataset_key", "task", "tp", "cls", "arm"]):
            rows.append(dict(kind="rare_not_detected", dataset_key=dk, task=tk, tp=tp,
                             cls=cl, arm=arm,
                             detail=f"n_k={int(g.n_k.iloc[0])} ultra-rare class, "
                                    f"{len(g)}/{len(g)} seeds generated none at all"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _worker(args: Tuple[str, str]):
    dskey, task = args
    try:
        return (dskey, task), run_one(dskey, task), None
    except Exception:
        return (dskey, task), None, traceback.format_exc()


def main(dskeys: Optional[List[str]] = None, workers: int = 12) -> int:
    C.ensure_dirs()
    jobs = [(d, t) for d in (dskeys or C.ALL_DSKEYS) for t in C.TASKS]
    comp, rare, errors = [], [], {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for key, res, err in ex.map(_worker, jobs):
            if err:
                errors[str(key)] = err
                print(f"[FAIL] {key}\n{err}", flush=True)
            else:
                comp += res["comp"]
                rare += res["rare"]
                print(f"[ok] {key[0]:5s} {key[1]:15s} comp={len(res['comp'])} "
                      f"rare={len(res['rare'])}", flush=True)

    dc, dr = pd.DataFrame(comp), pd.DataFrame(rare)
    if len(dc):
        dc.sort_values(["dataset_key", "task", "tp", "seed", "arm"]).to_csv(
            C.DIR_METRICS / "composition_per_tp.csv", index=False)
    if len(dr):
        dr.sort_values(["dataset_key", "task", "tp", "cls", "seed", "arm"]).to_csv(
            C.DIR_METRICS / "rare_fate_recall.csv", index=False)
    for task in C.TASKS:
        _agg_arm_dataset(dc, dr, task).to_csv(
            C.DIR_METRICS / f"main_table_{task}.csv", index=False)
    _paired_tests(dc).to_csv(C.DIR_METRICS / "scditt_vs_arms_tests.csv", index=False)
    _failure_list(dc, dr).to_csv(C.DIR_METRICS / "failure_list.csv", index=False)
    print(f"\n[written] {C.DIR_METRICS}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main([a for a in sys.argv[1:] if not a.startswith("-")] or None))
