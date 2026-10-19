from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]                                   # scDiTT_code/
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config62 as C                                             # noqa: E402
from exps.task6_bio_exp.common import data_io as io              # noqa: E402
from exps.task6_bio_exp.common import embedding                  # noqa: E402


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def stratified_split(strata: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    is_train = np.zeros(len(strata), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        k = min(max(1, int(round(frac * len(idx)))), len(idx))
        is_train[idx[:k]] = True
    return is_train


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def gate_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str],
                 majority_class: str) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 precision_recall_fscore_support)
    K = len(classes)
    lab = np.arange(K)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=lab, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=lab)
    maj = classes.index(majority_class)
    per_class = {}
    for i, c in enumerate(classes):
        n_i = int(sup[i])
        off = cm[i].astype(float).copy()
        off[i] = -1.0
        j = int(np.argmax(off))
        per_class[c] = {
            "support": n_i,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "absorbed_into_majority": float(cm[i, maj] / n_i) if n_i else float("nan"),
            "top_confusion_class": classes[j] if n_i else None,
            "top_confusion_frac": float(cm[i, j] / n_i) if n_i else float("nan"),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=lab, average="macro",
                                   zero_division=0)),
        "n_eval": int(len(y_true)),
        "classes": list(classes),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rownorm":
            (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist(),
        "per_class": per_class,
    }


def _expand_proba(P: np.ndarray, present: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((P.shape[0], K), dtype=np.float32)
    out[:, np.asarray(present, dtype=int)] = P.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_for_dataset(dskey: str) -> dict:
    t0 = time.time()
    ds, label_key = C.DATASET_SPECS[dskey]
    log: List[str] = []

    obs = io.load_labels(ds, "interpolation", 42)
    y_str = obs[label_key].astype(str).values
    tp = np.asarray(obs["tp"]).astype(int)

    adata = io.load_real(ds)
    if adata.n_obs != len(obs):
        raise RuntimeError(f"{dskey}: training-source Data({adata.n_obs}) and latents({len(obs)}) have different row counts")
    if not np.array_equal(np.asarray(adata.obs[io.TIME_KEY].values).astype(int), tp):
        raise RuntimeError(f"{dskey}: training-source Data and latents.h5ad disagree on tp row by row -- "
                           "most likely the wrong data version")
    import scipy.sparse as sp
    Xr = adata.X
    X = np.ascontiguousarray(Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr),
                             dtype=np.float32)
    del adata, Xr

    classes = sorted(set(y_str))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_idx[v] for v in y_str], dtype=np.int64)
    majority_class = classes[int(np.bincount(y, minlength=len(classes)).argmax())]
    log.append(f"cells={len(y)} genes={X.shape[1]} classes={len(classes)} "
               f"majority={majority_class}")

    C.ensure_dirs()
    for task in C.TASKS:
        m = ~np.isin(tp, io.held_out_tps(ds, task))
        np.savez_compressed(C.DIR_CLF / f"{dskey}_{task}_train_stats.npz",
                            mu=X[m].mean(axis=0).astype(np.float32),
                            sd=X[m].std(axis=0).astype(np.float32),
                            n_train_cells=np.int64(m.sum()),
                            train_tps=np.array(sorted(set(tp[m].tolist())), dtype=np.int32))

    strata = np.array([f"{a}|{b}" for a, b in zip(tp, y_str)])
    is_train = stratified_split(strata, C.TRAIN_FRAC, C.SPLIT_SEED)
    log.append(f"split: reference(70%)={int(is_train.sum())} gate(30%)={int((~is_train).sum())}")

    rng = np.random.default_rng(C.SPLIT_SEED)
    tr_idx = np.flatnonzero(is_train)
    fit_idx = (tr_idx if len(tr_idx) <= C.PCA_FIT_MAX_CELLS
               else np.sort(rng.choice(tr_idx, C.PCA_FIT_MAX_CELLS, replace=False)))
    emb = embedding.RealFittedEmbedding.fit(X[fit_idx], n_pcs=C.N_PCS, seed=C.SPLIT_SEED)
    Z = emb.transform(X)
    log.append(f"PCA fit on {len(fit_idx)} real reference cells -> {emb.n_pcs} PCs, "
               f"cum.EVR={float(emb.explained_variance_ratio.sum()):.4f}")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    te_mask = ~is_train
    try:
        hv = KNeighborsClassifier(n_neighbors=C.KNN_K, metric="cosine", n_jobs=-1)
        hv.fit(X[is_train], y[is_train])
        rb2 = gate_metrics(y[te_mask], hv.predict(X[te_mask]), classes, majority_class)
        for k in ("confusion_matrix", "confusion_matrix_rownorm", "classes"):
            rb2.pop(k, None)
        rb2["gate_pass"] = bool(rb2["accuracy"] > C.GATE_ACC_MIN
                                and rb2["macro_f1"] > C.GATE_MACRO_F1_MIN)
    except Exception as e:                                   # pragma: no cover
        rb2 = {"error": repr(e)}
    del X

    knn = KNeighborsClassifier(n_neighbors=C.KNN_K, weights="uniform", n_jobs=-1)
    knn.fit(Z[is_train], y[is_train])

    y_pred = np.full(len(y), -1, dtype=np.int64)
    y_proba = np.zeros((len(y), len(classes)), dtype=np.float32)

    te = np.flatnonzero(~is_train)
    if len(te):
        y_pred[te] = knn.predict(Z[te])
        y_proba[te] = _expand_proba(knn.predict_proba(Z[te]), knn.classes_, len(classes))

    ytr = y[is_train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skf = StratifiedKFold(n_splits=C.OOF_FOLDS, shuffle=True, random_state=C.SPLIT_SEED)
        for a, b in skf.split(np.zeros(len(ytr)), ytr):
            m = KNeighborsClassifier(n_neighbors=C.KNN_K, weights="uniform", n_jobs=-1)
            m.fit(Z[tr_idx[a]], ytr[a])
            gi = tr_idx[b]
            y_pred[gi] = m.predict(Z[gi])
            y_proba[gi] = _expand_proba(m.predict_proba(Z[gi]), m.classes_, len(classes))
    assert (y_pred >= 0).all()

    gate = gate_metrics(y[te], y_pred[te], classes, majority_class)
    gate.update({
        "dataset_key": dskey, "dataset": ds, "label_key": label_key,
        "classifier": f"kNN k={C.KNN_K} on PCA-{emb.n_pcs} (fit on real reference cells only)",
        "n_cells_total": int(len(y)), "n_reference": int(is_train.sum()),
        "majority_class": majority_class,
        "thresholds": {"accuracy": C.GATE_ACC_MIN, "macro_f1": C.GATE_MACRO_F1_MIN,
                       "per_class_recall": C.GATE_RARE_RECALL_MIN,
                       "absorbed_into_majority": C.GATE_ABSORB_MAX},
        "pca_cum_explained_variance": float(emb.explained_variance_ratio.sum()),
    })
    gate["pass_accuracy"] = bool(gate["accuracy"] > C.GATE_ACC_MIN)
    gate["pass_macro_f1"] = bool(gate["macro_f1"] > C.GATE_MACRO_F1_MIN)
    gate["gate_pass"] = bool(gate["pass_accuracy"] and gate["pass_macro_f1"])

    per_tp: Dict[str, Dict[str, dict]] = {}
    yte, pte, tte = y[te], y_pred[te], tp[te]
    for t in sorted(set(tte.tolist())):
        m = tte == t
        d = {}
        for i, c in enumerate(classes):
            sel = m & (yte == i)
            n_i = int(sel.sum())
            d[c] = {"support": n_i,
                    "recall": float((pte[sel] == i).mean()) if n_i else float("nan")}
        per_tp[str(int(t))] = d
    gate["per_tp_per_class_recall"] = per_tp

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Z[is_train], y[is_train])
        rb = gate_metrics(y[te], lr.predict(Z[te]), classes, majority_class)
        for k in ("confusion_matrix", "confusion_matrix_rownorm", "classes"):
            rb.pop(k, None)
        rb["gate_pass"] = bool(rb["accuracy"] > C.GATE_ACC_MIN
                               and rb["macro_f1"] > C.GATE_MACRO_F1_MIN)
        gate["robustness_logreg"] = rb
    except Exception as e:                                   # pragma: no cover
        gate["robustness_logreg"] = {"error": repr(e)}
    gate["robustness_hvg_cosine"] = rb2

    import joblib
    joblib.dump({"embedding": emb, "n_pcs": emb.n_pcs, "n_genes": emb.n_genes,
                 "fit_cells": int(len(fit_idx)), "dataset": ds, "dataset_key": dskey},
                C.DIR_CLF / f"{dskey}_pca50_real_fit.pkl")
    joblib.dump({"knn": knn, "classes": classes, "k": C.KNN_K,
                 "reference_mask": is_train, "dataset_key": dskey, "label_key": label_key},
                C.DIR_CLF / f"{dskey}_knn_reference.pkl")
    np.save(C.DIR_CLF / f"{dskey}_real_pca.npy", Z)
    np.savez_compressed(C.DIR_CLF / f"{dskey}_real_pred.npz",
                        classes=np.array(classes, dtype=object),
                        y_true=y.astype(np.int16), y_pred=y_pred.astype(np.int16),
                        y_proba=y_proba, tp=tp.astype(np.int32), is_reference=is_train)
    (C.DIR_CLF / f"gate_report_{dskey}.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    check = verify_frozen(dskey, ds, y_str, tp)
    gate["_log"] = log
    gate["_seconds"] = round(time.time() - t0, 1)
    return {"gate": gate, "check": check}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def verify_frozen(dskey: str, ds: str, y_str: np.ndarray, tp: np.ndarray) -> List[dict]:
    rows: List[dict] = []
    classes = sorted(set(y_str))
    for task in C.TASKS:
        for t in io.held_out_tps(ds, task):
            m = tp == t
            n_tot = int(m.sum())
            counts = {c: int((y_str[m] == c).sum()) for c in classes}
            frozen = C.frozen_rare_for(dskey, task, t)
            implied = C.rule_implied_rare(counts, n_tot)
            rng_spec = C.FROZEN_RARE_RANGES.get((dskey, task), {})
            listed = C.has_frozen_list(dskey, task)

            for c, p_expect in frozen.items():
                if c not in counts:
                    rows.append(dict(dataset_key=dskey, task=task, tp=int(t), cls=c,
                                     status="MISSING_CLASS", p_ref_frozen_pct=p_expect,
                                     p_ref_actual_pct=None, n_k=None, n_tp=n_tot,
                                     category=None, frozen_listed=True,
                                     note="a class in the frozen list does not exist in the real labels"))
                    continue
                n_k = counts[c]
                p_act = 100.0 * n_k / n_tot
                if p_expect is not None:
                    ok = abs(p_act - p_expect) <= C.PREF_VERIFY_TOL_PCT
                    status = "OK" if ok else "MISMATCH"
                    note = "" if ok else f"pre-registered {p_expect:.2f}% vs measured {p_act:.4f}%"
                else:
                    lo, hi = rng_spec.get(c, (None, None))
                    if lo is None:
                        status, note = "OK_RANGE_UNSPEC", "the pre-registration gives an interval; no bounds were recorded"
                    else:
                        ok = (lo - C.PREF_VERIFY_TOL_PCT) <= p_act <= (hi + C.PREF_VERIFY_TOL_PCT)
                        status = "OK_RANGE" if ok else "MISMATCH_RANGE"
                        note = "" if ok else f"pre-registered interval [{lo},{hi}]% vs measured {p_act:.4f}%"
                rows.append(dict(dataset_key=dskey, task=task, tp=int(t), cls=c,
                                 status=status, p_ref_frozen_pct=p_expect,
                                 p_ref_actual_pct=round(p_act, 4), n_k=n_k, n_tp=n_tot,
                                 category=C.rare_category(n_k, n_k / n_tot),
                                 frozen_listed=True, note=note))

            for c, cat in sorted(implied.items()):
                if c in frozen:
                    continue
                n_k = counts[c]
                rows.append(dict(
                    dataset_key=dskey, task=task, tp=int(t), cls=c,
                    status="RULE_ONLY" if listed else "RULE_DERIVED",
                    p_ref_frozen_pct=None, p_ref_actual_pct=round(100.0 * n_k / n_tot, 4),
                    n_k=n_k, n_tp=n_tot, category=cat, frozen_listed=False,
                    note=("matched by the rare-class rule but absent from the pre-registered list (recorded only, not in the main table)" if listed
                          else "no per-cell frozen list for this (DS,task); derived from the rare-class rule")))
    return rows


# ---------------------------------------------------------------------------
def _worker(dskey: str):
    try:
        return dskey, build_for_dataset(dskey), None
    except Exception:
        return dskey, None, traceback.format_exc()


def write_gate_summary(results: Dict[str, dict], errors: Dict[str, str]) -> None:
    L: List[str] = []
    A = L.append
    q = lambda s: '"' + s + '"'                                   # noqa: E731
    A("# 6-2 🚦 Validity gate overview\n")
    A("> This file is generated automatically by `run_gate.py`. **It licenses every downstream 6-2 conclusion.**")
    A("> The gate is evaluated on **real data only**, before any generated data is labelled"
      " (pre-registered analysis plan, Step 3).\n")
    A(f"- Classifier (primary convention): kNN k={C.KNN_K}, feature space PCA-{C.N_PCS}, "
      "**fitted only on the 70% reference set of real cells**")
    A(f"- Split: real cells stratified by `tp x celltype` {int(C.TRAIN_FRAC*100)}/"
      f"{100 - int(C.TRAIN_FRAC*100)}; 70% builds the reference library, 30% is the gate evaluation set")
    A(f"- Thresholds: accuracy > {C.GATE_ACC_MIN}, **macro-F1 > {C.GATE_MACRO_F1_MIN} (primary criterion)**, "
      f"per-class recall for each pre-registered rare class > {C.GATE_RARE_RECALL_MIN}, "
      f"fraction of rare cells absorbed into the majority class < {C.GATE_ABSORB_MAX}\n")

    A("## 1. Gate results\n")
    A("Primary convention = kNN k=15 on PCA-50. Two **pre-registered robustness conventions** are also reported"
      " (pre-registered analysis plan: swap the classifier / swap the feature space), "
      "which answer whether a different choice could rescue a failing gate.\n")
    A("| Dataset | Label column | Classes | Gate-set n | accuracy | **macro-F1** | Verdict | "
      "LogReg(balanced) acc / macro-F1 | all-HVG cosine acc / macro-F1 |")
    A("|---|---|---|---|---|---|---|---|---|")

    def _rs(r):
        if "accuracy" not in r:
            return "—"
        return (f"{r['accuracy']:.4f} / {r['macro_f1']:.4f} "
                f"{'✅' if r.get('gate_pass') else '❌'}")

    for k in C.ALL_DSKEYS:
        if k not in results:
            A(f"| {k} | - | - | - | - | - | ❌ run failed | - | - |")
            continue
        g = results[k]["gate"]
        A(f"| {k} | `{g['label_key']}` | {len(g['classes'])} | {g['n_eval']} | "
          f"{g['accuracy']:.4f}{'' if g['pass_accuracy'] else ' ❌'} | "
          f"**{g['macro_f1']:.4f}**{'' if g['pass_macro_f1'] else ' ❌'} | "
          f"{'✅ pass' if g['gate_pass'] else '❌ **fail -> not evaluable**'} | "
          f"{_rs(g.get('robustness_logreg', {}))} | "
          f"{_rs(g.get('robustness_hvg_cosine', {}))} |")

    A("\n## 2. Per-class recall of the pre-registered rare classes (gate set, real data)\n")
    A(f"> recall ≤ {C.GATE_RARE_RECALL_MIN} classes have their downstream proportion conclusions marked **not evaluable**; "
      "classes with `n<20` only ever report detection (0/1).\n")
    A("| Dataset | Class | Gate-set support | recall | Absorbed into majority | Main misclassification target | Verdict |")
    A("|---|---|---|---|---|---|---|")
    for k in C.ALL_DSKEYS:
        if k not in results:
            continue
        g = results[k]["gate"]
        rare = sorted({c for (dk, _tk, _tp), d in C.FROZEN_RARE.items() if dk == k for c in d})
        if not rare:
            rare = sorted({r["cls"] for r in results[k]["check"]
                           if r["category"] in ("recall", "detection")})
        for c in rare:
            pc = g["per_class"].get(c)
            if pc is None:
                continue
            ok = pc["recall"] > C.GATE_RARE_RECALL_MIN
            A(f"| {k} | `{c}` | {pc['support']} | {pc['recall']:.3f} | "
              f"{pc['absorbed_into_majority']:.3f} | "
              f"`{pc['top_confusion_class']}` ({pc['top_confusion_frac']:.3f}) | "
              f"{'✅' if ok else '❌ not evaluable'} |")

    A("\n## 3. Frozen rare-class list check (`metrics/frozen_rare_check.csv`)\n")
    df = pd.DataFrame([r for res in results.values() for r in res["check"]])
    if len(df):
        for st, n in df["status"].value_counts().items():
            A(f"- `{st}`: {n} rows")
        bad = df[df["status"].astype(str).str.startswith("MISMATCH")
                 | (df["status"] == "MISSING_CLASS")]
        if len(bad):
            A("\n### ⚠️ Entries inconsistent with the pre-registration (**reported as-is, no silent corrections**)\n")
            A("| Dataset | task | tp | Class | Status | Note |")
            A("|---|---|---|---|---|---|")
            for _, r in bad.iterrows():
                A(f"| {r['dataset_key']} | {r['task']} | {r['tp']} | `{r['cls']}` | "
                  f"{r['status']} | {r['note']} |")
        else:
            A(f"\n**Every entry with a recorded `p_ref` in the pre-registered list matches the real data"
              f" (tolerance {C.PREF_VERIFY_TOL_PCT}%).**")
        ro = df[df["status"] == "RULE_ONLY"]
        if len(ro):
            A(f"\n`RULE_ONLY` = (DS,task,tp,class) matched by the rare-class rule but absent from the pre-registered list, "
              f"total **{len(ro)}** rows.")
            A("They are **excluded from the main table** (the pre-registration wins) but are all recorded in `metrics/frozen_rare_check.csv` "
              "and `metrics/rare_fate_recall.csv` (`frozen_listed=False`) to prevent selective reporting.")

    A("\n## 4. Implementation conventions (where the pre-registered plan is ambiguous)\n")
    A("The three literal pre-registered statements below degenerate or are unimplementable; the chosen handling is recorded explicitly here, "
      "**without modifying the pre-registration file**.\n")
    A("### 4.1 Scope of negative control 2 (label shuffling): `within the same tp` -> **pooled across tps**\n")
    A("- The pre-registration says \"real cells, randomly permute the type labels **within the same tp**\".")
    A("- **The literal convention degenerates identically for every metric here**: proportion metrics only see the marginal distribution, "
      "and permuting within a tp preserves that tp's marginal exactly -> `JSD = 0`, `L1 = 0`, `rare recall = 1`; "
      f"that beats even the positive control and cannot be a {q('lower bound')}.")
    A("- The same pre-registration **expects** this arm to give \"composition ~ marginal prior\" and \"random rare recall\", "
      "and only **pooled cross-tp permutation** satisfies both.")
    A(f"- -> The implementation uses `NEG_SHUFFLE_SCOPE = {q(C.NEG_SHUFFLE_SCOPE)}`: "
      "cells at eval_tp draw labels without replacement from the dataset-wide label pool, so `q ~ global marginal prior`, "
      "a genuine lower bound that ignores time and guesses from the overall prior.")
    A("- The literal convention **needs no computation**: by construction it is always `JSD=0 / L1=0 / rare recall=1`.\n")
    A("### 4.2 bootstrap = resampling **with replacement**\n")
    A(f"- The pre-registration aligns both sides to `n_eff = min(n_real, {C.N_EFF_CAP})`, "
      f"`B = {C.BOOTSTRAP_B}`, reporting `mean ± 95% CI`.")
    A("- Implemented as \"subsample without replacement to n_eff\", the real side is the identity map whenever `n_real <= 2000`, "
      "so the variance is always 0 and the 95% CI collapses to a point.")
    A("- -> Implemented instead as a standard non-parametric bootstrap: draw `n_eff` cells **with replacement** from each side."
      "The definition of `n_eff` is exactly as pre-registered.")
    A("- The positive control uses **the same `n_eff` and the same bootstrap** (required by the pre-registration)."
      "Note each of its two pools holds only `n_real/2` cells, so when `n_real < 4*n_eff` it draws "
      "`n_eff` cells with replacement from a half-pool -- this makes the positive control's sampling noise slightly **too small**, "
      "i.e. the positive control is a **slightly optimistic** floor; it errs against us, so it is safe as an upper bound.\n")
    A("### 4.3 Classifier predictions for real cells use out-of-fold\n")
    A("- The positive control splits the tp's real cells in half at random and pushes half A through the classifier. But about 70% of half A "
      "is itself in the kNN reference library and self-matches, systematically under-estimating the positive control's classifier error "
      "(under-estimating the positive control inflates our relative performance).")
    A(f"- -> Real cells inside the reference library always use **{C.OOF_FOLDS}-fold out-of-fold** predictions, "
      "the gate set (30%) is predicted with the full reference library; the merged result is written to `classifier/{DS}_real_pred.npz`.\n")

    A("## 5. Impact of known artefacts on this experiment\n")
    A("Real data has 92-95% exact zeros and 0% negative values; scDiTT and all 6 baselines have "
      "**0% exact zeros and 25-40% negative values**.")
    A("PCA is fitted on real cells and generated cells are transformed into it, so this **domain shift propagates straight into the kNN neighbourhood structure**.")
    A("The gate is evaluated on real data only and **cannot cover that domain shift** -- an inherent methodological limitation of this experiment, "
      "with per-arm zero-structure profiles recorded in `metrics/zero_structure_profile.csv` for reference.\n")

    if errors:
        A("## 6. Run failures\n")
        for k, e in errors.items():
            A(f"### {k}\n```\n{e}\n```")

    (C.RESULTS / "GATE_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {C.RESULTS / 'GATE_SUMMARY.md'}")


def _reload_results(dskeys: List[str]) -> Dict[str, dict]:
    chk_path = C.DIR_METRICS / "frozen_rare_check.csv"
    chk = pd.read_csv(chk_path) if chk_path.exists() else pd.DataFrame()
    out: Dict[str, dict] = {}
    for k in dskeys:
        p = C.DIR_CLF / f"gate_report_{k}.json"
        if not p.exists():
            continue
        rows = (chk[chk.dataset_key == k].to_dict("records")
                if len(chk) else [])
        out[k] = {"gate": json.loads(p.read_text()), "check": rows}
    return out


def main(dskeys: Optional[List[str]] = None, summary_only: bool = False) -> int:
    C.ensure_dirs()
    dskeys = dskeys or C.ALL_DSKEYS
    if summary_only:
        res = _reload_results(dskeys)
        if not res:
            print("no usable gate_report_*.json")
            return 1
        write_gate_summary(res, {})
        return 0
    results: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=min(len(dskeys), 6)) as ex:
        for dskey, res, err in ex.map(_worker, dskeys):
            if err:
                errors[dskey] = err
                print(f"[FAIL] {dskey}\n{err}", flush=True)
            else:
                results[dskey] = res
                g = res["gate"]
                print(f"[ok] {dskey:5s} acc={g['accuracy']:.4f} macroF1={g['macro_f1']:.4f} "
                      f"pass={g['gate_pass']}  ({g['_seconds']}s)", flush=True)

    if results:
        pd.DataFrame([r for res in results.values() for r in res["check"]]).to_csv(
            C.DIR_METRICS / "frozen_rare_check.csv", index=False)
        write_gate_summary(results, errors)
    return 0 if not errors else 1


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(main(_args or None, summary_only="--summary-only" in sys.argv))
