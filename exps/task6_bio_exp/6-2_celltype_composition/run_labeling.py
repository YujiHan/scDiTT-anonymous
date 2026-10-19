"""6-2 Step 4: label all 10 arms (same PCA, same kNN reference library).

**Prerequisite**: ``run_gate.py`` must have been run (the gate precedes any labelling of generated data).

The 10 arms (order taken from ``common/controls.py: ARM_ORDER``)::

    neg_mean_noise | neg_shuffle | PI-SDE | scIMF | scNODE | MIOFlow | PRESCIENT
    | scDiffEq | scDiTT | pos_split_half

Usage::

    cd scDiTT_code
    PYTHONPATH=. python exps/task6_bio_exp/6-2_celltype_composition/run_labeling.py

Artefacts::

    labels/{DS}_{task}_seed{N}_{arm}_labels.npz   consumed directly by 6-6 / 6-10 / 6-11
    metrics/zero_structure_profile.csv            per-arm zero-inflation artefact profile (seed42)
    labels/provenance.json                        whether each baseline run came from REDONE or own
"""
from __future__ import annotations

import json
import sys
import time
import traceback
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
from exps.task6_bio_exp.common import artifacts                 # noqa: E402
from exps.task6_bio_exp.common import controls                  # noqa: E402
from exps.task6_bio_exp.common import data_io as io             # noqa: E402
from exps.task6_bio_exp.common import sampling                  # noqa: E402


def _expand_proba(P: np.ndarray, present: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((P.shape[0], K), dtype=np.float32)
    out[:, np.asarray(present, dtype=int)] = P.astype(np.float32)
    return out


def _save(dskey: str, task: str, seed: int, arm: str, classes: List[str],
          eval_tps: List[int], per_tp: Dict[int, dict], extra: dict) -> None:
    payload = {"classes": np.array(classes, dtype=object),
               "eval_tps": np.array(eval_tps, dtype=np.int32),
               "arm": arm, "dataset_key": dskey, "task": task, "seed": np.int64(seed)}
    payload.update({k: np.array(v) for k, v in extra.items()})
    for t, d in per_tp.items():
        for k, v in d.items():
            payload[f"tp{t}_{k}"] = v
    np.savez_compressed(C.labels_npz_path(dskey, task, seed, arm), **payload)


# ---------------------------------------------------------------------------
def run_one(dskey: str, task: str) -> dict:
    t0 = time.time()
    import joblib
    ds, label_key = C.DATASET_SPECS[dskey]

    pca = joblib.load(C.DIR_CLF / f"{dskey}_pca50_real_fit.pkl")
    ref = joblib.load(C.DIR_CLF / f"{dskey}_knn_reference.pkl")
    emb, knn, classes = pca["embedding"], ref["knn"], list(ref["classes"])
    K = len(classes)
    rp = np.load(C.DIR_CLF / f"{dskey}_real_pred.npz", allow_pickle=True)
    y_true, y_pred_real, proba_real = rp["y_true"], rp["y_pred"], rp["y_proba"]
    tp_all = rp["tp"]

    eval_tps = io.held_out_tps(ds, task)
    train_tps = [t for t in sorted(set(tp_all.tolist())) if t not in set(eval_tps)]
    real_train = np.concatenate(
        [v for _, v in sorted(io.real_by_tp(ds, train_tps).items())], axis=0)
    io.assert_gene_dim_match(ds, real_train)

    prov: List[dict] = []
    zprof: List[dict] = []
    real_eval = None

    for seed in C.SEEDS:
        for method in [C.OUR_ARM] + io.ALL_METHODS:
            try:
                b = (io.load_scditt(ds, task, seed) if method == C.OUR_ARM
                     else io.load_baseline(method, ds, task, seed))
                pred = b.pred_by_tp()
            except Exception as e:
                prov.append({"dataset_key": dskey, "task": task, "seed": seed,
                             "arm": method, "ok": False, "provenance": None,
                             "path": "", "error": repr(e)})
                continue
            per_tp = {}
            for t in eval_tps:
                Xg = np.ascontiguousarray(pred[t], dtype=np.float32)
                Zg = emb.transform(Xg)
                per_tp[t] = {"pred": knn.predict(Zg).astype(np.int16),
                             "proba": _expand_proba(knn.predict_proba(Zg),
                                                    knn.classes_, K)}
                if seed == C.SEEDS[0]:
                    zprof.append(dict(dataset_key=dskey, task=task, seed=seed, tp=int(t),
                                      arm=method, **artifacts.profile(Xg, method)))
            _save(dskey, task, seed, method, classes, eval_tps, per_tp,
                  {"provenance": b.provenance, "source_path": str(b.path)})
            prov.append({"dataset_key": dskey, "task": task, "seed": seed, "arm": method,
                         "ok": True, "provenance": b.provenance, "path": str(b.path),
                         "error": ""})
            del pred, b

        rng = np.random.default_rng(10_000 + seed)
        per_tp = {}
        for t in eval_tps:
            Xn = controls.negative_mean_noise(real_train, C.N_EFF_CAP, rng)
            Zn = emb.transform(Xn)
            per_tp[t] = {"pred": knn.predict(Zn).astype(np.int16),
                         "proba": _expand_proba(knn.predict_proba(Zn), knn.classes_, K)}
            if seed == C.SEEDS[0]:
                zprof.append(dict(dataset_key=dskey, task=task, seed=seed, tp=int(t),
                                  arm="neg_mean_noise",
                                  **artifacts.profile(Xn, "neg_mean_noise")))
        _save(dskey, task, seed, "neg_mean_noise", classes, eval_tps, per_tp,
              {"provenance": "control", "train_tps": np.array(train_tps, dtype=np.int32)})

        rng = np.random.default_rng(20_000 + seed)
        shuffled = controls.negative_shuffle_labels(y_true, rng)
        per_tp = {}
        for t in eval_tps:
            m = tp_all == t
            lab = shuffled[m].astype(np.int16)
            oh = np.zeros((len(lab), K), dtype=np.float32)
            oh[np.arange(len(lab)), lab.astype(int)] = 1.0
            per_tp[t] = {"pred": lab, "proba": oh}
        _save(dskey, task, seed, "neg_shuffle", classes, eval_tps, per_tp,
              {"provenance": "control", "scope": C.NEG_SHUFFLE_SCOPE})

        rng = np.random.default_rng(30_000 + seed)
        per_tp = {}
        for t in eval_tps:
            idx = np.flatnonzero(tp_all == t)
            a, bb = sampling.split_half(idx, len(idx) // 2, rng)
            per_tp[t] = {
                "pred": y_pred_real[a].astype(np.int16),
                "proba": proba_real[a].astype(np.float32),
                "ref": y_true[bb].astype(np.int16),
            }
        _save(dskey, task, seed, "pos_split_half", classes, eval_tps, per_tp,
              {"provenance": "control"})

        if seed == C.SEEDS[0]:
            if real_eval is None:
                real_eval = io.real_by_tp(ds, eval_tps)
            for t in eval_tps:
                zprof.append(dict(dataset_key=dskey, task=task, seed=seed, tp=int(t),
                                  arm="real", **artifacts.profile(real_eval[t], "real")))

    return {"provenance": prov, "zero_profile": zprof,
            "seconds": round(time.time() - t0, 1),
            "n_files": len(list(C.DIR_LABELS.glob(f"{dskey}_{task}_*")))}


def _worker(args: Tuple[str, str]):
    dskey, task = args
    try:
        return (dskey, task), run_one(dskey, task), None
    except Exception:
        return (dskey, task), None, traceback.format_exc()


def main(dskeys: Optional[List[str]] = None, workers: int = 12) -> int:
    C.ensure_dirs()
    jobs = [(d, t) for d in (dskeys or C.ALL_DSKEYS) for t in C.TASKS]
    prov, zprof, errors = [], [], {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for key, res, err in ex.map(_worker, jobs):
            if err:
                errors[str(key)] = err
                print(f"[FAIL] {key}\n{err}", flush=True)
            else:
                prov += res["provenance"]
                zprof += res["zero_profile"]
                print(f"[ok] {key[0]:5s} {key[1]:15s} files={res['n_files']:3d} "
                      f"({res['seconds']}s)", flush=True)

    if prov:
        (C.DIR_LABELS / "provenance.json").write_text(
            json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
        bad = [p for p in prov if not p["ok"]]
        print(f"\nprovenance: {len(prov)} runs, missing={len(bad)}")
        for p in bad[:20]:
            print("   MISSING", p["dataset_key"], p["task"], p["seed"], p["arm"], p["error"])
    if zprof:
        pd.DataFrame(zprof).to_csv(C.DIR_METRICS / "zero_structure_profile.csv", index=False)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main([a for a in sys.argv[1:] if not a.startswith("-")] or None))
