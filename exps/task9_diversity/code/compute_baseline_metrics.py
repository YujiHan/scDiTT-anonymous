from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

METHODS = ["PRESCIENT", "PI-SDE", "scDiffEq", "MIOFlow", "scNODE", "scIMF"]
SERVERS = ["A", "B", "C"]


def find_pred(method: str, ds: str, task: str, seed: int,
              variant: str = "prediction.npy") -> Path | None:
    for s in SERVERS:
        p = (C.BASELINE_ROOT / s / "baseline_results" / method / ds / task /
             f"seed{seed}" / variant)
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="*", default=C.DATASETS)
    ap.add_argument("--task", nargs="*", default=C.TASKS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--all_seeds", nargs="*", type=int,
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    device_str = C.set_gpu(args.gpu)
    import metrics_lib as ML  # noqa: E402
    from compute_metrics import REAL_SUBSAMPLE_TAG, load_floor
    device = ML.get_device(device_str)

    rows, diag = [], []
    for ds in args.ds:
        real = C.load_real_by_tp(ds)
        pca = C.get_pca(ds)
        for task in args.task:
            floor = load_floor(ds, task)
            test_tps = floor["test_tps"]
            real_sub = {t: ML.subsample(real[t], floor["stages"][str(t)]["n"],
                                        [0, t, REAL_SUBSAMPLE_TAG])
                        for t in test_tps}
            for method in args.methods:
                p = find_pred(method, ds, task, args.train_seed)
                if p is None:
                    print(f"[skip] {method}/{ds}/{task}/seed{args.train_seed}: no prediction.npy")
                    continue
                pred = np.load(p)                     # (T, 2000, G)
                assert pred.shape[-1] == real[test_tps[0]].shape[1], (
                    f"gene dim mismatch {pred.shape[-1]} vs {real[test_tps[0]].shape[1]}")
                for t in test_tps:
                    st = floor["stages"][str(t)]
                    n_t = st["n"]
                    Gt = pred[int(t)]
                    G = ML.subsample(Gt, n_t, [0, t, 0])
                    fid = ML.w2_safe(G, real_sub[t], pca, device)
                    f_self = float("nan")
                    if len(Gt) // n_t >= 2:
                        f_self = ML.block_floor(Gt, n_t, pca, device, R=20,
                                                base_seed=4242, max_blocks=4)["F"]
                    rows.append(dict(
                        dataset=ds, task=task, stage=int(t), method=method,
                        cfg_weight=np.nan, train_seed=args.train_seed,
                        K=1, n_t=n_t, N_t=st["N"],
                        floor_real=st["F"], floor_self=f_self,
                        fid_raw=fid, div_raw=np.nan,
                        fid_norm=fid / st["F"],
                        div_norm_self=np.nan, div_norm_real=np.nan,
                        fid_sd=np.nan, div_sd=np.nan,
                        pred_source=str(p),
                    ))
                    print(f"[b] {method:10s} {ds}/{task} tp{t} n={n_t} "
                          f"fid={fid:.3f} ({fid/st['F']:.3f}) "
                          f"F_self={f_self:.3f}", flush=True)

                preds = {}
                for s in args.all_seeds:
                    q = find_pred(method, ds, task, s)
                    if q is not None:
                        preds[s] = np.load(q)
                if len(preds) >= 2:
                    for t in test_tps:
                        n_t = floor["stages"][str(t)]["n"]
                        Gs = {s: ML.to_pca(ML.subsample(a[int(t)], n_t, [0, t, 0]), pca)
                              for s, a in preds.items()}
                        ks = sorted(Gs)
                        d = [ML.w2_pca(Gs[a], Gs[b], device)
                             for a, b in itertools.combinations(ks, 2)]
                        diag.append(dict(
                            dataset=ds, task=task, stage=int(t), method=method,
                            n_seeds=len(ks), seeds=",".join(map(str, ks)),
                            n_t=n_t, floor_real=floor["stages"][str(t)]["F"],
                            trainseed_spread_raw=float(np.mean(d)),
                            trainseed_spread_over_floor_real=float(
                                np.mean(d) / floor["stages"][str(t)]["F"]),
                        ))

    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.METRICS_DIR / "raw_baseline.csv", index=False)
    pd.DataFrame(diag).to_csv(
        C.METRICS_DIR / "diag_baseline_trainseed_spread.csv", index=False)
    print(f"\n[b] wrote raw_baseline.csv ({len(rows)} rows) + "
          f"diag_baseline_trainseed_spread.csv ({len(diag)} rows)")
    print("[b] note: div_raw is all NaN -- baseline sampling diversity needs K independent"
          "predictions after retraining (see this file's docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
