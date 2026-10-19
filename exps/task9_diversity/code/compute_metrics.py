from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

REAL_SUBSAMPLE_TAG = 7777


def load_floor(ds: str, task: str) -> dict:
    p = C.FLOOR_DIR / ds / task / "floor.json"
    if not p.exists():
        raise FileNotFoundError(f"floor not computed yet: {p}")
    with open(p) as fh:
        return json.load(fh)


def load_self_floor(ds: str, task: str, train_seed: int) -> dict | None:
    p = C.FLOOR_DIR / ds / task / f"self_floor_trainseed{train_seed}.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def baseline_run_dir(method: str, ds: str, task: str, train_seed: int) -> Path:
    return C.RUNS_DIR / "baseline" / method / ds / task / f"trainseed{train_seed}"


def pred_path(method: str, ds: str, task: str, cfg_w: float,
              train_seed: int, eval_seed: int) -> Path:
    if method == "scDiTT":
        return C.pred_dir(ds, task, cfg_w, train_seed, eval_seed) / "preds.npz"
    return (baseline_run_dir(method, ds, task, train_seed) /
            f"predseed{eval_seed}" / "preds.npz")


def load_pred_group(method: str, ds: str, task: str, cfg_w: float,
                    train_seed: int, eval_seeds) -> tuple[np.ndarray, list[int]]:
    arrs, tps = [], None
    for e in eval_seeds:
        p = pred_path(method, ds, task, cfg_w, train_seed, e)
        if not p.exists():
            raise FileNotFoundError(p)
        z = np.load(p)
        a, t = z["pred"], [int(x) for x in z["tps"]]
        if tps is None:
            tps = t
        elif tps != t:
            raise RuntimeError(f"test_tps mismatch: {tps} vs {t}")
        arrs.append(a)
    return np.stack(arrs, axis=0), tps


def fid_div_for_stage(G_list: list[np.ndarray], real_sub: np.ndarray,
                      pca, device) -> tuple[float, float, list[float], list[float]]:
    import metrics_lib as ML
    Gp = [ML.to_pca(g, pca) for g in G_list]
    Rp = ML.to_pca(real_sub, pca)
    fids = [ML.w2_pca(g, Rp, device) for g in Gp]
    divs = [ML.w2_pca(Gp[i], Gp[j], device)
            for i, j in itertools.combinations(range(len(Gp)), 2)]
    return float(np.mean(fids)), float(np.mean(divs)), fids, divs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="*", default=C.DATASETS)
    ap.add_argument("--task", nargs="*", default=C.TASKS)
    ap.add_argument("--cfg", nargs="*", type=float, default=C.CFG_WEIGHTS)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--eval_seeds", nargs="*", type=int, default=C.EVAL_SEEDS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--method", default="scDiTT",
                    help="scDiTT, or a baseline method name under runs/baseline/<METHOD>/")
    ap.add_argument("--out", default=None, help="output csv (default metrics/raw_<method>_<tag>.csv)")
    args = ap.parse_args()

    method = args.method
    cfgs = args.cfg if method == "scDiTT" else [float("nan")]

    device_str = C.set_gpu(args.gpu)
    import metrics_lib as ML  # noqa: E402
    device = ML.get_device(device_str)

    rows = []
    for ds in args.ds:
        real = C.load_real_by_tp(ds)
        pca = C.get_pca(ds)
        for task in args.task:
            floor = load_floor(ds, task)
            self_floor = (load_self_floor(ds, task, args.train_seed)
                          if method == "scDiTT" else None)
            if method == "scDiTT" and self_floor is None:
                print(f"[warn] {ds}/{task}: no self_floor, div_norm_self recorded as NaN "
                      f"(run compute_self_floor.py first)")
            test_tps = floor["test_tps"]
            real_sub = {}
            for t in test_tps:
                n_t = floor["stages"][str(t)]["n"]
                real_sub[t] = ML.subsample(real[t], n_t, [0, t, REAL_SUBSAMPLE_TAG])

            bl_self = {}
            if method != "scDiTT":
                sfp = baseline_run_dir(method, ds, task,
                                       args.train_seed) / "selffloor_preds.npz"
                if sfp.exists():
                    z = np.load(sfp)
                    sf, sf_tps = z["pred"], [int(x) for x in z["tps"]]
                    for ti, t in enumerate(sf_tps):
                        n_t = floor["stages"][str(t)]["n"]
                        bl_self[t] = ML.block_floor(
                            sf[ti], n_t, pca, device, R=20, base_seed=4242,
                            max_blocks=4)["F"]
                else:
                    print(f"[warn] {method}/{ds}/{task}: no selffloor_preds.npz")

            for w in cfgs:
                t0 = time.time()
                preds, tps = load_pred_group(method, ds, task, w,
                                             args.train_seed, args.eval_seeds)
                assert tps == test_tps, f"{tps} != {test_tps}"
                assert preds.shape[-1] > 1000, "latent space, not gene space (pitfall 8)"
                for ti, t in enumerate(test_tps):
                    st = floor["stages"][str(t)]
                    n_t = st["n"]
                    G_list = [ML.subsample(preds[i, ti], n_t, [0, t, i])
                              for i in range(preds.shape[0])]
                    fid, div, fids, divs = fid_div_for_stage(
                        G_list, real_sub[t], pca, device)
                    if method == "scDiTT":
                        f_self = (self_floor["cfg"][f"{w:g}"][str(t)]["F"]
                                  if self_floor is not None else float("nan"))
                    else:
                        f_self = bl_self.get(t, float("nan"))
                    rows.append(dict(
                        dataset=ds, task=task, stage=int(t), method=method,
                        cfg_weight=w, train_seed=args.train_seed,
                        K=len(G_list), n_t=n_t, N_t=st["N"],
                        floor_real=st["F"], floor_self=f_self,
                        fid_raw=fid, div_raw=div,
                        fid_norm=fid / st["F"],
                        div_norm_self=div / f_self,
                        div_norm_real=div / st["F"],
                        fid_sd=float(np.std(fids, ddof=1)),
                        div_sd=float(np.std(divs, ddof=1)),
                    ))
                    print(f"[m] {method} {ds}/{task} cfg={w:g} tp{t} n={n_t} "
                          f"F_real={st['F']:.3f} F_self={f_self:.3f} | "
                          f"fid={fid:.3f}({fid/st['F']:.3f}) "
                          f"div={div:.3f}(self {div/f_self:.3f} / "
                          f"real {div/st['F']:.3f})", flush=True)
                print(f"[m] {method} {ds}/{task} cfg={w:g} done "
                      f"({time.time()-t0:.0f}s)", flush=True)

    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.ds) + "_" + "-".join(args.task)
    out = (Path(args.out) if args.out
           else C.METRICS_DIR / f"raw_{method}_{tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[m] wrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
