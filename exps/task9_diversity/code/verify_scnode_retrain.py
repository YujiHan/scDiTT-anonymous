from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

METHOD = "scNODE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="*", default=C.DATASETS)
    ap.add_argument("--task", nargs="*", default=C.TASKS)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    device_str = C.set_gpu(args.gpu)
    import metrics_lib as ML  # noqa: E402
    from compute_baseline_metrics import find_pred
    from compute_metrics import REAL_SUBSAMPLE_TAG, baseline_run_dir, load_floor
    device = ML.get_device(device_str)

    rows = []
    for ds in args.ds:
        real = C.load_real_by_tp(ds)
        pca = C.get_pca(ds)
        for task in args.task:
            floor = load_floor(ds, task)
            repro_p = (baseline_run_dir(METHOD, ds, task, args.train_seed) /
                       "prediction_reproduce.npy")
            arch_p = find_pred(METHOD, ds, task, args.train_seed)
            if not repro_p.exists() or arch_p is None:
                print(f"[skip] {ds}/{task}: repro={repro_p.exists()} arch={arch_p}")
                continue
            repro, arch = np.load(repro_p), np.load(arch_p)
            for t in floor["test_tps"]:
                st = floor["stages"][str(t)]
                n_t = st["n"]
                Rs = ML.subsample(real[t], n_t, [0, t, REAL_SUBSAMPLE_TAG])
                Rp = ML.to_pca(Rs, pca)
                fa = ML.w2_pca(ML.to_pca(ML.subsample(arch[int(t)], n_t, [0, t, 0]),
                                         pca), Rp, device)
                fr = ML.w2_pca(ML.to_pca(ML.subsample(repro[int(t)], n_t, [0, t, 0]),
                                         pca), Rp, device)
                rows.append(dict(dataset=ds, task=task, stage=int(t), n_t=n_t,
                                 floor_real=st["F"],
                                 fid_archived=fa, fid_reproduce=fr,
                                 fid_norm_archived=fa / st["F"],
                                 fid_norm_reproduce=fr / st["F"],
                                 rel_diff=abs(fr - fa) / fa))
                print(f"[chk] {ds}/{task} tp{t}: archived={fa:.3f} "
                      f"({fa/st['F']:.3f})  reproduce={fr:.3f} "
                      f"({fr/st['F']:.3f})  rel_diff={abs(fr-fa)/fa:.3%}",
                      flush=True)

    df = pd.DataFrame(rows)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.METRICS_DIR / "scnode_retrain_check.csv", index=False)
    if len(df):
        print(f"\n[chk] fid_norm: archived mean={df.fid_norm_archived.mean():.3f}, "
              f"reproduce mean={df.fid_norm_reproduce.mean():.3f}")
        print(f"[chk] |rel_diff| median={df.rel_diff.median():.2%}  "
              f"max={df.rel_diff.max():.2%}")
    print(f"[chk] wrote {C.METRICS_DIR/'scnode_retrain_check.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
