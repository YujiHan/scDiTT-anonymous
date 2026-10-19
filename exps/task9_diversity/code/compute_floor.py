from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="*", default=C.DATASETS)
    ap.add_argument("--task", nargs="*", default=C.TASKS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device_str = C.set_gpu(args.gpu)
    import metrics_lib as ML  # noqa: E402
    from src.data.splits import get_tp_split

    device = ML.get_device(device_str)

    for ds in args.ds:
        real = C.load_real_by_tp(ds)
        pca = C.get_pca(ds)
        for task in args.task:
            out = C.FLOOR_DIR / ds / task / "floor.json"
            if out.exists() and not args.force:
                print(f"[skip] {out} exists")
                continue
            _, test_tps = get_tp_split(ds, task)
            rec = {"dataset": ds, "task": task,
                   "test_tps": [int(t) for t in test_tps],
                   "pca_dim": 50, "sinkhorn_blur": 0.05, "stages": {}}
            for t in test_tps:
                Rt = real[int(t)]
                N_t = len(Rt)
                n_t = C.n_t_for(N_t)
                R = ML.floor_R_for(N_t)
                t0 = time.time()
                f = ML.sampling_floor(Rt, n_t, pca, device, R=R)
                f["seconds"] = round(time.time() - t0, 1)
                rec["stages"][str(int(t))] = f
                print(f"[floor] {ds}/{task} tp{t}: N={N_t} n={n_t} "
                      f"M={f['M_blocks']} R={R} pairs={f['n_pairs']} "
                      f"F={f['F']:.4f} +- {f['F_std']:.4f} ({f['seconds']}s)",
                      flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as fh:
                json.dump(rec, fh, indent=2)
            print(f"[floor] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
