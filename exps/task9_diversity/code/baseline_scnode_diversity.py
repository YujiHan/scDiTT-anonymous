from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_BRUNS = _EXP / "baseline-code" / "baseline_runs"

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("task9_common", _HERE / "common.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["task9_common"] = C
_spec.loader.exec_module(C)

sys.path.insert(0, str(_BRUNS / "scNODE_model" / "scNODE"))   # -> model, optim
sys.path.insert(0, str(_BRUNS))                               # -> common, configs

METHOD = "scNODE"
DATA_ROOT = C.ARCHIVE_ROOT
SELF_FLOOR_CELLS = 9216


def out_root(ds: str, task: str, train_seed: int) -> Path:
    return C.RUNS_DIR / "baseline" / METHOD / ds / task / f"trainseed{train_seed}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--pred_seeds", nargs="*", type=int, default=C.EVAL_SEEDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = out_root(args.ds, args.task, args.train_seed)
    if (out / "meta.json").exists() and not args.force:
        print(f"[skip] {out} done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    torch.distributions.Distribution.set_default_validate_args(False)
    from common import seeds as seedmod, splits as bsplits
    from common.reduce import sample_initial
    from common.sc_dataset import load_h5ad
    import configs
    from optim.running import (constructscNODEModel, scNODEPredict,
                               scNODETrainWithPreTrain)

    cfg = configs.get(METHOD)
    p = cfg["params"]
    n_sim = cfg["n_sim_cells"]

    ds = load_h5ad(args.ds, data_root=str(DATA_ROOT))
    train_tps, test_tps = bsplits.get_split(ds.code, args.task, n_tps=ds.n_tps)
    all_tps = list(range(ds.n_tps))
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[scNODE] {args.ds}/{args.task} seed{args.train_seed} on {dev} | "
          f"train_tps={train_tps} test_tps={test_tps} n_genes={ds.n_genes}",
          flush=True)

    seedmod.set_seeds(args.train_seed)
    train_data = [torch.FloatTensor(ds.cells_at(t)).to(dev) for t in train_tps]
    train_tps_t = torch.FloatTensor(train_tps)
    model = constructscNODEModel(
        ds.n_genes, latent_dim=p["latent_dim"],
        enc_latent_list=p["enc_latent_list"], dec_latent_list=p["dec_latent_list"],
        drift_latent_size=p["drift_latent_size"],
        latent_enc_act=p["latent_enc_act"], latent_dec_act=p["latent_dec_act"],
        drift_act=p["drift_act"], ode_method=p["ode_method"],
    ).to(dev)
    t0 = time.time()
    model, loss_list, *_, best_state = scNODETrainWithPreTrain(
        train_data, train_tps_t, model, latent_coeff=p["latent_coeff"],
        epochs=p["epochs"], iters=p["iters"], batch_size=p["batch_size"],
        lr=p["lr"], pretrain_iters=p["pretrain_iters"],
        pretrain_lr=p["pretrain_lr"], kl_coeff=p["kl_coeff"])
    print(f"[scNODE] trained in {time.time()-t0:.0f}s, "
          f"final loss={loss_list[-1][0]:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    t0_tp = train_tps[0]
    tps_t = torch.FloatTensor(all_tps)

    def predict_with_seed(s: int, n_cells: int | None = None) -> np.ndarray:
        n_cells = n_sim if n_cells is None else n_cells
        init = sample_initial(ds.cells_at(t0_tp), 2 * n_cells, seed=s)
        first = torch.FloatTensor(init).to(dev)
        seedmod.set_seeds(s)
        pred = scNODEPredict(model, first, tps_t, n_cells=n_cells)  # (N, T, G)
        return np.ascontiguousarray(np.transpose(pred, (1, 0, 2))).astype(np.float32)

    repro = predict_with_seed(args.train_seed)
    arch = None
    for s in ("A", "B", "C"):
        q = (C.BASELINE_ROOT / s / "baseline_results" / METHOD / args.ds /
             args.task / f"seed{args.train_seed}" / "prediction.npy")
        if q.exists():
            arch = np.load(q)
            break
    repro_err = None
    if arch is not None and arch.shape == repro.shape:
        repro_err = {
            "max_abs": float(np.abs(arch - repro).max()),
            "rel_fro": float(np.linalg.norm(arch - repro) /
                             max(np.linalg.norm(arch), 1e-12)),
            "archived": str(q),
        }
        print(f"[scNODE] reproduce vs archived: max|Δ|={repro_err['max_abs']:.4e} "
              f"rel_fro={repro_err['rel_fro']:.4e}", flush=True)
    else:
        print("[scNODE] the archived prediction.npy is missing or has the wrong shape; skipping the reproduction check")
    np.save(out / "prediction_reproduce.npy", repro)

    sf = predict_with_seed(9999, n_cells=SELF_FLOOR_CELLS)
    sf_arr = np.stack([sf[int(t)] for t in test_tps], axis=0).astype(np.float32)
    np.savez_compressed(out / "selffloor_preds.npz", pred=sf_arr,
                        tps=np.asarray(test_tps, dtype=np.int64))
    print(f"[scNODE] self-floor pool shape={sf_arr.shape}", flush=True)

    for e in args.pred_seeds:
        pr = predict_with_seed(e)
        d = out / f"predseed{e}"
        d.mkdir(parents=True, exist_ok=True)
        arr = np.stack([pr[int(t)] for t in test_tps], axis=0).astype(np.float32)
        assert arr.shape[-1] > 1000, "gene dim <= 1000"
        np.savez_compressed(d / "preds.npz", pred=arr,
                            tps=np.asarray(test_tps, dtype=np.int64))
        print(f"[scNODE] predseed={e} shape={arr.shape}", flush=True)

    with open(out / "meta.json", "w") as fh:
        json.dump({
            "method": METHOD, "dataset": args.ds, "task": args.task,
            "train_seed": args.train_seed, "pred_seeds": args.pred_seeds,
            "n_sim_cells": n_sim, "n_genes": int(ds.n_genes),
            "self_floor_cells": SELF_FLOOR_CELLS, "self_floor_seed": 9999,
            "train_tps": train_tps, "test_tps": test_tps,
            "params": p, "device": str(dev),
            "weights": "best (min train loss)",
            "local_changes": [
                "predict seed decoupled from train seed",
                "torch.distributions validate_args disabled (assert-only, the "
                "Normal object never enters the loss; see code comment)",
            ],
            "reproduce_vs_archived": repro_err,
            "randomness_sources": [
                "initial-cell resampling: sample_initial(t0, 2*n_sim, seed) with replacement",
                "VAE latent draw: dynamic_model.predict -> _sampleGaussian(mu, sigma)",
            ],
        }, fh, indent=2)
    print(f"[scNODE] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
