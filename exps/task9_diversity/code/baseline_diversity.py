from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
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

_REPO = str(C.REPO_ROOT)
while _REPO in sys.path:
    sys.path.remove(_REPO)

EPOCHS = 1000
SELF_FLOOR_CELLS = 9216
SELF_FLOOR_SEED = 9999

METHODS = ["PRESCIENT", "PI-SDE", "scIMF", "scDiffEq", "MIOFlow"]
INNER = {"PRESCIENT": "prescient_model", "PI-SDE": "PI-SDE_model",
         "scIMF": "scIMF_model", "scDiffEq": "scDiffEq_model",
         "MIOFlow": "MIOFlow_model"}


def out_root(method: str, ds: str, task: str, train_seed: int) -> Path:
    return C.RUNS_DIR / "baseline" / method / ds / task / f"trainseed{train_seed}"


# ===========================================================================
# ===========================================================================
def build_prescient(ds, task, seed, dev, cfg, train_tps, test_tps, all_tps):
    import prescient_helpers as H
    from common import seeds as seedmod

    p = dict(cfg["params"])
    p["train_epochs"] = EPOCHS
    train_mask = np.isin(ds.tp, train_tps)
    expr_train = np.ascontiguousarray(ds.X[train_mask]).astype(np.float32)
    tp_train = ds.tp[train_mask].astype(int)
    celltype = np.repeat("NA", expr_train.shape[0])
    data_dict, scaler, pca = H.build_data_dict(
        expr_train, tp_train, ds.genes, celltype, train_tps,
        num_pcs=p["k_dim"], seed=seed)
    tmp = Path(tempfile.mkdtemp(prefix=f"prescient_div_{ds.code}_"))
    config = H.make_config(seed, p, str(tmp))
    model, best_state, config, loss_list = H.train(
        data_dict, config, train_tps, train_batch=p["train_batch"], gpu=0)
    print(f"[train] PRESCIENT {len(loss_list)} epochs, "
          f"best_epoch={best_state.get('epoch')}", flush=True)

    n_tps = len(all_tps)

    def predict(s, n_cells, tps):
        import torch
        from prescient.train.model import AutoGenerator
        from prescient.simulate.sim import simulate
        net = AutoGenerator(config)
        net.load_state_dict(best_state["model_state_dict"])
        net.to(dev)
        xp_t0 = data_dict["xp"][0].detach().cpu().numpy()
        rng = np.random.default_rng(s)
        idx = rng.choice(len(xp_t0), n_cells, replace=True)
        xp_init = np.ascontiguousarray(xp_t0[idx]).astype(np.float32)
        num_steps = (n_tps - 1) * 10
        sim = simulate(xp_init, np.zeros(n_cells), np.zeros(n_cells), None, net,
                       config, 1, n_cells, num_steps, dev, None, None)[0]
        return {int(t): scaler.inverse_transform(
                    pca.inverse_transform(sim[int(t) * 10])).astype(np.float32)
                for t in tps}

    return predict, lambda: shutil.rmtree(tmp, ignore_errors=True), \
        {"train_epochs": EPOCHS, "k_dim": p["k_dim"]}


def _track_a_common(ds, train_tps, n_pcs, seed):
    from common.reduce import fit_reducer
    train_mask = np.isin(ds.tp, train_tps)
    reducer = fit_reducer(ds.X[train_mask], n_pcs=n_pcs, seed=seed)
    return reducer, train_mask


def build_pisde(ds, task, seed, dev, cfg, train_tps, test_tps, all_tps):
    import pisde_helpers as H
    n_pcs = cfg["n_pcs"]
    reducer, _ = _track_a_common(ds, train_tps, n_pcs, seed)
    Z_by_tp = {t: reducer.transform(ds.cells_at(t)) for t in all_tps}

    tmp = Path(tempfile.mkdtemp(prefix=f"pisde_div_{ds.code}_"))
    data_pt = tmp / "data.pt"
    H.build_data_pt(Z_by_tp, all_tps, str(data_pt))
    args = H.build_args(seed, ds.code, str(data_pt), str(tmp / "out"),
                        len(all_tps), device_int=0, use_cuda=(dev.type == "cuda"))
    args.train_epochs = EPOCHS
    from src.train import run_leaveout
    from src.config_Veres import init_config
    config = run_leaveout(args, init_config, leaveouts=list(test_tps))
    best_pt = config.train_pt.format("best")
    print(f"[train] PI-SDE run_leaveout done (train_epochs={EPOCHS})", flush=True)

    def predict(s, n_cells, tps):
        pca_pred = H.predict(config, best_pt, Z_by_tp[0], all_tps, n_cells, dev, s)
        return {int(t): reducer.inverse(pca_pred[int(t)]).astype(np.float32)
                for t in tps}

    return predict, lambda: shutil.rmtree(tmp, ignore_errors=True), \
        {"train_epochs": EPOCHS, "n_pcs": n_pcs}


def build_scimf(ds, task, seed, dev, cfg, train_tps, test_tps, all_tps):
    import torch
    import scimf_helpers as H
    n_pcs = cfg["n_pcs"]
    reducer, _ = _track_a_common(ds, train_tps, n_pcs, seed)
    Z_by_tp = {t: reducer.transform(ds.cells_at(t)) for t in all_tps}

    tmp = Path(tempfile.mkdtemp(prefix=f"scimf_div_{ds.code}_"))
    data_list = [torch.FloatTensor(np.ascontiguousarray(Z_by_tp[t])) for t in all_tps]
    args = H.build_args(seed, ds.code, str(tmp), device_num=0,
                        cuda=(dev.type == "cuda"))
    args.train_epochs = EPOCHS
    import train as scimf_train
    scimf_train.load_data = H.make_load_data(data_list, len(all_tps), str(tmp / "out"))
    config = scimf_train.run(args, leaveouts=list(test_tps))
    best_pt = config.train_pt.format("best")
    print(f"[train] scIMF train.run done (train_epochs={EPOCHS})", flush=True)

    def predict(s, n_cells, tps):
        pca_pred = H.predict(config, best_pt, Z_by_tp[0], all_tps, n_cells, dev, s)
        return {int(t): reducer.inverse(pca_pred[int(t)]).astype(np.float32)
                for t in tps}

    return predict, lambda: shutil.rmtree(tmp, ignore_errors=True), \
        {"train_epochs": EPOCHS, "n_pcs": n_pcs}


def build_scdiffeq(ds, task, seed, dev, cfg, train_tps, test_tps, all_tps):
    import copy
    import lightning
    import scdiffeq_helpers as H

    p = dict(cfg["params"])
    p["train_epochs"] = EPOCHS
    n_pcs = cfg["n_pcs"]
    if int(p["latent_dim"]) != int(n_pcs):
        raise ValueError("scDiffEq VAE-off requires latent_dim == n_pcs")

    class _LossLogCB(lightning.pytorch.callbacks.Callback):
        def __init__(self, total):
            super().__init__()
            self.total, self.best_loss, self.best_state = int(total), float("inf"), None
            self._ep = []

        def on_train_batch_end(self, tr, pl, outputs, batch, idx):
            try:
                sl = getattr(pl, "sinkhorn_loss", None)
                if sl is not None:
                    self._ep.append(float(sl.sum().item()))
            except Exception:
                pass

        def on_train_epoch_end(self, tr, pl):
            try:
                if self._ep:
                    v = sum(self._ep) / len(self._ep)
                else:
                    tl = getattr(pl, "total_loss", None)
                    v = float(tl.sum().item()) if tl is not None else float("inf")
                self._ep = []
                if v < self.best_loss:
                    self.best_loss, self.best_state = v, copy.deepcopy(pl.state_dict())
                k = tr.current_epoch + 1
                if k == 1 or k == self.total or k % max(1, self.total // 50) == 0:
                    print(f"[loss] step={k}/{self.total} loss={v:.4f}", flush=True)
            except Exception:
                pass

    reducer, train_mask = _track_a_common(ds, train_tps, n_pcs, seed)
    X_pca_train = reducer.transform(ds.X[train_mask])
    tp_train = ds.tp[train_mask].astype(int)
    Z_t0 = reducer.transform(ds.cells_at(0))

    tmp = Path(tempfile.mkdtemp(prefix=f"scdiffeq_div_{ds.code}_"))
    cwd0 = os.getcwd()
    adata = H.build_adata(ds.X[train_mask], X_pca_train, tp_train, ds.genes)
    os.chdir(tmp)
    try:
        model = H.build_model(adata, p, seed, working_dir=tmp)
        cb = _LossLogCB(EPOCHS)
        model.fit(train_epochs=EPOCHS, pretrain_epochs=int(p["pretrain_epochs"]),
                  accelerator=("gpu" if dev.type == "cuda" else "cpu"), devices=1,
                  train_callbacks=[cb])
    finally:
        os.chdir(cwd0)
    if cb.best_state is not None:
        model.DiffEq.load_state_dict(cb.best_state)
    print(f"[train] scDiffEq fit done (train_epochs={EPOCHS}, "
          f"best_loss={cb.best_loss:.4f})", flush=True)

    def predict(s, n_cells, tps):
        pca_pred = H.predict(model, Z_t0, all_tps, n_cells, dev, s)
        return {int(t): reducer.inverse(pca_pred[int(t)]).astype(np.float32)
                for t in tps}

    return predict, lambda: shutil.rmtree(tmp, ignore_errors=True), \
        {"train_epochs": EPOCHS, "n_pcs": n_pcs, "best_loss": float(cb.best_loss)}


def build_mioflow(ds, task, seed, dev, cfg, train_tps, test_tps, all_tps):
    import mioflow_helpers as H
    from mioflow.mioflow import MIOFlow

    p = dict(cfg["params"])
    p["n_epochs"] = EPOCHS
    n_pcs = cfg["n_pcs"]
    use_cuda = (dev.type == "cuda")
    dev_str = "cuda" if use_cuda else "cpu"
    reducer, train_mask = _track_a_common(ds, train_tps, n_pcs, seed)
    X_pca_train = reducer.transform(ds.X[train_mask])
    samples = ds.tp[train_mask].astype(int)
    uniq_train = sorted(set(int(t) for t in train_tps))

    tmp = Path(tempfile.mkdtemp(prefix=f"mioflow_div_{ds.code}_"))
    gaga = H.fit_gaga_with_phate(
        X_pca_train, latent_dim=p["gaga_latent_dim"], hidden_dims=p["gaga_encoder"],
        enc_ep=p["gaga_epochs_enc"], dec_ep=p["gaga_epochs_dec"],
        lr=p["gaga_lr"], batch=p["gaga_batch"], device=dev_str, seed=seed)
    adata = H.build_adata(ds.X[train_mask], X_pca_train, samples, ds.genes)
    mf = MIOFlow(
        adata, gaga_model=gaga, gaga_input_key="X_pca", obs_time_key="day",
        debug_level="info", hidden_dim=p["ode_hidden"], use_cuda=use_cuda,
        momentum_beta=0.0, n_epochs=p["n_epochs"], lambda_ot=p["lambda_ot"],
        use_density_loss=False, lambda_energy=p["lambda_energy"], energy_time_steps=10,
        learning_rate=p["lr"], sample_size=p["sample_size"], grad_clip=p["grad_clip"],
        n_bins=p["n_bins"], n_trajectories=p["n_trajectories"], exp_dir=str(tmp))
    mf.fit()
    _best = getattr(mf.ode_model, "_best_state", None)
    if _best is not None:
        mf.ode_model.load_state_dict(_best)
    print(f"[train] MIOFlow fit done (n_epochs={EPOCHS}, "
          f"best_state={'yes' if _best is not None else 'no'})", flush=True)

    def predict(s, n_cells, tps):
        full = H.predict(mf, reducer, all_tps, uniq_train, n_cells, s)  # (T, N, G)
        return {int(t): full[int(t)].astype(np.float32) for t in tps}

    return predict, lambda: shutil.rmtree(tmp, ignore_errors=True), \
        {"n_epochs": EPOCHS, "n_pcs": n_pcs}


BUILDERS = {"PRESCIENT": build_prescient, "PI-SDE": build_pisde,
            "scIMF": build_scimf, "scDiffEq": build_scdiffeq,
            "MIOFlow": build_mioflow}


def main() -> int:
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--task", required=True,
                    choices=["interpolation", "extrapolation"])
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--pred_seeds", nargs="*", type=int, default=C.EVAL_SEEDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--self_floor_cells", type=int, default=SELF_FLOOR_CELLS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    EPOCHS = args.epochs

    out = out_root(args.method, args.ds, args.task, args.train_seed)
    if (out / "meta.json").exists() and not args.force:
        print(f"[skip] {out} done")
        return 0

    if str(args.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = (args.device.split(":")[1]
                                              if ":" in args.device else "0")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(_BRUNS / INNER[args.method]))     # -> <m>_helpers
    if args.method == "PI-SDE":
        sys.path.insert(0, str(_BRUNS / "PI-SDE_model" / "PI-SDE"))    # -> src
    elif args.method == "PRESCIENT":
        sys.path.insert(0, str(_BRUNS / "prescient_model" / "prescient"))
    sys.path.insert(0, str(_BRUNS))                          # -> common, configs
    if args.method == "scIMF":
        import scimf_helpers as _SH
        _SH.setup_imports(str(_BRUNS / "scIMF_model" / "scIMF"))

    import torch
    from common import seeds as seedmod, splits as bsplits
    from common.sc_dataset import load_h5ad
    import configs

    cfg = configs.get(args.method)
    n_sim = cfg["n_sim_cells"]
    ds = load_h5ad(args.ds, data_root=str(C.ARCHIVE_ROOT))
    train_tps, test_tps = bsplits.get_split(ds.code, args.task, n_tps=ds.n_tps)
    all_tps = list(range(ds.n_tps))
    dev = torch.device("cuda:0" if (str(args.device).startswith("cuda")
                                    and torch.cuda.is_available()) else "cpu")
    print(f"[{args.method}] {args.ds}/{args.task} seed{args.train_seed} on {dev} | "
          f"train_tps={train_tps} test_tps={test_tps} n_genes={ds.n_genes} "
          f"epochs={EPOCHS}", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    seedmod.set_seeds(args.train_seed)
    t0 = time.time()
    predict, cleanup, info = BUILDERS[args.method](
        ds, args.task, args.train_seed, dev, cfg, train_tps, test_tps, all_tps)
    train_s = time.time() - t0
    print(f"[{args.method}] trained in {train_s:.0f}s", flush=True)

    try:
        def run_predict(s, n_cells, tps):
            seedmod.set_seeds(s)
            return predict(s, n_cells, tps)

        repro = run_predict(args.train_seed, n_sim, all_tps)
        repro_arr = np.stack([repro[int(t)] for t in all_tps]).astype(np.float32)
        np.save(out / "prediction_reproduce.npy", repro_arr)
        repro_err = None
        for srv in ("A", "B", "C"):
            q = (C.BASELINE_ROOT / srv / "baseline_results" / args.method /
                 args.ds / args.task / f"seed{args.train_seed}" / "prediction.npy")
            if q.exists():
                arch = np.load(q)
                if arch.shape == repro_arr.shape:
                    repro_err = {
                        "max_abs": float(np.abs(arch - repro_arr).max()),
                        "rel_fro": float(np.linalg.norm(arch - repro_arr) /
                                         max(np.linalg.norm(arch), 1e-12)),
                        "archived": str(q),
                        "note": "epochs differ from the archive (1000 vs the method default), not comparable",
                    }
                    print(f"[{args.method}] vs archived: "
                          f"rel_fro={repro_err['rel_fro']:.4f} (different epochs, recorded only)",
                          flush=True)
                break
        del repro, repro_arr

        sf = run_predict(SELF_FLOOR_SEED, args.self_floor_cells, test_tps)
        sf_arr = np.stack([sf[int(t)] for t in test_tps]).astype(np.float32)
        assert sf_arr.shape[-1] > 1000, "gene dim <= 1000"
        np.savez_compressed(out / "selffloor_preds.npz", pred=sf_arr,
                            tps=np.asarray(test_tps, dtype=np.int64))
        print(f"[{args.method}] self-floor pool {sf_arr.shape}", flush=True)
        del sf, sf_arr

        for e in args.pred_seeds:
            pr = run_predict(e, n_sim, test_tps)
            arr = np.stack([pr[int(t)] for t in test_tps]).astype(np.float32)
            d = out / f"predseed{e}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "preds.npz", pred=arr,
                                tps=np.asarray(test_tps, dtype=np.int64))
            print(f"[{args.method}] predseed={e} {arr.shape}", flush=True)
    finally:
        cleanup()

    with open(out / "meta.json", "w") as fh:
        json.dump({
            "method": args.method, "dataset": args.ds, "task": args.task,
            "train_seed": args.train_seed, "pred_seeds": args.pred_seeds,
            "epochs": EPOCHS, "n_sim_cells": n_sim, "n_genes": int(ds.n_genes),
            "self_floor_cells": args.self_floor_cells,
            "self_floor_seed": SELF_FLOOR_SEED,
            "train_tps": train_tps, "test_tps": test_tps,
            "train_seconds": round(train_s, 1), "device": str(dev),
            "weights": "best", "build_info": info,
            "local_changes": [
                "predict seed decoupled from train seed",
                f"epochs fixed to {EPOCHS} for all methods (for comparability)",
            ],
            "reproduce_vs_archived": repro_err,
        }, fh, indent=2)
    print(f"[{args.method}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
