#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ARMS, DATASETS, PROBE_DIR, TASKS  # noqa: E402
from probe_common import (inpaint, load_ctx, load_model,  # noqa: E402
                          nearest_observed_tp, sample_condition)
from src.data.coupling import compute_coupling  # noqa: E402


# ---------------------------------------------------------------- CSI
def csi(P: np.ndarray, R: int) -> dict:
    if P.shape[0] < 2 or R < 2:
        return {"csi": None}
    mean_r = P.mean(axis=1)                       # [S, N, D]
    v_between = mean_r.var(axis=0, ddof=1)        # [N, D]
    v_within = P.var(axis=1, ddof=1).mean(axis=0)  # [N, D]
    v_cond = np.clip(v_between - v_within / R, 0.0, None)
    denom = v_cond.sum() + v_within.sum()
    return {
        "csi": float(v_cond.sum() / denom) if denom > 0 else 0.0,
        "var_cond": float(v_cond.sum()),
        "var_noise": float(v_within.sum()),
    }


def knn_labels(query: np.ndarray, ref: np.ndarray, ref_lab: np.ndarray,
               k: int = 15, device: str = "cuda") -> np.ndarray:
    q = torch.from_numpy(query).to(device)
    r = torch.from_numpy(ref).to(device)
    d = torch.cdist(q, r)                                   # [Nq, Nr]
    idx = d.topk(min(k, r.shape[0]), largest=False).indices.cpu().numpy()
    out = []
    for row in idx:
        labs, cnt = np.unique(ref_lab[row], return_counts=True)
        out.append(labs[cnt.argmax()])
    return np.asarray(out)


def lineage_consistency(ctx: dict, pred: np.ndarray, anchors: np.ndarray,
                        t_test: int, rng: np.random.Generator) -> dict:
    Z_t = ctx["Z_by_tp"][t_test]
    if len(Z_t) < 10:
        return {}

    eps = float(ctx["cfg"]["coupling"]["epsilon"])
    res = compute_coupling(anchors.astype(np.float32), Z_t.astype(np.float32),
                           epsilon=eps, normalize_cost=True, max_cells=2000,
                           rng=rng, ot_type="CPU_COST")
    pi = res["pi"]
    Zb = Z_t[res["idx_b"]]                           # [M', D]
    mu = pi @ Zb

    d = torch.cdist(torch.from_numpy(pred).float(),
                    torch.from_numpy(mu).float()).numpy()   # [N, N]
    N = d.shape[0]
    rank = (d < d[np.arange(N), np.arange(N)][:, None]).sum(axis=1)
    top1 = float((rank == 0).mean())
    pct = float(1.0 - rank.mean() / max(N - 1, 1))

    out = {"ot_top1": top1, "ot_top1_chance": 1.0 / N,
           "ot_pct_rank": pct, "ot_pct_rank_chance": 0.5, "N": N}

    if ctx["celltype_key"] is not None:
        ct_t = ctx["ct_by_tp"][t_test][res["idx_b"]]        # [M']
        ct_ot = ct_t[pi.argmax(axis=1)]
        ct_pred = knn_labels(pred.astype(np.float32), Zb.astype(np.float32),
                             ct_t, k=15, device=str(ctx["device"]))
        agree = float((ct_pred == ct_ot).mean())
        shuf = float(np.mean([(ct_pred == rng.permutation(ct_ot)).mean()
                              for _ in range(20)]))
        out.update({"ct_agree": agree, "ct_agree_shuffled": shuf,
                    "ct_lift": agree - shuf})
    return out


def probe_one(ds: str, task: str, seed: int, arm: str, S: int, R: int,
              device: str = "cuda") -> dict | None:
    ctx = load_ctx(ds, task, seed, arm, device=device)
    model = load_model(ctx, arm)
    if model is None:
        print(f"  [skip] {ds}/{task}/seed{seed}/{arm}: no best.pt")
        return None

    rng = np.random.default_rng(1234 + seed)
    conds = [sample_condition(ctx, rng) for _ in range(S)]

    test = ctx["test_tps"]
    preds = np.zeros((S, R, len(test), ctx["N"], ctx["D"]), dtype=np.float32)
    for s, (x_obs, _) in enumerate(conds):
        for r in range(R):
            x = inpaint(ctx, model, x_obs, noise_seed=1000 + r)
            preds[s, r] = x[0, test].cpu().numpy()

    per_tp_csi = {}
    for i, t in enumerate(test):
        per_tp_csi[int(t)] = csi(preds[:, :, i], R)
    csi_vals = [v["csi"] for v in per_tp_csi.values() if v.get("csi") is not None]

    lin = {}
    x_obs0, sel0 = conds[0]
    for i, t in enumerate(test):
        t_prev = nearest_observed_tp(ctx, t)
        if t_prev not in sel0:
            continue
        anchors = ctx["Z_by_tp"][t_prev][sel0[t_prev]]         # [N, D]
        pred_t = preds[0, :, i].mean(axis=0)
        lin[int(t)] = lineage_consistency(ctx, pred_t, anchors, int(t),
                                          np.random.default_rng(seed))

    del model
    torch.cuda.empty_cache()

    def _m(key):
        vals = [v[key] for v in lin.values() if key in v]
        return float(np.mean(vals)) if vals else None

    rec = {
        "dataset": ds, "task": task, "seed": seed, "arm": arm,
        "S": S, "R": R, "test_tps": test, "N": ctx["N"],
        "csi": float(np.mean(csi_vals)) if csi_vals else None,
        "csi_per_tp": per_tp_csi,
        "lineage": lin,
        "lineage_summary": {
            "ot_top1": _m("ot_top1"), "ot_top1_chance": _m("ot_top1_chance"),
            "ot_pct_rank": _m("ot_pct_rank"),
            "ct_agree": _m("ct_agree"), "ct_agree_shuffled": _m("ct_agree_shuffled"),
            "ct_lift": _m("ct_lift"),
        },
    }
    ls = rec["lineage_summary"]
    print(f"    CSI={rec['csi']:.4f}" if rec["csi"] is not None else "    CSI=NA",
          f"| OT-top1={ls['ot_top1']:.4f} (chance {ls['ot_top1_chance']:.4f})"
          if ls["ot_top1"] is not None else "",
          f"| pct_rank={ls['ot_pct_rank']:.3f}" if ls["ot_pct_rank"] is not None else "",
          f"| ct_lift={ls['ct_lift']:+.3f}" if ls["ct_lift"] is not None else "")
    return rec


def make_fig(records: list[dict], out_dir: Path) -> None:
    if not records:
        return
    arms = [a for a in ARMS if any(r["arm"] == a for r in records)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def _bar(ax, key, title, chance=None, sub=None):
        vals, errs = [], []
        for a in arms:
            v = [(r["lineage_summary"][key] if sub else r[key])
                 for r in records if r["arm"] == a
                 and (r["lineage_summary"][key] if sub else r[key]) is not None]
            vals.append(np.mean(v) if v else 0.0)
            errs.append(np.std(v) if len(v) > 1 else 0.0)
        ax.bar(arms, vals, yerr=errs, capsize=3,
               color=["#2c7fb8", "#7fcdbb", "#fdae61", "#d7191c"][:len(arms)])
        if chance is not None:
            ax.axhline(chance, color="gray", ls="--", lw=1, label="chance")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    _bar(axes[0], "csi", "Conditioning sensitivity (CSI)\n"
         "fraction of prediction variance explained by the condition", chance=0.0)
    ch = np.mean([r["lineage_summary"]["ot_top1_chance"] for r in records
                  if r["lineage_summary"]["ot_top1_chance"] is not None] or [0])
    _bar(axes[1], "ot_top1", "Lineage consistency:\nOT-descendant top-1 retrieval",
         chance=ch, sub=True)
    _bar(axes[2], "ct_lift", "Celltype agreement − shuffled baseline",
         chance=0.0, sub=True)
    fig.suptitle("Chain B-2: does conditioning information actually flow in? "
                 "(error bars = std over ds/task/seed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_conditioning.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--S", type=int, default=4, help="number of conditioning sets")
    ap.add_argument("--R", type=int, default=4, help="noise repeats per condition")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        for ds in args.datasets:
            for task in args.tasks:
                for arm in args.arms:
                    print(f"[probe-cond] {ds}/{task}/seed{seed}/{arm}")
                    r = probe_one(ds, task, seed, arm, args.S, args.R,
                                  device=args.device)
                    if r:
                        records.append(r)

    out = PROBE_DIR / "conditioning.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    make_fig(records, PROBE_DIR)

    print("\n=== summary (averaged across ds/task/seed) ===")
    print(f"{'arm':10s} {'CSI':>8s} {'OT-top1':>9s} {'chance':>8s} {'pct_rank':>9s} {'ct_lift':>8s}")
    for a in ARMS:
        rs = [r for r in records if r["arm"] == a]
        if not rs:
            continue
        def _m(k, sub=False):
            v = [(r["lineage_summary"][k] if sub else r[k]) for r in rs
                 if (r["lineage_summary"][k] if sub else r[k]) is not None]
            return np.mean(v) if v else float("nan")
        print(f"{a:10s} {_m('csi'):8.4f} {_m('ot_top1', True):9.4f} "
              f"{_m('ot_top1_chance', True):8.4f} {_m('ot_pct_rank', True):9.3f} "
              f"{_m('ct_lift', True):8.3f}")
    print(f"\nWritten: {out}\n         {PROBE_DIR}/fig_conditioning.png")


if __name__ == "__main__":
    main()
