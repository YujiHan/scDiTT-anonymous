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

from blocks_ablation import instrument  # noqa: E402
from common import DATASETS, PROBE_DIR, TASKS  # noqa: E402
from probe_common import (load_ctx, load_model, obs_mask_of,  # noqa: E402
                          sample_condition)
from src.diffusion.sampling import flow_euler_step  # noqa: E402


@torch.no_grad()
def inpaint_capture(ctx: dict, model, rec, x_obs: torch.Tensor,
                    noise_seed: int, capture_steps: list[int]) -> torch.Tensor:
    device, num_steps = ctx["device"], ctx["num_steps"]
    obs_mask = obs_mask_of(ctx)
    B, T, N, D = x_obs.shape

    taus = torch.linspace(0.0, 1.0, num_steps + 1, device=device).clamp(1e-4, 1 - 1e-4)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(noise_seed))
    x = torch.randn(B, T, N, D, device=device, generator=gen)
    om = obs_mask.view(B, T, 1, 1).float()
    x = x * (1.0 - om) + x_obs * om

    for i in range(num_steps):
        rec.on, rec.step = (i in capture_steps), i
        tau_cur, tau_next = taus[i].expand(B), taus[i + 1].expand(B)
        v = model(x, tau_cur, obs_mask, ctx["t_real"])
        x = flow_euler_step(x, v, tau_cur, tau_next)
        x = x * (1.0 - om) + x_obs * om
    rec.on = False
    return x


def time_stats(ctx: dict, A: np.ndarray) -> dict:
    T = ctx["T"]
    train, test = ctx["train_tps"], ctx["test_tps"]
    t_real = ctx["t_real"].cpu().numpy()

    uni_mass = len(train) / T
    out = {"mass_obs": [], "mass_nearest2": [], "dist_mean": [],
           "uniform_mass_obs": uni_mass, "uniform_dist_mean": [], "self_obs": []}

    for q in test:
        row = A[q]
        out["mass_obs"].append(float(row[train].sum()))
        near2 = sorted(train, key=lambda k: abs(t_real[k] - t_real[q]))[:2]
        out["mass_nearest2"].append(float(row[near2].sum()))
        d = np.abs(t_real - t_real[q])
        out["dist_mean"].append(float((row * d).sum()))
        out["uniform_dist_mean"].append(float(d.mean()))

    for q in train:
        out["self_obs"].append(float(A[q, q]))

    return {k: (float(np.mean(v)) if isinstance(v, list) and v else v)
            for k, v in out.items()}


def cell_stats(ctx: dict, A_by_tp: np.ndarray, sel_rows: dict,
               topk: int = 5) -> dict:
    N = ctx["N"]
    if ctx["celltype_key"] is None:
        return {"purity": None, "baseline": None, "entropy_ratio": None}

    pur, base, ent = [], [], []
    for t in ctx["train_tps"]:
        if t not in sel_rows:
            continue
        ct = ctx["ct_by_tp"][t][sel_rows[t]]
        A = A_by_tp[t].copy()                                # [N, N]
        np.fill_diagonal(A, -np.inf)
        top = np.argpartition(-A, kth=topk, axis=1)[:, :topk]  # [N, topk]
        match = (ct[top] == ct[:, None])                     # [N, topk]
        pur.append(match.mean())

        cmap = {c: n for c, n in zip(*np.unique(ct, return_counts=True))}
        base.append(np.mean([(cmap[c] - 1) / max(N - 1, 1) for c in ct]))

        P = A_by_tp[t]
        H = -(P * np.log(np.clip(P, 1e-12, None))).sum(axis=1)
        ent.append((H / np.log(N)).mean())

    return {"purity": float(np.mean(pur)), "baseline": float(np.mean(base)),
            "entropy_ratio": float(np.mean(ent))}


def probe_one(ds: str, task: str, seed: int, arm: str, capture_steps: list[int],
              device: str = "cuda") -> dict | None:
    ctx = load_ctx(ds, task, seed, arm, device=device)
    model = load_model(ctx, arm)
    if model is None:
        print(f"  [skip] {ds}/{task}/seed{seed}/{arm}: no best.pt")
        return None

    rec = instrument(model)
    rng = np.random.default_rng(ctx["cfg"]["inference"]["eval_seed"])
    x_obs, sel_rows = sample_condition(ctx, rng)
    inpaint_capture(ctx, model, rec, x_obs, noise_seed=0,
                    capture_steps=capture_steps)

    n_blocks = ctx["cfg"]["model"]["num_blocks"]
    per_block_time, per_block_cell = [], []
    time_mats = []

    for b in range(n_blocks):
        t_mats = [rec.time[(b, s)].numpy() for s in capture_steps
                  if (b, s) in rec.time]
        if t_mats:
            A = np.mean(t_mats, axis=0)                       # [T, T]
            time_mats.append(A)
            per_block_time.append({"block": b, **time_stats(ctx, A)})

        c_mats = [rec.cell[(b, s)].numpy() for s in capture_steps
                  if (b, s) in rec.cell]
        if c_mats:
            Ac = np.mean(c_mats, axis=0)                      # [T, N, N]
            per_block_cell.append({"block": b, **cell_stats(ctx, Ac, sel_rows)})

    rec.clear()
    del model
    torch.cuda.empty_cache()

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "dataset": ds, "task": task, "seed": seed, "arm": arm,
        "T": ctx["T"], "N": ctx["N"],
        "train_tps": ctx["train_tps"], "test_tps": ctx["test_tps"],
        "celltype_key": ctx["celltype_key"],
        "capture_steps": capture_steps,
        "time_attn": {
            "per_block": per_block_time,
            "mean_matrix": (np.mean(time_mats, axis=0).tolist() if time_mats else None),
            "summary": {k: _mean(per_block_time, k) for k in
                        ("mass_obs", "mass_nearest2", "dist_mean", "self_obs")},
            "uniform_mass_obs": (per_block_time[0]["uniform_mass_obs"]
                                 if per_block_time else None),
            "uniform_dist_mean": (per_block_time[0]["uniform_dist_mean"]
                                  if per_block_time else None),
        },
        "cell_attn": {
            "per_block": per_block_cell,
            "summary": {k: _mean(per_block_cell, k) for k in
                        ("purity", "baseline", "entropy_ratio")},
        },
    }


def make_figs(records: list[dict], out_dir: Path) -> None:
    full = [r for r in records if r["arm"] == "full"]
    if not full:
        return

    keys = sorted({(r["dataset"], r["task"]) for r in full})
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (ds, task) in zip(axes, keys):
        rs = [r for r in full if (r["dataset"], r["task"]) == (ds, task)
              and r["time_attn"]["mean_matrix"]]
        if not rs:
            continue
        A = np.mean([np.array(r["time_attn"]["mean_matrix"]) for r in rs], axis=0)
        im = ax.imshow(A, cmap="viridis", aspect="auto")
        for t in rs[0]["test_tps"]:
            ax.axhline(t, color="red", lw=0.8, alpha=0.6)
            ax.axvline(t, color="red", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{ds} / {task}\n(red = held-out tp)", fontsize=9)
        ax.set_xlabel("key tp"); ax.set_ylabel("query tp")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("time-attention: mean attention matrix "
                 "(full arm; averaged over blocks/steps/seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_time_attention.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ds, task in keys:
        rs = [r for r in full if (r["dataset"], r["task"]) == (ds, task)]
        if not rs:
            continue
        pb = [r["time_attn"]["per_block"] for r in rs]
        if pb and pb[0]:
            m = np.mean([[b["mass_obs"] for b in blocks] for blocks in pb], axis=0)
            axes[0].plot(m, marker="o", ms=3, label=f"{ds}/{task[:6]}")
            axes[0].axhline(rs[0]["time_attn"]["uniform_mass_obs"], color="gray",
                            ls=":", lw=1)
        cb = [r["cell_attn"]["per_block"] for r in rs]
        if cb and cb[0] and cb[0][0]["purity"] is not None:
            p = np.mean([[b["purity"] for b in blocks] for blocks in cb], axis=0)
            base = np.mean([[b["baseline"] for b in blocks] for blocks in cb])
            axes[1].plot(p, marker="o", ms=3, label=f"{ds}/{task[:6]}")
            axes[1].axhline(base, color="gray", ls=":", lw=1)
    axes[0].set_title("time-attn: attention mass on observed tps\n"
                      "(held-out queries; gray = uniform baseline)", fontsize=9)
    axes[0].set_xlabel("block"); axes[0].set_ylabel("mass on observed tps")
    axes[0].legend(fontsize=7)
    axes[1].set_title("cell-attn: top-5 same-celltype rate\n"
                      "(gray = uniform-attention baseline)", fontsize=9)
    axes[1].set_xlabel("block"); axes[1].set_ylabel("purity@5")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cell_attention.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["full"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--capture_steps", nargs="+", type=int, default=[3, 10, 17],
                    help="which diffusion steps to collect attention at (20 steps in total)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        for ds in args.datasets:
            for task in args.tasks:
                for arm in args.arms:
                    print(f"[probe-attn] {ds}/{task}/seed{seed}/{arm}")
                    r = probe_one(ds, task, seed, arm, args.capture_steps,
                                  device=args.device)
                    if r is None:
                        continue
                    records.append(r)
                    ta, ca = r["time_attn"]["summary"], r["cell_attn"]["summary"]
                    if ta["mass_obs"] is not None:
                        print(f"    time-attn: mass_obs={ta['mass_obs']:.3f} "
                              f"(uniform baseline {r['time_attn']['uniform_mass_obs']:.3f}), "
                              f"nearest2={ta['mass_nearest2']:.3f}, "
                              f"dist={ta['dist_mean']:.2f} "
                              f"(uniform {r['time_attn']['uniform_dist_mean']:.2f})")
                    if ca["purity"] is not None:
                        print(f"    cell-attn: purity@5={ca['purity']:.3f} "
                              f"(uniform baseline {ca['baseline']:.3f}), "
                              f"entropy_ratio={ca['entropy_ratio']:.3f}")

    out = PROBE_DIR / "attention_maps.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    make_figs(records, PROBE_DIR)
    print(f"\nWritten: {out}\n         {PROBE_DIR}/fig_time_attention.png"
          f"\n         {PROBE_DIR}/fig_cell_attention.png")


if __name__ == "__main__":
    main()
