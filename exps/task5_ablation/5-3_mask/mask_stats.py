#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import common
from common import ARMS, DATASETS, METRIC_DIR, TASKS, get_split, get_T
from mask_arms import ARM_DESC, build_sampler


def _analyse_sample(obs: np.ndarray, loss: np.ndarray, T: int) -> dict:
    o_idx = np.flatnonzero(obs)
    g_idx = np.flatnonzero(loss)
    if o_idx.size == 0:
        return {"full": True}                    # CFG null-condition
    if g_idx.size == 0:
        return {"empty": True}

    o_min, o_max = int(o_idx[0]), int(o_idx[-1])
    n_width1 = n_bounded = n_trailing = 0
    horizon = 0
    gap_widths: list[int] = []
    unobs = ~obs

    for g in g_idx:
        g = int(g)
        has_left, has_right = g > o_min, g < o_max
        if has_left and has_right:
            n_bounded += 1
            if obs[g - 1] and obs[g + 1]:
                n_width1 += 1
            lo = g
            while lo - 1 >= 0 and unobs[lo - 1]:
                lo -= 1
            hi = g
            while hi + 1 < T and unobs[hi + 1]:
                hi += 1
            gap_widths.append(hi - lo + 1)
        elif not has_right:
            n_trailing += 1
            horizon = max(horizon, g - o_max)

    return {"full": False, "empty": False,
            "n_width1": n_width1, "n_bounded": n_bounded,
            "n_trailing": n_trailing, "horizon": horizon,
            "gap_widths": gap_widths}


def analyse_arm(arm: str, ds: str, task: str, n: int, seed: int = 0) -> dict:
    train_tps, test_tps = get_split(ds, task)
    T = get_T(ds)
    L = len(train_tps)
    k_test = len(test_tps)

    obs_test = np.zeros(T, dtype=bool)
    obs_test[train_tps] = True
    loss_test = np.zeros(T, dtype=bool)
    loss_test[test_tps] = True
    ref = _analyse_sample(obs_test, loss_test, T)

    sampler = build_sampler(arm, train_tps, T, rng=np.random.default_rng(seed))
    obs_base = np.zeros(T, dtype=bool)
    obs_base[train_tps] = True

    n_full = n_empty = 0
    n_has_width1 = n_has_bounded = n_has_trailing = n_horizon_ge_k = 0
    masked = np.zeros(n, dtype=np.int32)
    per_tp = np.zeros(T, dtype=np.int64)
    horizon_hist: Counter = Counter()
    width_hist: Counter = Counter()
    width1_cnt: Counter = Counter()
    pattern_hits: Counter = Counter()

    for j in range(n):
        m, pname = sampler.sample_with_name()
        pattern_hits[pname] += 1
        loss = obs_base & m
        obs = obs_base & ~m
        masked[j] = int(loss.sum())
        per_tp += loss

        r = _analyse_sample(obs, loss, T)
        if r.get("full"):
            n_full += 1
            continue
        if r.get("empty"):
            n_empty += 1
            continue
        if r["n_width1"]:
            n_has_width1 += 1
        if r["n_bounded"]:
            n_has_bounded += 1
        if r["n_trailing"]:
            n_has_trailing += 1
        if r["horizon"] >= k_test:
            n_horizon_ge_k += 1
        horizon_hist[r["horizon"]] += 1
        width1_cnt[r["n_width1"]] += 1
        for w in r["gap_widths"]:
            width_hist[w] += 1

    return {
        "arm": arm, "dataset": ds, "task": task, "n_samples": n,
        "train_tps": train_tps, "test_tps": test_tps, "T": T, "L": L, "k_test": k_test,
        "ref_test": {k: v for k, v in ref.items() if k != "gap_widths"},
        "p_full": n_full / n,
        "p_empty": n_empty / n,
        "p_width1": n_has_width1 / n,
        "p_bounded": n_has_bounded / n,
        "mean_width1_per_sample": float(sum(k * v for k, v in width1_cnt.items()) / n),
        "p_trailing": n_has_trailing / n,
        "p_horizon_ge_k": n_horizon_ge_k / n,
        "mean_horizon": float(sum(k * v for k, v in horizon_hist.items()) / n),
        "masked_frac_mean": float(masked.mean() / L),
        "masked_frac_std": float(masked.std() / L),
        "obs_count_mean": float(L - masked.mean()),
        "obs_count_at_test": L,
        "per_tp_loss_prob": {str(t): float(per_tp[t] / n) for t in train_tps},
        "horizon_hist": {str(k): v / n for k, v in sorted(horizon_hist.items())},
        "bounded_gap_width_hist": {str(k): v / n for k, v in sorted(width_hist.items())},
        "pattern_hits": {k: v / n for k, v in sorted(pattern_hits.items())},
    }


def _md(recs: list[dict]) -> str:
    L = [
        "# Stage A -- direct diagnostics of the mask distribution (pure CPU, no training)",
        "",
        "> ## ⚠️ To be clear up front: **the \"coverage\" in this file is not an evaluation metric**",
        ">",
        "> It is **purely a statistic about the mask sampler itself** -- no model, no training, no data, just calling",
        "> `MaskSampler.sample()` 200,000 times and looking at the masks it produces. Hence it runs in seconds.",
        ">",
        "> The question it answers is: **what fraction of the problems the model practises during training are the same kind as the one it faces at test time?**",
        "> (During training the loss falls only on masked `train_tps`; the model is **never** directly supervised on `test_tps` --",
        "> so the sampler's entire role is to make the model practise on train_tps a **skill transferable to test_tps**.)",
        ">",
        "> **Ablation conclusions still come only from the 5 official post-training metrics** (`w2_avg` etc., see `main_table.md`);",
        "> that does not change. This diagnostic serves as a **mechanistic explanation**: once the downstream table exists, if an arm collapses on a task",
        "> one can point at the 0 here and say \"it never practised this pattern once, so this follows by construction\" instead of guessing vaguely.",
        "> (Mirroring \"stage C: measure coupling quality directly\" in 5-1_OT -- that independent evidence chain is exactly why it is credible.)",
        ">",
        "> **Necessary but not sufficient**: coverage = 0 is a strong signal (a problem never practised is unlikely to be solved);",
        "> but a difference such as 0.27 vs 0.21 is **not enough to predict the downstream winner** -- more practice is not better learning,",
        "> and training on a single pattern may overfit to that pattern and lose generality.",
        "",
        "Decomposes every training sample on the **full time axis** (`O`=observed, `G`=loss target, `test_tps`=permanent hole), "
        "quantifying whether each arm's mask distribution covers the pattern actually required at test time.",
        "",
        "| Pattern | Definition | Who needs it |",
        "|---|---|---|",
        "| **width1** | both neighbours g-1, g+1 of loss target g are observed (hole width=1) | **interpolation**: the test tp is exactly bracketed by two observed train tps |",
        "| **horizon>=k** | at least k consecutive steps predicted after the last observation | **extrapolation**: ZB predicts 2 steps, GA predicts 3 |",
        "| **p_full** | everything masked (no observations at all) | **CFG null-condition** (needed by inference at cfg_weight=3.0) |",
        "",
    ]
    for ds in DATASETS:
        for task in TASKS:
            sub = [r for r in recs if r["dataset"] == ds and r["task"] == task]
            if not sub:
                continue
            r0 = sub[0]
            need = "width1 (interpolation)" if task == "interpolation" else \
                   f"horizon≥{r0['k_test']} (extrapolation)"
            L += [
                f"## {ds} / {task} -- the pattern this task really needs: **{need}**",
                "",
                f"`train_tps={r0['train_tps']}`  `test_tps={r0['test_tps']}`  "
                f"(T={r0['T']}, L={r0['L']}, k={r0['k_test']})",
                "",
                f"reference pattern at test time: {r0['ref_test']}",
                "",
                "| Arm | p_full<br>(CFG) | **p_width1**<br>(interp. pattern) | p_bounded<br>(loose) | "
                f"**p_horizon≥{r0['k_test']}**<br>(extrap. pattern) | mean horizon | mean mask rate | "
                f"mean observed tps<br>(at test time={r0['L']}) |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in sub:
                w1 = f"{r['p_width1']:.3f}"
                hz = f"{r['p_horizon_ge_k']:.3f}"
                if task == "interpolation":
                    w1 = f"**{w1}**" + (" 🔴" if r["p_width1"] < 0.01 else "")
                else:
                    hz = f"**{hz}**" + (" 🔴" if r["p_horizon_ge_k"] < 0.01 else "")
                L.append(
                    f"| `{r['arm']}` | {r['p_full']:.3f} | {w1} | {r['p_bounded']:.3f} | "
                    f"{hz} | {r['mean_horizon']:.2f} | {r['masked_frac_mean']:.3f} ± "
                    f"{r['masked_frac_std']:.3f} | {r['obs_count_mean']:.2f} |")
            L.append("")
    L += [
        "## How to read this",
        "",
        "1. **`p_full` is comparable across the four arms (5-8%)** -- the training budget of the CFG unconditional branch is consistent "
        "(`src/diffusion/cfg.py`: the `full` pattern carries the null-condition)."
        "This is a prerequisite for a fair comparison: downstream differences **cannot** be explained by \"one arm's CFG branch was never trained\"."
        "(`random` is slightly higher because `uniform` itself masks everything with about 2% probability.)",
        "2. **🔴 = this arm has ~0 coverage on this task**, i.e. it **never practised what it must do at test time** -- "
        "that is the mechanistic explanation of the downstream matched/mismatched difference, and it is predictable **before training**.",
        "3. **`mix` is non-zero on both patterns** -> one model covers interpolation and extrapolation at once, exactly the intent of the mixed mask.",
        "4. **Training is harder than testing**: the mean number of observed tps < L at test time -- the training context is always sparser than the test context "
        "(a deliberate curriculum).",
        "",
        "> Note: during training the loss falls only on masked **train_tps**; the model is **never** directly supervised on test_tps."
        "So the sampler's entire role is to make the model practise on train_tps a **skill transferable to test_tps**."
        "Coverage of 0 = that skill was never practised at all.",
        "",
    ]
    return "\n".join(L)


def _figure(recs: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [(ds, t) for ds in DATASETS for t in TASKS]
    colors = {"mix": "#d62728", "random": "#7f7f7f",
              "tail_only": "#1f77b4", "scatter_only": "#2ca02c"}
    fig, axes = plt.subplots(2, len(cells), figsize=(4.9 * len(cells), 8.4))

    for c, (ds, task) in enumerate(cells):
        sub = [r for r in recs if r["dataset"] == ds and r["task"] == task]
        if not sub:
            continue
        r0 = sub[0]

        ax = axes[0, c]
        for r in sub:
            tps = [int(t) for t in r["per_tp_loss_prob"]]
            ax.plot(tps, list(r["per_tp_loss_prob"].values()), marker="o", ms=4,
                    label=r["arm"], color=colors.get(r["arm"]))
        for t in r0["test_tps"]:
            ax.axvline(t, color="k", ls=":", lw=1, alpha=.6)
        ax.set_title(f"{ds} / {task}\nP(tp is a loss target); dotted = test_tps",
                     fontsize=10)
        ax.set_xlabel("time point"); ax.set_ylabel("P(masked)")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = axes[1, c]
        keys = ["p_full", "p_width1", f"p_horizon_ge_k"]
        labels = ["p_full\n(CFG null)", "p_width1\n(INTERP shape)",
                  f"p_horizon>={r0['k_test']}\n(EXTRAP shape)"]
        x = np.arange(len(keys)); w = 0.2
        for i, r in enumerate(sub):
            ax.bar(x + (i - 1.5) * w, [r[k] for k in keys], w, label=r["arm"],
                   color=colors.get(r["arm"]))
        need = 1 if task == "interpolation" else 2
        ax.axvspan(need - .5, need + .5, color="gold", alpha=.18, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("probability"); ax.set_ylim(0, 1.02)
        ax.set_title("shape coverage (gold = what THIS task needs)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="direct diagnostics of the mask distribution (stage A)")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=ARMS)
    args = p.parse_args()

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    for ds in DATASETS:
        for task in TASKS:
            print(f"--- {ds} / {task} "
                  f"(requires: {'width1' if task == 'interpolation' else 'horizon>=k'})")
            for arm in args.arms:
                r = analyse_arm(arm, ds, task, n=args.n, seed=args.seed)
                recs.append(r)
                flag_i = " 🔴" if r["p_width1"] < 0.01 else "  "
                flag_e = " 🔴" if r["p_horizon_ge_k"] < 0.01 else "  "
                print(f"  {arm:13s} full={r['p_full']:.3f}  "
                      f"width1={r['p_width1']:.3f}{flag_i}  "
                      f"horizon>={r['k_test']}={r['p_horizon_ge_k']:.3f}{flag_e}  "
                      f"mean_horizon={r['mean_horizon']:.2f}  "
                      f"masked={r['masked_frac_mean']:.2f}")

    (METRIC_DIR / "mask_stats.json").write_text(
        json.dumps({"arms": {a: ARM_DESC[a] for a in args.arms}, "records": recs},
                   ensure_ascii=False, indent=2))
    (METRIC_DIR / "mask_stats.md").write_text(_md(recs))
    try:
        _figure(recs, METRIC_DIR / "fig_mask_stats.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    print(f"\n-> {METRIC_DIR/'mask_stats.json'}")
    print(f"-> {METRIC_DIR/'mask_stats.md'}")
    print(f"-> {METRIC_DIR/'fig_mask_stats.png'}")


if __name__ == "__main__":
    main()
