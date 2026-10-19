#!/usr/bin/env python
"""UMAP visualisation of the baseline comparison methods on the task1 time-series prediction task
(one method / one random seed).

The same two-panel figure as the scDiTT version (``umap_visualize.py``): Ground Truth on the left, the
method name on the right, with **interpolation + extrapolation** test time points drawn in one figure and
all real cells as a grey background.
**The same embedding cache is reused** (``_embedding_cache/<DS>_umap_embedding.joblib``), so the grey
background and the Ground Truth panel are **point-for-point identical** between the baseline figures and
the scDiTT figures, making them directly comparable.

Baseline results are scattered across several read-only directories, and some failed seeds are not in the
method's own directory but in the final rerun directory ``baseline_standard/baseline_results``. For every
``(method, dataset, task, seed)`` this script checks both directories and **prefers the status==ok result
from the rerun directory**, falling back to the ok result in the method's own directory
(``--dry-run`` prints just this provenance table).

Data sources (read-only, never modified)
-------------------------
  the method's own directory (own):
    A .../baseline_results/A/baseline_results/{PI-SDE,scIMF,scNODE}
    B .../baseline_results/B/baseline_results/MIOFlow
    C .../baseline_results/C/baseline_results/{PRESCIENT,scDiffEq}
  rerun directory (redone, preferred):
    .../baseline_standard/baseline_results/<method>
  each seed directory: <root>/<method>/<DS>/<task>/seed<N>/prediction.npy   (T, n_sim, n_genes)
                       <root>/<method>/<DS>/<task>/seed<N>/meta.json        (test_tps / status / n_genes)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umap_visualize import (  # noqa: E402
    PROJ_ROOT,
    DEFAULT_OUT_DIR,
    build_or_load_embedding,
    project_pred_test_cells,
    plot_seed_figure,
    tp_color_list,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
A = Path("")
B = Path("")
C = Path("")
REDONE = Path("")
OWN_PARENT = {
    "PI-SDE": A, "scIMF": A, "scNODE": A,
    "MIOFlow": B,
    "PRESCIENT": C, "scDiffEq": C,
}

ALL_METHODS = ["PI-SDE", "scIMF", "scNODE", "MIOFlow", "PRESCIENT", "scDiffEq"]
ALL_DATASETS = ["ZB", "DR", "GA", "MA", "SC"]
ALL_SEEDS = [42, 43, 44, 45, 46]

INTERP_TASK = "interpolation"
EXTRAP_TASK = "extrapolation"
PRED_FILE = "prediction.npy"

H5AD = {
    "ZB": ("Data/ZB/zebrafish_hvg_annotated.h5ad", "tp"),
    "DR": ("Data/DR/DR_Calderon_2022_hvg_annotated_subsampled.h5ad", "tp"),
    "GA": ("Data/GA/garcia_alonso.h5ad", "tp"),
    "MA": ("Data/MA/ma_forecast.h5ad", "tp"),
    "SC": ("Data/SC/schiebinger_2019_preprocess_subset.h5ad", "tp"),
}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _dir_ok(d: Path) -> bool:
    mp, pp = d / "meta.json", d / PRED_FILE
    if not mp.exists() or not pp.exists():
        return False
    try:
        return json.load(open(mp)).get("status") == "ok"
    except Exception:
        return False


def resolve_seed_dir(method: str, ds: str, task: str, seed: int):
    red = REDONE / method / ds / task / f"seed{seed}"
    own = OWN_PARENT[method] / method / ds / task / f"seed{seed}"
    if _dir_ok(red):
        return red, "REDONE"
    if _dir_ok(own):
        return own, "own"
    return None, None


def read_test_tps(seed_dir: Path):
    m = seed_dir / "metrics.json"
    if m.exists():
        try:
            j = json.load(open(m))
            if j.get("eval_tps"):
                return list(j["eval_tps"])
        except Exception:
            pass
    return list(json.load(open(seed_dir / "meta.json"))["test_tps"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--refit", action="store_true",
                    help="force refitting the embedding (the default reuses the cache -> background consistent across models)")
    ap.add_argument("--gray-size", type=float, default=6.0)
    ap.add_argument("--color-size", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="only print the (method,ds,task,seed) -> source directory table, do not plot")
    args = ap.parse_args()

    cache_dir = args.out_dir / "_embedding_cache"

    print("=" * 78, flush=True)
    print("source-directory check (REDONE=taken from the rerun directory baseline_standard, own=the method's own directory):", flush=True)
    n_redone = n_own = n_missing = 0
    plan = {}
    for method in args.methods:
        for ds in args.datasets:
            for seed in args.seeds:
                i_dir, i_src = resolve_seed_dir(method, ds, INTERP_TASK, seed)
                e_dir, e_src = resolve_seed_dir(method, ds, EXTRAP_TASK, seed)
                for task, src in (("interp", i_src), ("extrap", e_src)):
                    if src == "REDONE":
                        n_redone += 1
                    elif src == "own":
                        n_own += 1
                    else:
                        n_missing += 1
                if i_dir is None or e_dir is None:
                    print(f"  [MISSING] {method:9s} {ds} seed{seed}: "
                          f"interp={i_src} extrap={e_src}", flush=True)
                    continue
                if i_src == "REDONE" or e_src == "REDONE":
                    print(f"  [REDONE ] {method:9s} {ds} seed{seed}: "
                          f"interp={i_src:6s} extrap={e_src}", flush=True)
                plan[(method, ds, seed)] = (i_dir, e_dir)
    print("-" * 78, flush=True)
    print(f"task slot count (2 per combo: interp+extrap): "
          f"own={n_own} REDONE={n_redone} MISSING={n_missing}", flush=True)
    print(f"plottable combos (both tasks ok): {len(plan)}", flush=True)
    print("=" * 78, flush=True)
    if args.dry_run:
        return

    for ds in args.datasets:
        ds_combos = [(m, s) for (m, d, s) in plan if d == ds]
        if not ds_combos:
            continue
        rel, time_key = H5AD[ds]
        embed = build_or_load_embedding(
            ds, PROJ_ROOT / rel, time_key, cache_dir=cache_dir, refit=args.refit,
        )
        colors = tp_color_list(embed["all_tps"])
        n_genes = embed["n_genes"]

        for method in args.methods:
            for seed in args.seeds:
                if (method, ds, seed) not in plan:
                    continue
                i_dir, e_dir = plan[(method, ds, seed)]
                i_tps = read_test_tps(i_dir)
                e_tps = read_test_tps(e_dir)
                extrap_only = [t for t in e_tps if t not in set(i_tps)]
                overlap = sorted(set(i_tps) & set(e_tps))
                if overlap:
                    print(f"[warn] {method} {ds} seed{seed}: tp {overlap} interpolation/extrapolation overlap, "
                          f"taking interpolation; extrapolation only contributes {extrap_only}", flush=True)
                gt_test_tps = sorted(set(i_tps) | set(e_tps))

                pred_points = {}
                pred_points.update(
                    project_pred_test_cells(embed, i_dir / PRED_FILE, i_tps, n_genes))
                pred_points.update(
                    project_pred_test_cells(embed, e_dir / PRED_FILE, extrap_only, n_genes))

                ds_out = args.out_dir / ds
                ds_out.mkdir(parents=True, exist_ok=True)
                out_path = ds_out / f"{ds}_{method}_seed{seed}.png"
                plot_seed_figure(
                    embed, pred_points, gt_test_tps, method, ds, seed,
                    colors, out_path, gray_s=args.gray_size, color_s=args.color_size,
                    dpi=args.dpi,
                )

    print("[main] done.", flush=True)


if __name__ == "__main__":
    main()
