from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # .../exps/task5_ablation/5-2_attention
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "results" / "task5_ablation" / "5-2_attention"
RUNS_DIR = OUT_ROOT / "runs"
METRIC_DIR = OUT_ROOT / "metrics"
PROBE_DIR = OUT_ROOT / "probes"
LOG_DIR = OUT_ROOT / "logs"
CLAIM_DIR = OUT_ROOT / "claims"

#     export SCDITT_CACHE_ROOT=results/task1_ts_pred
CACHE_ROOT = os.environ.get("SCDITT_CACHE_ROOT", str(OUT_ROOT / "cache"))

ARMS = ["full", "no_time", "no_attn", "no_tpe"]
DATASETS = ["ZB", "DR"]
TASKS = ["extrapolation"]
SEEDS = [42, 43, 44, 45, 46]

# One budget for everything, matching the Training Details appendix: the full
# model and every ablation arm are trained for the same number of steps, so a
# row of Table S5 differs from the row above it only by the component removed.
TOTAL_STEPS = 40000
ABLATION_STEPS = TOTAL_STEPS
OT_TYPE = "CPU"
MAX_CELLS = 2000

ARM_DESC = {
    "full":    "cell-attn + time-attn (ours)",
    "no_cell": "time-attn only (cross-cell interaction removed)",
    "no_time": "cell-attn only (severs the only cross-time conditioning path)",
    "no_attn": "both removed (per-cell per-tp MLP diffusion, the full lower bound)",
    "no_tpe":  "no_attn with the time position embedding also removed "
               "(pure FFN with no time awareness; the end of the ladder / absolute lower bound)",
}

ARM_SPECS: dict[str, tuple[bool, bool]] = {
    "full":    (True, True),
    "no_cell": (False, True),
    "no_time": (True, False),
    "no_attn": (False, False),
    "no_tpe":  (False, False),
}


def default_steps(ds: str) -> int:
    return TOTAL_STEPS


def run_dir(ds: str, task: str, seed: int, arm: str) -> Path:
    return RUNS_DIR / f"{ds.upper()}_{task}_seed{int(seed)}" / arm


def job_id(ds: str, task: str, seed: int, arm: str) -> str:
    return f"{ds.upper()}_{task}_seed{int(seed)}_{arm}"


def is_done(ds: str, task: str, seed: int, arm: str) -> bool:
    return (run_dir(ds, task, seed, arm) / "metrics.json").exists()


def all_jobs() -> list[tuple[str, str, int, str]]:
    jobs = []
    for seed in SEEDS:
        for ds in DATASETS:
            for task in TASKS:
                for arm in ARMS:
                    jobs.append((ds, task, seed, arm))
    return jobs


def build_cfg(ds: str, task: str, seed: int, arm: str,
              total_steps: int | None = None,
              cache_root: str | None = None,
              output_dir: str | Path | None = None,
              ex_steps: int | None = None) -> dict:
    """(dataset, task, seed, arm) -> the official config dict.

    Goes through src.utils.config.build_config: default.yaml -> dataset/<ds>.yaml -> task/<task>.yaml,
    so the per-dataset tuned N / peak_lr / batch_size take effect automatically and **all four arms match exactly**.
    The latents/VAE/OT paths are determined uniquely by cache_paths(ds, task, seed, ot_type, results_root) --
    all four arms share one cache.

    ex_steps: given only when task=='extrapolation_steps' (for the multi-step extrapolation decay curve);
    it decides how many trailing tps are held out.
    """
    from src.utils.config import build_config

    cache_root = str(cache_root or CACHE_ROOT).rstrip("/")
    out = Path(output_dir) if output_dir else run_dir(ds, task, seed, arm)

    args = argparse.Namespace(
        dataset=ds.lower(), task=task, ex_steps=ex_steps, ot_type=OT_TYPE,
        seed=int(seed), output_dir=str(out), results_root=cache_root,
        resume_from=None, smoke_test=False,
    )
    cfg = build_config(args, extras=None, cfg_root=str(REPO_ROOT / "configs"))
    if total_steps is None:
        total_steps = default_steps(ds)
    cfg["training"]["total_steps"] = int(total_steps)

    for key in ("h5ad_path", "latents_path", "vae_ckpt_path", "coupling_cache_dir"):
        v = cfg["dataset"].get(key)
        if v and not str(v).startswith("/"):
            cfg["dataset"][key] = str(REPO_ROOT / str(v))
    if os.environ.get("SCDITT_NUM_WORKERS"):
        cfg["data"]["num_workers"] = int(os.environ["SCDITT_NUM_WORKERS"])
    cfg["ablation"] = {"arm": arm, "exp": "5-2_attention", "ex_steps": ex_steps}
    return cfg


def cfg_paths(ds: str, task: str, seed: int, cache_root: str | None = None) -> dict:
    from src.data.splits import cache_paths
    cache_root = str(cache_root or CACHE_ROOT).rstrip("/")
    return cache_paths(ds.upper(), task, None, int(seed),
                       ot_type=OT_TYPE, results_root=cache_root)


def cache_ready(ds: str, task: str, seed: int, cache_root: str | None = None) -> bool:
    from src.data.splits import coupling_complete, vae_cache_valid
    p = cfg_paths(ds, task, seed, cache_root)
    if not vae_cache_valid(ds.upper(), task, None, int(seed),
                           vae_ckpt_path=p["vae_ckpt_path"],
                           latents_path=p["latents_path"]):
        return False
    return coupling_complete(ds.upper(), task, None, int(seed),
                             ot_type=OT_TYPE,
                             coupling_cache_dir=p["coupling_cache_dir"],
                             latents_path=p["latents_path"])


def ensure_dirs() -> None:
    for d in (RUNS_DIR, METRIC_DIR, PROBE_DIR, LOG_DIR, CLAIM_DIR):
        d.mkdir(parents=True, exist_ok=True)
