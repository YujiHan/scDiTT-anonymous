"""Config loader: default -> dataset -> task -> experiment -> CLI overrides."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..data.splits import cache_paths, task_tag


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _coerce(val: str) -> Any:
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def apply_dot_override(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def build_config(
    args: argparse.Namespace,
    extras: list[str] | None = None,
    cfg_root: str | Path = "configs",
) -> dict:
    cfg_root = Path(cfg_root)
    cfg = load_yaml(cfg_root / "default.yaml")

    ds = args.dataset.lower()
    task = args.task

    cfg = deep_merge(cfg, load_yaml(cfg_root / "dataset" / f"{ds}.yaml"))
    cfg = deep_merge(cfg, load_yaml(cfg_root / "task" / f"{task}.yaml"))

    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "output_dir", None) is not None:
        cfg["output_dir"] = args.output_dir
    if getattr(args, "resume_from", None) is not None:
        cfg["resume_from"] = args.resume_from

    cfg.setdefault("task", {})
    if getattr(args, "ex_steps", None) is not None:
        cfg["task"]["ex_steps"] = args.ex_steps
    if cfg["task"].get("name") == "extrapolation_steps" \
            and cfg["task"].get("ex_steps") is None:
        raise ValueError(
            "task 'extrapolation_steps' requires --ex_steps N (integer, 0..T-1).")

    if getattr(args, "smoke_test", False):
        cfg["smoke_test"] = True
        if cfg.get("data", {}).get("length_override") is not None:
            cfg["data"]["length"] = cfg["data"]["length_override"]

    overridden: set[str] = set()
    if extras:
        i = 0
        while i < len(extras):
            tok = extras[i]
            if tok.startswith("--"):
                key = tok[2:]
                if "=" in key:
                    key, val = key.split("=", 1)
                    apply_dot_override(cfg, key, _coerce(val))
                    overridden.add(key)
                    i += 1
                    continue
                if i + 1 < len(extras):
                    val = extras[i + 1]
                    apply_dot_override(cfg, key, _coerce(val))
                    overridden.add(key)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

    results_root = str(
        getattr(args, "results_root", None) or cfg.get("results_root") or "results"
    ).rstrip("/")
    cfg["results_root"] = results_root

    if cfg.get("output_dir") is None:
        tag = task_tag(cfg["task"]["name"], cfg["task"].get("ex_steps"))
        cfg["output_dir"] = (
            f"{results_root}/model_checkpoints/"
            f"{cfg['dataset']['name']}_{tag}_seed{cfg['seed']}"
        )

    cfg["training"]["seed"] = cfg.get("seed", cfg["training"].get("seed", 0))

    cfg.setdefault("coupling", {})
    ot_type = (getattr(args, "ot_type", None)
               or cfg["coupling"].get("ot_type") or "CPU")
    ot_type = str(ot_type).upper()
    cfg["coupling"]["ot_type"] = ot_type

    _paths = cache_paths(
        cfg["dataset"]["name"], cfg["task"]["name"],
        cfg["task"].get("ex_steps"), cfg["training"]["seed"],
        ot_type=ot_type, results_root=results_root,
    )
    # These three are derived from results_root by default, but an explicit
    # --dataset.<key> on the command line wins. That is what lets several runs
    # share one autoencoder while reading different coupling caches, which is
    # how the trajectory-construction ablation keeps everything but the kernel
    # fixed (exps/task5_ablation/5-4_OT_real/).
    for _key in ("vae_ckpt_path", "latents_path", "coupling_cache_dir"):
        if f"dataset.{_key}" not in overridden:
            cfg["dataset"][_key] = _paths[_key]

    return cfg


def get_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=["zb", "dr", "sc", "ga", "ma",
                            "ZB", "DR", "SC", "GA", "MA"])
    p.add_argument("--task", type=str, required=True,
                   choices=["interpolation", "extrapolation",
                            "extrapolation_steps", "all"])
    p.add_argument("--ex_steps", type=int, default=None,
                   help="only for task=extrapolation_steps: number of trailing steps held out as the test set")
    p.add_argument("--ot_type", type=str, default=None,
                   choices=["CPU", "CPU_COST", "CUDA"],
                   help="which OT coupling cache to use (the directory carries an _{OT_TYPE} suffix); "
                        "must match the --ot_type used during preprocessing (default CPU)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--results_root", type=str, default=None,
                   help="output/cache root directory, used to isolate experiments (e.g. results/task1_ts_pred); "
                        "determines which VAE/OT cache is read; must match preprocessing (default results)")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--smoke_test", action="store_true")
    return p


def save_config(cfg: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
