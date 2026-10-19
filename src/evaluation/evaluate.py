"""End-to-end evaluation: checkpoint path -> the unified baseline 5-metric metrics_*.json.

scDiTT and every published baseline are scored by one protocol, implemented here:
  * metrics: w2_avg / mmd_avg / pearson_mean / sampled_pearson / pg_w1_mean
    (the shared implementations live in unified_metrics.py; the paper reports
    w2_avg and sampled_pearson).
  * eval-tp selection (see scoring.eval_tps):
      - interpolation / extrapolation -> the held-out test_tps, written to metrics.json;
      - extrapolation_steps -> the fixed sets alltp / lasttp, independent of ex_step,
        written to metrics_alltp.json / metrics_lasttp.json respectively.
        The horizon tables of the appendix report the lasttp scheme.
  * the JSON written here is the same record the baseline reruns produce, so the
    two are read by one aggregation script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.splits import get_tp_split
from ..data.vae import load_vae
from ..diffusion.schedulers import build_scheduler
from ..model.dit import SCDiT
from ..model.model_utils import load_checkpoint
from ..training.ema import EMA
from ..utils.logging import get_logger
from . import scoring
from .inference import predict_test_tps_from_adata


def evaluate_model(
    checkpoint_path: str | Path,
    dataset_name: str,
    task_name: str,
    num_repeats: int = 12,
    cfg_weight: float = 3.0,
    num_steps: int = 20,
    use_ema: bool = True,
    N_eval: int = 256,
    eval_seed: int = 0,
    n_steps: Optional[int] = None,
    device: Optional[str] = None,
    config_override: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
    n_pred_cells: int = 2000,
    save_predictions: bool = True,
) -> dict:
    """Load from a checkpoint, run one full evaluation and write baseline-format metrics_*.json.

    The checkpoint must contain a 'config' field (the product of build_config).

    Returns:
        ``{stem: record}`` -- stem is "metrics" / "metrics_alltp" / "metrics_lasttp",
        and record is the complete dict written to JSON (containing ``avg`` / ``per_tp`` / ``eval_tps`` ...).
    """
    logger = get_logger("scdit.eval")
    device = torch.device(device if device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = config_override if config_override is not None else ckpt.get("config")
    if cfg is None:
        raise RuntimeError(f"No config in {checkpoint_path}; pass config_override")

    # build model
    model = SCDiT(
        D_latent=cfg["model"]["D_latent"],
        D_model=cfg["model"]["D_model"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        T_max=max(32, cfg["dataset"]["T"]),
    ).to(device)

    # load weights
    if use_ema and ckpt.get("ema") is not None:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
        logger.info("Loaded EMA weights for evaluation.")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded raw (non-EMA) weights.")
    model.eval()

    # VAE decoder
    vae_ckpt = cfg["dataset"].get("vae_ckpt_path")
    if vae_ckpt and Path(vae_ckpt).exists():
        vae = load_vae(vae_ckpt, device=str(device))
    else:
        vae = None
        logger.warning(f"VAE ckpt not found at {vae_ckpt}; will compare in latent space")

    if n_steps is None:
        n_steps = cfg.get("task", {}).get("ex_steps")
    train_tps, test_tps = get_tp_split(dataset_name, task_name, n_steps=n_steps)
    T = cfg["dataset"]["T"]
    time_key = cfg["dataset"]["time_key"]

    scheduler = build_scheduler(cfg["diffusion"])

    logger.info(f"Running inference with {num_repeats} repeats, "
                f"cfg={cfg_weight}, steps={num_steps} (flow_euler), "
                f"n_pred_cells={n_pred_cells}")

    result = predict_test_tps_from_adata(
        model=model,
        vae=vae,
        latents_path=cfg["dataset"]["latents_path"],
        h5ad_path=cfg["dataset"]["h5ad_path"],
        train_tps=train_tps,
        test_tps=test_tps,
        T=T,
        time_key=time_key,
        real_time_key=cfg["dataset"].get("real_time_key"),
        scheduler=scheduler,
        N=N_eval,
        num_repeats=num_repeats,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        device=device,
        eval_seed=eval_seed,
        n_eval_cells=n_pred_cells,
        predict_all_tps=True,
    )
    pred_by_tp = result["predictions_gene"]
    real_by_tp = result["true_gene_list"]
    all_tps = list(range(T))

    if task_name == "extrapolation_steps":
        schemes = [("alltp", "metrics_alltp"), ("lasttp", "metrics_lasttp")]
    else:
        schemes = [(None, "metrics")]

    seed = int(cfg.get("seed", 0))
    ex_step = int(n_steps) if n_steps is not None else None

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    for scheme, stem in schemes:
        etps = scoring.eval_tps(task_name, T, test_tps, scheme=scheme)
        res = scoring.score(pred_by_tp, real_by_tp, etps, all_tps=all_tps,
                            include_w2=True, device=device)
        record = {
            "checkpoint": str(checkpoint_path),
            "dataset": dataset_name,
            "task": task_name,
            "seed": seed,
            "ex_step": ex_step,
            "variant": "best",
            "scheme": scheme,
            "eval_tps": [int(t) for t in etps],
            "avg": res["avg"],
            "per_tp": {int(t): res["per_tp"][int(t)] for t in etps},
            "status": "ok",
        }
        records[stem] = record

        avg = res["avg"]
        logger.info(
            f"[eval] scheme={scheme} eval_tps={record['eval_tps']} | " +
            " ".join(f"{k}={avg[k]:.4f}" for k in scoring.METRIC_KEYS))

        if out is not None:
            with open(out / f"{stem}.json", "w") as f:
                json.dump(record, f, indent=2)
            logger.info(f"[eval] wrote {out / (stem + '.json')}")

    if out is not None and save_predictions:
        if vae is None:
            logger.warning("VAE missing; pred_genes.npy will be latent space "
                           "(D_latent) rather than gene space.")
        pred_genes = np.stack(
            [pred_by_tp[t] for t in range(T)], axis=0
        ).astype(np.float32)
        np.save(out / "pred_genes.npy", pred_genes)
        logger.info(f"Saved pred_genes.npy {pred_genes.shape} "
                    f"(T, n_pred_cells, n_genes) to {out}")

    return records
