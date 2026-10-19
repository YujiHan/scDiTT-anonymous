from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import SCTrajectoryDataset
from ..data.mask_sampler import MaskSampler
from ..data.splits import get_tp_split, load_t_real, task_tag, vae_cache_valid
from ..data.vae import load_vae
from ..diffusion.losses import diffusion_loss, min_snr_weight
from ..diffusion.schedulers import build_scheduler
from ..evaluation import scoring
from ..evaluation.inference import predict_test_tps_from_adata
from ..model.dit import SCDiT
from ..model.model_utils import (count_params, load_checkpoint,
                                  save_checkpoint)
from ..utils.logging import get_logger
from ..utils.seeds import set_all_seeds, worker_init_fn
from .callbacks import TrainLossProbe, ValidationCallback, build_probe_batches
from .ema import EMA
from .optim import build_optimizer, build_scheduler as build_lr_scheduler


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def train_model(config: dict) -> None:
    set_all_seeds(config.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("scdit.train", output_dir / "train.log")

    # --- split ---
    ds_name = config["dataset"]["name"]
    task_name = config["task"]["name"]
    n_steps = config.get("task", {}).get("ex_steps")
    train_tps, test_tps = get_tp_split(ds_name, task_name, n_steps=n_steps)
    T = config["dataset"]["T"]
    time_key = config["dataset"]["time_key"]
    real_time_key = config["dataset"].get("real_time_key")
    logger.info(f"Dataset={ds_name}, Task={task_name}, T={T}")
    logger.info(f"train_tps={train_tps}")
    logger.info(f"test_tps={test_tps}")

    # --- t_real ---
    # The values PE(t_k) sees. real_time_key=None -> the stage index 0..T-1;
    # naming a column -> that stage's physical acquisition time, so the encoding
    # carries the true (possibly uneven) spacing. See configs/dataset/*.yaml.
    latents_path = config["dataset"]["latents_path"]
    t_real_list = load_t_real(latents_path, time_key=time_key,
                              real_time_key=real_time_key)
    if len(t_real_list) != T:
        logger.warning(f"len(t_real)={len(t_real_list)} != T={T}; truncating")
        t_real_list = t_real_list[:T]
    logger.info(f"t_real (real_time_key={real_time_key!r}): "
                + ", ".join(f"{v:g}" for v in t_real_list))

    if not vae_cache_valid(ds_name, task_name, n_steps,
                           config["training"]["seed"],
                           vae_ckpt_path=config["dataset"].get("vae_ckpt_path"),
                           latents_path=latents_path):
        logger.warning(
            "[LEAKAGE] the VAE ckpt does not record train_tps or disagrees with the current split; the latent space may already contain "
            "test information. Rerun run/preprocess_vae.py (bash: --force-vae) and retrain."
            f"ckpt={config['dataset'].get('vae_ckpt_path')}")

    # --- dataset + loader ---
    _root = str(config.get("results_root") or "results").rstrip("/")
    coupling_cache_dir = config["dataset"].get("coupling_cache_dir") or (
        f"{_root}/coupling_cache/{ds_name}_{task_tag(task_name, n_steps)}")
    mask_sampler = MaskSampler(
        train_tps=train_tps, T=T,
        rng=np.random.default_rng(config.get("seed", 0) + 12345),
    )
    ds_length = config["data"].get("length", 100000)
    if config["data"].get("length_override") is not None:
        ds_length = int(config["data"]["length_override"])
    ds = SCTrajectoryDataset(
        latents_path=latents_path,
        coupling_cache_dir=coupling_cache_dir,
        train_tps=train_tps,
        test_tps=test_tps,
        T=T,
        N=config["data"]["N"],
        t_real=t_real_list,
        time_key=time_key,
        real_time_key=real_time_key,
        mask_sampler=mask_sampler,
        length=ds_length,
        seed=config.get("seed", 0),
    )
    loader = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=config["data"].get("num_workers", 4) > 0,
    )

    # --- model ---
    model = SCDiT(
        D_latent=config["model"]["D_latent"],
        D_model=config["model"]["D_model"],
        num_blocks=config["model"]["num_blocks"],
        num_heads=config["model"]["num_heads"],
        T_max=max(32, T),
    ).to(device)
    pcount = count_params(model)
    logger.info(f"Model params: total={pcount['total']/1e6:.2f}M "
                f"trainable={pcount['trainable']/1e6:.2f}M")

    ema = EMA(model, decay=config["training"]["ema_decay"])

    # --- optimizer + lr scheduler ---
    optimizer = build_optimizer(
        model.parameters(),
        peak_lr=config["training"]["peak_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    lr_sched = build_lr_scheduler(
        optimizer,
        warmup_steps=config["training"]["warmup_steps"],
        total_steps=config["training"]["total_steps"],
        min_ratio=config["training"]["min_lr_ratio"],
    )

    # --- diffusion scheduler ---
    diff_sched = build_scheduler(config["diffusion"])
    logger.info("Diffusion scheduler: flow_matching (uniform sampling)")

    # --- resume ---
    start_step = 0
    if config.get("resume_from"):
        ckpt = load_checkpoint(config["resume_from"], map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            lr_sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0)) + 1
        logger.info(f"Resumed from {config['resume_from']} at step {start_step}")

    # --- VAE for decoding during eval ---
    vae_ckpt = config["dataset"].get("vae_ckpt_path")
    vae = None
    if vae_ckpt and Path(vae_ckpt).exists():
        try:
            vae = load_vae(vae_ckpt, device=str(device))
            logger.info(f"Loaded VAE decoder for eval: {vae_ckpt}")
        except Exception as e:
            logger.warning(f"Could not load VAE: {e}. Eval will skip decode.")

    use_bf16 = config["training"].get("use_bf16", True) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    n_probe = int(config["training"].get("probe_batches", 8) or 8)
    probe = TrainLossProbe(
        batches=build_probe_batches(
            ds, n_batches=n_probe,
            batch_size=config["training"]["batch_size"],
            seed=config.get("seed", 0) + 777),
        scheduler=diff_sched,
        device=device,
        seed=config.get("seed", 0) + 999,
        use_min_snr=config["diffusion"].get("use_min_snr", False),
        min_snr_gamma=config["diffusion"].get("min_snr_gamma", 5.0),
        use_bf16=use_bf16,
        amp_dtype=amp_dtype,
    )
    logger.info(f"[probe] selection signal = EMA diffusion loss on a fixed training probe set "
                f"({n_probe} batches × {config['training']['batch_size']}); "
                f"test metrics do not feed best.pt")

    _best_ckpt = output_dir / "best.pt"
    if start_step > 0 and _best_ckpt.exists():
        try:
            _prev = load_checkpoint(_best_ckpt, map_location="cpu")
            if _prev.get("selection_metric") == "probe_train_loss" \
                    and _prev.get("best_probe_loss") is not None:
                probe.best = float(_prev["best_probe_loss"])
                probe.best_step = int(_prev.get("step", 0))
                logger.info(f"[probe] resume: inheriting the probe loss of the existing best.pt "
                            f"{probe.best:.6f} (step {probe.best_step})")
        except Exception as e:
            logger.warning(f"[probe] cannot read the probe loss of the existing best.pt: {e}")

    val_cb = ValidationCallback(
        dataset_name=ds_name, task_type=task_name,
        num_repeats=config["training"].get("val_num_repeats", 4),
        cfg_weight=config["inference"]["cfg_weight"],
        num_steps=config["inference"]["num_steps"],
        N_eval=config["inference"]["N_eval"],
        eval_seed=config["inference"]["eval_seed"],
        device=str(device),
    )

    def _eval_fn(**kwargs):
        return evaluate_model_inline(
            model=kwargs["model"],
            vae=vae,
            dataset_name=ds_name,
            task_name=task_name,
            config=config,
            device=device,
            num_repeats=kwargs["num_repeats"],
            cfg_weight=kwargs["cfg_weight"],
            num_steps=kwargs["num_steps"],
            N_eval=kwargs["N_eval"],
            eval_seed=kwargs["eval_seed"],
        )

    # --- main loop ---
    data_iter = _infinite_loader(loader)
    total_steps = config["training"]["total_steps"]
    grad_clip = config["training"]["grad_clip"]

    logger.info(f"Start training from step {start_step} to {total_steps}")
    model.train()

    log_every = 100

    for step in range(start_step, total_steps):
        batch = next(data_iter)
        batch = _move_batch_to_device(batch, device)

        x0 = batch["x"]                 # [B, T, N, D_latent]
        obs_mask = batch["obs_mask"]    # [B, T]
        loss_mask = batch["loss_mask"]  # [B, T]
        t_real = batch["t_real"][0]     # [T]

        B = x0.shape[0]

        # Paper: one training step of Methodology -> "Masked flow matching
        # objective". One tau ~ U(0,1) is drawn per example and all T slots are
        # corrupted at that shared level; z_tau = (1-tau)*eps + tau*z_0 and the
        # regression target is the constant velocity v* = z_0 - eps. Which slots
        # are evidence and which are targets reaches the model only through
        # obs_mask (ME(m_k) of Eq. (3)) and loss_mask (l_k of Eq. (4)).
        tau = diff_sched.sample_timesteps(B, device)
        noise = torch.randn_like(x0)
        xt = diff_sched.add_noise(x0, noise, tau)
        v_target = diff_sched.compute_target(x0, noise, tau)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            v_pred = model(xt, tau, obs_mask, t_real)
            snr_w = None
            if config["diffusion"].get("use_min_snr", False):
                snr = diff_sched.snr(tau)
                snr_w = min_snr_weight(snr,
                                       gamma=config["diffusion"].get("min_snr_gamma", 5.0),
                                       prediction_type="v_prediction")
            # Paper: Eq. (4).
            ld = diffusion_loss(v_pred, v_target, loss_mask, snr_w)
            loss = ld["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()

        ema.update(model)

        if step % log_every == 0:
            logger.info(f"step {step:6d} loss={loss.item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}")

        val_tick = (step > 0 and step % config["training"]["val_every"] == 0) \
            or (step == total_steps - 1)
        if val_tick:
            probe_loss = probe.run(model, ema, step)
            if probe.update_best(probe_loss, step):
                save_checkpoint(output_dir / "best.pt",
                                model, ema, optimizer, lr_sched,
                                step=step, config=config,
                                extra={"selection_metric": "probe_train_loss",
                                       "best_probe_loss": probe_loss},
                                save_optimizer=False)
                logger.info(f"[probe step={step}] train_loss={probe_loss:.6f} "
                            f"-> saved best.pt")
            else:
                logger.info(f"[probe step={step}] train_loss={probe_loss:.6f} "
                            f"(best={probe.best:.6f} @ step {probe.best_step})")

        if val_tick and vae is not None:
            try:
                val_metrics = val_cb.run(model, ema, step, _eval_fn)
                logger.info(f"[val-diagnostic step={step}] " +
                            " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                val_cb.best = min(val_cb.best,
                                  val_metrics.get("val/avg_wd", float("inf")))
            except Exception as e:
                logger.warning(f"Diagnostic validation failed at step {step}: {e}")

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        logger.warning("No best.pt was produced during training "
                       "(probe never ran/improved); "
                       "saving final EMA weights as best.pt fallback.")
        save_checkpoint(best_path, model, ema, optimizer, lr_sched,
                        step=total_steps, config=config, save_optimizer=False)
    logger.info(f"Training finished. Best checkpoint: {best_path}")

    import json as _json
    _best_val = val_cb.best if val_cb.best != float("inf") else None
    _best_probe = probe.best if probe.best != float("inf") else None
    (output_dir / "stop_step.json").write_text(_json.dumps({
        "stop_step": int(total_steps - 1),
        "total_steps": int(total_steps),
        "selection_metric": "probe_train_loss",
        "best_probe_loss": _best_probe,
        "best_step": probe.best_step,
        "diagnostic_best_val_w2": _best_val,
        "probe_history": [{"step": s, "train_loss": v} for s, v in probe.history],
    }, indent=2))
    logger.info(f"[stop] completed {total_steps} steps "
                f"best_probe_loss={_best_probe} best_step={probe.best_step} "
                f"(diagnostic_best_val_w2={_best_val})")

    if not config.get("smoke_test", False):
        infer = config.get("inference", {})
        logger.info("Running final evaluation on best.pt (unified baseline 5-metric protocol)...")
        try:
            from ..evaluation.evaluate import evaluate_model
            records = evaluate_model(
                checkpoint_path=best_path,
                dataset_name=ds_name,
                task_name=task_name,
                num_repeats=infer.get("num_repeats", 12),
                cfg_weight=infer.get("cfg_weight", 3.0),
                num_steps=infer.get("num_steps", 20),
                use_ema=True,
                N_eval=infer.get("N_eval", 256),
                eval_seed=infer.get("eval_seed", 0),
                n_steps=n_steps,
                device=str(device),
                output_dir=output_dir,
            )
            for stem, rec in records.items():
                avg = rec["avg"]
                logger.info(f"[final eval] {stem}.json -> " +
                            " ".join(f"{k}={avg[k]:.4f}" for k in scoring.METRIC_KEYS))
        except Exception as e:
            logger.warning(f"Final evaluation failed: {e}")


def evaluate_model_inline(
    model: torch.nn.Module,
    vae,
    dataset_name: str,
    task_name: str,
    config: dict,
    device: torch.device,
    num_repeats: int,
    cfg_weight: float,
    num_steps: int,
    N_eval: int,
    eval_seed: int,
) -> dict:
    from ..diffusion.schedulers import build_scheduler
    diff_sched = build_scheduler(config["diffusion"])

    n_steps = config.get("task", {}).get("ex_steps")
    train_tps, test_tps = get_tp_split(dataset_name, task_name, n_steps=n_steps)
    T = config["dataset"]["T"]
    time_key = config["dataset"]["time_key"]

    val_tps = [T - 1] if task_name == "extrapolation_steps" else list(test_tps)

    result = predict_test_tps_from_adata(
        model=model,
        vae=vae,
        latents_path=config["dataset"]["latents_path"],
        h5ad_path=config["dataset"]["h5ad_path"],
        train_tps=train_tps,
        test_tps=val_tps,
        T=T,
        time_key=time_key,
        real_time_key=config["dataset"].get("real_time_key"),
        scheduler=diff_sched,
        N=N_eval,
        num_repeats=num_repeats,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        device=device,
        eval_seed=eval_seed,
    )
    metrics = scoring.w2_only(
        result["predictions_gene"],
        result["true_gene_list"],
        tps=val_tps,
        all_tps=list(range(T)),
        device=device,
    )
    return metrics
