"""The two autoregressive schedules: training pattern and sampling loop.

Paper reference
---------------
Supplementary material, "Autoregressive Variants of scDiTT" (Appendix J); the
arms plotted in Figure 3 of the main paper and tabulated in the
extrapolation-horizon tables.

Both arms are the scDiTT backbone unchanged -- same architecture, rectified-flow
objective, optimiser, schedule and per-dataset settings, and, for a given
(dataset, horizon, seed), the same autoencoder checkpoint and optimal-transport
couplings as the corresponding scDiTT run. **Only the conditioning pattern and
the sampling loop differ**, which is what makes the comparison attribute the
difference to the generation schedule rather than to model capacity.

Everything here is applied by monkey-patching the two names the standard
pipeline binds, so `src/` runs unmodified:

    patch_dataset(arm)   ->  src.training.train.SCTrajectoryDataset
    patch_sampler(arm)   ->  src.evaluation.inference.masked_diffusion_inpaint
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

ARMS = ("prefix", "markov")

# As in scDiTT, the observation mask is emptied with this probability so the
# unconditional branch classifier-free guidance needs is one the model has seen.
UNCOND_PROB = 0.05


# ---------------------------------------------------------------------------
# Training: what a generated stage may condition on
# ---------------------------------------------------------------------------
def make_pattern_sampler(arm: str) -> Callable:
    """(rng, train_tps, T) -> (obs_mask, loss_mask), both [T] bool.

    A training example samples a split point k among the observed stages.
    `prefix` sets the observation mask to every observed stage j <= k, `markov`
    to stage k alone, and the loss supervises the single stage k+1. Stages in
    neither role are zeroed by the dataset before noise is added.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm} (expected one of {ARMS})")

    def sampler(rng, train_tps, T):
        train_tps = list(train_tps)
        obs = np.zeros(T, dtype=bool)
        loss = np.zeros(T, dtype=bool)
        L = len(train_tps)
        if L < 2:
            # Degenerate split: nothing to condition on, supervise what there is.
            if L == 1:
                loss[train_tps[0]] = True
            return obs, loss

        i = int(rng.integers(0, L - 1))       # k = train_tps[i]
        loss[train_tps[i + 1]] = True         # the single supervised stage k+1

        if rng.random() < UNCOND_PROB:
            return obs, loss                  # unconditional branch

        if arm == "prefix":
            for t in train_tps[:i + 1]:
                obs[t] = True
        else:                                  # markov
            obs[train_tps[i]] = True
        return obs, loss

    return sampler


def patch_dataset(arm: str) -> None:
    """Make the training loop build trajectories with the AR conditioning pattern."""
    import src.training.train as train_mod
    from src.data.dataset import SCTrajectoryDataset

    pattern = make_pattern_sampler(arm)

    class _ARDataset(SCTrajectoryDataset):
        def __init__(self, *a, **kw):
            kw["pattern_sampler"] = pattern
            super().__init__(*a, **kw)

    _ARDataset.__name__ = f"SCTrajectoryDataset[AR-{arm}]"
    train_mod.SCTrajectoryDataset = _ARDataset


# ---------------------------------------------------------------------------
# Inference: generate stage by stage, writing each result back as evidence
# ---------------------------------------------------------------------------
def make_ar_sampler(arm: str) -> Callable:
    """A drop-in replacement for `masked_diffusion_inpaint` with the same signature.

    Generation proceeds from the first withheld stage to the last. Each stage
    receives the full budget of `num_steps` Euler steps with classifier-free
    guidance, so at horizon l the arm spends 40*l network evaluations against
    scDiTT's 40, and each generated stage is written back into the context as
    clean evidence: the prefix arm then conditions on a mixture of real
    observations and its own output, the Markov arm on the previous stage alone.

    Stages that are neither evidence nor the current target are held at zero
    throughout, matching how they are zeroed during training.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm} (expected one of {ARMS})")

    from src.diffusion.cfg import cfg_predict
    from src.diffusion.sampling import flow_euler_step

    @torch.no_grad()
    def ar_inpaint(model, x_obs, obs_mask, t_real, scheduler,
                   num_steps: int = 20, cfg_weight: float = 3.0,
                   device=None, rng_generator=None):
        if device is None:
            device = x_obs.device
        B, T, N, D = x_obs.shape

        ctx = x_obs.clone()                    # evidence: real, then generated
        avail = obs_mask.clone()               # which stages may be conditioned on
        targets = [t for t in range(T) if not bool(obs_mask[0, t].item())]

        taus = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        taus = taus.clamp(1e-4, 1.0 - 1e-4)

        for t in targets:
            cond = torch.zeros_like(avail)
            if arm == "prefix":
                cond[:, :t] = avail[:, :t]     # every earlier available stage
            elif t >= 1:
                cond[:, t - 1] = avail[:, t - 1]   # the previous stage alone

            cond_f = cond.view(B, T, 1, 1).float()
            # Inert = neither evidence nor the stage being generated now.
            inert = torch.ones(B, T, 1, 1, device=device)
            inert[:, t] = 0.0
            inert = inert * (1.0 - cond_f)

            x = torch.randn(B, T, N, D, device=device, generator=rng_generator)
            x = x * (1.0 - cond_f) + ctx * cond_f
            x = x * (1.0 - inert)

            for i in range(num_steps):
                tau = taus[i].expand(B)
                tau_next = taus[i + 1].expand(B)
                v = cfg_predict(model, x, tau, cond, t_real, cfg_weight)
                x = flow_euler_step(x, v, tau, tau_next)
                x = x * (1.0 - cond_f) + ctx * cond_f
                x = x * (1.0 - inert)

            ctx[:, t] = x[:, t]
            avail[:, t] = True

        return ctx

    return ar_inpaint


def patch_sampler(arm: str) -> None:
    """Make every prediction path go through the AR sampling loop."""
    import src.evaluation.inference as inf_mod
    inf_mod.masked_diffusion_inpaint = make_ar_sampler(arm)


def patch_all(arm: str) -> None:
    patch_dataset(arm)
    patch_sampler(arm)
