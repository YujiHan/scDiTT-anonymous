import torch

from src.diffusion.schedulers import FlowMatchingScheduler


def test_flow_matching_roundtrip():
    sched = FlowMatchingScheduler()
    B, T, N, D = 2, 3, 4, 8
    x0 = torch.randn(B, T, N, D)
    noise = torch.randn_like(x0)
    tau = sched.sample_timesteps(B, device=torch.device("cpu"))
    xt = sched.add_noise(x0, noise, tau)
    v = sched.compute_target(x0, noise, tau)
    x0_hat = sched.v_to_x0(v, xt, tau)
    eps_hat = sched.v_to_eps(v, xt, tau)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, noise, atol=1e-4)


def test_flow_matching_snr_monotonic():
    sched = FlowMatchingScheduler()
    tau = torch.linspace(0.05, 0.95, 10)
    snr = sched.snr(tau)
    diffs = snr[1:] - snr[:-1]
    assert (diffs >= -1e-6).all()
