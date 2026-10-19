"""SCDiT forward shape + permutation equivariance + gradient flow."""
import torch

from src.model.dit import SCDiT


def _tiny_model(D_latent=16, D_model=32, num_blocks=2, num_heads=4, T_max=16):
    return SCDiT(D_latent=D_latent, D_model=D_model,
                 num_blocks=num_blocks, num_heads=num_heads,
                 T_max=T_max)


def test_forward_shape():
    B, T, N, D = 2, 6, 8, 16
    model = _tiny_model(D_latent=D)
    x = torch.randn(B, T, N, D)
    tau = torch.rand(B)
    obs_mask = torch.randint(0, 2, (B, T)).bool()
    t_real = torch.arange(T, dtype=torch.float32)
    out = model(x, tau, obs_mask, t_real)
    assert out.shape == (B, T, N, D)


def test_cell_permutation_equivariance():
    B, T, N, D = 1, 4, 5, 16
    model = _tiny_model(D_latent=D)
    model.eval()
    x = torch.randn(B, T, N, D)
    tau = torch.rand(B)
    obs_mask = torch.ones(B, T).bool()
    t_real = torch.arange(T, dtype=torch.float32)
    with torch.no_grad():
        out1 = model(x, tau, obs_mask, t_real)
        perm = torch.randperm(N)
        x_perm = x[:, :, perm, :]
        out2 = model(x_perm, tau, obs_mask, t_real)
    assert torch.allclose(out1[:, :, perm, :], out2, atol=1e-5)


def test_gradient_flow_to_all_params():
    B, T, N, D = 2, 4, 8, 16
    model = _tiny_model(D_latent=D)
    x = torch.randn(B, T, N, D)
    tau = torch.rand(B)
    obs_mask = torch.ones(B, T).bool()
    t_real = torch.arange(T, dtype=torch.float32)
    out = model(x, tau, obs_mask, t_real)
    loss = out.mean()
    loss.backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"grad is None for {n}"
