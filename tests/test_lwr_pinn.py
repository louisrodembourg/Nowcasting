import torch

from src.pinns.lwr_pinn import FourierEmbedding, LWRPINN, _pde_residual, total_loss


def _collocation(n: int = 64):
    x = torch.rand(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True)
    return x, t


def test_outputs_are_bounded_densities_and_speeds():
    model = LWRPINN()
    rho, v = model(torch.rand(32, 1), torch.rand(32, 1))
    assert rho.shape == v.shape == (32, 1)
    assert torch.all((rho > 0) & (rho < 1))
    assert torch.all((v > 0) & (v < 1))


def test_fourier_embedding_dimension_and_checkpoint_buffer():
    embedding = FourierEmbedding(input_dim=2, n_freqs=16)
    assert embedding(torch.rand(5, 2)).shape == (5, 2 * 16 + 2)
    assert "embedding.B" in LWRPINN(n_freqs=16).state_dict()


def test_pde_residual_vanishes_for_travelling_wave():
    # rho(x, t) = f(x - c t) with constant speed c solves  d rho/dt + d(rho v)/dx = 0
    c = 0.4
    x, t = _collocation()
    rho = torch.sin(3.0 * (x - c * t)) ** 2
    v = torch.full_like(rho, c)
    assert _pde_residual(rho, v, x, t, alpha_courant=1.0).item() < 1e-10


def test_pde_residual_detects_non_conservative_field():
    x, t = _collocation()
    rho = t * x            # density grows everywhere: mass is not conserved
    v = torch.zeros_like(rho)
    assert _pde_residual(rho, v, x, t).item() > 0.1


def test_total_loss_is_differentiable_with_and_without_fourier():
    for n_freqs in (0, 8):
        model = LWRPINN(n_freqs=n_freqs)
        x_col, t_col = _collocation()
        loss, parts = total_loss(
            model,
            torch.rand(16, 1), torch.rand(16, 1), torch.rand(16, 1), torch.rand(16, 1),
            x_col, t_col, alpha_courant=10.0,
        )
        loss.backward()
        assert set(parts) == {"data", "pde", "kin", "total"}
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
