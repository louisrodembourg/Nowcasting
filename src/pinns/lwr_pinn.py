"""
Phase 3 — PINN with LWR equation + kinematic constraints (Alam et al. 2025).

Architecture:
  MLP (x, t) → (ρ, v)
  Loss = Data (MSE) + λ_pde * LWR_residual + λ_kin * Kinematic_penalty

LWR in Courant-corrected normalized coordinates:
  ∂ρ̂/∂t̂ + α · ∂(ρ̂·v̂)/∂x̂ = 0

  where α = v_max [km/h] · T_episode [h] / L_channel [km]  (Courant number)

  Without α the two terms have very different physical scales
  (factor ~40× for Houston crises), biasing the learned dynamics.
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LWRPINN(nn.Module):
    def __init__(self, hidden_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        rho = torch.sigmoid(out[:, 0:1])
        v   = torch.sigmoid(out[:, 1:2])
        return rho, v


# ─── internal helpers (take pre-computed rho, v to avoid double forward) ─────

def _pde_residual(
    rho: torch.Tensor,
    v: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    alpha_courant: float = 1.0,
) -> torch.Tensor:
    """
    Courant-corrected LWR residual: ∂ρ̂/∂t̂ + α · ∂(ρ̂v̂)/∂x̂ = 0

    rho, v must be outputs of model(x_col, t_col) with create_graph enabled
    (i.e. x_col and t_col must have requires_grad=True).
    """
    flux = rho * v
    drho_dt  = torch.autograd.grad(
        rho,  t_col, grad_outputs=torch.ones_like(rho),  create_graph=True
    )[0]
    dflux_dx = torch.autograd.grad(
        flux, x_col, grad_outputs=torch.ones_like(flux), create_graph=True
    )[0]
    residual = drho_dt + alpha_courant * dflux_dx
    return (residual ** 2).mean()


def _kinematic_penalty(
    rho: torch.Tensor,
    v: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    a_max: float = 0.1,
    grad_max: float = 0.5,
) -> torch.Tensor:
    """
    Soft kinematic constraints on pre-computed (rho, v):
      - |∂v̂/∂t̂| ≤ a_max   (acceleration limit for cargo ships)
      - |∂v̂/∂x̂| ≤ grad_max (spatial smoothness)

    Density bounds [0,1] are guaranteed by Sigmoid — no explicit penalty needed.
    """
    dv_dt = torch.autograd.grad(
        v, t_col, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]
    dv_dx = torch.autograd.grad(
        v, x_col, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]
    loss  = torch.mean(torch.relu(torch.abs(dv_dt) - a_max))
    loss  = loss + torch.mean(torch.relu(torch.abs(dv_dx) - grad_max))
    return loss


# ─── public API ──────────────────────────────────────────────────────────────

def data_loss(
    model: LWRPINN,
    x_data: torch.Tensor,
    t_data: torch.Tensor,
    rho_obs: torch.Tensor,
    v_obs: torch.Tensor,
) -> torch.Tensor:
    rho_pred, v_pred = model(x_data, t_data)
    return ((rho_pred - rho_obs) ** 2).mean() + ((v_pred - v_obs) ** 2).mean()


def total_loss(
    model: LWRPINN,
    x_data: torch.Tensor,
    t_data: torch.Tensor,
    rho_obs: torch.Tensor,
    v_obs: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    lambda_pde: float = 0.1,
    lambda_kin: float = 0.05,
    alpha_courant: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Hybrid loss: L_data + λ_pde · L_LWR + λ_kin · L_kinematic.

    The model is evaluated on collocation points exactly once.
    The resulting (rho_col, v_col) tensors are shared between the PDE
    residual and the kinematic penalty, halving compute and graph memory
    compared to calling pde_loss + kinematic_loss separately.

    Parameters
    ----------
    alpha_courant : Courant number = v_max [km/h] * T_ep [h] / L [km].
                   Pass 1.0 to reproduce legacy (un-scaled) behaviour.
    """
    l_data = data_loss(model, x_data, t_data, rho_obs, v_obs)

    # Single forward pass on collocation points — shared between PDE and kinematic
    rho_col, v_col = model(x_col, t_col)
    l_pde = _pde_residual(rho_col, v_col, x_col, t_col, alpha_courant)
    l_kin = _kinematic_penalty(rho_col, v_col, x_col, t_col)

    loss = l_data + lambda_pde * l_pde + lambda_kin * l_kin
    return loss, {
        "data":  float(l_data.item()),
        "pde":   float(l_pde.item()),
        "kin":   float(l_kin.item()),
        "total": float(loss.item()),
    }
