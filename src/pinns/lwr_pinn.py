"""
Phase 3 — PINN with LWR equation + kinematic constraints (Alam et al. 2025).

Architecture:
  MLP (x, t) → (ρ, v)
  Loss = Data (MSE) + λ_pde * LWR_residual + λ_kin * Kinematic_penalty

LWR: ∂ρ/∂t + ∂(ρv)/∂x = 0

Kinematic constraints (Alam et al. 2025):
  - v ∈ [0, v_max]        — speed bounds
  - |∂v/∂t| ≤ a_max       — acceleration limit for cargo ships
  - |∂v/∂x| ≤ grad_max    — smooth spatial speed variation
  - ρ ≥ 0, ρ ≤ ρ_max      — density bounds
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LWRPINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=64):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        rho = torch.sigmoid(out[:, 0:1])
        v = torch.sigmoid(out[:, 1:2])
        return rho, v


def pde_loss(model, x_col, t_col):
    """
    LWR residual: ∂ρ/∂t + ∂(ρv)/∂x = 0
    """
    rho, v = model(x_col, t_col)
    flux = rho * v

    drho_dt = torch.autograd.grad(
        rho, t_col, grad_outputs=torch.ones_like(rho), create_graph=True
    )[0]

    dflux_dx = torch.autograd.grad(
        flux, x_col, grad_outputs=torch.ones_like(flux), create_graph=True
    )[0]

    residual = drho_dt + dflux_dx
    return (residual**2).mean()


def data_loss(model, x_data, t_data, rho_obs, v_obs):
    rho_pred, v_pred = model(x_data, t_data)
    l_rho = ((rho_pred - rho_obs) ** 2).mean()
    l_v = ((v_pred - v_obs) ** 2).mean()
    return l_rho + l_v


def kinematic_loss(model, x_col, t_col, v_max=1.0, a_max=0.1, grad_max=0.5):
    """
    Kinematic constraints from Alam et al. 2025:
    - acceleration limit: |∂v/∂t| ≤ a_max
    - spatial gradient limit: |∂v/∂x| ≤ grad_max
    - speed within bounds (via sigmoid output, automatically [0,1])
    """
    rho, v = model(x_col, t_col)

    dv_dt = torch.autograd.grad(
        v, t_col, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    dv_dx = torch.autograd.grad(
        v, x_col, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    loss = torch.mean(torch.relu(torch.abs(dv_dt) - a_max))
    loss += torch.mean(torch.relu(torch.abs(dv_dx) - grad_max))
    loss += torch.mean(torch.relu(rho - 1.0))
    loss += torch.mean(torch.relu(-rho))

    return loss


def total_loss(
    model,
    x_data,
    t_data,
    rho_obs,
    v_obs,
    x_col,
    t_col,
    lambda_pde=0.1,
    lambda_kin=0.05,
):
    l_data = data_loss(model, x_data, t_data, rho_obs, v_obs)
    l_pde = pde_loss(model, x_col, t_col)
    l_kin = kinematic_loss(model, x_col, t_col)

    loss = l_data + lambda_pde * l_pde + lambda_kin * l_kin
    return loss, {
        "data": float(l_data.item()),
        "pde": float(l_pde.item()),
        "kin": float(l_kin.item()),
        "total": float(loss.item()),
    }
