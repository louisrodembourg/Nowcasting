"""
Phase 3 — PINN avec équation LWR (Lighthill-Whitham-Richards).

Architecture :
    - Réseau MLP (x, t) → ρ  où x = position normalisée, t = temps normalisé
    - v est contraint par Greenshields : v = v(ρ)
    - Loss totale = L_data + λ_pde·L_PDE + λ_bc·L_BC + λ_kin·L_KIN

Équation LWR (loi de conservation du trafic) :
    ∂ρ/∂t + ∂(ρ·v(ρ))/∂x = 0

Modèle de vitesse de Greenshields (linéaire) :
    v(ρ) = v_max · (1 - ρ/ρ_max)

Contrainte cinématique (Alam et al. 2025) :
    |∂v/∂t| ≤ a_max  — pénalise les accélérations irréalistes pour un cargo lourd.

Pour LA, x représente la longitude normalisée sur la bbox de San Pedro Bay.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Réseau MLP
# ---------------------------------------------------------------------------

class LWRPINN(nn.Module):
    """
    Physics-Informed Neural Network pour le modèle LWR.

    Input  : (x, t) — position et temps normalisés ∈ [0, 1]
    Output : rho — densité normalisée ∈ [0, 1]
    """

    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_size:   int = 64,
        v_max:   float = 1.0,
        rho_max: float = 1.0,
    ):
        super().__init__()
        self.v_max   = v_max
        self.rho_max = rho_max

        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

    def greenshields_v(self, rho: torch.Tensor) -> torch.Tensor:
        return self.v_max * (1.0 - rho / self.rho_max)


# ---------------------------------------------------------------------------
# Fonctions de loss
# ---------------------------------------------------------------------------

def pde_loss(
    model: LWRPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
) -> torch.Tensor:
    """
    Résidu de l'équation LWR : ∂ρ/∂t + ∂(ρ·v)/∂x = 0
    x_col, t_col : (N, 1), requires_grad=True
    """
    rho = model(x_col, t_col)
    v      = model.greenshields_v(rho)
    flux   = rho * v

    drho_dt = torch.autograd.grad(
        rho, t_col,
        grad_outputs=torch.ones_like(rho),
        create_graph=True,
    )[0]

    dflux_dx = torch.autograd.grad(
        flux, x_col,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
    )[0]

    return (drho_dt + dflux_dx).pow(2).mean()


def data_loss(
    model:   LWRPINN,
    x_data:  torch.Tensor,
    t_data:  torch.Tensor,
    rho_obs: torch.Tensor,
    v_obs:   torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Écart MSE entre prédiction et observations AIS.
    weights : (N, 1) — pondération par gravity_score (Alam et al. 2025).
    """
    rho_pred = model(x_data, t_data)
    v_pred   = model.greenshields_v(rho_pred)
    if weights is not None:
        loss_rho = (weights * (rho_pred - rho_obs).pow(2)).mean()
        loss_v   = (weights * (v_pred   - v_obs  ).pow(2)).mean()
    else:
        loss_rho = (rho_pred - rho_obs).pow(2).mean()
        loss_v   = (v_pred   - v_obs  ).pow(2).mean()
    return loss_rho + loss_v


def boundary_loss(
    model:  LWRPINN,
    x_bc:   torch.Tensor,
    t_bc:   torch.Tensor,
    rho_bc: torch.Tensor,
) -> torch.Tensor:
    """Condition aux limites : densité aux entrées/sorties du port."""
    rho_pred = model(x_bc, t_bc)
    return (rho_pred - rho_bc).pow(2).mean()


def kinematic_loss(
    model: LWRPINN,
    x_kin: torch.Tensor,
    t_kin: torch.Tensor,
    a_max: float = 0.05,
) -> torch.Tensor:
    """
    Contrainte cinématique (Alam et al. 2025).
    Pénalise les variations de vitesse |∂v/∂t| > a_max — irréalistes pour
    un cargo lourd sur des moyennes journalières.
    t_kin doit avoir requires_grad=True.
    """
    rho = model(x_kin, t_kin)
    v   = model.greenshields_v(rho)
    dv_dt = torch.autograd.grad(
        v, t_kin,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
    )[0]
    return torch.relu(dv_dt.abs() - a_max).pow(2).mean()


def total_loss(
    model:   LWRPINN,
    x_data:  torch.Tensor, t_data:  torch.Tensor,
    rho_obs: torch.Tensor, v_obs:   torch.Tensor,
    x_col:   torch.Tensor, t_col:   torch.Tensor,
    x_bc:    torch.Tensor, t_bc:    torch.Tensor, rho_bc: torch.Tensor,
    x_kin:   torch.Tensor | None = None,
    t_kin:   torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    lambda_pde: float = 0.1,
    lambda_bc:  float = 0.1,
    lambda_kin: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """
    Loss totale pondérée.
    Retourne (loss, dict des composantes pour logging/visualisation).
    """
    l_data = data_loss(model, x_data, t_data, rho_obs, v_obs, weights)
    l_pde  = pde_loss(model, x_col, t_col)
    l_bc   = boundary_loss(model, x_bc, t_bc, rho_bc)

    l_kin = torch.zeros(1, device=x_data.device)[0]
    if x_kin is not None and t_kin is not None:
        l_kin = kinematic_loss(model, x_kin, t_kin)

    loss = l_data + lambda_pde * l_pde + lambda_bc * l_bc + lambda_kin * l_kin
    return loss, {
        "data":  l_data.item(),
        "pde":   l_pde.item(),
        "bc":    l_bc.item(),
        "kin":   l_kin.item(),
        "total": loss.item(),
    }
