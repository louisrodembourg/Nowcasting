"""
Phase 3 — PINN avec équation LWR (Lighthill-Whitham-Richards).

Architecture :
  - Réseau MLP (x, t) → (ρ, v)  où x = position normalisée, t = temps normalisé
  - Loss totale = Loss_données + λ_pde * Loss_PDE + λ_bc * Loss_BC

Équation LWR (loi de conservation du trafic) :
    ∂ρ/∂t + ∂(ρ·v(ρ))/∂x = 0

Modèle de vitesse de Greenshields (linéaire) :
    v(ρ) = v_max · (1 - ρ/ρ_max)

Ce modèle est adapté aux détroits/canaux (flux 1D).
Pour Houston, x représente la position le long du chenal (LON normalisé).
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Utilise MPS si disponible (Apple Silicon), sinon CPU
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
    Output : (rho, v) — densité et vitesse normalisées ∈ [0, 1]

    Architecture : MLP avec activations tanh (smooth → dérivées propres).
    """

    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_size:   int = 64,
        v_max:   float = 1.0,   # vitesse max normalisée
        rho_max: float = 1.0,   # densité max normalisée
    ):
        super().__init__()
        self.v_max   = v_max
        self.rho_max = rho_max

        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 2), nn.Sigmoid()]  # output ∈ [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x, t : tensors of shape (N, 1), requires_grad=True for PDE loss.
        Returns (rho, v) each of shape (N, 1).
        """
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        rho = out[:, 0:1]
        v   = out[:, 1:2]
        return rho, v

    def greenshields_v(self, rho: torch.Tensor) -> torch.Tensor:
        """Greenshields speed-density relation: v = v_max * (1 - rho/rho_max)."""
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
    Loss PDE : résidu de l'équation LWR au points de collocation.
    ∂ρ/∂t + ∂(ρ·v)/∂x = 0

    x_col, t_col : (N_col, 1), requires_grad=True
    """
    rho, _ = model(x_col, t_col)
    v      = model.greenshields_v(rho)
    flux   = rho * v   # ρ·v

    # Gradients automatiques
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

    residual = drho_dt + dflux_dx
    return (residual ** 2).mean()


def data_loss(
    model: LWRPINN,
    x_data: torch.Tensor,
    t_data: torch.Tensor,
    rho_obs: torch.Tensor,
    v_obs:   torch.Tensor,
) -> torch.Tensor:
    """
    Loss données : écart entre prédiction et observations AIS normalisées.
    """
    rho_pred, v_pred = model(x_data, t_data)
    loss_rho = ((rho_pred - rho_obs) ** 2).mean()
    loss_v   = ((v_pred   - v_obs)   ** 2).mean()
    return loss_rho + loss_v


def boundary_loss(
    model: LWRPINN,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    rho_bc: torch.Tensor,
) -> torch.Tensor:
    """
    Loss conditions aux limites : densité aux entrées/sorties du chenal.
    """
    rho_pred, _ = model(x_bc, t_bc)
    return ((rho_pred - rho_bc) ** 2).mean()


def total_loss(
    model:   LWRPINN,
    x_data:  torch.Tensor, t_data:  torch.Tensor,
    rho_obs: torch.Tensor, v_obs:   torch.Tensor,
    x_col:   torch.Tensor, t_col:   torch.Tensor,
    x_bc:    torch.Tensor, t_bc:    torch.Tensor, rho_bc: torch.Tensor,
    lambda_pde: float = 0.1,
    lambda_bc:  float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Loss totale pondérée.
    Retourne (loss, dict de composantes pour le logging).
    """
    l_data = data_loss(model, x_data, t_data, rho_obs, v_obs)
    l_pde  = pde_loss(model, x_col, t_col)
    l_bc   = boundary_loss(model, x_bc, t_bc, rho_bc)

    loss = l_data + lambda_pde * l_pde + lambda_bc * l_bc
    return loss, {
        "data": l_data.item(),
        "pde":  l_pde.item(),
        "bc":   l_bc.item(),
        "total": loss.item(),
    }
