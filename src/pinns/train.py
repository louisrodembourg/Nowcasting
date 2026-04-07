"""
Phase 3 — Entraînement du PINN LWR sur Harvey.

Fenêtre d'entraînement : 10 jours avant Harvey + fenêtre de crise + 10 jours après.
Default : 2017-08-15 → 2017-09-10

Les données AIS (ρ, v) sont extraites de la matrice de features Phase 1 et
normalisées dans [0,1]. Les points de collocation PDE sont échantillonnés
uniformément dans le domaine (x, t).

Le modèle entraîné est sauvegardé dans outputs/models/lwr_pinn.pt.

Usage:
    python src/pinns/train.py
    python src/pinns/train.py --epochs 3000 --lr 1e-3
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import torch
import torch.optim as optim

from src.pinns.lwr_pinn import LWRPINN, total_loss, get_device

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/features/houston_daily_features.parquet")
MODEL_PATH    = Path("outputs/models/lwr_pinn.pt")

# Fenêtre Harvey
TRAIN_START   = date(2017, 8, 15)
TRAIN_END     = date(2017, 9, 10)

# Hyperparamètres par défaut
DEFAULT_EPOCHS      = 2000
DEFAULT_LR          = 5e-4
DEFAULT_N_COL       = 2000   # points de collocation PDE
DEFAULT_LAMBDA_PDE  = 0.1
DEFAULT_LAMBDA_BC   = 0.1
LOG_EVERY           = 200


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def prepare_training_data(
    train_start: date,
    train_end:   date,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Extrait et normalise les données AIS pour l'entraînement.

    x = LON normalisé ∈ [0,1] (position le long du chenal)
    t = jour normalisé ∈ [0,1] (depuis train_start)
    ρ = utilization_rate_rho ∈ [0,1]  (déjà une fraction)
    v = SOG_median normalisé ∈ [0,1]  (/ SOG_max observé)

    Retourne un dict de tenseurs sur `device`.
    """
    df = (
        pl.read_parquet(FEATURES_PATH)
        .sort("date")
        .filter(pl.col("date").is_between(train_start, train_end))
    )

    if len(df) == 0:
        raise ValueError(f"Aucune donnée entre {train_start} et {train_end}")

    n_days  = (train_end - train_start).days + 1
    dates   = df["date"].to_list()

    # t normalisé
    t_vals  = np.array([(d - train_start).days / max(n_days - 1, 1) for d in dates])

    # x : on utilise une position x=0.5 (centroid du chenal) —
    # on n'a pas de coordonnée x par jour dans les features agrégées.
    # Les trajectoires détaillées (trajectory.py) donneraient un x continu,
    # mais pour le training sur features journalières, x=0.5 est un proxy valide.
    x_vals  = np.full(len(df), 0.5)

    # ρ = utilization_rate_rho (déjà ∈ [0,1])
    rho_vals = df["utilization_rate_rho"].to_numpy().astype(float)

    # v = SOG_mean normalisé (SOG_median=0 pour la plupart des jours car navires stationnaires)
    sog = df["SOG_mean"].to_numpy().astype(float)
    sog_max = sog.max()
    v_vals = sog / sog_max if sog_max > 0 else sog

    def t_(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=grad).reshape(-1, 1)

    x_data   = t_(x_vals)
    t_data   = t_(t_vals)
    rho_obs  = t_(rho_vals)
    v_obs    = t_(v_vals)

    # Points de collocation PDE : grille uniforme (x, t) ∈ [0,1]²
    n_col   = DEFAULT_N_COL
    x_col_np = np.random.uniform(0, 1, n_col)
    t_col_np = np.random.uniform(0, 1, n_col)
    x_col    = t_(x_col_np, grad=True)
    t_col    = t_(t_col_np, grad=True)

    # Conditions aux limites : ρ = baseline aux bords temporels (t=0 et t=1)
    n_bc   = 50
    x_bc_np = np.random.uniform(0, 1, n_bc)
    t_bc_np = np.concatenate([np.zeros(n_bc // 2), np.ones(n_bc // 2)])
    rho_bc_np = np.full(n_bc, float(rho_vals[rho_vals > 0].mean()))
    x_bc     = t_(x_bc_np)
    t_bc     = t_(t_bc_np)
    rho_bc   = t_(rho_bc_np)

    log.info("Training data: %d days | rho=[%.3f, %.3f] | v=[%.3f, %.3f]",
             len(df), rho_vals.min(), rho_vals.max(), v_vals.min(), v_vals.max())

    return {
        "x_data": x_data, "t_data": t_data,
        "rho_obs": rho_obs, "v_obs": v_obs,
        "x_col": x_col, "t_col": t_col,
        "x_bc": x_bc, "t_bc": t_bc, "rho_bc": rho_bc,
    }


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train(
    epochs:      int   = DEFAULT_EPOCHS,
    lr:          float = DEFAULT_LR,
    lambda_pde:  float = DEFAULT_LAMBDA_PDE,
    lambda_bc:   float = DEFAULT_LAMBDA_BC,
    train_start: date  = TRAIN_START,
    train_end:   date  = TRAIN_END,
) -> LWRPINN:
    device = get_device()
    log.info("Device: %s", device)

    data  = prepare_training_data(train_start, train_end, device)
    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5)

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # Refresh collocation points every 500 epochs (avoid overfitting to grid)
        if epoch % 500 == 1:
            n_col = DEFAULT_N_COL
            data["x_col"] = torch.rand(n_col, 1, device=device, requires_grad=True)
            data["t_col"] = torch.rand(n_col, 1, device=device, requires_grad=True)

        loss, components = total_loss(
            model,
            data["x_data"], data["t_data"], data["rho_obs"], data["v_obs"],
            data["x_col"], data["t_col"],
            data["x_bc"],  data["t_bc"],  data["rho_bc"],
            lambda_pde=lambda_pde,
            lambda_bc=lambda_bc,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss)

        history.append(components["total"])

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  total=%.6f  data=%.6f  pde=%.6f  bc=%.6f",
                epoch, epochs,
                components["total"], components["data"],
                components["pde"],   components["bc"],
            )

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "train_start": train_start.isoformat(),
        "train_end":   train_end.isoformat(),
        "best_loss":   best_loss,
        "history":     history,
        "epochs":      epochs,
        "lambda_pde":  lambda_pde,
        "lambda_bc":   lambda_bc,
    }, MODEL_PATH)
    log.info("Modèle sauvegardé → %s  (best_loss=%.6f)", MODEL_PATH, best_loss)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINN LWR Training")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde", type=float, default=DEFAULT_LAMBDA_PDE)
    parser.add_argument("--lambda-bc",  type=float, default=DEFAULT_LAMBDA_BC)
    parser.add_argument("--train-start", default=TRAIN_START.isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--train-end",   default=TRAIN_END.isoformat(),   metavar="YYYY-MM-DD")
    args = parser.parse_args()

    train(
        epochs=args.epochs, lr=args.lr,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
