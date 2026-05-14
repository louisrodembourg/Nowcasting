"""
Phase 3 — Entraînement du PINN LWR.

Nouveautés vs version Houston :
  - Spatialisation réelle : les points de collocation PDE sont échantillonnés
    depuis les positions LON des zones constituantes Phase 2 (au lieu de x=0.5).
  - Pondération gravity_score : les jours de crise contribuent plus à la data loss.
  - Loss cinématique (Alam et al. 2025) : pénalise |∂v/∂t| > a_max.

Usage:
    python src/pinns/train.py --location la
    python src/pinns/train.py --location la --epochs 3000 --lr 1e-3
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

# Hyperparamètres par défaut
DEFAULT_EPOCHS     = 2000
DEFAULT_LR         = 5e-4
DEFAULT_N_COL      = 2000
DEFAULT_N_KIN      = 500
DEFAULT_LAMBDA_PDE = 0.3
DEFAULT_LAMBDA_BC  = 0.1
DEFAULT_LAMBDA_KIN = 0.1
LOG_EVERY          = 200

# LA bbox longitude (pour normalisation x si pas de fichier zones)
LA_LON_MIN = -118.35
LA_LON_MAX = -118.05


# ---------------------------------------------------------------------------
# Préparation des données d'entraînement avec spatialisation
# ---------------------------------------------------------------------------

def prepare_training_data(
    train_start:   date,
    train_end:     date,
    device:        torch.device,
    features_path: Path,
    zones_path:    Path | None = None,
    gravity_path:  Path | None = None,
    n_col:         int  = DEFAULT_N_COL,
    n_kin:         int  = DEFAULT_N_KIN,
    n_bc:          int  = 50,
) -> dict[str, torch.Tensor | tuple]:
    """
    Prépare les tenseurs d'entraînement pour le PINN LWR.

    Spatialisation :
      Si zones_path est fourni et existe, les points de collocation PDE sont
      tirés depuis les positions LON réelles des zones constituantes Phase 2.
      Sinon, grille uniforme [0,1] (fallback).

    Pondération gravity :
      Si gravity_path est fourni, les jours à fort gravity_score contribuent
      davantage à la data loss (weight = 1 + gravity_score, normalisé à μ=1).

    Retourne un dict de tenseurs sur `device` + métadonnées (lon_range, sog_max).
    """
    df = (
        pl.read_parquet(features_path)
        .sort("date")
        .filter(pl.col("date").is_between(train_start, train_end))
    )
    if len(df) == 0:
        raise ValueError(f"Aucune donnée entre {train_start} et {train_end}")

    n_days = (train_end - train_start).days + 1
    dates  = df["date"].to_list()
    t_vals = np.array([(d - train_start).days / max(n_days - 1, 1) for d in dates])

    # ρ : blocked_capacity normalisée pour LA (pic de crise = haute capacité bloquée)
    #     utilization_rate_rho pour Houston (port fermé = chute de ρ)
    if "blocked_capacity" in df.columns and df["blocked_capacity"].max() > 0:
        cap_arr  = df["blocked_capacity"].to_numpy().astype(float)
        cap_95   = float(np.percentile(cap_arr[cap_arr > 0], 95)) if cap_arr.max() > 0 else 1.0
        rho_vals = np.clip(cap_arr / cap_95, 0.0, 1.0)
        log.info("ρ = blocked_capacity normalisée (95p=%.0f)", cap_95)
    else:
        rho_vals = df["utilization_rate_rho"].to_numpy().astype(float)
        log.info("ρ = utilization_rate_rho")

    sog      = df["SOG_mean"].to_numpy().astype(float)
    sog_max  = sog.max() if sog.max() > 0 else 1.0
    v_vals   = sog / sog_max

    # Observation : centroïde du port à x=0.5
    x_vals = np.full(len(df), 0.5)

    # --- Pondération par gravity score ---
    weights_np = np.ones(len(df))
    if gravity_path is not None and Path(gravity_path).exists():
        try:
            gdf = pl.read_parquet(gravity_path).sort("date")
            date_to_gs = dict(zip(gdf["date"].to_list(), gdf["gravity_score"].to_list()))
            for i, d in enumerate(dates):
                weights_np[i] = 1.0 + float(date_to_gs.get(d, 0.0))
            w_mean = weights_np.mean()
            if w_mean > 0:
                weights_np /= w_mean
            log.info("Gravity weights: min=%.3f  max=%.3f  mean=%.3f",
                     weights_np.min(), weights_np.max(), weights_np.mean())
        except Exception as exc:
            log.warning("Impossible de charger gravity_path (%s) — poids uniformes", exc)

    # --- Positions spatiales depuis les zones constituantes ---
    lon_min, lon_max = LA_LON_MIN, LA_LON_MAX
    if zones_path is not None and Path(zones_path).exists():
        try:
            zones    = pl.read_parquet(zones_path).filter(pl.col("is_constituent"))
            lon_arr  = zones["lon"].to_numpy().astype(float)
            lon_min  = float(lon_arr.min())
            lon_max  = float(lon_arr.max())
            x_zones  = (lon_arr - lon_min) / max(lon_max - lon_min, 1e-8)
            log.info("Zones constituantes : %d zones  lon=[%.4f, %.4f]",
                     len(zones), lon_min, lon_max)
        except Exception as exc:
            log.warning("Impossible de charger zones_path (%s) — grille uniforme", exc)
            x_zones = np.linspace(0, 1, 100)
    else:
        x_zones = np.linspace(0, 1, 100)
        log.info("Pas de fichier zones — grille uniforme de 100 points")

    def mk(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=grad).reshape(-1, 1)

    x_data   = mk(x_vals)
    t_data   = mk(t_vals)
    rho_obs  = mk(rho_vals)
    v_obs    = mk(v_vals)
    weights  = mk(weights_np)

    # Collocation PDE : positions zones × temps aléatoire
    idx_col = np.random.choice(len(x_zones), n_col, replace=True)
    x_col   = mk(x_zones[idx_col], grad=True)
    t_col   = mk(np.random.uniform(0, 1, n_col), grad=True)

    # Points cinématiques
    idx_kin = np.random.choice(len(x_zones), n_kin, replace=True)
    x_kin   = mk(x_zones[idx_kin])
    t_kin   = mk(np.random.uniform(0, 1, n_kin), grad=True)

    # Conditions aux limites : densité baseline aux extrémités temporelles
    rho_bc_val  = float(rho_vals[rho_vals > 0].mean()) if rho_vals.max() > 0 else 0.5
    x_bc_np     = np.random.uniform(0, 1, n_bc)
    t_bc_np     = np.concatenate([np.zeros(n_bc // 2), np.ones(n_bc // 2)])
    rho_bc_np   = np.full(n_bc, rho_bc_val)
    x_bc        = mk(x_bc_np)
    t_bc        = mk(t_bc_np)
    rho_bc      = mk(rho_bc_np)

    log.info("Training data: %d jours | rho=[%.3f, %.3f] | v=[%.3f, %.3f]",
             len(df), rho_vals.min(), rho_vals.max(), v_vals.min(), v_vals.max())

    return {
        "x_data": x_data, "t_data": t_data,
        "rho_obs": rho_obs, "v_obs": v_obs,
        "weights": weights,
        "x_col": x_col, "t_col": t_col,
        "x_bc": x_bc, "t_bc": t_bc, "rho_bc": rho_bc,
        "x_kin": x_kin, "t_kin": t_kin,
        "lon_range": (lon_min, lon_max),
        "sog_max": sog_max,
        "x_zones": x_zones,
    }


# ---------------------------------------------------------------------------
# Refresh des points stochastiques (évite l'overfitting à la grille)
# ---------------------------------------------------------------------------

def _refresh_stochastic_points(data: dict, device: torch.device) -> None:
    """Retire et remplace les points de collocation et cinématiques."""
    x_zones = data["x_zones"]
    n_col   = len(data["x_col"])
    n_kin   = len(data["x_kin"])

    def mk(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=grad).reshape(-1, 1)

    idx_col = np.random.choice(len(x_zones), n_col, replace=True)
    data["x_col"] = mk(x_zones[idx_col], grad=True)
    data["t_col"] = mk(np.random.uniform(0, 1, n_col), grad=True)

    idx_kin = np.random.choice(len(x_zones), n_kin, replace=True)
    data["x_kin"] = mk(x_zones[idx_kin])
    data["t_kin"] = mk(np.random.uniform(0, 1, n_kin), grad=True)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train(
    epochs:        int   = DEFAULT_EPOCHS,
    lr:            float = DEFAULT_LR,
    lambda_pde:    float = DEFAULT_LAMBDA_PDE,
    lambda_bc:     float = DEFAULT_LAMBDA_BC,
    lambda_kin:    float = DEFAULT_LAMBDA_KIN,
    train_start:   date | str = date(2020, 1, 1),
    train_end:     date | str = date(2020, 12, 31),
    features_path: Path | str = Path("data/features/la_daily_features.parquet"),
    zones_path:    Path | str | None = Path("data/features/la_constituent_zones.parquet"),
    gravity_path:  Path | str | None = Path("data/features/la_gravity_daily.parquet"),
    model_path:    Path | str = Path("outputs/models/la_lwr_pinn.pt"),
) -> LWRPINN:
    if isinstance(train_start, str):
        train_start = date.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = date.fromisoformat(train_end)

    device = get_device()
    log.info("Device: %s", device)

    data = prepare_training_data(
        train_start, train_end, device,
        features_path=Path(features_path),
        zones_path=Path(zones_path) if zones_path else None,
        gravity_path=Path(gravity_path) if gravity_path else None,
    )

    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5)

    best_loss  = float("inf")
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        if epoch % 500 == 1 and epoch > 1:
            _refresh_stochastic_points(data, device)

        loss, components = total_loss(
            model,
            data["x_data"], data["t_data"], data["rho_obs"], data["v_obs"],
            data["x_col"], data["t_col"],
            data["x_bc"],  data["t_bc"],  data["rho_bc"],
            x_kin=data["x_kin"], t_kin=data["t_kin"],
            weights=data["weights"],
            lambda_pde=lambda_pde,
            lambda_bc=lambda_bc,
            lambda_kin=lambda_kin,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss)

        history.append(components)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  total=%.6f  data=%.6f  pde=%.6f  kin=%.6f  bc=%.6f",
                epoch, epochs,
                components["total"], components["data"],
                components["pde"],   components["kin"], components["bc"],
            )

    if best_state:
        model.load_state_dict(best_state)

    mp = Path(model_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "train_start": train_start.isoformat(),
        "train_end":   train_end.isoformat(),
        "best_loss":   best_loss,
        "history":     history,
        "epochs":      epochs,
        "lambda_pde":  lambda_pde,
        "lambda_bc":   lambda_bc,
        "lambda_kin":  lambda_kin,
        "lon_range":   data["lon_range"],
        "sog_max":     data["sog_max"],
    }, mp)
    log.info("Modèle sauvegardé → %s  (best_loss=%.6f)", mp, best_loss)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINN LWR Training")
    parser.add_argument("--location",    default="la", choices=["houston", "la"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde",  type=float, default=DEFAULT_LAMBDA_PDE)
    parser.add_argument("--lambda-bc",   type=float, default=DEFAULT_LAMBDA_BC)
    parser.add_argument("--lambda-kin",  type=float, default=DEFAULT_LAMBDA_KIN)
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end",   default="2020-12-31")
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--zones-path",    default=None)
    parser.add_argument("--gravity-path",  default=None)
    parser.add_argument("--model-path",    default=None)
    args = parser.parse_args()

    loc = args.location
    features_path = Path(args.features_path) if args.features_path else Path(
        f"data/features/{loc}_daily_features.parquet"
    )
    zones_path = Path(args.zones_path) if args.zones_path else Path(
        f"data/features/{loc}_constituent_zones.parquet"
    )
    gravity_path = Path(args.gravity_path) if args.gravity_path else Path(
        f"data/features/{loc}_gravity_daily.parquet"
    )
    model_path = Path(args.model_path) if args.model_path else Path(
        f"outputs/models/{loc}_lwr_pinn.pt"
    )

    train(
        epochs=args.epochs, lr=args.lr,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_kin=args.lambda_kin,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        features_path=features_path,
        zones_path=zones_path,
        gravity_path=gravity_path,
        model_path=model_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
