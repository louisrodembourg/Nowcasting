"""
Phase 3 — Entraînement du PINN LWR.

Pré-requis : le manifold géospatial Phase 2 doit avoir été exécuté :
    python run_phase2.py --location <loc> ...
    → génère data/features/{loc}_constituent_zones.parquet  (requis)
              data/features/{loc}_gravity_daily.parquet     (optionnel, pondération)

Fonctionnement :
  - Spatialisation réelle : les points de collocation PDE sont échantillonnés
    depuis les positions LON des zones constituantes (manifold géospatial Phase 2).
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
DEFAULT_EPOCHS          = 3000
DEFAULT_LR              = 5e-4
DEFAULT_N_COL           = 2000
DEFAULT_N_KIN           = 500
DEFAULT_LAMBDA_PDE      = 0.3
DEFAULT_LAMBDA_BC       = 0.1
DEFAULT_LAMBDA_KIN      = 0.1
LOG_EVERY               = 200
DEFAULT_CURRICULUM      = 500   # epochs data-only avant montée progressive physique
DEFAULT_OBS_PER_DAY     = 1     # point spatial par observation (x=0.5 centroïde du port)
DEFAULT_LRA_ALPHA       = 0.9   # EMA pour LRA
DEFAULT_LRA_MAX         = 0.5   # cap λ_pde_max (évite la solution triviale ρ=const)
DEFAULT_LRA_FREQ        = 50    # fréquence de mise à jour LRA (epochs)


# ---------------------------------------------------------------------------
# Préparation des données d'entraînement avec spatialisation
# ---------------------------------------------------------------------------

def prepare_training_data_perzone(
    train_start:       date,
    train_end:         date,
    device:            torch.device,
    zone_features_path: Path,
    zones_path:        Path,
    gravity_path:      Path | None = None,
    n_col:             int  = DEFAULT_N_COL,
    n_kin:             int  = DEFAULT_N_KIN,
    n_bc:              int  = 50,
    max_obs:           int  = 20_000,
) -> dict:
    """
    Variante per-zone de prepare_training_data.

    Charge houston_zone_daily_features.parquet (produit par features_per_zone.py)
    et construit des observations spatiales réelles :
      x = lon_norm (position de la zone sur [0,1])
      t = (date - train_start) / n_train_days
      rho_obs = rho_norm (densité per-zone)
      v_obs   = 1 - rho_norm  (Greenshields implicite à l'obs.)

    Remplace l'approche centroïde x=0.5 par ~147k observations spatiales.
    max_obs : plafond de sous-échantillonnage (évite un batch trop lourd sur MPS).
    """
    df_zones_feat = (
        pl.read_parquet(zone_features_path)
        .sort(["date", "lon"])
        .filter(
            pl.col("date").is_between(train_start, train_end)
            & (pl.col("vessel_count") > 0)
        )
    )
    if len(df_zones_feat) == 0:
        raise ValueError(
            f"Aucune observation per-zone entre {train_start} et {train_end} dans {zone_features_path}"
        )

    n_train_days = (train_end - train_start).days + 1
    dates_z      = df_zones_feat["date"].to_list()
    t_arr        = np.array([(d - train_start).days / max(n_train_days - 1, 1) for d in dates_z])
    x_arr        = df_zones_feat["lon_norm"].to_numpy().astype(float)
    rho_arr      = df_zones_feat["rho_norm"].to_numpy().astype(float)
    v_arr        = df_zones_feat["v_norm"].to_numpy().astype(float)  # SOG réel

    # Pondération gravity (optionnel)
    weights_arr = np.ones(len(df_zones_feat))
    if gravity_path is not None and Path(gravity_path).exists():
        try:
            gdf = pl.read_parquet(gravity_path).with_columns(
                pl.col("date").cast(pl.Utf8).str.to_date().alias("date")
            )
            date_to_gs = dict(zip(gdf["date"].to_list(), gdf["gravity_score"].to_list()))
            for i, d in enumerate(dates_z):
                weights_arr[i] = 1.0 + float(date_to_gs.get(d, 0.0))
            w_mean = weights_arr.mean()
            if w_mean > 0:
                weights_arr /= w_mean
        except Exception as exc:
            log.warning("gravity_path illisible (%s) — poids uniformes", exc)

    # Sous-échantillonnage si trop de points
    n_obs = len(x_arr)
    if n_obs > max_obs:
        idx = np.random.choice(n_obs, max_obs, replace=False)
        x_arr = x_arr[idx]; t_arr = t_arr[idx]
        rho_arr = rho_arr[idx]; v_arr = v_arr[idx]
        weights_arr = weights_arr[idx]
        log.info("Sous-échantillonnage obs : %d → %d", n_obs, max_obs)
    log.info(
        "Observations per-zone : %d points  x=[%.3f,%.3f]  rho=[%.3f,%.3f]  v=[%.3f,%.3f]",
        len(x_arr), x_arr.min(), x_arr.max(),
        rho_arr.min(), rho_arr.max(), v_arr.min(), v_arr.max(),
    )

    # Zones pour collocation PDE et BC
    zp    = Path(zones_path)
    zones = pl.read_parquet(zp).filter(pl.col("is_constituent"))
    lon_z = zones["lon"].to_numpy().astype(float)
    lon_min, lon_max = float(lon_z.min()), float(lon_z.max())
    x_zones_col = (lon_z - lon_min) / max(lon_max - lon_min, 1e-8)

    def mk(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=grad).reshape(-1, 1)

    x_data  = mk(x_arr)
    t_data  = mk(t_arr)
    rho_obs = mk(rho_arr)
    v_obs   = mk(v_arr)
    weights = mk(weights_arr)

    idx_col = np.random.choice(len(x_zones_col), n_col, replace=True)
    x_col   = mk(x_zones_col[idx_col], grad=True)
    t_col   = mk(np.random.uniform(0, 1, n_col), grad=True)

    idx_kin = np.random.choice(len(x_zones_col), n_kin, replace=True)
    x_kin   = mk(x_zones_col[idx_kin])
    t_kin   = mk(np.random.uniform(0, 1, n_kin), grad=True)

    rho_bc_val = float(rho_arr.mean()) if rho_arr.max() > 0 else 0.3
    x_bc  = mk(np.random.uniform(0, 1, n_bc))
    t_bc  = mk(np.concatenate([np.zeros(n_bc // 2), np.ones(n_bc // 2)]))
    rho_bc = mk(np.full(n_bc, rho_bc_val))

    return {
        "x_data": x_data, "t_data": t_data,
        "rho_obs": rho_obs, "v_obs": v_obs, "weights": weights,
        "x_col": x_col, "t_col": t_col,
        "x_bc": x_bc, "t_bc": t_bc, "rho_bc": rho_bc,
        "x_kin": x_kin, "t_kin": t_kin,
        "lon_range": (lon_min, lon_max),
        "sog_max": 1.0,
        "x_zones": x_zones_col,
    }


def prepare_training_data(
    train_start:   date,
    train_end:     date,
    device:        torch.device,
    features_path: Path,
    zones_path:    Path,
    gravity_path:  Path | None = None,
    n_col:         int  = DEFAULT_N_COL,
    n_kin:         int  = DEFAULT_N_KIN,
    n_bc:          int  = 50,
    obs_per_day:   int  = DEFAULT_OBS_PER_DAY,
) -> dict[str, torch.Tensor | tuple]:
    """
    Prépare les tenseurs d'entraînement pour le PINN LWR.

    Spatialisation (manifold géospatial obligatoire) :
      Les points de collocation PDE sont tirés depuis les positions LON réelles
      des zones constituantes Phase 2 (zones_path doit exister).

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

    # ρ : waiting_capacity normalisée pour LA (pic de crise = haute capacité bloquée)
    #     utilization_rate_rho pour Houston (port fermé = chute de ρ)
    if "waiting_capacity" in df.columns and df["waiting_capacity"].max() > 0:
        cap_arr  = df["waiting_capacity"].to_numpy().astype(float)
        cap_95   = float(np.percentile(cap_arr[cap_arr > 0], 95)) if cap_arr.max() > 0 else 1.0
        rho_vals = np.clip(cap_arr / cap_95, 0.0, 1.0)
        log.info("ρ = waiting_capacity normalisée (95p=%.0f)", cap_95)
    else:
        rho_vals = df["utilization_rate_rho"].to_numpy().astype(float)
        log.info("ρ = utilization_rate_rho")

    sog      = df["SOG_mean"].to_numpy().astype(float)
    sog_max  = sog.max() if sog.max() > 0 else 1.0
    v_vals   = sog / sog_max

    # Observation : multi-point spatial (obs_per_day zones constituantes par jour)
    # Chaque jour contribue K observations à différentes positions du domaine [0,1]
    # avec la même ρ_obs/v_obs → cohérence avec les collocation points PDE

    # --- Pondération par gravity score ---
    weights_np = np.ones(len(df))
    if gravity_path is not None and Path(gravity_path).exists():
        try:
            gdf = pl.read_parquet(gravity_path).with_columns(
                pl.col("date").cast(pl.Utf8).str.to_date().alias("date")
            ).sort("date")
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

    # --- Positions spatiales depuis les zones constituantes (manifold géospatial) ---
    zp = Path(zones_path)
    if not zp.exists():
        raise FileNotFoundError(
            f"Manifold géospatial introuvable : {zp}\n"
            "Lancez d'abord run_phase2.py pour générer les zones constituantes."
        )
    zones   = pl.read_parquet(zp).filter(pl.col("is_constituent"))
    if len(zones) == 0:
        raise ValueError(f"Aucune zone constituante dans {zp} — relancez run_phase2.py")
    lon_arr = zones["lon"].to_numpy().astype(float)
    lon_min = float(lon_arr.min())
    lon_max = float(lon_arr.max())
    x_zones = (lon_arr - lon_min) / max(lon_max - lon_min, 1e-8)
    log.info("Zones constituantes : %d zones  lon=[%.4f, %.4f]", len(zones), lon_min, lon_max)

    # Observations spatiales : x=0.5 (centroïde port) pour données agrégées quotidiennes
    # Distribuer les observations sur plusieurs zones forcerait ρ uniforme en x → trivial LWR
    # Le paramètre obs_per_day est conservé pour des extensions futures avec données per-zone
    K = max(1, obs_per_day)
    n_days = len(df)
    if K == 1:
        # Mode standard : centroïde du port (physiquement correct pour données agrégées)
        x_vals   = np.full(n_days, 0.5)
        v_vals_r = v_vals
        weights_r = weights_np
    else:
        # Mode multi-zone (expérimental — nécessite des features par zone)
        idx_obs  = np.random.choice(len(x_zones), size=(n_days, K), replace=True)
        x_vals   = x_zones[idx_obs].ravel()
        t_vals   = np.repeat(t_vals,    K)
        rho_vals = np.repeat(rho_vals,  K)
        v_vals_r = np.repeat(v_vals,    K)
        weights_r = np.repeat(weights_np, K)

    def mk(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=grad).reshape(-1, 1)

    x_data   = mk(x_vals)
    t_data   = mk(t_vals)
    rho_obs  = mk(rho_vals)
    v_obs    = mk(v_vals_r)
    weights  = mk(weights_r)
    log.info("Observations : %d points (x=%.3f±%.3f)", len(x_vals),
             float(x_vals.mean()), float(x_vals.std()) if K > 1 else 0.0)

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
    epochs:             int   = DEFAULT_EPOCHS,
    lr:                 float = DEFAULT_LR,
    lambda_pde:         float = DEFAULT_LAMBDA_PDE,
    lambda_bc:          float = DEFAULT_LAMBDA_BC,
    lambda_kin:         float = DEFAULT_LAMBDA_KIN,
    train_start:        date | str = date(2020, 1, 1),
    train_end:          date | str = date(2020, 12, 31),
    features_path:      Path | str = Path("data/features/la_daily_features.parquet"),
    zones_path:         Path | str = Path("data/features/la_constituent_zones.parquet"),
    gravity_path:       Path | str | None = Path("data/features/la_gravity_daily.parquet"),
    model_path:         Path | str = Path("outputs/models/la_lwr_pinn.pt"),
    curriculum_warmup:  int   = DEFAULT_CURRICULUM,
    obs_per_day:        int   = DEFAULT_OBS_PER_DAY,
    use_lra:            bool  = True,
    lra_alpha:          float = DEFAULT_LRA_ALPHA,
    lra_max:            float = DEFAULT_LRA_MAX,
    lra_freq:           int   = DEFAULT_LRA_FREQ,
    zone_features_path: Path | str | None = None,
    max_obs:            int   = 20_000,
) -> LWRPINN:
    if isinstance(train_start, str):
        train_start = date.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = date.fromisoformat(train_end)

    device = get_device()
    log.info("Device: %s", device)

    if zone_features_path is not None and Path(zone_features_path).exists():
        log.info("Mode per-zone : %s (max_obs=%d)", zone_features_path, max_obs)
        data = prepare_training_data_perzone(
            train_start, train_end, device,
            zone_features_path=Path(zone_features_path),
            zones_path=Path(zones_path),
            gravity_path=Path(gravity_path) if gravity_path else None,
            max_obs=max_obs,
        )
    else:
        data = prepare_training_data(
            train_start, train_end, device,
            features_path=Path(features_path),
            zones_path=Path(zones_path),
            gravity_path=Path(gravity_path) if gravity_path else None,
            obs_per_day=obs_per_day,
        )

    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=300, factor=0.5)

    best_loss  = float("inf")
    best_state = None
    history: list[dict] = []

    # LRA state
    lambda_pde_ema = float(lambda_pde)
    if use_lra:
        log.info("LRA activé (α=%.2f, λ_max=%.1f, freq=%d)", lra_alpha, lra_max, lra_freq)
    if curriculum_warmup > 0:
        log.info("Curriculum : %d epochs data-only → %d epochs ramp → physique complète",
                 curriculum_warmup, curriculum_warmup)

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        if epoch % 500 == 1 and epoch > 1:
            _refresh_stochastic_points(data, device)

        # Curriculum : montée progressive des termes physiques
        if curriculum_warmup > 0:
            if epoch < curriculum_warmup:
                lam_pde_eff = lam_bc_eff = lam_kin_eff = 0.0
            elif epoch < curriculum_warmup * 2:
                progress    = (epoch - curriculum_warmup) / float(curriculum_warmup)
                lam_pde_eff = lambda_pde_ema * progress
                lam_bc_eff  = lambda_bc  * progress
                lam_kin_eff = lambda_kin * progress
            else:
                lam_pde_eff = lambda_pde_ema
                lam_bc_eff  = lambda_bc
                lam_kin_eff = lambda_kin
        else:
            lam_pde_eff = lambda_pde_ema
            lam_bc_eff  = lambda_bc
            lam_kin_eff = lambda_kin

        loss, components = total_loss(
            model,
            data["x_data"], data["t_data"], data["rho_obs"], data["v_obs"],
            data["x_col"], data["t_col"],
            data["x_bc"],  data["t_bc"],  data["rho_bc"],
            x_kin=data["x_kin"], t_kin=data["t_kin"],
            weights=data["weights"],
            lambda_pde=lam_pde_eff,
            lambda_bc=lam_bc_eff,
            lambda_kin=lam_kin_eff,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # LRA : met à jour lambda_pde_ema tous les lra_freq epochs (hors phase warmup)
        if use_lra and epoch % lra_freq == 0 and epoch >= curriculum_warmup:
            try:
                from src.pinns.lwr_pinn import data_loss as _dl, pde_loss as _pl
                with torch.enable_grad():
                    l_d = _dl(model, data["x_data"], data["t_data"],
                               data["rho_obs"], data["v_obs"], data["weights"])
                    l_p = _pl(model, data["x_col"], data["t_col"])
                grads_d = torch.autograd.grad(l_d, model.parameters(),
                                               retain_graph=True, allow_unused=True)
                grads_p = torch.autograd.grad(l_p, model.parameters(),
                                               retain_graph=True, allow_unused=True)
                mean_d = float(torch.stack([g.abs().mean() for g in grads_d if g is not None]).mean())
                mean_p = float(torch.stack([g.abs().mean() for g in grads_p if g is not None]).mean())
                if mean_p > 1e-12:
                    lra_raw = mean_d / mean_p
                    lambda_pde_ema = lra_alpha * lambda_pde_ema + (1.0 - lra_alpha) * lra_raw
                    lambda_pde_ema = min(lambda_pde_ema, lra_max)
            except Exception:
                pass  # LRA non critique, on continue si erreur

        opt.step()
        scheduler.step(loss)

        history.append({**components, "lambda_pde_eff": lam_pde_eff})

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  total=%.6f  data=%.6f  pde=%.6f  kin=%.6f  bc=%.6f  λ_pde=%.4f",
                epoch, epochs,
                components["total"], components["data"],
                components["pde"],   components["kin"], components["bc"],
                lam_pde_eff,
            )

    if best_state:
        model.load_state_dict(best_state)

    mp = Path(model_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":      model.state_dict(),
        "train_start":      train_start.isoformat(),
        "train_end":        train_end.isoformat(),
        "best_loss":        best_loss,
        "history":          history,
        "epochs":           epochs,
        "lambda_pde":       lambda_pde,
        "lambda_pde_final": lambda_pde_ema,
        "lambda_bc":        lambda_bc,
        "lambda_kin":       lambda_kin,
        "lon_range":        data["lon_range"],
        "sog_max":          data["sog_max"],
        "arch_params": {
            "hidden_layers": 4,
            "hidden_size":   64,
        },
    }, mp)
    log.info("Modèle sauvegardé → %s  (best_loss=%.6f)", mp, best_loss)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINN LWR Training")
    parser.add_argument("--location",    default="la", choices=["houston", "la"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde",         type=float, default=DEFAULT_LAMBDA_PDE)
    parser.add_argument("--lambda-bc",          type=float, default=DEFAULT_LAMBDA_BC)
    parser.add_argument("--lambda-kin",         type=float, default=DEFAULT_LAMBDA_KIN)
    parser.add_argument("--train-start",        default="2020-01-01")
    parser.add_argument("--train-end",          default="2020-12-31")
    parser.add_argument("--curriculum-warmup",  type=int,   default=DEFAULT_CURRICULUM)
    parser.add_argument("--obs-per-day",        type=int,   default=DEFAULT_OBS_PER_DAY)
    parser.add_argument("--lra",                action="store_true", default=True)
    parser.add_argument("--no-lra",             dest="lra", action="store_false")
    parser.add_argument("--lra-alpha",          type=float, default=DEFAULT_LRA_ALPHA)
    parser.add_argument("--lra-max",            type=float, default=DEFAULT_LRA_MAX)
    parser.add_argument("--lra-freq",           type=int,   default=DEFAULT_LRA_FREQ)
    parser.add_argument("--features-path",      default=None)
    parser.add_argument("--zones-path",          default=None)
    parser.add_argument("--gravity-path",        default=None)
    parser.add_argument("--model-path",          default=None)
    parser.add_argument("--zone-features-path",  default=None,
                        help="Chemin vers houston_zone_daily_features.parquet (per-zone)")
    parser.add_argument("--max-obs",             type=int, default=20_000,
                        help="Plafond d'observations per-zone (sous-échantillonnage)")
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
    zone_features_path = Path(args.zone_features_path) if args.zone_features_path else None

    train(
        epochs=args.epochs, lr=args.lr,
        lambda_pde=args.lambda_pde, lambda_bc=args.lambda_bc, lambda_kin=args.lambda_kin,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        features_path=features_path,
        zones_path=zones_path,
        gravity_path=gravity_path,
        model_path=model_path,
        curriculum_warmup=args.curriculum_warmup,
        obs_per_day=args.obs_per_day,
        use_lra=args.lra,
        lra_alpha=args.lra_alpha,
        lra_max=args.lra_max,
        lra_freq=args.lra_freq,
        zone_features_path=zone_features_path,
        max_obs=args.max_obs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
