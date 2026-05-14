"""
Phase 3 — Visualisation des résultats du PINN LWR.

Trois graphiques :
  1. plot_density_timeseries  — ρ(t) prédit vs observé + gravity_score + TTC
  2. plot_spacetime_heatmap   — champ ρ(x, t) prédit sur une grille 60×60
  3. plot_loss_history        — décomposition de la loss d'entraînement

Usage autonome :
    python src/pinns/visualize_pinn.py --location la
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import torch

from src.pinns.lwr_pinn import LWRPINN, get_device

log = logging.getLogger(__name__)

FIGURE_DIR = Path("outputs/figures")


# ---------------------------------------------------------------------------
# 1. Série temporelle ρ prédit vs observé
# ---------------------------------------------------------------------------

def plot_density_timeseries(
    df_result:    pl.DataFrame,
    df_features:  pl.DataFrame,
    gravity_df:   pl.DataFrame | None = None,
    peak_date:    date | str           = date(2020, 8, 11),
    location:     str                  = "la",
    output_dir:   Path                 = FIGURE_DIR,
    ma_window:    int                  = 14,
) -> Path:
    """
    Série temporelle ρ_pred vs ρ_obs autour du pic de crise.
    Superpose le gravity_score (axe secondaire) et marque le TTC.
    """
    if isinstance(peak_date, str):
        peak_date = date.fromisoformat(peak_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{location}_pinn_timeseries.png"

    # Prépare les données observées autour du pic (± 60 jours)
    window_start = peak_date - timedelta(days=60)
    window_end   = peak_date + timedelta(days=90)

    obs = (
        df_features
        .filter(pl.col("date").is_between(window_start, window_end))
        .sort("date")
    )
    obs_dates = [d for d in obs["date"].to_list()]

    # Normalise blocked_capacity comme dans train.py (95e percentile)
    if "blocked_capacity" in obs.columns and obs["blocked_capacity"].max() > 0:
        full_cap = df_features["blocked_capacity"].to_numpy().astype(float)
        cap_95   = float(np.percentile(full_cap[full_cap > 0], 95))
        obs_rho  = np.clip(obs["blocked_capacity"].to_numpy().astype(float) / cap_95, 0.0, 1.0)
        rho_label = "Capacité bloquée ρ (norm.)"
    else:
        obs_rho   = obs["utilization_rate_rho"].to_numpy()
        rho_label = "Utilisation ρ"

    if len(obs_rho) >= max(ma_window, 1):
        kernel = np.ones(ma_window, dtype=float) / float(ma_window)
        obs_ma = np.convolve(obs_rho, kernel, mode="same")
    else:
        obs_ma = obs_rho

    pred_dates = [d for d in df_result["date"].to_list()]
    pred_rho   = df_result["rho_pred"].to_numpy()
    threshold  = float(df_result["rho_threshold"][0]) if len(df_result) > 0 else 0.0

    ttc_row  = df_result.filter(pl.col("time_to_clear_days").is_not_null())
    ttc_days = int(ttc_row["time_to_clear_days"][0]) if len(ttc_row) > 0 else None
    ttc_date = peak_date + timedelta(days=ttc_days) if ttc_days is not None else None

    fig, ax1 = plt.subplots(figsize=(13, 5))

    # Observé
    ax1.plot(obs_dates, obs_rho, color="#1565C0", lw=1.8, label="ρ observé (AIS)", zorder=3)
    ax1.plot(obs_dates, obs_ma, color="#455A64", lw=1.4, ls=":",
             label=f"Baseline MA({ma_window}j)", zorder=2)
    # Prédit
    ax1.plot(pred_dates, pred_rho, color="#E53935", lw=2.2, ls="--",
             label="ρ prédit (PINN LWR)", zorder=4)
    # Seuil TTC
    ax1.axhline(threshold, color="#43A047", lw=1.2, ls=":", label=f"Seuil TTC ({threshold:.3f})")
    # Pic
    ax1.axvline(peak_date, color="#FF6F00", lw=1.5, ls="-.", alpha=0.8,
                label=f"Pic ({peak_date})")
    # TTC
    if ttc_date:
        ax1.axvline(ttc_date, color="#6A1B9A", lw=2.0, alpha=0.9,
                    label=f"TTC = {ttc_days} jours")

    ax1.set_xlabel("Date")
    ax1.set_ylabel(rho_label, color="#1565C0")
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(axis="y", labelcolor="#1565C0")

    # Gravity score en superposition (axe secondaire)
    if gravity_df is not None and len(gravity_df) > 0:
        ax2 = ax1.twinx()
        grav = (
            gravity_df
            .filter(pl.col("date").is_between(window_start, window_end))
            .sort("date")
        )
        if len(grav) > 0:
            grav_dates = [d for d in grav["date"].to_list()]
            grav_score = grav["gravity_score"].to_numpy()
            ax2.fill_between(grav_dates, grav_score, alpha=0.15, color="#FF6F00",
                             label="Gravity score")
            ax2.set_ylabel("Gravity Score", color="#FF6F00")
            ax2.set_ylim(0, 1.5)
            ax2.tick_params(axis="y", labelcolor="#FF6F00")
            ax2.legend(loc="upper right", fontsize=8)

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=35, labelsize=8)

    ax1.legend(loc="upper left", fontsize=9)
    title_suffix = f" — TTC = {ttc_days} jours" if ttc_days else ""
    ax1.set_title(f"Port de {location.upper()} — PINN LWR : densité maritime{title_suffix}",
                  fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Série temporelle → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2. Heatmap spatio-temporelle ρ(x, t)
# ---------------------------------------------------------------------------

def plot_spacetime_heatmap(
    model_path:  Path | str = Path("outputs/models/la_lwr_pinn.pt"),
    train_start: date | str = date(2020, 1, 1),
    train_end:   date | str = date(2020, 12, 31),
    peak_date:   date | str = date(2020, 8, 11),
    location:    str        = "la",
    nx:          int        = 60,
    nt:          int        = 60,
    output_dir:  Path       = FIGURE_DIR,
) -> Path:
    """
    Évalue ρ(x, t) sur une grille nx×nt et produit une heatmap.
    x ∈ [0,1] correspond à la longitude normalisée du port.
    t ∈ [0,1] correspond à la fenêtre d'entraînement.
    """
    if isinstance(train_start, str):
        train_start = date.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = date.fromisoformat(train_end)
    if isinstance(peak_date, str):
        peak_date = date.fromisoformat(peak_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{location}_pinn_heatmap.png"

    device     = get_device()
    mp         = Path(model_path)
    checkpoint = torch.load(mp, map_location=device, weights_only=False)
    model      = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_train_days = (train_end - train_start).days + 1

    x_grid = np.linspace(0, 1, nx)
    t_grid = np.linspace(0, 1, nt)
    XX, TT = np.meshgrid(x_grid, t_grid)  # (nt, nx)

    x_flat = XX.ravel().astype(np.float32)
    t_flat = TT.ravel().astype(np.float32)

    x_t = torch.tensor(x_flat, device=device).reshape(-1, 1)
    t_t = torch.tensor(t_flat, device=device).reshape(-1, 1)

    with torch.no_grad():
        rho_flat = model(x_t, t_t)

    rho_grid = rho_flat.cpu().numpy().reshape(nt, nx)

    # Axes lisibles
    t_abs = [
        (train_start + timedelta(days=int(tt * (n_train_days - 1)))).isoformat()
        for tt in t_grid
    ]
    lon_range = checkpoint.get("lon_range", (-118.35, -118.05))
    x_abs     = np.linspace(lon_range[0], lon_range[1], nx)

    t_peak_norm = (peak_date - train_start).days / max(n_train_days - 1, 1)

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(
        rho_grid, aspect="auto", origin="lower",
        extent=[x_abs[0], x_abs[-1], 0, 1],
        cmap="RdYlGn_r", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Densité ρ normalisée")

    # Ligne de pic
    ax.axhline(t_peak_norm, color="white", lw=2, ls="--", alpha=0.9,
               label=f"Pic ({peak_date})")

    # Yticks → dates lisibles
    n_yticks = 8
    step_y   = max(1, nt // n_yticks)
    ax.set_yticks(t_grid[::step_y])
    ax.set_yticklabels(t_abs[::step_y], fontsize=7, rotation=30)

    ax.set_xlabel("Longitude (normalisée → Est)")
    ax.set_ylabel("Date")
    ax.set_title(f"Port de {location.upper()} — Champ ρ(x,t) prédit par le PINN LWR",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap spatio-temporelle → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Courbes de loss d'entraînement
# ---------------------------------------------------------------------------

def plot_loss_history(
    model_path: Path | str = Path("outputs/models/la_lwr_pinn.pt"),
    location:   str        = "la",
    output_dir: Path       = FIGURE_DIR,
) -> Path:
    """
    Trace la décomposition de la loss sur les epochs d'entraînement.
    Quatre composantes : data, pde, kin, bc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{location}_pinn_loss.png"

    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    history    = checkpoint.get("history", [])
    if not history:
        log.warning("Pas d'historique de loss dans le checkpoint")
        return out_path

    epochs = np.arange(1, len(history) + 1)
    keys   = ["total", "data", "pde", "kin", "bc"]
    colors = ["#212121", "#1565C0", "#E53935", "#FF6F00", "#43A047"]
    styles = ["-",       "--",      "-.",      ":",       (0,(3,1,1,1))]
    labels = ["Total", "Data", "PDE", "Cinématique", "BC"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for key, col, ls, lbl in zip(keys, colors, styles, labels):
        vals = [h.get(key, 0) for h in history]
        ax.semilogy(epochs, vals, color=col, ls=ls, lw=1.6, label=lbl)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (échelle log)")
    ax.set_title(f"Port de {location.upper()} — Décomposition de la loss PINN", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Loss history → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def visualize_all(
    location:     str       = "la",
    peak_date:    str       = "2020-08-11",
    train_start:  str       = "2020-01-01",
    train_end:    str       = "2020-12-31",
    output_dir:   Path      = FIGURE_DIR,
) -> dict[str, Path]:
    model_path    = Path(f"outputs/models/{location}_lwr_pinn.pt")
    features_path = Path(f"data/features/{location}_daily_features.parquet")
    ttc_path      = Path(f"data/features/{location}_time_to_clear.parquet")
    gravity_path  = Path(f"data/features/{location}_gravity_daily.parquet")

    for p in [model_path, features_path, ttc_path]:
        if not p.exists():
            log.error("Fichier manquant : %s", p)
            return {}

    df_result   = pl.read_parquet(ttc_path)
    df_features = pl.read_parquet(features_path).sort("date")
    gravity_df  = pl.read_parquet(gravity_path).sort("date") if gravity_path.exists() else None

    paths = {}

    paths["timeseries"] = plot_density_timeseries(
        df_result, df_features, gravity_df,
        peak_date=peak_date, location=location, output_dir=output_dir,
    )
    paths["heatmap"] = plot_spacetime_heatmap(
        model_path=model_path,
        train_start=train_start, train_end=train_end,
        peak_date=peak_date, location=location,
        output_dir=output_dir,
    )
    paths["loss"] = plot_loss_history(
        model_path=model_path, location=location, output_dir=output_dir,
    )
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 — Visualisation PINN")
    parser.add_argument("--location",    default="la")
    parser.add_argument("--peak-date",   default="2020-08-11")
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end",   default="2020-12-31")
    args = parser.parse_args()

    paths = visualize_all(
        location=args.location,
        peak_date=args.peak_date,
        train_start=args.train_start,
        train_end=args.train_end,
    )
    for name, p in paths.items():
        print(f"  {name:<14} → {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
