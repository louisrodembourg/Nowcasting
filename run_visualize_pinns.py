"""
Visualisation complète des résultats PINN — Phase 3.
Génère 3 figures par localisation :
  1. Timeseries : ρ prédit vs observé + gravity score + TTC
  2. Spacetime heatmap : champ ρ(x,t) sur la grille d'entraînement
  3. Loss history : décomposition des composantes de loss

Usage:
    python run_visualize_pinns.py
    python run_visualize_pinns.py --location la
    python run_visualize_pinns.py --location houston
"""
import argparse
import logging
from datetime import date, timedelta, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
import torch

from src.pinns.lwr_pinn import LWRPINN, get_device

log = logging.getLogger(__name__)
OUT = Path("outputs/figures")

# ── Couleurs ───────────────────────────────────────────────────────────────────
C_OBS    = "#1565C0"
C_PRED   = "#E53935"
C_THRESH = "#43A047"
C_PEAK   = "#FF6F00"
C_TTC    = "#6A1B9A"
C_GRAV   = "#FF6F00"
C_MA     = "#455A64"


def _to_dt(d) -> datetime:
    """Convertit date/str en datetime pour matplotlib."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime(d.year, d.month, d.day)
    return d


def _load_model(model_path: Path, device: torch.device):
    ck = torch.load(model_path, map_location=device, weights_only=False)
    ap = ck.get("arch_params", {"hidden_layers": 4, "hidden_size": 64})
    model = LWRPINN(
        hidden_layers=ap.get("hidden_layers", 4),
        hidden_size=ap.get("hidden_size", 64),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


# ── Figure 1 : Timeseries ─────────────────────────────────────────────────────

def plot_timeseries(loc: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{loc}_pinn_timeseries.png"

    feat_path  = Path(f"data/features/{loc}_daily_features.parquet")
    ttc_path   = Path(f"data/features/{loc}_time_to_clear.parquet")
    grav_path  = Path(f"data/features/{loc}_gravity_daily.parquet")
    model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")

    for p in [feat_path, ttc_path, model_path]:
        if not p.exists():
            log.error("Fichier manquant : %s", p)
            return out_path

    feat_df = pl.read_parquet(feat_path).sort("date")
    ttc_df  = pl.read_parquet(ttc_path)
    grav_df = pl.read_parquet(grav_path).with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date().alias("date")
    ).sort("date") if grav_path.exists() else None

    _, ck = _load_model(model_path, get_device())
    peak_date = date.fromisoformat(ck["train_end"]) - timedelta(days=180)

    # TTC info
    ttc_rows = ttc_df.filter(pl.col("time_to_clear_days").is_not_null())
    ttc_days = int(ttc_rows["time_to_clear_days"][0]) if len(ttc_rows) > 0 else None

    # Peak date from checkpoint comment or TTC parquet
    # Try to infer peak from TTC file start
    first_eval = ttc_df["date"][0]
    if isinstance(first_eval, str):
        first_eval = date.fromisoformat(first_eval)
    peak_date = first_eval

    ttc_date = peak_date + timedelta(days=ttc_days) if ttc_days else None
    threshold = float(ttc_df["rho_threshold"][0])

    # Fenêtre d'affichage : ± 90 jours autour du pic
    win_start = peak_date - timedelta(days=90)
    win_end   = peak_date + timedelta(days=min(ttc_days + 30 if ttc_days else 120, 180))

    # ρ observé normalisé
    window_feat = feat_df.filter(pl.col("date").is_between(win_start, win_end))
    if "waiting_capacity" in feat_df.columns and feat_df["waiting_capacity"].max() > 0:
        train_start = date.fromisoformat(ck["train_start"])
        train_end   = date.fromisoformat(ck["train_end"])
        train_cap   = feat_df.filter(
            pl.col("date").is_between(train_start, train_end)
        )["waiting_capacity"].to_numpy().astype(float)
        cap_95 = float(np.percentile(train_cap[train_cap > 0], 95))
        obs_rho = np.clip(window_feat["waiting_capacity"].to_numpy().astype(float) / cap_95, 0, 1)
        rho_label = "Capacité bloquée ρ (norm.)"
    else:
        obs_rho = window_feat["utilization_rate_rho"].to_numpy()
        rho_label = "Utilisation ρ"

    obs_dates = [_to_dt(d) for d in window_feat["date"].to_list()]
    ma14 = np.convolve(obs_rho, np.ones(14)/14, mode="same")

    pred_dates = [_to_dt(d) for d in ttc_df["date"].to_list()]
    pred_rho   = ttc_df["rho_pred"].to_numpy()

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax_grav) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=False)

    # Axe principal
    ax1.fill_between(obs_dates, obs_rho, alpha=0.25, color=C_OBS)
    ax1.plot(obs_dates, obs_rho, color=C_OBS, lw=1.5, label="ρ observé (AIS)")
    ax1.plot(obs_dates, ma14,   color=C_MA,  lw=1.2, ls=":", label="MA 14j observé")
    ax1.plot(pred_dates, pred_rho, color=C_PRED, lw=2.5, ls="--", label="ρ prédit (PINN LWR)")
    ax1.axhline(threshold, color=C_THRESH, lw=1.5, ls=":", label=f"Seuil TTC ({threshold:.3f})")
    ax1.axvline(_to_dt(peak_date), color=C_PEAK, lw=2, ls="-.", alpha=0.9,
                label=f"Pic ({peak_date})")
    if ttc_date:
        ax1.axvline(_to_dt(ttc_date), color=C_TTC, lw=2.5, alpha=0.95,
                    label=f"TTC = {ttc_days}j ({ttc_date})")
        ax1.axvspan(_to_dt(peak_date), _to_dt(ttc_date), alpha=0.05, color=C_PRED)

    ax1.set_ylabel(rho_label, fontsize=11)
    ax1.set_ylim(-0.02, 1.15)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    title_ttc = f"TTC = {ttc_days} jours" if ttc_days else "TTC non atteint"
    ax1.set_title(f"{loc.upper()} — PINN LWR : densité maritime  ({title_ttc})",
                  fontsize=13, fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # Gravity score (panneau bas)
    if grav_df is not None:
        grav_win = grav_df.filter(pl.col("date").is_between(win_start, win_end))
        if len(grav_win) > 0:
            gd = [_to_dt(d) for d in grav_win["date"].to_list()]
            gs = grav_win["gravity_score"].to_numpy().astype(float)
            gs_norm = gs / max(gs.max(), 1)
            ax_grav.fill_between(gd, gs_norm, alpha=0.6, color=C_GRAV)
            ax_grav.plot(gd, gs_norm, color=C_GRAV, lw=1.2)
            ax_grav.axvline(_to_dt(peak_date), color=C_PEAK, lw=2, ls="-.", alpha=0.7)
            if ttc_date:
                ax_grav.axvline(_to_dt(ttc_date), color=C_TTC, lw=2, alpha=0.7)
    ax_grav.set_ylabel("Gravity\nScore (norm.)", fontsize=9)
    ax_grav.set_ylim(0, 1.2)
    ax_grav.grid(True, alpha=0.2)
    ax_grav.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_grav.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_grav.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Timeseries → %s", out_path)
    return out_path


# ── Figure 2 : Spacetime heatmap ──────────────────────────────────────────────

def plot_heatmap(loc: str, out_dir: Path, nx: int = 80, nt: int = 80) -> Path:
    out_path = out_dir / f"{loc}_pinn_heatmap.png"
    model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    if not model_path.exists():
        return out_path

    device = get_device()
    model, ck = _load_model(model_path, device)

    train_start  = date.fromisoformat(ck["train_start"])
    train_end    = date.fromisoformat(ck["train_end"])
    lon_range    = ck.get("lon_range", (-95.3, -94.2) if loc == "houston" else (-118.5, -118.0))
    n_train_days = (train_end - train_start).days + 1

    x_grid = np.linspace(0, 1, nx, dtype=np.float32)
    t_grid = np.linspace(0, 1, nt, dtype=np.float32)
    XX, TT = np.meshgrid(x_grid, t_grid)

    x_t = torch.tensor(XX.ravel(), device=device).reshape(-1, 1)
    t_t = torch.tensor(TT.ravel(), device=device).reshape(-1, 1)

    with torch.no_grad():
        rho_flat = model(x_t, t_t).cpu().numpy().ravel()

    rho_grid = rho_flat.reshape(nt, nx)

    # Labels axes
    lon_labels = np.linspace(lon_range[0], lon_range[1], 6)
    t_dates    = [train_start + timedelta(days=int(tt * (n_train_days - 1))) for tt in t_grid]
    t_step     = max(1, nt // 8)
    t_ticks    = range(0, nt, t_step)
    t_labels   = [t_dates[i].strftime("%Y-%m-%d") for i in t_ticks]

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(rho_grid, aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdYlGn_r", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Densité ρ normalisée", fontsize=10)

    # Axe x → longitude
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{l:.2f}°" for l in lon_labels], fontsize=8)

    # Axe y → dates
    ax.set_yticks([i / nt for i in t_ticks])
    ax.set_yticklabels(t_labels, fontsize=7)

    ax.set_xlabel("Longitude normalisée (Est →)", fontsize=10)
    ax.set_ylabel("Date", fontsize=10)
    ax.set_title(f"{loc.upper()} — Champ ρ(x,t) prédit par le PINN LWR\n"
                 f"({train_start} → {train_end})", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap → %s", out_path)
    return out_path


# ── Figure 3 : Loss history ───────────────────────────────────────────────────

def plot_loss(loc: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{loc}_pinn_loss.png"
    model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    if not model_path.exists():
        return out_path

    ck      = torch.load(model_path, map_location="cpu", weights_only=False)
    history = ck.get("history", [])
    if not history:
        return out_path

    epochs = np.arange(1, len(history) + 1)
    keys   = ["total", "data", "pde", "kin", "bc"]
    labels = ["Total", "Data", "PDE résidu", "Cinématique", "BC"]
    colors = ["#212121", C_OBS, C_PRED, C_PEAK, C_THRESH]
    styles = ["-", "--", "-.", ":", (0,(3,1,1,1))]

    fig, (ax_loss, ax_lambda) = plt.subplots(2, 1, figsize=(12, 6),
                                              gridspec_kw={"height_ratios": [3, 1]},
                                              sharex=True)

    for key, lbl, col, ls in zip(keys, labels, colors, styles):
        vals = np.array([h.get(key, 0) for h in history])
        vals = np.where(vals <= 0, np.nan, vals)
        ax_loss.semilogy(epochs, vals, color=col, ls=ls, lw=1.5, label=lbl)

    ax_loss.set_ylabel("Loss (log)", fontsize=10)
    ax_loss.legend(fontsize=9, ncol=2)
    ax_loss.grid(True, which="both", alpha=0.25)
    ax_loss.set_title(f"{loc.upper()} — Convergence PINN  (best_loss={ck['best_loss']:.5f})",
                      fontsize=12, fontweight="bold")

    # Lambda PDE effectif (si disponible depuis LRA)
    lambda_vals = [h.get("lambda_pde_eff", np.nan) for h in history]
    if any(not np.isnan(v) for v in lambda_vals):
        lv = np.array(lambda_vals)
        lv[lv == 0] = np.nan
        ax_lambda.plot(epochs, lv, color="#9C27B0", lw=1.5, label="λ_pde effectif")
        ax_lambda.set_ylabel("λ_pde", fontsize=9)
        ax_lambda.legend(fontsize=8)
        ax_lambda.grid(True, alpha=0.25)
    else:
        ax_lambda.set_visible(False)

    ax_lambda.set_xlabel("Epoch", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Loss → %s", out_path)
    return out_path


# ── Figure 4 : Vue d'ensemble gravity_score 2017–2024 ─────────────────────────

def plot_gravity_overview(loc: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{loc}_gravity_overview.png"
    grav_path = Path(f"data/features/{loc}_gravity_daily.parquet")
    feat_path = Path(f"data/features/{loc}_daily_features.parquet")

    if not grav_path.exists() or not feat_path.exists():
        return out_path

    grav = pl.read_parquet(grav_path).with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date().alias("date")
    ).sort("date")

    feat = pl.read_parquet(feat_path).sort("date")
    if "waiting_capacity" in feat.columns:
        cap_arr = feat["waiting_capacity"].to_numpy().astype(float)
        cap_95  = float(np.percentile(cap_arr[cap_arr > 0], 95))
        feat = feat.with_columns((pl.col("waiting_capacity") / cap_95).clip(0, 1).alias("rho_norm"))
    else:
        feat = feat.with_columns(pl.col("utilization_rate_rho").alias("rho_norm"))

    gd = [_to_dt(d) for d in grav["date"].to_list()]
    gs = grav["gravity_score"].to_numpy().astype(float)
    gs_max = gs.max()
    gs_norm = gs / max(gs_max, 1)

    fd = [_to_dt(d) for d in feat["date"].to_list()]
    rho = feat["rho_norm"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # ρ observé
    ax1.fill_between(fd, rho, alpha=0.3, color=C_OBS)
    ax1.plot(fd, rho, color=C_OBS, lw=1.0, label="ρ observé (waiting_capacity normalisé)")
    rho_ma30 = np.convolve(rho, np.ones(30)/30, mode="same")
    ax1.plot(fd, rho_ma30, color=C_MA, lw=2, ls="-", label="MA 30j")

    # TTC markers
    ttc_path = Path(f"data/features/{loc}_time_to_clear.parquet")
    if ttc_path.exists():
        ttc_df = pl.read_parquet(ttc_path)
        ttc_rows = ttc_df.filter(pl.col("time_to_clear_days").is_not_null())
        if len(ttc_rows) > 0:
            ttc_days = int(ttc_rows["time_to_clear_days"][0])
            first_date = ttc_df["date"][0]
            if isinstance(first_date, str):
                first_date = date.fromisoformat(first_date)
            peak_dt  = _to_dt(first_date)
            clear_dt = _to_dt(first_date + timedelta(days=ttc_days))
            ax1.axvline(peak_dt,  color=C_PEAK, lw=2, ls="-.", alpha=0.9, label=f"Pic ({first_date})")
            ax1.axvline(clear_dt, color=C_TTC,  lw=2, ls="--", alpha=0.9,
                        label=f"TTC = {ttc_days}j ({(first_date + timedelta(days=ttc_days))})")
            ax1.axvspan(peak_dt, clear_dt, alpha=0.06, color=C_PRED)

    ax1.set_ylabel("ρ normalisé", fontsize=10)
    ax1.set_ylim(-0.02, 1.15)
    ax1.legend(fontsize=9, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_title(f"{loc.upper()} — Vue d'ensemble 2017–2024 : densité maritime & gravity score",
                  fontsize=13, fontweight="bold", pad=8)

    # Gravity score
    ax2.fill_between(gd, gs_norm, alpha=0.7, color=C_GRAV, label="Gravity score (normalisé)")
    ax2.plot(gd, gs_norm, color=C_GRAV, lw=1.0)
    # Mark top-5 gravity events
    top5_idx = np.argsort(gs_norm)[-5:]
    for idx in top5_idx:
        ax2.axvline(gd[idx], color="red", lw=0.8, alpha=0.5)

    ax2.set_ylabel("Gravity\nScore (norm.)", fontsize=9)
    ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Vue d'ensemble → %s", out_path)
    return out_path


# ── Figure 5 : Diagramme Fondamental (q-k diagram) ────────────────────────────

def plot_fundamental_diagram(loc: str, out_dir: Path) -> Path:
    """
    Diagramme Fondamental LWR : chaque jour = point (ρ, F=ρ·v) coloré par date.

    - Courbe théorique Greenshields : F = ρ·(1−ρ)  (v_max=ρ_max=1)
    - Point critique : ρ_c=0.5, F_c=0.25  (capacité maximale)
    - Couleur des points : progression temporelle (bleu → rouge)
    - Harvey mis en valeur (halo orange + annotation)
    - Inset zoom sur la zone réelle des données si ρ concentré sur [0.8, 1]
    """
    out_path = out_dir / f"{loc}_pinn_fundamental_diagram.png"

    feat_path  = Path(f"data/features/{loc}_daily_features.parquet")
    model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    ttc_path   = Path(f"data/features/{loc}_time_to_clear.parquet")

    if not feat_path.exists():
        log.error("Fichier manquant : %s", feat_path)
        return out_path

    _, ck = _load_model(model_path, get_device()) if model_path.exists() else (None, {})
    train_start = date.fromisoformat(ck.get("train_start", "2017-06-01"))
    train_end   = date.fromisoformat(ck.get("train_end",   "2018-06-30"))
    sog_max_ck  = float(ck.get("sog_max", 0.0))

    feat = (
        pl.read_parquet(feat_path)
        .sort("date")
        .filter(pl.col("date").is_between(train_start, train_end))
    )
    dates_list = feat["date"].to_list()
    n          = len(feat)

    # ρ observé
    if "waiting_capacity" in feat.columns and feat["waiting_capacity"].max() > 0:
        cap_arr = feat["waiting_capacity"].to_numpy().astype(float)
        cap_95  = float(np.percentile(cap_arr[cap_arr > 0], 95))
        rho_obs = np.clip(cap_arr / cap_95, 0.0, 1.0)
        rho_label = "Capacité bloquée ρ (norm.)"
    else:
        rho_obs   = feat["utilization_rate_rho"].to_numpy().astype(float)
        rho_label = "Utilisation ρ"

    sog     = feat["SOG_mean"].to_numpy().astype(float)
    sog_max = sog_max_ck if sog_max_ck > 0 else (sog.max() if sog.max() > 0 else 1.0)
    v_obs   = sog / sog_max
    F_obs   = rho_obs * v_obs

    # ρ / F prédit par le PINN (depuis TTC parquet)
    rho_pred_arr = v_pred_arr = F_pred_arr = pred_dates = None
    if ttc_path.exists():
        ttc_df      = pl.read_parquet(ttc_path).sort("date")
        rho_pred_arr = ttc_df["rho_pred"].to_numpy().astype(float)
        v_pred_arr   = ttc_df["v_pred"].to_numpy().astype(float)
        F_pred_arr   = rho_pred_arr * v_pred_arr
        pred_dates   = ttc_df["date"].to_list()

    # Masques temporels
    peak_dt = date.fromisoformat(str(pred_dates[0])[:10]) if pred_dates else train_end
    harvey_start = peak_dt - timedelta(days=13)
    harvey_end   = peak_dt + timedelta(days=3)
    harvey_mask  = np.array([
        harvey_start <= d <= harvey_end for d in dates_list
    ])
    ttc_days_val = None
    if ttc_path.exists():
        rows = pl.read_parquet(ttc_path).filter(
            pl.col("time_to_clear_days").is_not_null()
        )
        if len(rows) > 0:
            ttc_days_val = int(rows["time_to_clear_days"][0])

    # Colormap temporelle : bleu (début) → rouge (pic) → violet (fin)
    t_norm = np.array([(d - train_start).days for d in dates_list], dtype=float)
    t_norm /= max(t_norm.max(), 1)
    colors_obs = plt.cm.cool(t_norm)

    # ── Figure principale ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7))
    gs_fig = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax_main = fig.add_subplot(gs_fig[0])
    ax_zoom = fig.add_subplot(gs_fig[1])

    rho_range = np.linspace(0, 1, 400)
    F_green   = rho_range * (1 - rho_range)  # Greenshields normalisé

    for ax, title_suffix, xlim in [
        (ax_main, "  (domaine complet)", (0, 1.0)),
        (ax_zoom, f"  (zoom ρ∈[{rho_obs.min():.2f},{rho_obs.max():.2f}])",
         (max(0, rho_obs.min() - 0.05), min(1, rho_obs.max() + 0.05))),
    ]:
        # Courbe Greenshields
        ax.plot(rho_range, F_green, color="#212121", lw=2, ls="--",
                label="Greenshields  F=ρ(1−ρ)", zorder=2)
        # Point critique
        ax.scatter([0.5], [0.25], marker="*", s=180, color="#212121",
                   zorder=5, label="ρ_c = 0.5  (capacité max)")
        ax.axvline(0.5, color="#9E9E9E", lw=0.8, ls=":", alpha=0.6)

        # Zones régimes
        ax.axvspan(0, 0.5,  alpha=0.04, color="#1565C0", label="Régime fluide")
        ax.axvspan(0.5, 1.0, alpha=0.04, color="#E53935", label="Régime congestionné")

        # Observations normales
        normal_mask = ~harvey_mask
        ax.scatter(rho_obs[normal_mask], F_obs[normal_mask],
                   c=colors_obs[normal_mask], s=22, alpha=0.75,
                   zorder=3, label="Jours normaux")

        # Harvey (halo + points)
        if harvey_mask.sum() > 0:
            ax.scatter(rho_obs[harvey_mask], F_obs[harvey_mask],
                       s=90, color="#FF6F00", edgecolors="#BF360C", lw=1.2,
                       zorder=6, label=f"Harvey ({harvey_mask.sum()} jours)")
            # Flèche depuis le centroïde normal vers le centroïde Harvey
            rho_base_c = float(rho_obs[normal_mask].mean())
            F_base_c   = float(F_obs[normal_mask].mean())
            rho_harv_c = float(rho_obs[harvey_mask].mean())
            F_harv_c   = float(F_obs[harvey_mask].mean())
            ax.annotate(
                "", xy=(rho_harv_c, F_harv_c),
                xytext=(rho_base_c, F_base_c),
                arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=2),
                zorder=7,
            )

        # Prédictions PINN (si disponibles, en rouge pointillé)
        if F_pred_arr is not None and len(F_pred_arr) > 0:
            ax.plot(rho_pred_arr, F_pred_arr, color="#E53935", lw=1.5,
                    ls=":", alpha=0.7, label="PINN (ρ_pred, F_pred)", zorder=4)
            ax.scatter(rho_pred_arr[0], F_pred_arr[0],
                       marker="D", s=60, color="#E53935", zorder=5)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.01, max(F_obs.max(), 0.26) * 1.12)
        ax.set_xlabel(rho_label, fontsize=10)
        ax.set_ylabel("Flux  F = ρ · v  (normalisé)", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f"{loc.upper()} — Diagramme Fondamental LWR{title_suffix}\n"
            f"[{train_start} → {train_end}]"
            + (f"  ·  TTC = {ttc_days_val} j" if ttc_days_val else ""),
            fontsize=10, fontweight="bold", pad=6,
        )

    ax_main.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)

    # Colorbar temporelle
    sm = plt.cm.ScalarMappable(cmap="cool",
                                norm=plt.Normalize(0, (train_end - train_start).days))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_zoom, fraction=0.04, pad=0.02)
    cbar.set_label("Jours depuis début", fontsize=8)
    n_ticks = 5
    tick_days = np.linspace(0, (train_end - train_start).days, n_ticks, dtype=int)
    cbar.set_ticks(tick_days)
    cbar.set_ticklabels(
        [(train_start + timedelta(days=int(d))).strftime("%Y-%m") for d in tick_days],
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Diagramme Fondamental → %s", out_path)
    return out_path


# ── Figure 6 : Diagramme Fondamental per-zone ─────────────────────────────────

def plot_fundamental_diagram_perzone(loc: str, out_dir: Path) -> Path:
    """
    Diagramme Fondamental avec une observation par (zone, jour).
    Couleur : type de zone (docked=bleu / waiting=orange).
    Taille  : proportionnelle au vessel_count.
    Overlay : courbe Greenshields + prédictions PINN per-zone.
    """
    out_path = out_dir / f"{loc}_pinn_fd_perzone.png"

    zone_feat_path = Path(f"data/features/{loc}_zone_daily_features.parquet")
    model_path     = Path(f"outputs/models/{loc}_perzone_lwr_pinn.pt")
    if not model_path.exists():
        model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")

    if not zone_feat_path.exists():
        log.error("Fichier per-zone manquant : %s", zone_feat_path)
        return out_path

    _, ck = _load_model(model_path, get_device()) if model_path.exists() else (None, {})
    train_start  = date.fromisoformat(ck.get("train_start", "2017-06-01"))
    train_end    = date.fromisoformat(ck.get("train_end",   "2018-06-30"))
    n_train_days = (train_end - train_start).days + 1

    df = (
        pl.read_parquet(zone_feat_path)
        .filter(pl.col("vessel_count") > 0)
        .sort(["date", "lon"])
    )

    rho  = df["rho_norm"].to_numpy().astype(float)
    v    = df["v_norm"].to_numpy().astype(float)   # SOG_mean réel normalisé
    F    = rho * v
    types   = df["cluster_type"].to_list()
    counts  = df["vessel_count"].to_numpy().astype(float)
    dates_z = df["date"].to_list()

    t_norm = np.array([(d - train_start).days for d in dates_z], dtype=float)
    t_norm /= max(n_train_days - 1, 1)

    # Masque Harvey
    peak_dt      = date(2017, 9, 7) if loc == "houston" else date(2022, 1, 6)
    harvey_start = peak_dt - timedelta(days=13)
    harvey_end   = peak_dt + timedelta(days=3)
    harvey_mask  = np.array([harvey_start <= d <= harvey_end for d in dates_z])

    docked_mask  = np.array([t == "docked"  for t in types])
    waiting_mask = np.array([t == "waiting" for t in types])

    # Taille des points : proportionnelle au vessel_count (plafonné)
    sizes = np.clip(counts / float(np.percentile(counts, 95)), 0, 1) * 30 + 2

    # Prédictions PINN per-zone
    rho_pred_zone = v_pred_zone = None
    if model_path.exists():
        device   = get_device()
        model, _ = _load_model(model_path, device)
        lon_arr  = df["lon_norm"].to_numpy().astype(float)
        x_t  = torch.tensor(lon_arr.astype(np.float32), device=device).reshape(-1, 1)
        t_t  = torch.tensor(t_norm.astype(np.float32),  device=device).reshape(-1, 1)
        with torch.no_grad():
            rho_pred_zone = model(x_t, t_t).cpu().numpy().ravel()
        v_pred_zone   = model.v_max * (1.0 - rho_pred_zone / model.rho_max)

    # ── Figure ─────────────────────────────────────────────────────────────────
    rho_range = np.linspace(0, 1, 400)
    F_green   = rho_range * (1 - rho_range)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, mask, color, label in [
        (axes[0], docked_mask,  "#1565C0", "Zones quai (docked)"),
        (axes[1], waiting_mask, "#E65100", "Zones attente (waiting)"),
    ]:
        # Greenshields
        ax.plot(rho_range, F_green, color="#212121", lw=2, ls="--",
                label="Greenshields F=ρ(1−ρ)", zorder=2, alpha=0.6)
        ax.scatter([0.5], [0.25], marker="*", s=200, color="#212121",
                   zorder=6, label="ρ_c (capacité max)")
        ax.axvline(0.5, color="#BDBDBD", lw=0.8, ls=":")
        ax.axvspan(0, 0.5,  alpha=0.03, color="#1565C0")
        ax.axvspan(0.5, 1.0, alpha=0.03, color="#E53935")

        # Observations normales
        normal = mask & ~harvey_mask
        if normal.sum() > 0:
            sc = ax.scatter(
                rho[normal], F[normal],
                c=t_norm[normal], cmap="cool",
                s=sizes[normal], alpha=0.4, zorder=3,
                vmin=0, vmax=1, label=label,
            )

        # Harvey
        harv = mask & harvey_mask
        if harv.sum() > 0:
            ax.scatter(rho[harv], F[harv],
                       s=sizes[harv] * 3, color="#FF6F00",
                       edgecolors="#BF360C", lw=1, zorder=7,
                       label=f"Harvey ({harv.sum()} obs.)")

        # PINN per-zone overlay (sous-échantillonné pour lisibilité)
        if rho_pred_zone is not None:
            F_pred = rho_pred_zone[mask] * v_pred_zone[mask]
            step   = max(1, mask.sum() // 500)
            ax.scatter(rho_pred_zone[mask][::step], F_pred[::step],
                       marker="+", s=12, color="#E53935", alpha=0.3,
                       zorder=4, label="PINN prédit")

        ax.set_xlabel("Densité ρ normalisée", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.01, 0.28)
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.set_title(
            f"{loc.upper()} — {label}\n"
            f"[{train_start} → {train_end}]  ·  {mask.sum():,} obs.",
            fontsize=10, fontweight="bold",
        )

    axes[0].set_ylabel("Flux  F = ρ · v  (normalisé)", fontsize=10)

    # Colorbar temporelle (commune)
    sm = plt.cm.ScalarMappable(cmap="cool",
                                norm=plt.Normalize(0, n_train_days))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("Jours depuis début", fontsize=8)
    ticks = np.linspace(0, n_train_days, 5, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [(train_start + timedelta(days=int(d))).strftime("%Y-%m") for d in ticks],
        fontsize=7,
    )

    fig.suptitle(
        f"{loc.upper()} — Diagramme Fondamental per-zone  "
        f"(PINN entraîné sur {len(df):,} observations spatiales)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("FD per-zone → %s", out_path)
    return out_path


# ── Figure 7 : Profils spatiaux ρ(x) et F(x) par zone constituante ────────────

def plot_zone_flux_profiles(
    loc:       str,
    out_dir:   Path,
    n_bins:    int  = 40,
) -> Path:
    """
    Profils spatiaux ρ(x) et F(x) = ρ·v(ρ) évalués aux positions
    des zones constituantes (Phase 2), à 5 dates clés autour du pic.

    - Axe x : longitude (Ouest → Est)
    - Ligne lissée (n_bins bins) + nuage de points semi-transparent
    - Marqueurs | bas du graphe : ■ quai (bleu) · ● attente (orange)
    - Palette temporelle : bleu froid (baseline) → rouge (pic) → violet (TTC)
    """
    out_path = out_dir / f"{loc}_pinn_zone_flux.png"

    model_path = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    zones_path = Path(f"data/features/{loc}_constituent_zones.parquet")
    for p in [model_path, zones_path]:
        if not p.exists():
            log.error("Fichier manquant : %s", p)
            return out_path

    device = get_device()
    model, ck = _load_model(model_path, device)

    train_start  = date.fromisoformat(ck["train_start"])
    train_end    = date.fromisoformat(ck["train_end"])
    n_train_days = (train_end - train_start).days + 1

    # Zones constituantes triées ouest → est
    zones = (
        pl.read_parquet(zones_path)
        .filter(pl.col("is_constituent"))
        .sort("lon")
    )
    lon_arr  = zones["lon"].to_numpy().astype(float)
    lon_min, lon_max = float(lon_arr.min()), float(lon_arr.max())
    x_zones  = (lon_arr - lon_min) / max(lon_max - lon_min, 1e-8)
    types    = zones["cluster_type"].to_list()

    # Dates clés : depuis le fichier TTC
    ttc_path = Path(f"data/features/{loc}_time_to_clear.parquet")
    peak_dt  = train_start + timedelta(days=n_train_days // 2)
    ttc_days = None
    if ttc_path.exists():
        ttc_df   = pl.read_parquet(ttc_path)
        rows     = ttc_df.filter(pl.col("time_to_clear_days").is_not_null())
        if len(rows) > 0:
            ttc_days = int(rows["time_to_clear_days"][0])
        first = ttc_df["date"][0]
        peak_dt = date.fromisoformat(str(first)[:10])

    key_dates: list[tuple[date, str]] = [
        (peak_dt - timedelta(days=30), "J−30 (baseline)"),
        (peak_dt - timedelta(days=7),  "J−7"),
        (peak_dt,                      "Pic"),
        (peak_dt + timedelta(days=15), "J+15"),
    ]
    if ttc_days:
        key_dates.append((peak_dt + timedelta(days=ttc_days), f"TTC (J+{ttc_days})"))

    palette = plt.cm.plasma(np.linspace(0.05, 0.90, len(key_dates)))

    # Bins de longitude pour lisser les 1600+ zones
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_cx  = 0.5 * (bins[:-1] + bins[1:])           # centres en x normalisé
    bin_lon = bin_cx * (lon_max - lon_min) + lon_min  # → longitude réelle
    bin_idx = np.clip(np.digitize(x_zones, bins) - 1, 0, n_bins - 1)

    x_t = torch.tensor(x_zones.astype(np.float32), device=device).reshape(-1, 1)

    fig, (ax_rho, ax_flux) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    for (kd, label), col in zip(key_dates, palette):
        t_norm = float(np.clip(
            (kd - train_start).days / max(n_train_days - 1, 1), 0.0, 1.0
        ))
        t_t = torch.full_like(x_t, t_norm)

        with torch.no_grad():
            rho_np = model(x_t, t_t).cpu().numpy().ravel()

        v_np   = model.v_max * (1.0 - rho_np / model.rho_max)
        flux_np = rho_np * v_np

        # Moyenne par bin
        rho_bin  = np.full(n_bins, np.nan)
        flux_bin = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                rho_bin[b]  = rho_np[mask].mean()
                flux_bin[b] = flux_np[mask].mean()

        valid = ~np.isnan(rho_bin)
        lw    = 2.8 if "Pic" in label or "TTC" in label else 1.8
        ls    = "--" if "Pic" in label else "-"

        # Nuage semi-transparent
        ax_rho.scatter(lon_arr,  rho_np,  c=[col], s=3, alpha=0.06, zorder=1)
        ax_flux.scatter(lon_arr, flux_np, c=[col], s=3, alpha=0.06, zorder=1)

        # Courbe lissée
        ax_rho.plot(bin_lon[valid],  rho_bin[valid],  color=col, lw=lw, ls=ls,
                    label=label, zorder=4)
        ax_flux.plot(bin_lon[valid], flux_bin[valid], color=col, lw=lw, ls=ls,
                     label=label, zorder=4)

    # Marqueurs de type de zone en bas de chaque panneau
    docked_lon  = [l for l, t in zip(lon_arr, types) if t == "docked"]
    waiting_lon = [l for l, t in zip(lon_arr, types) if t == "waiting"]
    for ax in (ax_rho, ax_flux):
        ax.scatter(docked_lon,  np.zeros(len(docked_lon)),
                   marker="|", s=18, color="#1565C0", alpha=0.25, zorder=2,
                   label="Zone quai (docked)")
        ax.scatter(waiting_lon, np.zeros(len(waiting_lon)),
                   marker="|", s=18, color="#FF6F00", alpha=0.25, zorder=2,
                   label="Zone attente (waiting)")

    ax_rho.set_ylabel("Densité ρ  (normalisée)", fontsize=11)
    ax_rho.set_ylim(-0.04, 1.08)
    ax_rho.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.9)
    ax_rho.grid(True, alpha=0.2)
    ax_rho.set_title(
        f"{loc.upper()} — Profils spatiaux ρ(x) et Flux F(x) = ρ·v par zone constituante\n"
        f"Fenêtre [{train_start} → {train_end}]  ·  Pic : {peak_dt}"
        + (f"  ·  TTC = {ttc_days} j" if ttc_days else ""),
        fontsize=12, fontweight="bold", pad=8,
    )

    ax_flux.set_ylabel("Flux  F = ρ · v  (normalisé)", fontsize=11)
    ax_flux.set_ylim(-0.01, 0.30)
    ax_flux.set_xlabel("Longitude  (Ouest ←  → Est)", fontsize=11)
    ax_flux.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.9)
    ax_flux.grid(True, alpha=0.2)

    # Tick longitude
    n_xticks = 8
    xtick_lons = np.linspace(lon_min, lon_max, n_xticks)
    for ax in (ax_rho, ax_flux):
        ax.set_xticks(xtick_lons)
        ax.set_xticklabels([f"{l:.2f}°" for l in xtick_lons], fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Zone flux profiles → %s", out_path)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualisation PINN Phase 3")
    parser.add_argument("--location", default="all", choices=["la", "houston", "all"])
    parser.add_argument("--zone-flux-only", action="store_true",
                        help="Génère uniquement le graphique flux/densité par zone")
    parser.add_argument("--fd-only", action="store_true",
                        help="Génère uniquement le diagramme fondamental")
    parser.add_argument("--fd-perzone-only", action="store_true",
                        help="Génère uniquement le diagramme fondamental per-zone")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    locs = ["la", "houston"] if args.location == "all" else [args.location]

    for loc in locs:
        print(f"\n=== {loc.upper()} ===")
        if args.zone_flux_only:
            p5 = plot_zone_flux_profiles(loc, OUT)
            print(f"  zone_flux   → {p5}")
            continue
        if args.fd_only:
            p6 = plot_fundamental_diagram(loc, OUT)
            print(f"  fund_diag   → {p6}")
            continue
        if args.fd_perzone_only:
            p7 = plot_fundamental_diagram_perzone(loc, OUT)
            print(f"  fd_perzone  → {p7}")
            continue
        p1 = plot_timeseries(loc, OUT)
        print(f"  timeseries  → {p1}")
        p2 = plot_heatmap(loc, OUT)
        print(f"  heatmap     → {p2}")
        p3 = plot_loss(loc, OUT)
        print(f"  loss        → {p3}")
        p4 = plot_gravity_overview(loc, OUT)
        print(f"  overview    → {p4}")
        p5 = plot_zone_flux_profiles(loc, OUT)
        print(f"  zone_flux   → {p5}")
        p6 = plot_fundamental_diagram(loc, OUT)
        print(f"  fund_diag   → {p6}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
