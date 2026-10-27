"""
Entraîne un PINN (rho + v, architecture actuelle) par épisode,
extrait les résumés physiques sur grille dense, sauvegarde en Parquet.

Usage:
    python run_train_pinn_per_episode.py [--epochs 500] [--max-duration 30]

Output:
    data/features/la_pinn_episode_summaries.parquet
"""
import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.pinns.lwr_pinn import LWRPINN, get_device, total_loss
from src.pinns.pinn_pipeline import structure_episodes

# ── Params ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",       type=int,   default=500)
parser.add_argument("--max-duration", type=int,   default=30,
                    help="Exclure les épisodes > N jours (outliers pandémie)")
parser.add_argument("--n-colloc",     type=int,   default=800)
parser.add_argument("--lr",           type=float, default=5e-4)
parser.add_argument("--lambda-pde",   type=float, default=0.1)
parser.add_argument("--lambda-kin",   type=float, default=0.05)
args = parser.parse_args()

LOCATION       = "la"
GRAVITY_PATH   = Path("data/features/la_gravity_all_years.parquet")
OUT_PATH       = Path("data/features/la_pinn_episode_summaries.parquet")
GRAVITY_THRESH = 5_000
MIN_EP_DAYS    = 3
X_GRID         = 60
T_GRID         = 60

# ── Détection épisodes ────────────────────────────────────────────────────────
log.info("Détection épisodes 2017-2024...")
episodes, global_meta = structure_episodes(
    location=LOCATION,
    gravity_path=GRAVITY_PATH,
    gravity_threshold=GRAVITY_THRESH,
    min_episode_days=MIN_EP_DAYS,
    start_date=date(2017, 1, 1),
    end_date=date(2024, 10, 31),
)

# Filtrer méga-épisodes (outliers pandémie/crise congestion)
n_raw = len(episodes)
episodes = [ep for ep in episodes if ep.duration_days <= args.max_duration]
log.info(
    "%d épisodes retenus sur %d (filtre max_duration=%dj)",
    len(episodes), n_raw, args.max_duration,
)

device = get_device()
log.info("Device: %s", device)

channel_len = global_meta["channel_len_km"]
rho_max     = global_meta["global_rho_max"]
v_max       = global_meta["global_v_max"]


def train_episode_pinn(ep) -> LWRPINN:
    """Entraîne un PINN from scratch sur les observations d'un seul épisode."""
    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rho_norm = np.clip(ep.rho_physical / rho_max, 0, 1).astype(np.float32)
    v_norm   = np.clip(ep.v_physical   / v_max,   0, 1).astype(np.float32)

    x_data = torch.tensor(ep.x_norm.reshape(-1,1), dtype=torch.float32, device=device)
    t_data = torch.tensor(ep.t_norm.reshape(-1,1), dtype=torch.float32, device=device)
    rho_obs = torch.tensor(rho_norm.reshape(-1,1), dtype=torch.float32, device=device)
    v_obs   = torch.tensor(v_norm.reshape(-1,1),   dtype=torch.float32, device=device)

    v_max_kmh     = v_max * 1.852
    alpha_courant = v_max_kmh * (ep.duration_days * 24.0) / max(channel_len, 1e-8)

    x_col = torch.rand(args.n_colloc, 1, requires_grad=True, device=device)
    t_col = torch.rand(args.n_colloc, 1, requires_grad=True, device=device)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        if epoch % 100 == 1 and epoch > 1:
            x_col = torch.rand(args.n_colloc, 1, requires_grad=True, device=device)
            t_col = torch.rand(args.n_colloc, 1, requires_grad=True, device=device)

        optimizer.zero_grad()
        loss, _ = total_loss(
            model, x_data, t_data, rho_obs, v_obs,
            x_col, t_col,
            lambda_pde=args.lambda_pde,
            lambda_kin=args.lambda_kin,
            alpha_courant=alpha_courant,
        )
        loss.backward()
        optimizer.step()

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def extract_summaries(model: LWRPINN, ep) -> dict:
    """Extrait les résumés scalaires depuis les champs PINN sur grille dense."""
    model.eval().to(device)

    t_max_days = float(ep.duration_days) * 2.0 + 1.0

    x_grid = np.linspace(0.0, channel_len, X_GRID, dtype=np.float32)
    t_grid = np.linspace(0.0, t_max_days,  T_GRID, dtype=np.float32)
    xx, tt = np.meshgrid(x_grid, t_grid)

    x_norm_g = (xx.ravel() / channel_len).reshape(-1, 1)
    t_norm_g = (tt.ravel() / t_max_days).reshape(-1, 1)

    x_t = torch.tensor(x_norm_g, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_norm_g, dtype=torch.float32, device=device)

    with torch.no_grad():
        rho_pred, v_pred = model(x_t, t_t)

    rho_field = rho_pred.cpu().numpy().ravel() * rho_max   # vessels/km
    v_field   = v_pred.cpu().numpy().ravel()   * v_max     # knots

    rho_grid  = rho_field.reshape(T_GRID, X_GRID)   # (t, x)
    v_grid    = v_field.reshape(T_GRID, X_GRID)

    rho_max_t = rho_grid.max(axis=1)           # profil temporel de densité max

    peak_idx       = int(np.argmax(rho_max_t))
    peak_rho       = float(rho_max_t[peak_idx])
    t_to_peak      = float(t_grid[peak_idx])
    area_under_rho = float(np.trapezoid(rho_max_t, t_grid))

    post_t   = t_grid[peak_idx:]
    post_rho = rho_max_t[peak_idx:]
    decay_rate = float(np.polyfit(post_t, post_rho, 1)[0]) if len(post_t) >= 2 else 0.0

    # Flux à l'entrée du chenal (x ≤ 10% du chenal)
    entry_mask = (xx.ravel() / channel_len) <= 0.10
    flux_entry = float((rho_field * v_field)[entry_mask].mean()) if entry_mask.any() else 0.0

    v_mean_global = float(v_field.mean())

    return {
        "peak_rho":        peak_rho,
        "t_to_peak_days":  t_to_peak,
        "area_under_rho":  area_under_rho,
        "decay_rate":      decay_rate,
        "flux_entry_mean": flux_entry,
        "v_mean_global":   v_mean_global,
    }


# ── Boucle principale ─────────────────────────────────────────────────────────
records = []
t_start_total = time.time()

for i, ep in enumerate(episodes):
    t0 = time.time()

    if ep.n_points < 2:
        log.warning("[%d/%d] %s — trop peu de points (%d), skipped",
                    i+1, len(episodes), ep.episode_id, ep.n_points)
        continue

    log.info("[%d/%d] %s | %dj | %d pts AIS",
             i+1, len(episodes), ep.episode_id, ep.duration_days, ep.n_points)

    try:
        pinn   = train_episode_pinn(ep)
        summ   = extract_summaries(pinn, ep)
        del pinn  # libérer la mémoire GPU/MPS
        torch.mps.empty_cache() if device.type == "mps" else None

        elapsed = time.time() - t0
        log.info("  → peak_rho=%.2f  decay=%.4f  area=%.1f  [%.1fs]",
                 summ["peak_rho"], summ["decay_rate"], summ["area_under_rho"], elapsed)

        records.append({
            "episode_id":    ep.episode_id,
            "start_date":    ep.start_date,
            "end_date":      ep.end_date,
            "duration_days": ep.duration_days,
            "gravity_score": float(ep.trigger_score),
            "n_ais_points":  ep.n_points,
            **summ,
        })

    except Exception as exc:
        log.error("  ÉCHEC %s: %s", ep.episode_id, exc)

    # Estimation ETA
    done = i + 1
    elapsed_total = time.time() - t_start_total
    eta = elapsed_total / done * (len(episodes) - done)
    if done % 10 == 0 or done <= 3:
        log.info("  ETA: %dm%02ds restants (%d/%d épisodes)",
                 int(eta)//60, int(eta)%60, done, len(episodes))

df = pd.DataFrame(records)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

total_elapsed = time.time() - t_start_total
log.info("=== TERMINÉ ===")
log.info("%d épisodes traités en %dm%02ds", len(df), int(total_elapsed)//60, int(total_elapsed)%60)
log.info("Sauvegardé → %s", OUT_PATH)
log.info("Durées : min=%dj  median=%.0fj  max=%dj",
         int(df["duration_days"].min()),
         df["duration_days"].median(),
         int(df["duration_days"].max()))
log.info("peak_rho : min=%.1f  max=%.1f  std=%.2f",
         df["peak_rho"].min(), df["peak_rho"].max(), df["peak_rho"].std())
