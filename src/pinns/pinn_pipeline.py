"""
Phase 3 — PINN physical engine: 4-step pipeline for maritime traffic
dynamics during disruption events.

Implements a physics-informed neural network (PINN) modelling the
macroscopic Lighthill-Whitham-Richards (LWR) equation:
    ∂ρ/∂t + ∂(ρ·v)/∂x = 0

4-step structure:
  Step 1 — structure_episodes()        : split the data into episodes
                                         from the gravity score
  Step 2 — LWRPINN (lwr_pinn.py)      : MLP (x, t) → (ρ̂, v̂)
  Step 3 — hybrid loss (lwr_pinn.py)  : L_data + λ·L_LWR + λ_kin·L_kinematic
  Step 4 — train_on_episodes()         : episodic training loop
           extract_physical_profiles(): dense inference, with gaps filled
                                        by the physics

Usage:
    python src/pinns/pinn_pipeline.py --train  --location la --start 2019-01-01 --end 2019-12-31
    python src/pinns/pinn_pipeline.py --train  --location houston --start 2017-01-01 --end 2017-12-31
    python src/pinns/pinn_pipeline.py --infer  --model outputs/models/pinn_houston_episodes.pt --start 2020-08-02 --end 2020-12-30
    python src/pinns/pinn_pipeline.py --both   --location houston
    python src/pinns/pinn_pipeline.py --data-only --location houston --start 2017-01-01 --end 2017-12-31

    python src/pinns/pinn_pipeline.py --infer `
  --model outputs/models/pinn_la_episodes.pt `
  --start 2020-11-01 --end 2021-02-28 `
  --fine-tune `
  --fine-tune-epochs 300 `
  --fine-tune-lr 1e-4
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _force_unlink(path: Path) -> None:
    """Delete an existing file before writing, to avoid Windows file locks."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


import numpy as np
import pandas as pd
import polars as pl
import torch

from src.pinns.lwr_pinn import LWRPINN, get_device, total_loss
from src.pinns.data_prep import (
    CHANNEL_AXES,
    EPOCH_DATE,
    T_DAYS_MAX,
    build_rho_v_tensors,
    compute_channel_length,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/figures")
MODEL_DIR = Path("outputs/models")
FEATURES_DIR = Path("data/features")

# Collocation points for the physics constraint
N_COLLOC = 2000       # number of points per draw
COLLOC_REFRESH = 500  # resample every N epochs
LOG_EVERY = 200       # logging frequency

# ── Per-location parameters ──────────────────────────────────────────────────
# gravity_threshold: episode trigger threshold.
#   The gravity score distribution is bimodal (0 or very large).
#   Pick a threshold above the background noise but below the real peaks.
#   → Adjust these values if you get too many or too few episodes.
LOCATION_DEFAULTS: dict[str, dict] = {
    # Houston: non-zero scores 4k–370k | p50=37k | p75=83k | p90=134k
    # Harvey (Aug 2017) produces peaks >300k → a 50k threshold captures Harvey + large events
    "houston": {
        "gravity_threshold": 5_000,
        "min_episode_days": 2,
    },
    # LA: bimodal distribution — 250 days at 580–7500 (normal) + peaks 100k–214k (crises)
    # p90=5700, p95=7500, p99=109k → a 5000 threshold isolates real disruptions
    "la": {
        "gravity_threshold": 5_000,
        "min_episode_days": 3,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — STRUCTURING THE DATA INTO EPISODES
#
# An episode starts (t=0) when the gravity score exceeds a trigger threshold.
# For each episode we:
#   1. Extract the AIS data over the time window
#   2. Project geographic coordinates onto the 1D channel axis (x in km)
#   3. Compute vessel density (ρ, vessels/km) and mean speed (v, knots)
#   4. Normalise for the PINN:
#        x_norm = x_km / channel_len_km          ∈ [0, 1]
#        t_norm = (day − episode_start) / duration  ∈ [0, 1]  (episode-relative)
#        ρ_norm = ρ / ρ_max_global               ∈ [0, 1]
#        v_norm = v / v_max_global               ∈ [0, 1]
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """A disruption event extracted from the gravity score time series."""
    episode_id: str
    location: str
    start_date: date
    end_date: date
    trigger_score: float    # gravity score at trigger time
    duration_days: int
    # Normalised coordinates for the PINN (N observation points)
    x_norm: np.ndarray      # position along the channel ∈ [0, 1]
    t_norm: np.ndarray      # episode-relative time ∈ [0, 1]
    # Raw physical values (before global normalisation)
    rho_physical: np.ndarray  # density in vessels/km
    v_physical: np.ndarray    # speed in knots
    n_points: int
    channel_len_km: float


def _detect_episode_ranges(
    gravity_df: pl.DataFrame,
    threshold: float,
    min_days: int,
    pad_days: int = 0,
) -> list[tuple[date, date, float]]:
    """
    Scan the gravity score series and return the ranges
    (start_date, end_date, trigger_score) of each episode.

    An episode is a continuous run of days where gravity_score >= threshold,
    lasting at least min_days days.

    pad_days > 0 extends each episode beyond the trigger threshold to
    include the recovery phase — needed for the PINN to learn the
    density decay and estimate a realistic TTC.
    """
    sorted_df = gravity_df.sort("date")
    all_dates = [
        date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]
        for row in sorted_df.iter_rows(named=True)
    ]
    last_date = all_dates[-1] if all_dates else None

    episodes: list[tuple[date, date, float]] = []
    in_run = False
    run_start: Optional[date] = None
    trigger_score = 0.0

    for row in sorted_df.iter_rows(named=True):
        d = date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]
        score = float(row["gravity_score"])

        if not in_run and score >= threshold:
            in_run = True
            run_start = d
            trigger_score = score

        elif in_run and score < threshold:
            congestion_end = d - timedelta(days=1)
            if (congestion_end - run_start).days + 1 >= min_days:
                # Extend episode to include recovery tail
                if pad_days > 0 and last_date is not None:
                    padded_end = min(congestion_end + timedelta(days=pad_days), last_date)
                else:
                    padded_end = congestion_end
                episodes.append((run_start, padded_end, trigger_score))
            in_run = False

    # Close an episode that runs until the end of the series
    if in_run and last_date is not None:
        if (last_date - run_start).days + 1 >= min_days:
            episodes.append((run_start, last_date, trigger_score))

    return episodes


def structure_episodes(
    location: str = "houston",
    gravity_path: Optional[Path] = None,
    gravity_threshold: Optional[float] = None,
    min_episode_days: Optional[int] = None,
    dx_km: float = 2.0,
    constituent_path: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    pad_days: int = 0,
) -> tuple[list[Episode], dict]:
    """
    Step 1 — Split the data into disruption episodes and extract
    the physical observations (x, ρ, v) of each episode.

    Parameters
    ----------
    location          : 'houston' or 'la'
    gravity_path      : path to gravity_daily.parquet
                        (default: data/features/<location>_gravity_daily.parquet)
    gravity_threshold : episode trigger threshold
    min_episode_days  : minimum length for a run to count as an episode
    dx_km             : width of the spatial bins along the channel
    constituent_path  : optional filter on the constituent zones
    start_date        : lower bound of the scan period (inclusive)
    end_date          : upper bound of the scan period (inclusive)

    Returns
    -------
    episodes    : list of Episode objects (normalised coordinates for the PINN)
    global_meta : {'channel_len_km', 'global_rho_max', 'global_v_max', 'dx_km', 'location'}
    """
    # Apply per-location defaults when not provided
    loc_defaults = LOCATION_DEFAULTS.get(location, {})
    if gravity_threshold is None:
        gravity_threshold = float(loc_defaults.get("gravity_threshold", 5_000))
    if min_episode_days is None:
        min_episode_days = int(loc_defaults.get("min_episode_days", 3))

    if gravity_path is None:
        gravity_path = FEATURES_DIR / f"{location}_gravity_daily.parquet"
    if not gravity_path.exists():
        raise FileNotFoundError(f"Gravity score file not found: {gravity_path}")

    gravity_df = pl.read_parquet(gravity_path)

    # Filter by period if provided
    if start_date is not None:
        gravity_df = gravity_df.filter(pl.col("date") >= start_date.isoformat())
    if end_date is not None:
        gravity_df = gravity_df.filter(pl.col("date") <= end_date.isoformat())

    period_str = (
        f"{start_date or 'start'} -> {end_date or 'end'}"
        if (start_date or end_date)
        else "full period"
    )
    log.info(
        "Gravity score: %d rows | period: %s | file: %s",
        len(gravity_df), period_str, gravity_path,
    )

    date_ranges = _detect_episode_ranges(gravity_df, gravity_threshold, min_episode_days, pad_days=pad_days)
    if not date_ranges:
        log.warning(
            "No episode found (threshold=%.2f, min_days=%d). Try a lower threshold.",
            gravity_threshold, min_episode_days,
        )
        return [], {}

    log.info("%d candidate episode(s) detected (threshold=%.2f)", len(date_ranges), gravity_threshold)

    waypoints = CHANNEL_AXES[location]
    channel_len_km = compute_channel_length(waypoints)

    raw_episodes: list[Episode] = []

    for start_d, end_d, trigger_score in date_ranges:
        ep_id = f"{location}_{start_d.strftime('%Y%m%d')}_{end_d.strftime('%Y%m%d')}"
        duration = (end_d - start_d).days + 1
        log.info("Processing episode %s (%d days)", ep_id, duration)

        try:
            X_raw, y_raw, meta_ep = build_rho_v_tensors(
                start_d,
                end_d,
                location,
                dx_km=dx_km,
                constituent_path=str(constituent_path) if constituent_path else None,
                use_raw_velocity=True,
            )
        except Exception as exc:
            log.warning("Episode %s skipped: %s", ep_id, exc)
            continue

        if len(X_raw) == 0:
            log.warning("Episode %s skipped: no AIS data.", ep_id)
            continue

        # x_norm = x_km / channel_len is already in X_raw[:, 0] ∈ [0, 1]
        x_norm = X_raw[:, 0].astype(np.float32)

        # Convert global normalised time → episode-relative time ∈ [0, 1]
        # t_global_norm = (day - EPOCH_DATE).days / T_DAYS_MAX
        t_start_global = (start_d - EPOCH_DATE).days / T_DAYS_MAX
        t_end_global = (end_d - EPOCH_DATE).days / T_DAYS_MAX
        duration_norm = max(t_end_global - t_start_global, 1e-8)
        t_norm = np.clip(
            (X_raw[:, 1] - t_start_global) / duration_norm, 0.0, 1.0
        ).astype(np.float32)

        # Retrieve raw physical values (denormalised from the per-episode stats)
        # y_raw[:, 0] = (ρ - ρ_min) / (ρ_max - ρ_min)  →  ρ_physical = y_raw[:,0]*(ρ_max-ρ_min)+ρ_min
        rho_range = max(meta_ep["rho_max"] - meta_ep["rho_min"], 1e-8)
        rho_physical = (y_raw[:, 0] * rho_range + meta_ep["rho_min"]).astype(np.float32)

        # y_raw[:, 1] = v / v_max  →  v_physical = y_raw[:,1] * v_max
        v_physical = (y_raw[:, 1] * max(meta_ep["v_max"], 1e-8)).astype(np.float32)

        raw_episodes.append(Episode(
            episode_id=ep_id,
            location=location,
            start_date=start_d,
            end_date=end_d,
            trigger_score=trigger_score,
            duration_days=duration,
            x_norm=x_norm,
            t_norm=t_norm,
            rho_physical=rho_physical,
            v_physical=v_physical,
            n_points=len(x_norm),
            channel_len_km=channel_len_km,
        ))

    if not raw_episodes:
        log.error("No episode with AIS data found.")
        return [], {}

    # Global normalisation constants (99th percentile to ignore outliers)
    all_rho = np.concatenate([ep.rho_physical for ep in raw_episodes])
    all_v = np.concatenate([ep.v_physical for ep in raw_episodes])
    global_rho_max = float(max(np.percentile(all_rho, 99), 1e-4))
    global_v_max = float(max(np.percentile(all_v, 99), 1e-4))

    log.info(
        "Global normalisation: ρ_max=%.2f vessels/km  v_max=%.2f kn  channel=%.1f km",
        global_rho_max, global_v_max, channel_len_km,
    )
    log.info(
        "Total: %d observation points over %d episode(s)",
        sum(ep.n_points for ep in raw_episodes), len(raw_episodes),
    )

    global_meta = {
        "location": location,
        "channel_len_km": channel_len_km,
        "global_rho_max": global_rho_max,
        "global_v_max": global_v_max,
        "dx_km": dx_km,
    }

    return raw_episodes, global_meta


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — PINN ARCHITECTURE  (implemented in lwr_pinn.py)
#
# class LWRPINN(nn.Module):
#   Inputs : [x_norm, t_norm] ∈ [0, 1]²  (episode-relative normalised coordinates)
#   Outputs: [ρ̂_norm, v̂_norm] ∈ (0, 1)  (bounded by a sigmoid)
#   Corps   : Linear(2, H) → Tanh → [Linear(H, H) → Tanh] × (L−1) → Linear(H, 2)
#   Default: L=4 layers, H=64 units
#
# Physical meaning of the outputs:
#   ρ_physique [vessels/km] = ρ̂_norm × global_rho_max
#   v_physique [noeuds]     = v̂_norm × global_v_max
# ════════════════════════════════════════════════════════════════════════════

# (LWRPINN imported from src.pinns.lwr_pinn)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — HYBRID LOSS FUNCTION  (implemented in lwr_pinn.py)
#
#   L_total = L_data  +  λ_pde · L_LWR  +  λ_kin · L_kinematic
#
#   L_data  = MSE(ρ̂, ρ_obs) + MSE(v̂, v_obs)
#             computed on the AIS observation points
#
#   L_LWR   = mean[ (∂ρ̂/∂t + ∂(ρ̂·v̂)/∂x)² ]
#             LWR residual, computed with torch.autograd.grad
#             on random collocation points in [0,1]²
#
#   L_kin   = mean[ relu(|∂v̂/∂t| − a_max)  ]  (vessel acceleration limit)
#           + mean[ relu(|∂v̂/∂x| − g_max)  ]  (spatial smoothness)
#           (density bounds are enforced by the sigmoid outputs)
# ════════════════════════════════════════════════════════════════════════════

# (total_loss imported from src.pinns.lwr_pinn)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAINING LOOP + INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def train_on_episodes(
    episodes: list[Episode],
    global_meta: dict,
    epochs: int = 2000,
    lr: float = 5e-4,
    lambda_pde: float = 0.1,
    lambda_kin: float = 0.05,
    n_colloc: int = N_COLLOC,
    colloc_refresh: int = COLLOC_REFRESH,
    model_name: Optional[str] = None,
    warmup_epochs: int = 0,
    n_freqs: int = 0,
    fourier_sigma: float = 1.0,
) -> tuple[LWRPINN, dict]:
    """
    Step 4a — Train the PINN jointly on all historical episodes.

    A single model is trained on the pooled observations (x_norm, t_norm),
    normalised consistently across episodes through the global constants.

    The LWR physics constraint (Step 3) is evaluated on N_COLLOC random
    collocation points in [0,1]², resampled every COLLOC_REFRESH
    epochs to cover the whole domain.

    The saved checkpoint includes global_meta so that
    extract_physical_profiles can denormalise.

    Returns
    -------
    model   : trained LWRPINN (best state according to the total loss)
    history : per-epoch loss components
    """
    device = get_device()
    log.info("Device: %s", device)

    global_rho_max = global_meta["global_rho_max"]
    global_v_max = global_meta["global_v_max"]

    # Pool all episodes into a single, globally normalised (X, y) set
    x_list, t_list, rho_list, v_list = [], [], [], []
    for ep in episodes:
        rho_norm = np.clip(ep.rho_physical / global_rho_max, 0.0, 1.0)
        v_norm = np.clip(ep.v_physical / global_v_max, 0.0, 1.0)
        x_list.append(ep.x_norm)
        t_list.append(ep.t_norm)
        rho_list.append(rho_norm.astype(np.float32))
        v_list.append(v_norm.astype(np.float32))

    x_data = torch.tensor(np.concatenate(x_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_data = torch.tensor(np.concatenate(t_list), dtype=torch.float32, device=device).unsqueeze(1)
    rho_obs = torch.tensor(np.concatenate(rho_list), dtype=torch.float32, device=device).unsqueeze(1)
    v_obs = torch.tensor(np.concatenate(v_list), dtype=torch.float32, device=device).unsqueeze(1)

    log.info(
        "Pooled dataset: %d points from %d episode(s)",
        len(x_data), len(episodes),
    )

    # Courant number α = v_max [km/h] · T_ep_mean [h] / L [km]
    # Ensures the two PDE terms have the same physical scale in normalized space.
    v_max_kmh = global_meta["global_v_max"] * 1.852  # knots → km/h
    mean_duration_h = float(np.mean([ep.duration_days for ep in episodes])) * 24.0
    alpha_courant = v_max_kmh * mean_duration_h / max(global_meta["channel_len_km"], 1e-8)
    log.info(
        "Courant alpha = %.2f  (v_max=%.1f km/h  T_mean=%.0f h  L=%.1f km)",
        alpha_courant, v_max_kmh, mean_duration_h, global_meta["channel_len_km"],
    )

    model = LWRPINN(
        hidden_layers=4, hidden_size=64, n_freqs=n_freqs, fourier_sigma=fourier_sigma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)

    if n_freqs > 0:
        log.info("Fourier embedding enabled: n_freqs=%d, sigma=%.2f", n_freqs, fourier_sigma)
    if warmup_epochs > 0:
        log.info("Curriculum learning: warm-up over %d epochs", warmup_epochs)

    best_loss = float("inf")
    best_state = None
    history: dict = {"total": [], "data": [], "pde": [], "kin": []}

    # Initial draw of the collocation points (leaf tensors required by autograd)
    x_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)
    t_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)

    for epoch in range(1, epochs + 1):

        # Periodically resample the collocation points
        if epoch % colloc_refresh == 1 and epoch > 1:
            x_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)
            t_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)

        # Curriculum learning: ramp physics weights from 0 to lambda over warmup_epochs
        if warmup_epochs > 0:
            ramp = min(1.0, epoch / warmup_epochs)
            lambda_pde_eff = lambda_pde * ramp
            lambda_kin_eff = lambda_kin * ramp
        else:
            lambda_pde_eff = lambda_pde
            lambda_kin_eff = lambda_kin

        optimizer.zero_grad()

        loss, comps = total_loss(
            model,
            x_data, t_data, rho_obs, v_obs,
            x_col, t_col,
            lambda_pde=lambda_pde_eff,
            lambda_kin=lambda_kin_eff,
            alpha_courant=alpha_courant,
        )

        loss.backward()
        # Clear gradients accumulated on the collocation tensors (not handled by the optimizer)
        if x_col.grad is not None:
            x_col.grad = None
        if t_col.grad is not None:
            t_col.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        history["total"].append(comps["total"])
        history["data"].append(comps["data"])
        history["pde"].append(comps["pde"])
        history["kin"].append(comps["kin"])

        if comps["total"] < best_loss:
            best_loss = comps["total"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  total=%.5f  data=%.5f  pde=%.5f  kin=%.5f",
                epoch, epochs,
                comps["total"], comps["data"], comps["pde"], comps["kin"],
            )

    if best_state:
        model.load_state_dict(best_state)
        log.info("Best state restored (loss=%.6f)", best_loss)

    # Save the checkpoint with the normalisation metadata
    location = global_meta.get("location", "unknown")
    name = model_name or f"pinn_{location}_episodes"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{name}.pt"
    _force_unlink(save_path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "global_meta": global_meta,
            "best_loss": best_loss,
            "epochs": epochs,
            "n_episodes": len(episodes),
            "episode_ids": [ep.episode_id for ep in episodes],
            "n_freqs": n_freqs,
            "fourier_sigma": fourier_sigma,
        },
        save_path,
    )
    log.info("Checkpoint saved → %s", save_path)

    return model, history


# ════════════════════════════════════════════════════════════════════════════
# FINE-TUNING — Adapting the PINN to the data of a new event
# ════════════════════════════════════════════════════════════════════════════

def fine_tune_on_episode(
    model: LWRPINN,
    x_km: np.ndarray,
    t_days: np.ndarray,
    rho_physical: np.ndarray,
    v_physical: np.ndarray,
    global_meta: dict,
    t_max_days: float,
    epochs: int = 300,
    lr: float = 1e-4,
    lambda_pde: float = 0.1,
    lambda_kin: float = 0.05,
    n_colloc: int = 500,
) -> LWRPINN:
    """
    Fast fine-tuning of the PINN on the AIS observations of a new event.

    Starts from the pre-trained weights and fits the (ρ̂, v̂) field to the
    actual data of the current event while keeping the LWR physics
    constraint (∂ρ/∂t + ∂(ρv)/∂x = 0).

    Idea: pre-training learned the general physics of the channel
    (shock-wave shape, fundamental relation). Fine-tuning adjusts the
    amplitude and location specific to the observed event.
    Small LR (1e-4) so the learned physics is not overwritten.

    Parameters
    ----------
    model         : pre-trained LWRPINN (modified in place + returned)
    x_km          : x positions [km] of the AIS observations
    t_days        : time [days] of the observations (0 = episode onset)
    rho_physical  : density [vessels/km] of the observations
    v_physical    : speed [knots] of the observations
    global_meta   : normalisation metadata from the checkpoint
    t_max_days    : episode length in days (to normalise t_norm)
    epochs        : number of iterations (default: 300)
    lr            : learning rate (default: 1e-4, small to preserve the physics)
    lambda_pde    : weight of the LWR constraint
    lambda_kin    : weight of the kinematic constraint
    n_colloc      : collocation points for the physics constraint

    Returns
    -------
    fine-tuned model (same object, modified in place)
    """
    if len(x_km) == 0:
        log.warning("Fine-tuning: no AIS data available — step skipped.")
        return model

    device = get_device()
    model = model.to(device)
    model.train()

    channel_len_km = global_meta["channel_len_km"]
    global_rho_max = global_meta["global_rho_max"]
    global_v_max   = global_meta["global_v_max"]

    # Normalise the observations into the PINN [0, 1]² space
    x_norm   = np.clip(x_km       / max(channel_len_km, 1e-8), 0, 1).astype(np.float32)
    t_norm   = np.clip(t_days     / max(t_max_days,     1e-8), 0, 1).astype(np.float32)
    rho_norm = np.clip(rho_physical / max(global_rho_max, 1e-8), 0, 1).astype(np.float32)
    v_norm   = np.clip(v_physical  / max(global_v_max,   1e-8), 0, 1).astype(np.float32)

    x_t   = torch.tensor(x_norm,    device=device).unsqueeze(1)
    t_t   = torch.tensor(t_norm,    device=device).unsqueeze(1)
    rho_t = torch.tensor(rho_norm,  device=device).unsqueeze(1)
    v_t   = torch.tensor(v_norm,    device=device).unsqueeze(1)

    # Courant number for this fine-tuning (based on the current episode t_max_days)
    v_max_kmh     = global_v_max * 1.852
    alpha_courant = v_max_kmh * (t_max_days * 24.0) / max(channel_len_km, 1e-8)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    log.info(
        "Fine-tuning: %d pts AIS | %d epochs | lr=%.0e | alpha=%.2f | device=%s",
        len(x_km), epochs, lr, alpha_courant, device,
    )

    x_col: torch.Tensor | None = None
    t_col: torch.Tensor | None = None
    best_loss  = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        # Resample the collocation points every 100 epochs
        if (epoch - 1) % 100 == 0:
            x_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)
            t_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)

        optimizer.zero_grad()
        loss, _ = total_loss(
            model, x_t, t_t, rho_t, v_t,
            x_col, t_col,
            lambda_pde=lambda_pde,
            lambda_kin=lambda_kin,
            alpha_courant=alpha_courant,
        )
        loss.backward()
        if x_col.grad is not None:
            x_col.grad = None
        if t_col.grad is not None:
            t_col.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss  = lv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 100 == 0 or epoch == 1:
            log.info("  FT [%d/%d]  loss=%.6f  best=%.6f", epoch, epochs, lv, best_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    log.info("Fine-tuning done. best_loss=%.6f on %d AIS pts", best_loss, len(x_km))
    return model


def extract_physical_profiles(
    model: LWRPINN,
    episode_data: pd.DataFrame,
    global_meta: dict,
    x_grid_points: int = 100,
    t_grid_points: int = 100,
    t_max_days: Optional[float] = None,
) -> pd.DataFrame:
    """
    Step 4b — Dense inference on a new, ongoing episode.

    Takes the observations of the first hours/days of an episode, feeds the
    (x, t) coordinates to the trained PINN, and returns a clean DataFrame
    with smooth, continuous physical profiles of density and speed.

    Data gaps are filled by the LWR physics embedded in the network:
    the PINN predicts on a dense grid even where there is no
    AIS observation.

    Parameters
    ----------
    model        : trained LWRPINN
    episode_data : DataFrame with columns [x_km, t_days]
                     x_km   = position along the channel in km from the entrance
                     t_days = days elapsed since the episode start (t=0 at onset)
    global_meta  : normalisation metadata from training
                   (channel_len_km, global_rho_max, global_v_max)
    x_grid_points: spatial resolution of the output grid
    t_grid_points: temporal resolution of the output grid
    t_max_days   : prediction horizon in days
                   (default: observed window × 1.2 + 1 day)

    Returns
    -------
    DataFrame with columns:
        x_km      : position along the channel [km]
        t_days    : time since the episode onset [days]
        rho_hat   : predicted density [vessels/km]   — smoothed by LWR physics
        v_hat     : predicted speed [knots]          — smoothed by LWR physics
        flux_hat  : predicted flow [vessels·kn/km]   = rho_hat × v_hat
    """
    model.eval()
    device = get_device()
    model = model.to(device)

    channel_len_km = global_meta["channel_len_km"]
    global_rho_max = global_meta["global_rho_max"]
    global_v_max = global_meta["global_v_max"]

    # Prediction horizon
    if t_max_days is None:
        t_max_days = float(episode_data["t_days"].max()) * 1.2 + 1.0

    # Dense evaluation grid in physical coordinates
    x_grid = np.linspace(0.0, channel_len_km, x_grid_points, dtype=np.float32)
    t_grid = np.linspace(0.0, t_max_days, t_grid_points, dtype=np.float32)
    xx, tt = np.meshgrid(x_grid, t_grid)  # shape: (t_grid_points, x_grid_points)

    x_flat = xx.ravel()
    t_flat = tt.ravel()

    # Normalise into the PINN input space [0, 1]²
    x_norm = x_flat / max(channel_len_km, 1e-8)
    t_norm = t_flat / max(t_max_days, 1e-8)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(1)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        rho_pred_norm, v_pred_norm = model(x_tensor, t_tensor)

    # Denormalise back to physical units
    rho_hat = rho_pred_norm.cpu().numpy().ravel() * global_rho_max
    v_hat = v_pred_norm.cpu().numpy().ravel() * global_v_max

    return pd.DataFrame({
        "x_km": x_flat,
        "t_days": t_flat,
        "rho_hat": rho_hat,
        "v_hat": v_hat,
        "flux_hat": rho_hat * v_hat,
    })


def extract_pinn_summaries(profiles: pd.DataFrame) -> dict:
    """
    Extract scalar summaries of the predicted PINN field (ρ̂, v̂, flux̂).

    These scalars are the physical features of the Phase 4 XGBoost,
    which predicts the observed TTC (episode duration) from the characteristics
    of the congestion field reconstructed by the PINN.

    Returns
    -------
    dict with keys:
        peak_rho        [vessels/km] : maximum density over the episode
        t_to_peak_days  [days]       : time of the congestion peak
        area_under_rho  [v/km * d]   : time integral of max_x ρ(x,t)
        decay_rate      [v/km / d]   : post-peak clearing slope (negative = decay)
        flux_entry_mean [v*kn/km]    : mean flux over the first tenth of the channel
        v_mean_global   [knots]      : mean speed over the whole episode
    """
    t_series = (
        profiles.groupby("t_days")["rho_hat"]
        .max()
        .reset_index()
        .sort_values("t_days")
    )
    t_vals    = t_series["t_days"].values
    rho_max_t = t_series["rho_hat"].values

    if len(t_vals) == 0:
        return {
            "peak_rho": 0.0, "t_to_peak_days": 0.0,
            "area_under_rho": 0.0, "decay_rate": 0.0,
            "flux_entry_mean": 0.0, "v_mean_global": 0.0,
        }

    peak_idx  = int(np.argmax(rho_max_t))
    peak_rho  = float(rho_max_t[peak_idx])
    t_to_peak = float(t_vals[peak_idx])

    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was removed in NumPy 2
    area_under_rho = float(trapezoid(rho_max_t, t_vals)) if len(t_vals) > 1 else 0.0

    post_t   = t_vals[peak_idx:]
    post_rho = rho_max_t[peak_idx:]
    decay_rate = float(np.polyfit(post_t, post_rho, 1)[0]) if len(post_t) >= 2 else 0.0

    x_threshold     = profiles["x_km"].quantile(0.10)
    entry           = profiles[profiles["x_km"] <= x_threshold]
    flux_entry_mean = float(entry["flux_hat"].mean()) if len(entry) > 0 else 0.0

    v_mean_global = float(profiles["v_hat"].mean())

    return {
        "peak_rho":        peak_rho,
        "t_to_peak_days":  t_to_peak,
        "area_under_rho":  area_under_rho,
        "decay_rate":      decay_rate,
        "flux_entry_mean": flux_entry_mean,
        "v_mean_global":   v_mean_global,
    }


def plot_physical_profiles(
    profiles: pd.DataFrame,
    episode_data: Optional[pd.DataFrame] = None,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot the PINN physical profiles as 3 interactive heatmaps:
        - Density ρ(x, t)  [vessels/km]
        - Speed   v(x, t)  [knots]
        - Flow    q(x, t)  = ρ·v

    x axis (horizontal) = time since the episode onset [days]
    y axis (vertical)   = position along the channel [km]

    If episode_data is provided (columns [x_km, t_days]), the actual AIS
    observations are overlaid as a scatter on each panel.

    Parameters
    ----------
    profiles     : DataFrame from extract_physical_profiles
    episode_data : raw AIS observations to overlay (optional)
    title_suffix : text appended to the title (e.g. episode name)
    output_path  : output HTML path (default: outputs/figures/)

    Returns
    -------
    Path of the saved HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required for plotting: pip install plotly")

    # Pivot the long DataFrame → 2D grid for the heatmaps
    t_vals = np.sort(profiles["t_days"].unique())
    x_vals = np.sort(profiles["x_km"].unique())

    def _to_grid(col: str) -> np.ndarray:
        pivot = profiles.pivot_table(index="x_km", columns="t_days", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=x_vals, columns=t_vals)
        return pivot.values  # shape (n_x, n_t)

    rho_grid = _to_grid("rho_hat")
    v_grid = _to_grid("v_hat")
    flux_grid = _to_grid("flux_hat")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Density rho(x,t)  [vessels/km]",
            "Speed v(x,t)      [knots]",
            "Debit  q(x,t)     [vessels*kn/km]",
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colorscales = ["Blues", "RdYlGn_r", "Oranges"]
    grids = [rho_grid, v_grid, flux_grid]
    rows = [1, 2, 3]

    for row, grid, cscale in zip(rows, grids, colorscales):
        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=t_vals,
                y=x_vals,
                colorscale=cscale,
                zsmooth="best",
                colorbar=dict(len=0.28, y=1.0 - (row - 1) * 0.36, thickness=12),
                hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km  val=%{z:.2f}<extra></extra>",
            ),
            row=row, col=1,
        )

        # Overlay the actual AIS observations if available
        if episode_data is not None and "x_km" in episode_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=episode_data["t_days"],
                    y=episode_data["x_km"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="white",
                        line=dict(color="black", width=1),
                        symbol="circle",
                    ),
                    name="Observations AIS",
                    showlegend=(row == 1),
                    hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km<extra>AIS</extra>",
                ),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="x [km]", row=1, col=1)
    fig.update_yaxes(title_text="x [km]", row=2, col=1)
    fig.update_yaxes(title_text="x [km]", row=3, col=1)
    fig.update_xaxes(title_text="Time since onset [days]", row=3, col=1)

    title = "PINN LWR — Physical profiles"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=900,
        template="plotly_white",
        legend=dict(x=1.02, y=0.5),
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn_profiles"
        output_path = OUTPUT_DIR / f"{stem}_physical_profiles.png"

    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Plot saved -> %s", output_path)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# ADVANCED LWR PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_shockwave_heatmap(
    profiles: pd.DataFrame,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    LWR space-time diagram — shock wave.

    Houston waypoint convention (CHANNEL_AXES["houston"]):
      - x = 0 km : turning basin / terminals (inland, city side)
      - x = max  : Galveston Bay / Gulf entrance (seaward)
    The order follows the waypoints — x=0 is the first one.

      - X axis : space x [km]
      - Y axis : time t [hours] since the episode start
      - Colour : density ρ̂  (blue=free flow → dark red=saturated)

    The shock wave shows up as a dense red diagonal band on the
    terminal side (x≈0) propagating towards the entrance (large x) over time.

    Returns output path.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    profiles_h = profiles.copy()
    profiles_h["t_hours"] = profiles_h["t_days"] * 24.0

    t_u = np.sort(profiles_h["t_hours"].unique())
    x_u = np.sort(profiles_h["x_km"].unique())

    # z shape: (n_t, n_x) — t on rows (Y), x on columns (X)
    rho_grid = (
        profiles_h.pivot_table(index="t_hours", columns="x_km", values="rho_hat", aggfunc="mean")
        .reindex(index=t_u, columns=x_u)
        .values
    )

    colorscale = [
        [0.00, "#0a1628"],
        [0.25, "#1565C0"],
        [0.50, "#FFD600"],
        [0.75, "#E65100"],
        [1.00, "#7B0000"],
    ]

    rho_max = float(np.nanmax(rho_grid)) if not np.all(np.isnan(rho_grid)) else 1.0
    rho_mean = float(np.nanmean(rho_grid)) if not np.all(np.isnan(rho_grid)) else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=x_u,
            y=t_u,
            colorscale=colorscale,
            zsmooth="best",
            colorbar=dict(title=dict(text="rho<br>[v/km]", side="right"), thickness=16),
            hovertemplate=(
                "x = %{x:.1f} km<br>t = %{y:.1f} h<br>rho = %{z:.2f} v/km<extra></extra>"
            ),
        )
    )

    # Contour at the shock front (50% of the peak) to materialise the wave
    shock_level = 0.5 * rho_max
    if shock_level > rho_mean * 0.1 and not np.isnan(shock_level):
        fig.add_trace(
            go.Contour(
                z=rho_grid,
                x=x_u,
                y=t_u,
                contours=dict(
                    start=shock_level, end=shock_level, size=1,
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(color="white", size=10),
                ),
                line=dict(color="white", width=2, dash="dot"),
                showscale=False,
                name=f"Front ({shock_level:.1f} v/km)",
            )
        )

    title = "PINN LWR — Shock wave  |  rho(x, t)"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(title="Distance along the channel [km]  (entrance -> port)", showgrid=False),
        yaxis=dict(
            title="Time since onset [hours]",
            autorange="reversed",
            showgrid=False,
        ),
        height=620,
        template="plotly_dark",
        annotations=[
            dict(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text="<b>-- Shock front</b><br><i>propagates towards x=0 (sea)</i>",
                showarrow=False,
                font=dict(color="white", size=11),
                align="left",
                bgcolor="rgba(0,0,0,0.45)",
                bordercolor="white", borderwidth=1,
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_shockwave.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Shock-wave diagram saved -> %s", output_path)
    return output_path


def plot_ais_vs_pinn(
    profiles: pd.DataFrame,
    raw_x_km: np.ndarray,
    raw_t_days: np.ndarray,
    raw_rho: np.ndarray,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Side-by-side comparison: raw AIS data (with gaps) vs PINN output (smooth).

    Shows how the PINN fills satellite coverage gaps through the
    LWR physics constraint.

    Left panel : AIS observations binned on the PINN grid — white areas
                 = no AIS signal (coverage gaps).
    Right panel: dense PINN output — continuous and physically consistent
                 even where there is no observation.

    Parameters
    ----------
    profiles    : DataFrame from extract_physical_profiles (dense grid)
    raw_x_km    : x positions [km] of the AIS observations
    raw_t_days  : time [days] of the AIS observations (episode-relative)
    raw_rho     : density [vessels/km] of the AIS observations
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    t_u = np.sort(profiles["t_days"].unique())
    x_u = np.sort(profiles["x_km"].unique())

    pinn_grid = (
        profiles.pivot_table(index="t_days", columns="x_km", values="rho_hat", aggfunc="mean")
        .reindex(index=t_u, columns=x_u)
        .values
    )

    # Bin the raw AIS observations on the PINN grid
    raw_grid = np.full((len(t_u), len(x_u)), np.nan, dtype=np.float32)
    if len(raw_x_km) > 0:
        dx = (x_u[1] - x_u[0]) / 2 if len(x_u) > 1 else 0.5
        dt = (t_u[1] - t_u[0]) / 2 if len(t_u) > 1 else 0.5
        x_edges = np.concatenate([[x_u[0] - dx], (x_u[:-1] + x_u[1:]) / 2, [x_u[-1] + dx]])
        t_edges = np.concatenate([[t_u[0] - dt], (t_u[:-1] + t_u[1:]) / 2, [t_u[-1] + dt]])
        raw_acc = np.zeros((len(t_u), len(x_u)), dtype=np.float64)
        raw_cnt = np.zeros((len(t_u), len(x_u)), dtype=np.int32)
        for xi, ti, ri in zip(raw_x_km, raw_t_days, raw_rho):
            xi_i = int(np.clip(np.searchsorted(x_edges, xi, side="right") - 1, 0, len(x_u) - 1))
            ti_i = int(np.clip(np.searchsorted(t_edges, ti, side="right") - 1, 0, len(t_u) - 1))
            raw_acc[ti_i, xi_i] += ri
            raw_cnt[ti_i, xi_i] += 1
        mask = raw_cnt > 0
        raw_grid[mask] = (raw_acc[mask] / raw_cnt[mask]).astype(np.float32)

    zmax = float(np.nanmax(pinn_grid)) if not np.all(np.isnan(pinn_grid)) else 1.0
    colorscale = [
        [0.00, "#0a1628"],
        [0.25, "#1565C0"],
        [0.50, "#FFD600"],
        [0.75, "#E65100"],
        [1.00, "#7B0000"],
    ]
    coverage_pct = 100.0 * float(np.sum(~np.isnan(raw_grid))) / raw_grid.size

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Ground Truth AIS  ({coverage_pct:.0f}% coverage)",
            "PINN output  (continuous, no gaps)",
        ),
        horizontal_spacing=0.10,
    )

    for col, (grid, name) in enumerate([(raw_grid, "AIS brut"), (pinn_grid, "PINN")], start=1):
        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=x_u,
                y=t_u,
                colorscale=colorscale,
                zmin=0, zmax=zmax,
                zsmooth="best",
                showscale=(col == 2),
                colorbar=dict(title="rho<br>[v/km]", len=0.8, y=0.5, thickness=14)
                if col == 2 else None,
                hovertemplate=(
                    f"x=%{{x:.1f}}km  t=%{{y:.1f}}j<br>"
                    f"rho=%{{z:.2f}}<br><i>{name}</i><extra></extra>"
                ),
            ),
            row=1, col=col,
        )

    fig.update_yaxes(title_text="Time [days]", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="x [km]", row=1, col=1)
    fig.update_xaxes(title_text="x [km]", row=1, col=2)

    title = "PINN LWR — Physics smoothing: raw AIS vs. PINN output"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=550,
        template="plotly_dark",
        annotations=[
            dict(
                x=0.50, y=-0.14, xref="paper", yref="paper",
                text=(
                    "<i>White areas (left) = no AIS signal. "
                    "The PINN (right) fills these gaps through the LWR physics constraint.</i>"
                ),
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_ais_vs_pinn.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("AIS vs PINN comparison saved -> %s", output_path)
    return output_path


def plot_density_profiles_1d(
    profiles: pd.DataFrame,
    n_slices: int = 6,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    1D density profiles rho(x) at different times of an episode.

    Shows how the blocked-mass bump evolves: flat at first, it grows
    in front of the port entrance (high x) and propagates seaward.

    Each curve is a cross-section rho(x) at fixed t.
    Colour gradient: blue (start) -> red (end of the episode).

    Parameters
    ----------
    profiles   : DataFrame from extract_physical_profiles
    n_slices   : number of time slices to plot (default: 6)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    t_all = np.sort(profiles["t_days"].unique())

    indices = np.round(np.linspace(0, len(t_all) - 1, n_slices)).astype(int)
    t_slices = t_all[indices]

    # Gradient: blue (hue=240) -> red (hue=0)
    colors = [
        f"hsl({int(240 - 240 * i / max(n_slices - 1, 1))}, 80%, 50%)"
        for i in range(n_slices)
    ]

    fig = go.Figure()
    for i, (t_val, color) in enumerate(zip(t_slices, colors)):
        idx = int(np.argmin(np.abs(t_all - t_val)))
        t_actual = t_all[idx]
        slice_df = profiles[np.isclose(profiles["t_days"], t_actual)].sort_values("x_km")
        if len(slice_df) == 0:
            continue

        label = f"t = {t_actual*24:.0f}h" if t_actual < 2 else f"t = {t_actual:.1f}j"
        fill_color = color.replace("hsl", "hsla").replace(")", ", 0.08)")

        fig.add_trace(
            go.Scatter(
                x=slice_df["x_km"].values,
                y=slice_df["rho_hat"].values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                fill="tozeroy",
                fillcolor=fill_color,
                hovertemplate=(
                    f"x=%{{x:.1f}} km<br>rho=%{{y:.2f}} v/km<br>{label}<extra></extra>"
                ),
            )
        )

    title = "PINN LWR — Density profiles rho(x) at different times"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="Distance along the channel [km]  (entrance -> port)",
            showgrid=True, gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title="Density rho  [vessels/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        height=520,
        template="plotly_white",
        legend=dict(title="Time (blue=start, red=end)", x=1.02, y=0.5),
        annotations=[
            dict(
                x=0.50, y=-0.14, xref="paper", yref="paper",
                text=(
                    "<i>The bump forms at high x (port entrance) "
                    "and propagates towards x=0 (sea): LWR kinematic wave.</i>"
                ),
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_profiles_1d.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("1D profiles saved -> %s", output_path)
    return output_path


def plot_fundamental_diagram(
    profiles: pd.DataFrame,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Fundamental traffic diagram: flow q = rho * v as a function of rho.

    Under the LWR equation (Greenshields model), (rho, q) forms a
    parabola: q rises with rho up to the channel capacity,
    then collapses (jam — speed -> 0).

    If the PINN has learned the physics, the point cloud reproduces
    this parabola naturally, without it being imposed explicitly.
    This shows the PINN behaves like a physics simulator,
    not a black box.

    The cloud is coloured by time t [days] to show how the traffic
    state evolves.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    rho = profiles["rho_hat"].values.astype(np.float64)
    q   = profiles["flux_hat"].values.astype(np.float64)
    t   = profiles["t_days"].values

    rho_p99 = float(np.percentile(rho, 99))

    # Fit Greenshields: q = a*rho + b*rho^2 (no intercept, b < 0)
    valid = (rho > 0.01 * rho_p99) & np.isfinite(q) & np.isfinite(rho)
    has_fit = False
    rho_opt_fit = rho_jam_fit = q_max_fit = v_free_fit = 0.0
    rho_theory = q_theory = np.array([])

    if valid.sum() >= 10:
        A = np.stack([rho[valid], rho[valid] ** 2], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, q[valid], rcond=None)
        a_c, b_c = float(coeffs[0]), float(coeffs[1])
        if b_c < 0:
            has_fit    = True
            rho_jam_fit = -a_c / b_c
            v_free_fit  = a_c
            q_max_fit   = a_c ** 2 / (-4.0 * b_c)
            rho_opt_fit = rho_jam_fit / 2.0
            rho_theory  = np.linspace(0, min(rho_jam_fit * 1.05, rho_p99 * 2.5), 300)
            q_theory    = a_c * rho_theory + b_c * rho_theory ** 2

    fig = go.Figure()

    # Point cloud coloured by time
    fig.add_trace(
        go.Scatter(
            x=rho, y=q,
            mode="markers",
            marker=dict(
                size=5,
                color=t,
                colorscale="RdYlBu_r",
                colorbar=dict(title="t [days]", thickness=14),
                opacity=0.55,
                line=dict(width=0),
            ),
            name="Predictions PINN",
            hovertemplate=(
                "rho=%{x:.2f} v/km<br>q=%{y:.2f} v*kn/km<br>"
                "t=%{marker.color:.1f}j<extra></extra>"
            ),
        )
    )

    if has_fit:
        fig.add_trace(
            go.Scatter(
                x=rho_theory, y=q_theory,
                mode="lines",
                line=dict(color="red", width=2.5, dash="dash"),
                name=(
                    f"Greenshields ajuste<br>"
                    f"v_free={v_free_fit:.1f} kn, rho_jam={rho_jam_fit:.1f} v/km"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[rho_opt_fit], y=[q_max_fit],
                mode="markers+text",
                marker=dict(size=14, color="red", symbol="star"),
                text=[f"  q* = {q_max_fit:.2f}<br>  rho* = {rho_opt_fit:.2f}"],
                textposition="top right",
                name="Maximum capacity q*",
                textfont=dict(color="red", size=12),
            )
        )

    title = "PINN LWR — Fundamental traffic diagram  q = rho * v"
    if title_suffix:
        title += f"  |  {title_suffix}"

    caption = (
        "<i>The (Greenshields) parabola is a consequence of the LWR equation. "
        "If the PINN reproduces it naturally, it has learned the physics "
        "of maritime traffic — not a black box.</i>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="Density rho  [vessels/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        yaxis=dict(
            title="Flux q = rho * v  [vessels*kn/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        height=560,
        template="plotly_white",
        legend=dict(x=1.02, y=0.7),
        annotations=[
            dict(
                x=0.50, y=-0.13, xref="paper", yref="paper",
                text=caption,
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_fundamental_diagram.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Fundamental diagram saved -> %s", output_path)
    return output_path


def plot_congestion_clearance(
    profiles: pd.DataFrame,
    rho_threshold_frac: float = 0.7,
    ttc_metric: str = "mean",
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> tuple[Path, Optional[float]]:
    """
    Plot the congestion clearing dynamics and compute the TTC.

    Congestion is measured by a spatial aggregate of ρ(x,t) (mean or max
    along the channel) at each time. The TTC is the first day this value
    falls below the threshold = rho_threshold_frac × peak.

    Panel 1 — Clearing curve:
        aggregated ρ(x,t) over time, with:
        - red dotted threshold line
        - grey "active congestion" area
        - green vertical marker at the TTC
        - "TTC = X days" annotation

    Panel 2 — Spatial heatmap ρ(x,t):
        The same field as plot_physical_profiles, with the red isoline
        at the threshold level — shows which end of the channel
        clears first and how fast.

    Parameters
    ----------
    profiles            : DataFrame from extract_physical_profiles
    rho_threshold_frac  : fraction of the peak defining the return to normal.
                          With ttc_metric="mean" (default), 0.7 = congestion down to 70%
                          of its peak in mean density.
    ttc_metric          : 'mean' (default) or 'max'.
                          'mean' = spatial mean — overall channel load (recommended).
                          'max'  = spatial max — the port-side queue (always high).
    title_suffix        : text appended to the title
    output_path         : output path

    Returns
    -------
    (path_png, ttc_days)  — ttc_days is None if not reached within the window
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    # ── Clearing curve ───────────────────────────────────────────────────────
    # Compute both max and mean across x at each t
    t_grp = profiles.groupby("t_days")["rho_hat"]
    t_max_series  = t_grp.max().reset_index().sort_values("t_days")
    t_mean_series = t_grp.mean().reset_index().sort_values("t_days")

    t_vals_c  = t_max_series["t_days"].values
    rho_max_t = t_max_series["rho_hat"].values
    rho_mean_t= t_mean_series["rho_hat"].values

    # Select the primary metric for TTC
    if ttc_metric == "mean":
        rho_primary = rho_mean_t
        metric_label = "mean_x ρ(x,t)"
    else:
        rho_primary = rho_max_t
        metric_label = "max_x ρ(x,t)"

    rho_peak = float(rho_primary.max()) if len(rho_primary) > 0 else 1.0
    threshold = rho_threshold_frac * rho_peak

    # TTC = first t where primary metric < threshold
    ttc_days: Optional[float] = None
    below = np.where(rho_primary < threshold)[0]
    if len(below) > 0:
        ttc_days = float(t_vals_c[below[0]])

    # ── 2D grid for the heatmap ──────────────────────────────────────────────
    t_u = np.sort(profiles["t_days"].unique())
    x_u = np.sort(profiles["x_km"].unique())
    rho_grid = (
        profiles.pivot_table(index="x_km", columns="t_days", values="rho_hat", aggfunc="mean")
        .reindex(index=x_u, columns=t_u)
        .values
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.42, 0.58],
        subplot_titles=(
            f"Clearing curve — {metric_label}",
            "Density field ρ(x,t)  [vessels/km]  with threshold isoline",
        ),
        vertical_spacing=0.12,
    )

    # — Panel 1: clearing curve ————————————————————————————————————————
    # Grey "active congestion" area (rho > threshold)
    rho_clipped = np.where(rho_primary > threshold, rho_primary, threshold)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_vals_c, t_vals_c[::-1]]),
            y=np.concatenate([rho_clipped, np.full(len(t_vals_c), threshold)]),
            fill="toself",
            fillcolor="rgba(200, 50, 50, 0.15)",
            line=dict(width=0),
            name="Congested area",
            showlegend=True,
        ),
        row=1, col=1,
    )
    # Main curve (selected metric)
    fig.add_trace(
        go.Scatter(
            x=t_vals_c,
            y=rho_primary,
            mode="lines",
            line=dict(color="#1565C0", width=2.5),
            name=metric_label,
        ),
        row=1, col=1,
    )
    # Secondary curve (the other metric, in the background)
    secondary_y = rho_mean_t if ttc_metric == "max" else rho_max_t
    secondary_label = "mean_x ρ" if ttc_metric == "max" else "max_x ρ (contextuel)"
    fig.add_trace(
        go.Scatter(
            x=t_vals_c,
            y=secondary_y,
            mode="lines",
            line=dict(color="#90A4AE", width=1.2, dash="dot"),
            name=secondary_label,
            opacity=0.6,
        ),
        row=1, col=1,
    )
    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        annotation_text=f"Threshold ({rho_threshold_frac*100:.0f}% of peak = {threshold:.1f} v/km)",
        annotation_position="top right",
        row=1, col=1,
    )
    # Marqueur TTC
    if ttc_days is not None:
        fig.add_vline(
            x=ttc_days,
            line_dash="dot",
            line_color="green",
            line_width=2,
            row=1, col=1,
        )
        fig.add_annotation(
            x=ttc_days,
            y=rho_peak * 0.85,
            text=f"<b>TTC = {ttc_days:.1f} days</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(color="green", size=13),
            ax=40, ay=-30,
            row=1, col=1,
        )
    else:
        fig.add_annotation(
            x=t_vals_c[-1] * 0.6,
            y=rho_peak * 0.5,
            text="TTC not reached within the window",
            font=dict(color="orange", size=12),
            showarrow=False,
            row=1, col=1,
        )

    fig.update_yaxes(title_text=f"{metric_label}  [vessels/km]", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)

    # — Panel 2 : heatmap + isoline ──────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=t_u,
            y=x_u,
            colorscale="Blues",
            zsmooth="best",
            colorbar=dict(title="rho<br>[v/km]", len=0.45, y=0.22, thickness=14),
            hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km  rho=%{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Threshold isoline
    fig.add_trace(
        go.Contour(
            z=rho_grid,
            x=t_u,
            y=x_u,
            contours=dict(
                start=threshold, end=threshold, size=1,
                coloring="none",
                showlabels=True,
                labelfont=dict(color="red", size=11),
            ),
            line=dict(color="red", width=2, dash="dash"),
            showscale=False,
            name=f"Threshold {threshold:.1f} v/km",
        ),
        row=2, col=1,
    )
    # Vertical TTC line on the heatmap
    if ttc_days is not None:
        fig.add_vline(
            x=ttc_days,
            line_dash="dot",
            line_color="green",
            line_width=2,
            row=2, col=1,
        )

    fig.update_yaxes(title_text="x [km]", row=2, col=1)
    fig.update_xaxes(title_text="Time since onset [days]", row=2, col=1)

    # ── Mise en page ─────────────────────────────────────────────────────────
    ttc_label = f"{ttc_days:.1f}d" if ttc_days is not None else "not reached"
    title = f"PINN LWR — Congestion clearing ({metric_label})  |  TTC = {ttc_label}"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        height=750,
        template="plotly_white",
        legend=dict(x=1.02, y=0.85),
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_clearance.png"

    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info(
        "Clearance plot saved -> %s  (TTC=%s)",
        output_path, f"{ttc_days:.1f}j" if ttc_days is not None else "n/a",
    )
    return output_path, ttc_days


# ════════════════════════════════════════════════════════════════════════════
# NOWCAST VALIDATION — predicted vs actual TTC
# ════════════════════════════════════════════════════════════════════════════

def _compute_gt_ttc(
    gt_t_days: np.ndarray,
    gt_rho_physical: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """
    Compute the actual TTC from the ground-truth AIS data.

    Groups the AIS observations by day and computes max_x(ρ) at each
    discrete time step. The actual TTC is the first day where max_x(ρ) < threshold.

    Parameters
    ----------
    gt_t_days       : episode-relative time [days] of the GT observations
                      (t=0 = episode start, the GT window starts at obs_end_days)
    gt_rho_physical : corresponding density [vessels/km]
    threshold       : return-to-normal threshold [vessels/km] — same value
                      as the one used for the PINN-predicted TTC

    Returns
    -------
    ttc_days (float) or None if not reached within the GT window
    """
    if len(gt_t_days) == 0 or threshold <= 0:
        return None

    day_bins = np.floor(gt_t_days).astype(int)
    unique_days = np.unique(day_bins)

    for d in unique_days:
        mask = day_bins == d
        if float(gt_rho_physical[mask].max()) < threshold:
            return float(d) + 0.5   # middle of the day

    return None


def plot_nowcast_validation(
    profiles: pd.DataFrame,
    gt_t_days: np.ndarray,
    gt_rho_physical: np.ndarray,
    obs_end_days: float,
    ttc_predicted: Optional[float],
    ttc_actual: Optional[float],
    rho_threshold: float,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Nowcast validation: predicted curve (PINN) vs actual curve (AIS).

    Shows on the same chart:
      - Grey area : observation window [0, obs_end_days]
      - Blue area : prediction horizon [obs_end_days, t_max]
      - Vertical grey line : observation / prediction split (--nowcast-end)
      - Red dotted line    : max_x ρ̂(x,t) predicted by the PINN
      - Solid blue line    : actual max_x ρ(x,t) from the ground-truth AIS
      - Horizontal red line: TTC threshold
      - Green marker : predicted TTC (when the PINN curve drops below the threshold)
      - Orange marker: actual TTC    (when the GT curve drops below the threshold)

    Idea: during fine-tuning the PINN ONLY sees the observation window.
    This chart checks whether its extrapolation beyond it matches reality.

    Parameters
    ----------
    profiles        : DataFrame from extract_physical_profiles (PINN predictions)
    gt_t_days       : episode-relative time [days] of the GT observations
    gt_rho_physical : density [vessels/km] of the GT observations
    obs_end_days    : length of the observation window in days
    ttc_predicted   : TTC predicted by the PINN (days since onset), or None
    ttc_actual      : actual TTC computed from GT, or None
    rho_threshold   : threshold [vessels/km] defining the return to normal
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    # PINN curve: max_x ρ̂(x,t) at each time step
    t_series = (
        profiles.groupby("t_days")["rho_hat"]
        .max()
        .reset_index()
        .sort_values("t_days")
    )
    t_pinn   = t_series["t_days"].values
    rho_pinn = t_series["rho_hat"].values

    # GT curve: max_x ρ(x,t) per day
    gt_curve_t: np.ndarray = np.array([])
    gt_curve_rho: np.ndarray = np.array([])
    if len(gt_t_days) > 0:
        day_bins = np.floor(gt_t_days).astype(int)
        days_sorted = np.unique(day_bins)
        gt_curve_t   = np.array([float(d) + 0.5 for d in days_sorted])
        gt_curve_rho = np.array([float(gt_rho_physical[day_bins == d].max()) for d in days_sorted])

    t_max_plot = float(t_pinn[-1]) if len(t_pinn) > 0 else obs_end_days * 2
    if len(gt_curve_t) > 0:
        t_max_plot = max(t_max_plot, float(gt_curve_t[-1]))

    fig = go.Figure()

    # Coloured areas: observation (grey) / prediction (light blue)
    fig.add_vrect(
        x0=0, x1=obs_end_days,
        fillcolor="rgba(120,120,120,0.10)", line_width=0,
        layer="below",
        annotation_text="<b>Observation</b>", annotation_position="top left",
        annotation_font=dict(size=12, color="gray"),
    )
    fig.add_vrect(
        x0=obs_end_days, x1=t_max_plot * 1.02,
        fillcolor="rgba(21,101,192,0.04)", line_width=0,
        layer="below",
        annotation_text="<b>PINN prediction</b>", annotation_position="top left",
        annotation_font=dict(size=12, color="#1565C0"),
    )

    # Vertical observation / prediction split line
    fig.add_vline(
        x=obs_end_days,
        line_dash="dashdot", line_color="#555555", line_width=2,
        annotation_text=f"Fin obs. (j{obs_end_days:.0f})",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#555555"),
    )

    # TTC threshold (horizontal red line)
    fig.add_hline(
        y=rho_threshold,
        line_dash="dash", line_color="#C62828", line_width=1.2,
        annotation_text=f"TTC threshold = {rho_threshold:.1f} v/km",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#C62828"),
    )

    # Ground-truth AIS curve (solid blue)
    if len(gt_curve_t) > 0:
        fig.add_trace(go.Scatter(
            x=gt_curve_t, y=gt_curve_rho,
            mode="lines+markers",
            line=dict(color="#1565C0", width=2.5),
            marker=dict(size=6, color="#1565C0"),
            name="Ground-truth AIS (actual)",
        ))

    # PINN curve (red dotted)
    fig.add_trace(go.Scatter(
        x=t_pinn, y=rho_pinn,
        mode="lines",
        line=dict(color="#E53935", width=2.2, dash="dot"),
        name="PINN prediction",
    ))

    # Predicted TTC (green)
    if ttc_predicted is not None:
        fig.add_vline(
            x=ttc_predicted, line_dash="dot", line_color="#2E7D32", line_width=2,
        )
        fig.add_annotation(
            x=ttc_predicted, y=rho_threshold * 1.25,
            text=f"<b>Predicted TTC = {ttc_predicted:.1f}d</b>",
            font=dict(color="#2E7D32", size=12),
            showarrow=True, arrowcolor="#2E7D32", arrowhead=2, ax=35, ay=-30,
        )

    # Actual TTC (orange)
    if ttc_actual is not None:
        fig.add_vline(
            x=ttc_actual, line_dash="dot", line_color="#E65100", line_width=2,
        )
        fig.add_annotation(
            x=ttc_actual, y=rho_threshold * 1.55,
            text=f"<b>Actual TTC = {ttc_actual:.1f}d</b>",
            font=dict(color="#E65100", size=12),
            showarrow=True, arrowcolor="#E65100", arrowhead=2, ax=-35, ay=-30,
        )

    # Title with error
    if ttc_predicted is not None and ttc_actual is not None:
        err = abs(ttc_predicted - ttc_actual)
        err_str = f"  |  Error = {err:.1f}d"
    elif ttc_predicted is None and ttc_actual is None:
        err_str = "  |  TTC not reached (predicted + actual)"
    elif ttc_predicted is None:
        err_str = "  |  Predicted TTC: not reached within the horizon"
    else:
        err_str = "  |  Actual TTC: not reached within the GT window"

    title = f"PINN LWR — Validation Nowcast{err_str}"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="Time since onset [days]",
            showgrid=True, gridcolor="#e8e8e8",
        ),
        yaxis=dict(
            title="max ρ(x,t) [vessels/km]",
            showgrid=True, gridcolor="#e8e8e8", rangemode="tozero",
        ),
        height=520,
        template="plotly_white",
        legend=dict(x=1.02, y=0.75),
        annotations=[
            dict(
                x=0.50, y=-0.14, xref="paper", yref="paper",
                text=(
                    "<i>Red dotted line = PINN prediction (without seeing the future). "
                    "Blue line = actual AIS data. "
                    "Grey area = observation window used for fine-tuning.</i>"
                ),
                showarrow=False, font=dict(color="gray", size=10), align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_nowcast_validation.png"

    _force_unlink(output_path)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Nowcast validation saved -> %s", output_path)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# NEW PLOTS — PHASE 3 DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

def plot_loss_decomposition(
    loss_path: Path,
    warmup_epochs: int = 0,
    colloc_refresh: int = COLLOC_REFRESH,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Semi-log plot of training loss components (total, data, pde, kin) vs epochs.

    Requires the loss history saved as .npz (multi-component format from updated
    train_on_episodes). Falls back gracefully if only a legacy .npy (total only) exists.

    Marks collocation refresh epochs and curriculum warmup end on the plot.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    loss_path = Path(loss_path)
    # Try npz first, then npy legacy
    if not loss_path.exists():
        alt = loss_path.with_suffix(".npz" if loss_path.suffix == ".npy" else ".npy")
        if alt.exists():
            loss_path = alt
        else:
            log.warning("Loss file not found: %s", loss_path)
            return None

    if loss_path.suffix == ".npz":
        data = np.load(str(loss_path))
        history = {k: data[k].tolist() for k in data.files}
    else:
        total = np.load(str(loss_path)).tolist()
        history = {"total": total}

    epochs = len(history["total"])
    epoch_arr = np.arange(1, epochs + 1)

    colors = {"total": "#1A237E", "data": "#1565C0", "pde": "#E65100", "kin": "#2E7D32"}
    labels = {"total": "L total", "data": "L data (MSE)", "pde": "L LWR (PDE)", "kin": "L kinematic"}

    fig = go.Figure()
    for key in ["total", "data", "pde", "kin"]:
        if key not in history:
            continue
        fig.add_trace(
            go.Scatter(
                x=epoch_arr,
                y=history[key],
                mode="lines",
                name=labels[key],
                line=dict(color=colors[key], width=2 if key == "total" else 1.5,
                          dash="solid" if key == "total" else "dot" if key != "data" else "solid"),
                opacity=1.0 if key == "total" else 0.75,
            )
        )

    # Vertical markers: collocation refresh epochs
    for ep in range(colloc_refresh, epochs, colloc_refresh):
        fig.add_vline(x=ep, line_dash="dot", line_color="gray", line_width=0.8, opacity=0.5)

    # Curriculum warmup end
    if warmup_epochs > 0 and warmup_epochs < epochs:
        fig.add_vline(
            x=warmup_epochs, line_dash="dash", line_color="purple", line_width=1.5,
            annotation_text=f"fin warmup ({warmup_epochs})",
            annotation_position="top right",
            annotation_font=dict(color="purple", size=10),
        )

    fig.update_layout(
        title=dict(text="PINN — Loss decomposition by component", x=0.5, font=dict(size=14)),
        xaxis=dict(title="Epochs", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Loss (log scale)", type="log", showgrid=True, gridcolor="#e0e0e0"),
        height=480,
        template="plotly_white",
        legend=dict(x=1.02, y=0.85),
        annotations=[
            dict(
                x=0.5, y=-0.14, xref="paper", yref="paper",
                text="<i>Vertical grey lines = collocation resampling. "
                     "L_data should dominate early in training.</i>",
                showarrow=False, font=dict(color="gray", size=10), align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "pinn_loss_decomposition.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Loss decomposition saved -> %s", output_path)
    return output_path


def plot_lwr_residual_field(
    model: LWRPINN,
    global_meta: dict,
    t_max_days: float,
    n_grid: int = 60,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Heatmap of the LWR PDE residual |∂ρ/∂t + ∂(ρv)/∂x| across the x-t domain.

    This is the canonical PINN validation figure (Raissi et al. 2019 JCCP).
    A low residual everywhere proves mass conservation is enforced, not just fitted.
    White = perfect conservation, red = physics violation.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    device = get_device()
    model = model.to(device)
    model.train()  # enable grad tracking through the network

    channel_len_km = global_meta["channel_len_km"]

    x_arr = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    t_arr = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    xx, tt = np.meshgrid(x_arr, t_arr)  # (n_grid, n_grid), t on rows, x on cols

    x_flat = xx.ravel()
    t_flat = tt.ravel()

    x_tensor = torch.tensor(x_flat[:, None], dtype=torch.float32, device=device, requires_grad=True)
    t_tensor = torch.tensor(t_flat[:, None], dtype=torch.float32, device=device, requires_grad=True)

    rho, v = model(x_tensor, t_tensor)
    flux = rho * v

    drho_dt = torch.autograd.grad(
        rho, t_tensor,
        grad_outputs=torch.ones_like(rho),
        create_graph=False, retain_graph=True,
    )[0]
    dflux_dx = torch.autograd.grad(
        flux, x_tensor,
        grad_outputs=torch.ones_like(flux),
        create_graph=False,
    )[0]

    residual = (drho_dt + dflux_dx).abs().detach().cpu().numpy().ravel()
    model.eval()

    residual_grid = residual.reshape(n_grid, n_grid)  # (t_rows, x_cols)
    t_vals_phys = np.linspace(0.0, t_max_days, n_grid)
    x_vals_phys = np.linspace(0.0, channel_len_km, n_grid)

    res_mean = float(np.mean(residual))
    res_max  = float(np.max(residual))
    res_p95  = float(np.percentile(residual, 95))

    fig = go.Figure(
        go.Heatmap(
            z=residual_grid,
            x=x_vals_phys,
            y=t_vals_phys,
            colorscale=[
                [0.00, "white"],
                [0.05, "#AED6F1"],
                [0.25, "#F39C12"],
                [0.60, "#E74C3C"],
                [1.00, "#7B241C"],
            ],
            zmax=res_p95,
            zmin=0.0,
            zsmooth="best",
            colorbar=dict(title=dict(text="|∂ρ/∂t +<br>∂q/∂x|", side="right"), thickness=16),
            hovertemplate="x=%{x:.1f}km  t=%{y:.1f}d<br>residual=%{z:.5f}<extra></extra>",
        )
    )

    title = "PINN — LWR residual field  |∂ρ/∂t + ∂(ρv)/∂x|"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(title="Distance along the channel [km]", showgrid=False),
        yaxis=dict(title="Time since onset [days]", autorange="reversed", showgrid=False),
        height=560,
        template="plotly_white",
        annotations=[
            dict(
                x=0.01, y=0.03, xref="paper", yref="paper",
                text=(
                    f"<b>Mean residual: {res_mean:.5f}   |   max: {res_max:.4f}</b><br>"
                    "<i>Blanc = conservation parfaite  |  Rouge = violation LWR</i>"
                ),
                showarrow=False, font=dict(color="black", size=11), align="left",
                bgcolor="rgba(255,255,255,0.85)", bordercolor="gray", borderwidth=1,
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_lwr_residual.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info(
        "LWR residual field saved -> %s  (mean=%.5f, max=%.4f)",
        output_path, res_mean, res_max,
    )
    return output_path


def plot_characteristic_lines(
    model: LWRPINN,
    global_meta: dict,
    t_max_days: float,
    n_chars: int = 12,
    dt_days: float = 0.1,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Method of Characteristics: integrate dx/dt = v(x,t) from n_chars starting positions.

    Each line shows how a fluid parcel propagates through the channel.
    Converging lines signal shock wave formation (LWR signature).
    Superimposed on the density ρ(x,t) heatmap for context.

    Reference: Daganzo 1994, Newell 2002.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    device = get_device()
    model = model.eval().to(device)

    channel_len_km = global_meta["channel_len_km"]
    global_v_max   = global_meta["global_v_max"]
    global_rho_max = global_meta["global_rho_max"]

    # Build density heatmap (background)
    n_bg = 60
    x_bg = np.linspace(0.0, 1.0, n_bg, dtype=np.float32)
    t_bg = np.linspace(0.0, 1.0, n_bg, dtype=np.float32)
    xx_bg, tt_bg = np.meshgrid(x_bg, t_bg)
    with torch.no_grad():
        x_t_bg = torch.tensor(xx_bg.ravel()[:, None], dtype=torch.float32, device=device)
        t_t_bg = torch.tensor(tt_bg.ravel()[:, None], dtype=torch.float32, device=device)
        rho_bg, _ = model(x_t_bg, t_t_bg)
    rho_grid = rho_bg.cpu().numpy().ravel().reshape(n_bg, n_bg) * global_rho_max
    x_phys_bg = np.linspace(0.0, channel_len_km, n_bg)
    t_phys_bg = np.linspace(0.0, t_max_days, n_bg)

    # RK4 integration of dx/dt = v_physical [km/day]
    KM_PER_KNOT_PER_DAY = 1.852 * 24.0
    n_steps = max(1, int(t_max_days / dt_days))
    x_starts = np.linspace(0.0, channel_len_km, n_chars, dtype=np.float32)

    # Batch-integrate all characteristics simultaneously (one model call per step)
    trajectories_x = [x_starts.copy()]  # list of (n_chars,) arrays
    t_steps = [0.0]

    for step in range(n_steps):
        t_cur = step * dt_days
        x_cur = trajectories_x[-1].copy()

        x_norm = np.clip(x_cur / max(channel_len_km, 1e-8), 0.0, 1.0).astype(np.float32)
        t_norm_cur = float(np.clip(t_cur / max(t_max_days, 1e-8), 0.0, 1.0))

        x_t_cur = torch.tensor(x_norm[:, None], dtype=torch.float32, device=device)
        t_t_cur = torch.full((n_chars, 1), t_norm_cur, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, v_norm = model(x_t_cur, t_t_cur)

        v_phys = v_norm.cpu().numpy().ravel() * global_v_max  # knots
        dx = v_phys * KM_PER_KNOT_PER_DAY * dt_days
        x_new = np.clip(x_cur + dx, 0.0, channel_len_km)

        trajectories_x.append(x_new)
        t_steps.append(t_cur + dt_days)

    t_arr_chars = np.array(t_steps)
    traj_arr = np.array(trajectories_x)  # (n_steps+1, n_chars)

    # Plot
    # Convention: x on x-axis [km], t on y-axis [hours] reversed (like shockwave diagram)
    t_hours_bg = t_phys_bg * 24.0
    t_hours_chars = t_arr_chars * 24.0

    colorscale_bg = [
        [0.00, "#0a1628"], [0.25, "#1565C0"],
        [0.50, "#FFD600"], [0.75, "#E65100"], [1.00, "#7B0000"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=x_phys_bg,
            y=t_hours_bg,
            colorscale=colorscale_bg,
            zsmooth="best",
            colorbar=dict(title=dict(text="ρ<br>[v/km]", side="right"), thickness=14),
            hovertemplate="x=%{x:.1f}km  t=%{y:.1f}h  ρ=%{z:.2f}<extra></extra>",
            name="Density ρ(x,t)",
        )
    )

    # Draw characteristic lines
    palette = [f"hsl({int(240 * i / max(n_chars - 1, 1))}, 90%, 70%)" for i in range(n_chars)]
    for i in range(n_chars):
        fig.add_trace(
            go.Scatter(
                x=traj_arr[:, i],
                y=t_hours_chars,
                mode="lines",
                line=dict(color=palette[i], width=1.5),
                showlegend=(i == 0),
                name="Characteristics dx/dt=v",
                opacity=0.85,
                hovertemplate=f"Char {i+1}: x=%{{x:.1f}}km  t=%{{y:.1f}}h<extra></extra>",
            )
        )

    title = "PINN LWR — Characteristic lines dx/dt = v(x,t)"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(title="Distance along the channel [km]", showgrid=False),
        yaxis=dict(
            title="Time since onset [hours]",
            autorange="reversed", showgrid=False,
        ),
        height=620,
        template="plotly_dark",
        annotations=[
            dict(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text="<b>Coloured lines = trajectories dx/dt = v</b><br>"
                     "<i>Convergence → formation of an LWR shock</i>",
                showarrow=False, font=dict(color="white", size=11), align="left",
                bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1,
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_characteristic_lines.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Characteristic lines saved -> %s", output_path)
    return output_path


def plot_episode_gallery(
    model: LWRPINN,
    episodes: list[Episode],
    global_meta: dict,
    n_show: int = 6,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Gallery of ρ(x,t) heatmaps for the N largest episodes by trigger score.

    Shows that the PINN generalizes consistently across diverse congestion events
    (different durations, intensities, seasonal patterns).
    All episodes use t_norm ∈ [0,1] (episode-relative time), so shorter episodes
    appear compressed compared to longer ones.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    device = get_device()
    model = model.eval().to(device)

    channel_len_km = global_meta["channel_len_km"]
    global_rho_max = global_meta["global_rho_max"]

    # Select top episodes by trigger score
    sorted_eps = sorted(episodes, key=lambda ep: ep.trigger_score, reverse=True)
    top_eps = sorted_eps[:min(n_show, len(sorted_eps))]
    n_actual = len(top_eps)

    n_cols = min(3, n_actual)
    n_rows = (n_actual + n_cols - 1) // n_cols

    n_x, n_t = 20, 30
    x_norm_grid = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
    t_norm_grid = np.linspace(0.0, 1.0, n_t, dtype=np.float32)
    xx, tt = np.meshgrid(x_norm_grid, t_norm_grid)
    x_flat_norm = xx.ravel()
    t_flat_norm = tt.ravel()

    # Batch all episodes in one model call
    x_batch = np.tile(x_flat_norm, n_actual)
    t_batch = np.tile(t_flat_norm, n_actual)
    x_t = torch.tensor(x_batch[:, None], dtype=torch.float32, device=device)
    t_t = torch.tensor(t_batch[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho_norm_all, _ = model(x_t, t_t)
    rho_phys_all = rho_norm_all.cpu().numpy().ravel() * global_rho_max

    pts_per_ep = n_x * n_t

    subplot_titles = []
    for ep in top_eps:
        subplot_titles.append(
            f"{ep.start_date.strftime('%Y-%m')}  {ep.duration_days}j  "
            f"score={ep.trigger_score/1e3:.0f}k"
        )

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    colorscale = [
        [0.00, "#0a1628"], [0.25, "#1565C0"],
        [0.50, "#FFD600"], [0.75, "#E65100"], [1.00, "#7B0000"],
    ]
    zmax_global = float(np.max(rho_phys_all)) if len(rho_phys_all) > 0 else 1.0

    x_phys_grid = np.linspace(0.0, channel_len_km, n_x)

    for i, ep in enumerate(top_eps):
        row = i // n_cols + 1
        col = i % n_cols + 1

        rho_ep = rho_phys_all[i * pts_per_ep: (i + 1) * pts_per_ep]
        rho_grid = rho_ep.reshape(n_t, n_x)  # (t_rows, x_cols)
        t_phys_ep = t_norm_grid * ep.duration_days  # episode-relative days

        fig.add_trace(
            go.Heatmap(
                z=rho_grid,
                x=x_phys_grid,
                y=t_phys_ep,
                colorscale=colorscale,
                zmin=0, zmax=zmax_global,
                zsmooth="best",
                showscale=(i == n_actual - 1),
                colorbar=dict(title="ρ<br>[v/km]", len=0.5, y=0.25, thickness=12),
                hovertemplate=(
                    "x=%{x:.1f}km  t=%{y:.1f}j<br>ρ=%{z:.1f}<extra>"
                    + ep.episode_id + "</extra>"
                ),
            ),
            row=row, col=col,
        )
        if col == 1:
            fig.update_yaxes(title_text="t [j]", row=row, col=col)
        if row == n_rows:
            fig.update_xaxes(title_text="x [km]", row=row, col=col)

    location_label = global_meta.get("location", "").upper()
    fig.update_layout(
        title=dict(
            text=f"PINN — Multi-episode gallery {location_label}  "
                 f"(Top {n_actual} by gravity score)",
            x=0.5, font=dict(size=14),
        ),
        height=300 * n_rows + 100,
        template="plotly_dark",
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        loc = global_meta.get("location", "pinn")
        output_path = OUTPUT_DIR / f"{loc}_pinn_episode_gallery.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Multi-episode gallery saved -> %s  (%d episodes)", output_path, n_actual)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — PINN physical engine (steps 1–4)"
    )
    parser.add_argument("--location", default="houston", choices=["houston", "la"])

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--train", action="store_true",
                      help="Steps 1+2+3+4a: detect the episodes and train the PINN")
    mode.add_argument("--infer", action="store_true",
                      help="Step 4b: extract the physical profiles from a checkpoint")
    mode.add_argument("--both", action="store_true",
                      help="Train then infer (--train then --infer)")
    mode.add_argument("--data-only", action="store_true",
                      help="Step 1 only: print the detected episodes")

    # Data
    parser.add_argument("--start", help="Start of the inference window (YYYY-MM-DD)")
    parser.add_argument("--end",   help="End of the inference window   (YYYY-MM-DD)")
    parser.add_argument(
        "--nowcast-end", default=None,
        help=(
            "End date of the strict observation window (YYYY-MM-DD). "
            "Enables leakage-free nowcast mode: "
            "  --start → --nowcast-end : observation window (fine-tuning + normalisation) "
            "  --nowcast-end → --end   : ground truth loaded AFTER inference, for evaluation only. "
            "Requirement: --nowcast-end must be strictly before --end."
        ),
    )
    parser.add_argument("--gravity-path", type=Path, default=None,
                        help="Path to gravity_daily.parquet")
    parser.add_argument("--gravity-threshold", type=float, default=None,
                        help="Episode trigger threshold (default: value in LOCATION_DEFAULTS)")
    parser.add_argument("--min-episode-days", type=int, default=3,
                        help="Minimum episode length in days (default: 3)")
    parser.add_argument("--constituent-path", type=Path, default=None)
    parser.add_argument("--dx-km", type=float, default=2.0,
                        help="Width of the spatial bins along the channel (km)")

    # Model
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to the .pt checkpoint (for --infer)")
    parser.add_argument("--model-name", default=None,
                        help="Output file name (without extension)")

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-kin", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Curriculum learning: ramp the physics weights from 0 to λ over N epochs (0=off)")
    parser.add_argument("--n-freqs", type=int, default=0,
                        help="Fourier feature embedding: number of frequencies (0=off, suggested: 16)")
    parser.add_argument("--fourier-sigma", type=float, default=1.0,
                        help="Scale of the random frequency matrix (default: 1.0)")
    parser.add_argument("--pad-days", type=int, default=0,
                        help="Recovery days added after the end of each episode (suggested: 30)")

    # Inference grid resolution
    parser.add_argument("--x-grid", type=int, default=100)
    parser.add_argument("--t-grid", type=int, default=100)
    parser.add_argument("--t-max-days", type=float, default=None,
                        help="Prediction horizon in days")
    parser.add_argument("--rho-threshold-frac", type=float, default=0.7,
                        help="Fraction of the peak defining the return to normal "
                             "(default: 0.7 with metric=mean, i.e. 70%% of the peak)")
    parser.add_argument("--ttc-metric", default="mean", choices=["mean", "max"],
                        help="Spatial metric for the TTC: 'mean' (overall load, default) or 'max' (port queue)")

    # Fine-tuning
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune the PINN on the AIS data of the --start/--end window before inference")
    parser.add_argument("--fine-tune-epochs", type=int, default=300,
                        help="Number of fine-tuning epochs (default: 300)")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4,
                        help="Fine-tuning learning rate (default: 1e-4)")

    # Nowcasting: observation window + prediction horizon
    parser.add_argument(
        "--obs-days", type=int, default=None,
        help=(
            "Number of days from --start used as the observation window "
            "to calibrate the PINN (automatic fine-tuning). "
            "The model then predicts the clearing up to --t-max-days. "
            "E.g. --obs-days 7 calibrates on the first 7 days and predicts 60 days by default."
        ),
    )

    args = parser.parse_args()

    if not any([args.train, args.infer, args.both, args.data_only]):
        parser.print_help()
        return

    # ── Step 1: episode detection (train / both / data-only modes) ──────────
    episodes: list[Episode] = []
    global_meta: dict = {}

    if args.train or args.both or args.data_only:
        log.info("=== STEP 1: Structuring the data into episodes ===")
        # Resolve the threshold: --gravity-threshold > location default
        loc_defaults = LOCATION_DEFAULTS.get(args.location, {})
        effective_threshold = (
            args.gravity_threshold
            if args.gravity_threshold is not None
            else loc_defaults.get("gravity_threshold", 50_000)
        )
        effective_min_days = (
            args.min_episode_days
            if args.min_episode_days != 3
            else loc_defaults.get("min_episode_days", args.min_episode_days)
        )
        log.info(
            "Localisation=%s | threshold=%.0f | min_episode_days=%d",
            args.location, effective_threshold, effective_min_days,
        )
        train_start = date.fromisoformat(args.start) if args.start else None
        train_end = date.fromisoformat(args.end) if args.end else None
        episodes, global_meta = structure_episodes(
            location=args.location,
            gravity_path=args.gravity_path,
            gravity_threshold=effective_threshold,
            min_episode_days=effective_min_days,
            dx_km=args.dx_km,
            constituent_path=args.constituent_path,
            start_date=train_start,
            end_date=train_end,
            pad_days=args.pad_days,
        )

        if args.data_only:
            print(f"\n=== Detected episodes ({args.location}, threshold={effective_threshold:.0f}) ===")
            for ep in episodes:
                print(
                    f"  {ep.episode_id}  |  {ep.start_date} -> {ep.end_date}"
                    f"  ({ep.duration_days}j)  |  {ep.n_points} pts"
                    f"  |  trigger={ep.trigger_score:.3f}"
                )
            if global_meta:
                print(f"\n  rho_max global: {global_meta['global_rho_max']:.2f} vessels/km")
                print(f"  v_max global:   {global_meta['global_v_max']:.2f} knots")
            return

    # ── Steps 2-3-4a: training ───────────────────────────────────────────────
    trained_model: Optional[LWRPINN] = None

    if args.train or args.both:
        if not episodes:
            log.error("No episode available for training.")
            return

        log.info("=== STEP 2: LWRPINN architecture — MLP(x, t) → (ρ̂, v̂) ===")
        log.info("=== STEP 3: Hybrid LWR loss + kinematic constraints ===")
        log.info("=== STEP 4a: Episodic training loop ===")

        trained_model, history = train_on_episodes(
            episodes=episodes,
            global_meta=global_meta,
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_kin=args.lambda_kin,
            model_name=args.model_name or f"pinn_{args.location}_episodes",
            warmup_epochs=args.warmup_epochs,
            n_freqs=args.n_freqs,
            fourier_sigma=args.fourier_sigma,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        loss_path = OUTPUT_DIR / f"{args.location}_pinn_episode_loss.npz"
        _force_unlink(loss_path)
        np.savez(str(loss_path), **{k: np.array(v) for k, v in history.items()})
        log.info("Loss curve saved -> %s", loss_path)

        # Loss decomposition plot
        loss_decomp_path = plot_loss_decomposition(
            loss_path=loss_path,
            warmup_epochs=args.warmup_epochs,
            colloc_refresh=COLLOC_REFRESH,
            output_path=OUTPUT_DIR / f"{args.location}_pinn_loss_decomposition.png",
        )
        if loss_decomp_path:
            log.info("Loss decomposition -> %s", loss_decomp_path)

        # Episode gallery (requires model + episodes)
        if trained_model is not None and episodes:
            gallery_path = plot_episode_gallery(
                model=trained_model,
                episodes=episodes,
                global_meta=global_meta,
                n_show=6,
                output_path=OUTPUT_DIR / f"{args.location}_pinn_episode_gallery.png",
            )
            log.info("Episode gallery -> %s", gallery_path)

        if args.both:
            args.model = MODEL_DIR / f"{args.model_name or f'pinn_{args.location}_episodes'}.pt"

    # ── Step 4b: physical profile extraction ──────────────────────────────────
    if args.infer or args.both:
        log.info("=== STEP 4b: extract_physical_profiles ===")

        if args.model is None:
            log.error("--model required for --infer")
            return
        if not Path(args.model).exists():
            log.error("Checkpoint not found: %s", args.model)
            return

        # Load the model and metadata (or reuse them from --both)
        if args.both and trained_model is not None:
            infer_model = trained_model
            infer_meta = global_meta
        else:
            ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
            infer_meta = ckpt["global_meta"]
            ckpt_n_freqs = int(ckpt.get("n_freqs", 0))
            ckpt_sigma   = float(ckpt.get("fourier_sigma", 1.0))
            infer_model  = LWRPINN(
                hidden_layers=4, hidden_size=64,
                n_freqs=ckpt_n_freqs, fourier_sigma=ckpt_sigma,
            )
            infer_model.load_state_dict(ckpt["model_state"])
            log.info(
                "Checkpoint loaded: %s (best_loss=%.6f, %d episode(s))",
                args.model, ckpt["best_loss"], ckpt["n_episodes"],
            )

        # Raw AIS arrays for fine-tuning + plot_ais_vs_pinn
        raw_x_km_arr: np.ndarray = np.array([], dtype=np.float32)
        raw_t_arr:    np.ndarray = np.array([], dtype=np.float32)
        raw_rho_arr:  np.ndarray = np.array([], dtype=np.float32)
        raw_v_arr:    np.ndarray = np.array([], dtype=np.float32)

        # ── AIS loading: classic mode vs nowcast mode ───────────────────────
        #
        # Classic mode (without --nowcast-end):
        #   build_rho_v_tensors(start, end) → a single window
        #
        # Nowcast mode (with --nowcast-end) — NO DATA LEAKAGE:
        #   Call 1 (observation) : build_rho_v_tensors(start, nowcast_end)
        #     → used for fine-tuning AND defines the normalisation (rho_min/max, v_max)
        #   Call 2 (ground truth): build_rho_v_tensors(nowcast_end+1d, end)
        #     → loaded AFTER inference, only for post-hoc evaluation of the actual TTC
        #
        # Invariant: global_rho_max/global_v_max (from the checkpoint) and meta_obs
        # (from the observation window) NEVER see future data.

        # Ground-truth arrays (empty in classic mode)
        gt_x_km_arr: np.ndarray = np.array([], dtype=np.float32)
        gt_t_arr:    np.ndarray = np.array([], dtype=np.float32)
        gt_rho_arr:  np.ndarray = np.array([], dtype=np.float32)
        nowcast_end_d: Optional[date] = None
        obs_end_days: float = 0.0

        if args.start and args.end:
            start_d = date.fromisoformat(args.start)
            end_d   = date.fromisoformat(args.end)
            if end_d < start_d:
                log.error(
                    "Inverted dates: --start %s is after --end %s. "
                    "Check your arguments (e.g. --end %s).",
                    args.start, args.end,
                    (start_d + timedelta(days=120)).isoformat()[:7] + "-28",
                )
                raise SystemExit(1)

            # Resolve nowcast mode
            if args.nowcast_end is not None:
                nowcast_end_d = date.fromisoformat(args.nowcast_end)
                if nowcast_end_d >= end_d:
                    log.error(
                        "--nowcast-end (%s) must be strictly before --end (%s) "
                        "— the ground-truth window must not be empty.",
                        nowcast_end_d, end_d,
                    )
                    raise SystemExit(1)
                if nowcast_end_d <= start_d:
                    log.error(
                        "--nowcast-end (%s) must be after --start (%s).",
                        nowcast_end_d, start_d,
                    )
                    raise SystemExit(1)
                obs_load_end = nowcast_end_d
                obs_end_days = float((nowcast_end_d - start_d).days)
                log.info(
                    "Mode nowcast STRICT — observation [%s, %s] (%d j) | "
                    "ground truth [%s, %s] (%d j)",
                    start_d, nowcast_end_d, (nowcast_end_d - start_d).days,
                    nowcast_end_d + timedelta(1), end_d,
                    (end_d - nowcast_end_d - timedelta(1)).days + 1,
                )
            else:
                obs_load_end = end_d

            # Absolute time offset (shared by both calls)
            t_start_g = (start_d - EPOCH_DATE).days / T_DAYS_MAX

            # ── Call 1: observation window ───────────────────────────────────
            try:
                X_obs, y_obs, meta_obs = build_rho_v_tensors(
                    start_d, obs_load_end, infer_meta["location"],
                    dx_km=infer_meta["dx_km"],
                    use_raw_velocity=True,
                )
                if len(X_obs) > 0:
                    x_km_obs    = X_obs[:, 0] * infer_meta["channel_len_km"]
                    t_days_obs  = (X_obs[:, 1] - t_start_g) * T_DAYS_MAX
                    episode_data = pd.DataFrame({"x_km": x_km_obs, "t_days": t_days_obs})
                    # Denormalise with meta_obs only (no future leakage)
                    rho_range_obs = meta_obs["rho_max"] - meta_obs.get("rho_min", 0.0)
                    raw_rho_arr   = (y_obs[:, 0] * rho_range_obs + meta_obs.get("rho_min", 0.0)).astype(np.float32)
                    raw_v_arr     = (y_obs[:, 1] * max(meta_obs["v_max"], 1e-8)).astype(np.float32)
                    raw_x_km_arr  = x_km_obs.astype(np.float32)
                    raw_t_arr     = t_days_obs.astype(np.float32)
                    log.info(
                        "Observation loaded: %d AIS pts | rho_max_obs=%.2f v/km | v_max_obs=%.2f kn",
                        len(X_obs), meta_obs["rho_max"], meta_obs["v_max"],
                    )
                else:
                    log.warning("No AIS data in the observation window. Using the default grid.")
                    episode_data = pd.DataFrame({
                        "x_km": [0.0, infer_meta["channel_len_km"]],
                        "t_days": [0.0, float((obs_load_end - start_d).days)],
                    })
            except Exception as exc:
                log.warning("Could not load the observation AIS (%s). Using the default grid.", exc)
                episode_data = pd.DataFrame({
                    "x_km": [0.0, infer_meta["channel_len_km"]],
                    "t_days": [0.0, 30.0],
                })

            # ── Call 2: ground truth (nowcast mode only) ─────────────────────
            # Loaded HERE but used only AFTER inference, for evaluation.
            # global_rho_max / fine-tuning never touch this data.
            if nowcast_end_d is not None:
                gt_start_d = nowcast_end_d + timedelta(days=1)
                if gt_start_d <= end_d:
                    try:
                        X_gt, y_gt, meta_gt = build_rho_v_tensors(
                            gt_start_d, end_d, infer_meta["location"],
                            dx_km=infer_meta["dx_km"],
                            use_raw_velocity=True,
                        )
                        if len(X_gt) > 0:
                            x_km_gt   = X_gt[:, 0] * infer_meta["channel_len_km"]
                            t_days_gt = (X_gt[:, 1] - t_start_g) * T_DAYS_MAX
                            rho_range_gt = meta_gt["rho_max"] - meta_gt.get("rho_min", 0.0)
                            gt_rho_arr   = (y_gt[:, 0] * rho_range_gt + meta_gt.get("rho_min", 0.0)).astype(np.float32)
                            gt_x_km_arr  = x_km_gt.astype(np.float32)
                            gt_t_arr     = t_days_gt.astype(np.float32)
                            log.info(
                                "Ground truth loaded (post-hoc evaluation): %d AIS pts | [%s, %s]",
                                len(X_gt), gt_start_d, end_d,
                            )
                        else:
                            log.warning("No AIS data in the ground-truth window.")
                    except Exception as exc:
                        log.warning("Could not load the ground-truth AIS (%s).", exc)
        else:
            # Default grid: full channel over 30 days
            episode_data = pd.DataFrame({
                "x_km": [0.0, infer_meta["channel_len_km"]],
                "t_days": [0.0, 30.0],
            })

        # ── Observation window (--obs-days) + prediction horizon ────────────
        #
        # Nowcasting workflow:
        #   1. Load the first N days of the episode (--obs-days N)
        #   2. Fine-tune the PINN on these N days (calibration)
        #   3. Predict rho(x,t) up to --t-max-days (future horizon)
        #   4. Extract the TTC from the predicted clearing curve
        #
        # Without --obs-days the whole --start/--end range is used (classic mode).

        ft_x_km  = raw_x_km_arr.copy()
        ft_t_arr = raw_t_arr.copy()
        ft_rho   = raw_rho_arr.copy()
        ft_v     = raw_v_arr.copy()

        if args.obs_days is not None and len(raw_t_arr) > 0:
            obs_mask = raw_t_arr <= float(args.obs_days)
            ft_x_km  = raw_x_km_arr[obs_mask]
            ft_t_arr = raw_t_arr[obs_mask]
            ft_rho   = raw_rho_arr[obs_mask]
            ft_v     = raw_v_arr[obs_mask]
            log.info(
                "Observation window: first %d days | %d/%d AIS points kept for fine-tuning",
                args.obs_days, int(obs_mask.sum()), len(raw_t_arr),
            )

        # Prediction horizon
        t_max_days_infer = args.t_max_days
        if t_max_days_infer is None:
            if args.obs_days is not None:
                # Nowcasting: predict over 4x the observation window (min 30 days)
                t_max_days_infer = max(float(args.obs_days) * 4.0, 30.0)
                log.info(
                    "Prediction horizon: %.0f days (4x obs. window of %d d) "
                    "— override with --t-max-days",
                    t_max_days_infer, args.obs_days,
                )
            else:
                # Classic mode: slight extension beyond the observed data
                ep_t_max = float(episode_data["t_days"].max()) if len(episode_data) > 0 else 0.0
                t_max_days_infer = ep_t_max * 1.2 + 1.0

        # Fine-tuning: enabled by --fine-tune OR automatically with --obs-days
        do_fine_tune = args.fine_tune or (args.obs_days is not None) or (nowcast_end_d is not None)
        if do_fine_tune:
            if len(ft_x_km) == 0:
                log.warning(
                    "Fine-tuning requested but no AIS data available "
                    "(check --start/--end and that the AIS parquets exist)."
                )
            else:
                log.info("=== FINE-TUNING on the current event ===")
                infer_model = fine_tune_on_episode(
                    model=infer_model,
                    x_km=ft_x_km,
                    t_days=ft_t_arr,
                    rho_physical=ft_rho,
                    v_physical=ft_v,
                    global_meta=infer_meta,
                    t_max_days=t_max_days_infer,
                    epochs=args.fine_tune_epochs,
                    lr=args.fine_tune_lr,
                )

        profiles = extract_physical_profiles(
            model=infer_model,
            episode_data=episode_data,
            global_meta=infer_meta,
            x_grid_points=args.x_grid,
            t_grid_points=args.t_grid,
            t_max_days=t_max_days_infer,
        )

        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        model_stem = Path(args.model).stem
        out_path = FEATURES_DIR / f"{model_stem}_profiles.parquet"
        _force_unlink(out_path)
        profiles.to_parquet(out_path, index=False)
        log.info("Physical profiles saved -> %s", out_path)

        # Plots
        infer_period = (
            f"{args.start}_to_{args.end}" if args.start and args.end else model_stem
        )

        # Results sub-folder: <location>_<start>_pinn_results/
        location_label = infer_meta["location"]
        date_label = (args.start or model_stem).replace("-", "")[:8]  # YYYYMMDD
        png_dir = OUTPUT_DIR / f"{location_label}_{date_label}_pinn_results"
        png_dir.mkdir(parents=True, exist_ok=True)
        log.info("PNG results -> %s", png_dir)

        title_tag = f"{location_label.upper()} | {infer_period}"

        plot_path = plot_physical_profiles(
            profiles=profiles,
            episode_data=episode_data if len(episode_data) > 2 else None,
            title_suffix=title_tag,
            output_path=png_dir / "01_profiles_rho_v_q.png",
        )

        clearance_path, ttc_days = plot_congestion_clearance(
            profiles=profiles,
            rho_threshold_frac=args.rho_threshold_frac,
            ttc_metric=args.ttc_metric,
            title_suffix=title_tag,
            output_path=png_dir / "02_clearance_ttc.png",
        )

        shock_path = plot_shockwave_heatmap(
            profiles=profiles,
            title_suffix=title_tag,
            output_path=png_dir / "03_shockwave_lwr.png",
        )

        ais_path = plot_ais_vs_pinn(
            profiles=profiles,
            raw_x_km=raw_x_km_arr,
            raw_t_days=raw_t_arr,
            raw_rho=raw_rho_arr,
            title_suffix=title_tag,
            output_path=png_dir / "04_ais_vs_pinn.png",
        )

        slices_path = plot_density_profiles_1d(
            profiles=profiles,
            n_slices=6,
            title_suffix=title_tag,
            output_path=png_dir / "05_profiles_1d_slices.png",
        )

        fd_path = plot_fundamental_diagram(
            profiles=profiles,
            title_suffix=title_tag,
            output_path=png_dir / "06_fundamental_diagram.png",
        )

        # Figure 08 — LWR residual field
        residual_path = plot_lwr_residual_field(
            model=infer_model,
            global_meta=infer_meta,
            t_max_days=t_max_days_infer,
            title_suffix=title_tag,
            output_path=png_dir / "08_lwr_residual.png",
        )

        # Figure 09 — Characteristic lines
        chars_path = plot_characteristic_lines(
            model=infer_model,
            global_meta=infer_meta,
            t_max_days=t_max_days_infer,
            title_suffix=title_tag,
            output_path=png_dir / "09_characteristic_lines.png",
        )

        # ── Plot 07: nowcast validation (only with --nowcast-end) ────────────
        # TTC threshold in physical units (consistent with plot_congestion_clearance)
        _rho_peak_pinn = float(
            profiles.groupby("t_days")["rho_hat"].agg(args.ttc_metric).values.max()
        ) if len(profiles) > 0 else 1.0
        ttc_threshold_val = args.rho_threshold_frac * _rho_peak_pinn

        ttc_actual: Optional[float] = None
        if nowcast_end_d is not None:
            ttc_actual = _compute_gt_ttc(gt_t_arr, gt_rho_arr, ttc_threshold_val)
            plot_nowcast_validation(
                profiles=profiles,
                gt_t_days=gt_t_arr,
                gt_rho_physical=gt_rho_arr,
                obs_end_days=obs_end_days,
                ttc_predicted=ttc_days,
                ttc_actual=ttc_actual,
                rho_threshold=ttc_threshold_val,
                title_suffix=title_tag,
                output_path=png_dir / "07_nowcast_validation.png",
            )

        # ── Prediction summary ───────────────────────────────────────────────
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  PINN LWR — PREDICTION DE RESORPTION | {location_label.upper()}")
        print(sep)

        if args.start and args.end:
            print(f"  Period           : {args.start}  ->  {args.end}")

        if nowcast_end_d is not None:
            print(f"  Mode             : STRICT NOWCAST (no data leakage)")
            print(f"  Observation      : {args.start}  ->  {nowcast_end_d}  ({obs_end_days:.0f} j)")
            print(f"  Ground truth     : {nowcast_end_d + timedelta(1)}  ->  {args.end}")
        elif args.obs_days is not None:
            obs_end_d_disp = date.fromisoformat(args.start) + timedelta(days=args.obs_days)
            print(f"  Mode             : nowcasting (--obs-days={args.obs_days})")
            print(f"  Observation      : {args.start}  ->  {obs_end_d_disp}  ({args.obs_days} j)")
        else:
            print(f"  Mode             : reconstruction (full observed period)")
        print(f"  Horizon          : {t_max_days_infer:.0f} days from the start")

        print(f"\n  --- Diagnostic physique ---")
        print(f"  Chenal     : {infer_meta['channel_len_km']:.1f} km")
        print(f"  rho_hat    : [{profiles['rho_hat'].min():.2f}, {profiles['rho_hat'].max():.2f}] vessels/km")
        print(f"  v_hat      : [{profiles['v_hat'].min():.2f}, {profiles['v_hat'].max():.2f}] knots")
        print(f"  TTC threshold : {ttc_threshold_val:.2f} v/km  ({args.rho_threshold_frac*100:.0f}% of peak)")

        print(f"\n  --- Time to Clear (TTC) ---")
        if ttc_days is not None:
            ttc_abs = ""
            if args.start:
                ttc_date = date.fromisoformat(args.start) + timedelta(days=int(ttc_days))
                ttc_abs  = f"  (date estimee : {ttc_date})"
            print(f"  Predicted TTC = {ttc_days:.1f} d since onset{ttc_abs}")
        else:
            print(f"  Predicted TTC = not reached within the {t_max_days_infer:.0f} d horizon")
            print(f"                  Re-run with --t-max-days {int(t_max_days_infer * 2)}")

        if nowcast_end_d is not None:
            if ttc_actual is not None:
                ttc_actual_date = date.fromisoformat(args.start) + timedelta(days=int(ttc_actual))
                print(f"  Actual TTC    = {ttc_actual:.1f} d since onset  (date: {ttc_actual_date})")
            else:
                print(f"  Actual TTC    = not reached within the ground-truth window")
            if ttc_days is not None and ttc_actual is not None:
                err = abs(ttc_days - ttc_actual)
                sign = "+" if ttc_days > ttc_actual else "-"
                print(f"  Error         = {sign}{err:.1f} d  ({'over-estimated' if ttc_days > ttc_actual else 'under-estimated'})")
            elif ttc_days is None and ttc_actual is not None:
                remaining_horizon = ttc_actual - t_max_days_infer
                print(f"  (horizon too short: actual TTC at {ttc_actual:.1f}d, horizon={t_max_days_infer:.0f}d)")

        print(f"\n  Parquet : {out_path}")
        print(f"  Figures : {png_dir}/")
        print(f"    01_profiles_rho_v_q.png       heatmaps rho / v / debit")
        print(f"    02_clearance_ttc.png           clearing curve + TTC")
        print(f"    03_shockwave_lwr.png           LWR shock wave")
        print(f"    04_ais_vs_pinn.png             AIS brut vs PINN lisse")
        print(f"    05_profiles_1d_slices.png      coupes rho(x) a 6 instants")
        print(f"    06_fundamental_diagram.png     fundamental diagram")
        if nowcast_end_d is not None:
            print(f"    07_nowcast_validation.png     predicted vs actual + TTC error")
        print(f"    08_lwr_residual.png            champ residuel LWR")
        print(f"    09_characteristic_lines.png    characteristic lines dx/dt=v")
        print(sep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
