"""
Phase 4 — Financial pipeline: maritime nowcasting → market signal.

Target variable (Y): change of a financial index over N days.

Target financial assets — automatic routing per port:
  - Houston → WTI crude oil (daily): oil port (~70% energy traffic)
              CSV source: stooq.com/q/d/l/?s=cl.f&i=d  (col=Close)
  - LA      → S&P 500     (daily): US consumer-goods gateway
              CSV source: stooq.com/q/d/l/?s=%5Espx&i=d  (col=Close)
  - Manual override via --financial for any other index (BDI, FBX, GSCSI…)

Explanatory variables (X): gravity score + PINN time-to-clear (TTC) and field summaries.

Step 1: build_events_dataset()      → gravity score + PINN TTC for each episode
Step 2: align_with_financials()     → merge with the target asset, N-day forward return
Step 3: run_correlation_analysis()  → Pearson / Spearman + XGBRegressor (TimeSeriesSplit)
Step 4: generate_trading_signal()   → structured alert adapted to the port (WTI or S&P 500)

Commands (single line each):

  Houston → WTI :
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_houston_episodes.pt --location houston --wti data/financial/WTI.csv --start 2017-01-01 --end 2017-12-31 --n-days 5 10 15 30 --target return_10d

  LA → S&P 500 :
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_la_episodes.pt --location la --sp500 data/financial/SP500.csv --start 2017-01-01 --end 2017-12-31 --gravity-threshold 1000 --n-days 5 10 15 30 --target return_10d

  Manual override (e.g. BDI):
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_houston_episodes.pt --location houston --financial data/financial/BDI.csv --price-col Close --start 2017-01-01 --end 2017-12-31 --n-days 5 10 15 30 --target return_10d

Notes:
  - LA gravity threshold     : 1000 (low bimodal distribution, default=5000 too strict)
  - Houston gravity threshold: 50000 (LOCATION_DEFAULTS, tuned for Harvey)
  - WTI CSV source: stooq.com ticker cl.f (Close column, daily)
  - SP500 CSV: data/financial/SP500.csv (converted from Investing.com)
  - Gravity data: data/features/<location>_gravity_daily.parquet
  - Use --gravity-only for long periods without AIS parquets
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.pinns.lwr_pinn import LWRPINN, get_device
from src.pinns.pinn_pipeline import (
    Episode,
    LOCATION_DEFAULTS,
    extract_physical_profiles,
    extract_pinn_summaries,
    structure_episodes,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/financial")
DEFAULT_N_DAYS = [5, 10, 15, 30]
DEFAULT_RHO_THRESHOLD_FRAC = 0.3

# Port → target financial asset
# Economic rationale:
#   Houston: ~70% oil/gas traffic       → WTI is the most directly affected asset
#   LA     : US consumer-goods gateway  → the S&P 500 captures the retail supply-chain impact
LOCATION_ASSET_MAP: dict[str, dict] = {
    "houston": {
        "asset_name":  "CRUDE OIL (WTI)",
        "asset_short": "WTI",
        "consequence": (
            "reflecting an energy supply shock "
            "(Houston: ~70% oil and gas traffic)."
        ),
    },
    "la": {
        "asset_name":  "S&P 500",
        "asset_short": "SP500",
        "consequence": (
            "reflecting an impact on the supply chain "
            "of US consumer goods."
        ),
    },
}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ════════════════════════════════════════════════════════════════════════════

def _compute_ttc(
    profiles: pd.DataFrame,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
    fallback_days: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """
    Compute the time-to-clear from a DataFrame of PINN profiles.
    TTC = first day where max_x ρ(x,t) < rho_threshold_frac × initial peak.

    If the threshold is never reached within the PINN window, fallback_days
    (= observed episode duration) is used as a conservative lower bound.

    Returns
    -------
    (ttc_value, source)  where source ∈ {"pinn", "duration_fallback", "unavailable"}
    """
    t_series = (
        profiles.groupby("t_days")["rho_hat"]
        .max()
        .reset_index()
        .sort_values("t_days")
    )
    rho_max_t = t_series["rho_hat"].values
    t_vals    = t_series["t_days"].values

    if len(rho_max_t) == 0:
        if fallback_days is not None:
            return fallback_days, "duration_fallback"
        return None, "unavailable"

    threshold = rho_threshold_frac * float(rho_max_t[0])
    below = np.where(rho_max_t < threshold)[0]

    if len(below) > 0:
        return float(t_vals[below[0]]), "pinn"

    # The PINN did not predict a resolution → fall back to the observed duration
    if fallback_days is not None:
        return fallback_days, "duration_fallback"
    return None, "unavailable"


def _load_checkpoint(model_path: Path) -> tuple[LWRPINN, dict]:
    """Load a PINN checkpoint and return (model, global_meta)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LWRPINN(
        hidden_layers=4, hidden_size=64,
        n_freqs=int(ckpt.get("n_freqs", 0)),
        fourier_sigma=float(ckpt.get("fourier_sigma", 1.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(
        "Checkpoint loaded: %s | best_loss=%.6f | %d training episode(s)",
        model_path, ckpt["best_loss"], ckpt["n_episodes"],
    )
    return model, ckpt["global_meta"]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Physical metrics extraction
# ════════════════════════════════════════════════════════════════════════════

def build_events_dataset(
    model: LWRPINN,
    global_meta: dict,
    location: str,
    gravity_threshold: Optional[float] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    gravity_path: Optional[Path] = None,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
    x_grid_points: int = 80,
    t_grid_points: int = 80,
) -> pd.DataFrame:
    """
    Step 1 — Extract gravity score + TTC for each disruption episode.

    For each episode detected by structure_episodes():
      1. Rebuild the physical coordinates (x_km, t_days) from the Episode.
      2. Call extract_physical_profiles() → dense PINN field (ρ̂, v̂).
      3. Compute the TTC with _compute_ttc() (no PNG).

    Parameters
    ----------
    model             : pre-trained LWRPINN (from checkpoint)
    global_meta       : normalisation metadata of the checkpoint
    location          : 'houston' or 'la'
    gravity_threshold : trigger threshold (default: LOCATION_DEFAULTS)
    start_date        : lower bound of the scan period
    end_date          : upper bound of the scan period
    gravity_path      : custom path to gravity_daily.parquet
    rho_threshold_frac: fraction of the peak defining the TTC threshold
    x_grid_points     : spatial resolution of the PINN grid
    t_grid_points     : temporal resolution of the PINN grid

    Returns
    -------
    DataFrame pandas:
        episode_id    | start_date | end_date | duration_days
        gravity_score | ttc_pinn   | n_ais_points | PINN summaries
    """
    if gravity_threshold is None:
        gravity_threshold = LOCATION_DEFAULTS.get(location, {}).get("gravity_threshold", 50_000)

    log.info(
        "=== STEP 1: Build events dataset | %s | threshold=%.0f | %s -> %s",
        location, gravity_threshold,
        start_date or "start", end_date or "end",
    )

    episodes, _ = structure_episodes(
        location=location,
        gravity_path=gravity_path,
        gravity_threshold=gravity_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    if not episodes:
        log.warning("No episode detected — returning an empty DataFrame.")
        return pd.DataFrame(columns=[
            "episode_id", "start_date", "end_date",
            "duration_days", "gravity_score", "ttc_days", "ttc_source", "n_ais_points",
        ])

    log.info("%d episodes detected. Extracting PINN profiles...", len(episodes))
    model.eval()
    records = []

    for ep in episodes:
        log.info(
            "  [%s] %s -> %s (%dj, %d pts AIS, trigger=%.0f)",
            ep.episode_id, ep.start_date, ep.end_date,
            ep.duration_days, ep.n_points, ep.trigger_score,
        )

        ttc, ttc_source = None, "unavailable"
        if ep.n_points > 0:
            # Rebuild physical coordinates from the normalised values
            episode_data = pd.DataFrame({
                "x_km":   ep.x_norm * ep.channel_len_km,
                "t_days": ep.t_norm * ep.duration_days,
            })
            # Extrapolate far enough beyond the episode to observe resolution.
            # Minimum 90 days: typical LA port congestion clears in 2-8 weeks post-episode.
            t_max = max(float(episode_data["t_days"].max()) * 3.0 + 30.0, 90.0)

            try:
                profiles = extract_physical_profiles(
                    model=model,
                    episode_data=episode_data,
                    global_meta=global_meta,
                    x_grid_points=x_grid_points,
                    t_grid_points=t_grid_points,
                    t_max_days=t_max,
                )
                ttc, ttc_source = _compute_ttc(
                    profiles, rho_threshold_frac,
                    fallback_days=float(ep.duration_days),
                )
                pinn_summaries = extract_pinn_summaries(profiles)
                log.info(
                    "    TTC = %.1f days [%s] | peak_rho=%.2f | decay=%.4f",
                    ttc, ttc_source,
                    pinn_summaries["peak_rho"], pinn_summaries["decay_rate"],
                )
            except Exception as exc:
                log.warning("    PINN profile error (%s) — fallback duration", exc)
                ttc, ttc_source = float(ep.duration_days), "duration_fallback"
                pinn_summaries = {
                    "peak_rho": float("nan"), "t_to_peak_days": float("nan"),
                    "area_under_rho": float("nan"), "decay_rate": float("nan"),
                    "flux_entry_mean": float("nan"), "v_mean_global": float("nan"),
                }
        else:
            log.warning("    No AIS point — fallback duration")
            ttc, ttc_source = float(ep.duration_days), "duration_fallback"
            pinn_summaries = {
                "peak_rho": float("nan"), "t_to_peak_days": float("nan"),
                "area_under_rho": float("nan"), "decay_rate": float("nan"),
                "flux_entry_mean": float("nan"), "v_mean_global": float("nan"),
            }

        records.append({
            "episode_id":    ep.episode_id,
            "start_date":    ep.start_date,
            "end_date":      ep.end_date,
            "duration_days": ep.duration_days,   # observed TTC — Phase 4 XGBoost label
            "gravity_score": float(ep.trigger_score),
            "ttc_pinn":      ttc,                # TTC from PINN thresholding (reference)
            "ttc_source":    ttc_source,
            "n_ais_points":  ep.n_points,
            **pinn_summaries,                    # peak_rho, t_to_peak_days, area_under_rho, decay_rate, flux_entry_mean, v_mean_global
        })

    df = pd.DataFrame(records)
    n_pinn     = (df["ttc_source"] == "pinn").sum()
    n_fallback = (df["ttc_source"] == "duration_fallback").sum()
    log.info(
        "Dataset: %d episodes | TTC: %d via PINN, %d via duration fallback, %d unavailable",
        len(df), n_pinn, n_fallback, (df["ttc_source"] == "unavailable").sum(),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 1b — Fast mode: episodes from the gravity parquet (no PINN)
# ════════════════════════════════════════════════════════════════════════════

def build_events_from_gravity(
    location: str,
    gravity_threshold: Optional[float] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    gravity_path: Optional[Path] = None,
    min_episode_days: int = 3,
    gap_tolerance_days: int = 1,
) -> pd.DataFrame:
    """
    --gravity-only mode: build the episode dataset directly from the
    gravity parquet, without loading AIS data or running the PINN.

    TTC = observed episode duration (duration_fallback).
    Gravity score = max score over the episode.

    Advantage: processes whole years in a few seconds.
    Typical use: long periods (2021-2024) when the AIS parquets have
    been deleted, or for a quick correlation.

    Parameters
    ----------
    gap_tolerance_days : number of "below threshold" days tolerated inside
                         an episode (avoids fragmenting on weekends).
    """
    if gravity_threshold is None:
        gravity_threshold = LOCATION_DEFAULTS.get(location, {}).get("gravity_threshold", 50_000)

    # Gravity file path
    if gravity_path is None:
        gravity_path = Path(f"data/features/{location}_gravity_daily.parquet")
    if not gravity_path.exists():
        raise FileNotFoundError(f"Gravity parquet not found: {gravity_path}")

    df = pd.read_parquet(gravity_path)
    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "=== STEP 1 (gravity-only) | %s | threshold=%.0f | %d days | %s -> %s",
        location, gravity_threshold, len(df),
        df["date"].iloc[0].date() if len(df) else "?",
        df["date"].iloc[-1].date() if len(df) else "?",
    )

    # ── Group into episodes ─────────────────────────────────────────────────
    above = df["gravity_score"] >= gravity_threshold
    records = []
    i = 0
    while i < len(df):
        if not above.iloc[i]:
            i += 1
            continue

        # Start of an episode
        ep_start = i
        j = i + 1
        gap = 0
        while j < len(df):
            if above.iloc[j]:
                gap = 0
                j += 1
            elif gap < gap_tolerance_days:
                gap += 1
                j += 1
            else:
                break
        # Remove trailing gap days at the end of the episode
        ep_end = j - gap - 1

        ep_df   = df.iloc[ep_start : ep_end + 1]
        dur     = len(ep_df)

        if dur >= min_episode_days:
            start_d = ep_df["date"].iloc[0].date()
            end_d   = ep_df["date"].iloc[-1].date()
            g_score = float(ep_df["gravity_score"].max())
            ep_id   = f"{location}_{str(start_d).replace('-', '')}_{str(end_d).replace('-', '')}"

            records.append({
                "episode_id":    ep_id,
                "start_date":    start_d,
                "end_date":      end_d,
                "duration_days": dur,
                "gravity_score": g_score,
                "ttc_days":      float(dur),
                "ttc_source":    "duration_fallback",
                "n_ais_points":  0,
            })
        i = ep_end + 1

    result = pd.DataFrame(records)
    log.info(
        "%d episodes detected (threshold=%.0f, min_days=%d)",
        len(result), gravity_threshold, min_episode_days,
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Alignment with financial data
# ════════════════════════════════════════════════════════════════════════════

def load_financial_data(path: Path, date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    """
    Load a financial index CSV file.
    Expected columns: Date (YYYY-MM-DD) and a price column (e.g. Close, Price, BDI).
    Returns a DataFrame with a DatetimeIndex and the price_col column.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {path}. Columns: {list(df.columns)}")
    if price_col not in df.columns:
        available = [c for c in df.columns if c != date_col]
        raise ValueError(
            f"Column '{price_col}' not found in {path}. "
            f"Available columns: {available}. Use --price-col."
        )
    df = df[[date_col, price_col]].dropna().sort_values(date_col)
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)
    log.info(
        "Financial data: %d trading days | %s -> %s | col='%s'",
        len(df), df.index.min().date(), df.index.max().date(), price_col,
    )
    return df


def select_financial_data(
    location: str,
    wti_path: Optional[Path],
    sp500_path: Optional[Path],
    date_col: str = "Date",
    price_col: str = "Close",
) -> tuple[pd.DataFrame, dict]:
    """
    Select and load the target financial asset for the studied port.

    Houston → WTI crude oil (oil port, ~70% energy traffic)
    LA      → S&P 500       (US consumer-goods gateway)

    Parameters
    ----------
    location   : 'houston' or 'la'
    wti_path   : path to the WTI CSV     (required if location=houston)
    sp500_path : path to the S&P 500 CSV (required if location=la)
    date_col   : date column in the CSVs
    price_col  : price column in the CSVs (typically 'Close' for Stooq)

    Returns
    -------
    (financial_df, asset_info_dict)
      asset_info_dict : LOCATION_ASSET_MAP entry (asset_name, asset_short, consequence)
    """
    asset_info = LOCATION_ASSET_MAP.get(location)
    if asset_info is None:
        raise ValueError(
            f"Location '{location}' unknown in LOCATION_ASSET_MAP. "
            f"Valeurs acceptees : {list(LOCATION_ASSET_MAP.keys())}"
        )

    if location == "houston":
        if wti_path is None:
            raise ValueError(
                "location=houston requires --wti <path/WTI.csv>. "
                "Free source: stooq.com/q/d/l/?s=cl.f&i=d"
            )
        if not wti_path.exists():
            raise FileNotFoundError(f"WTI file not found: {wti_path}")
        log.info("Houston → target financial asset: %s (%s)", asset_info["asset_name"], wti_path)
        df = load_financial_data(wti_path, date_col, price_col)

    else:  # la
        if sp500_path is None:
            raise ValueError(
                "location=la requires --sp500 <path/SP500.csv>. "
                "Free source: stooq.com/q/d/l/?s=%%5Espx&i=d"
            )
        if not sp500_path.exists():
            raise FileNotFoundError(f"S&P 500 file not found: {sp500_path}")
        log.info("LA → target financial asset: %s (%s)", asset_info["asset_name"], sp500_path)
        df = load_financial_data(sp500_path, date_col, price_col)

    return df, asset_info


def align_with_financials(
    events_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    price_col: str = "Close",
    n_days_list: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Step 2 — Merge the episodes with the financial index and compute
    N-day forward returns.

    Market closures (weekends, holidays) are handled with a forward-fill
    on a continuous calendar → an episode date always falls on a valid price.
    Works for any frequency: daily (BDI), weekly (FBX/WCI), monthly (GSCSI).

    N-day return = (P[D+N] - P[D]) / |P[D]| × 100
    where D = episode start_date.
    |P[D]| in the denominator handles indices that can be negative (e.g. GSCSI).

    Parameters
    ----------
    events_df    : output of build_events_dataset
    financial_df : output of load_financial_data (DatetimeIndex)
    price_col    : name of the price column in financial_df
    n_days_list  : forward horizons in calendar days
                   (default: [5, 10, 15, 30] for daily data;
                    use [30, 60, 90] for monthly data)

    Returns
    -------
    DataFrame enriched with:
        index_at_event : index value on the episode day
        return_{N}d    : % change of the index at D+N (one column per horizon)
    """
    if n_days_list is None:
        n_days_list = DEFAULT_N_DAYS

    # Continuous calendar + forward-fill for days without a quote
    full_idx = pd.date_range(financial_df.index.min(), financial_df.index.max(), freq="D")
    fin = financial_df[[price_col]].reindex(full_idx).ffill()

    result = events_df.copy()
    result["event_dt"] = pd.to_datetime(result["start_date"])

    index_at_event = []
    fwd_returns = {n: [] for n in n_days_list}

    for _, row in result.iterrows():
        t0 = row["event_dt"]
        p0 = float(fin.loc[t0, price_col]) if t0 in fin.index else np.nan
        index_at_event.append(p0)

        for n in n_days_list:
            t_n = t0 + pd.Timedelta(days=n)
            denom = abs(p0) if not np.isnan(p0) else 0.0
            if t_n in fin.index and denom > 1e-10:
                p_n = float(fin.loc[t_n, price_col])
                fwd_returns[n].append((p_n - p0) / denom * 100.0)
            else:
                fwd_returns[n].append(np.nan)

    result["index_at_event"] = index_at_event
    for n in n_days_list:
        result[f"return_{n}d"] = fwd_returns[n]

    result = result.drop(columns=["event_dt"])

    valid = result["index_at_event"].notna().sum()
    log.info(
        "Financial alignment (%s): %d episodes | %d with an available value | horizons: %s d",
        price_col, len(result), valid, n_days_list,
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Correlations + XGBoost
# ════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(
    dataset: pd.DataFrame,
    features: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Step 3a — Pearson and Spearman correlations between physical metrics
    [gravity_score, ttc_days] and financial returns [return_Nd].

    Parameters
    ----------
    dataset  : output of align_with_financials
    features : explanatory variables (default: ['gravity_score', 'ttc_days'])
    targets  : target variables (default: all 'return_*' columns)

    Returns
    -------
    DataFrame: feature × target with pearson_r, pearson_p, spearman_r, spearman_p, n_obs
    """
    if features is None:
        features = ["gravity_score", "ttc_days"]
    if targets is None:
        targets = sorted(c for c in dataset.columns if c.startswith("return_"))

    rows = []
    for feat in features:
        for tgt in targets:
            sub = dataset[[feat, tgt]].dropna()
            n = len(sub)
            if n < 4:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r,  pearson_p  = stats.pearsonr(sub[feat], sub[tgt])
                spearman_r, spearman_p = stats.spearmanr(sub[feat], sub[tgt])
            rows.append({
                "feature":    feat,
                "target":     tgt,
                "pearson_r":  round(float(pearson_r),  4) if not np.isnan(pearson_r)  else np.nan,
                "pearson_p":  round(float(pearson_p),  4) if not np.isnan(pearson_p)  else np.nan,
                "spearman_r": round(float(spearman_r), 4) if not np.isnan(spearman_r) else np.nan,
                "spearman_p": round(float(spearman_p), 4) if not np.isnan(spearman_p) else np.nan,
                "n_obs":      n,
            })

    return pd.DataFrame(rows)


def train_xgboost(
    dataset: pd.DataFrame,
    target_col: str,
    features: Optional[list[str]] = None,
    n_splits: int = 3,
) -> tuple:
    """
    Step 3b — XGBRegressor with chronological validation (TimeSeriesSplit).

    Strictly chronological train/test split to avoid any data
    leakage: future episodes never contaminate training.

    Parameters
    ----------
    dataset    : output of align_with_financials
    target_col : target column (e.g. 'return_10d')
    features   : X features (default: ['gravity_score', 'ttc_days'])
    n_splits   : number of chronological folds

    Returns
    -------
    (xgb_model_final, metrics_dict, feature_importance_df)
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        raise ImportError("Required: pip install xgboost scikit-learn")

    if features is None:
        features = ["gravity_score", "ttc_days"]

    # Automatically drop all-NaN features (e.g. ttc_days if never reached)
    available = [f for f in features if f in dataset.columns and dataset[f].notna().any()]
    dropped = [f for f in features if f not in available]
    if dropped:
        log.warning("Features ignored (all NaN): %s", dropped)
    features = available
    if not features:
        raise ValueError("No valid feature — all columns are NaN.")

    sort_col = "start_date" if "start_date" in dataset.columns else None
    sub = dataset[features + [target_col]].dropna()
    if sort_col:
        sub = dataset[features + [target_col, sort_col]].dropna().sort_values(sort_col)
        sub = sub.drop(columns=[sort_col])

    X = sub[features].values
    y = sub[target_col].values

    min_samples = 4
    if len(X) < min_samples:
        raise ValueError(
            f"Not enough data for XGBoost ({len(X)} rows after dropna — "
            f"minimum {min_samples}). Widen the time window or "
            f"lower gravity_threshold."
        )

    n_splits = min(n_splits, len(X) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info(
        "=== STEP 3b: XGBoost | target=%s | features=%s | %d obs | %d folds",
        target_col, features, len(X), n_splits,
    )

    rmse_scores, mae_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        m = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])

        rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        mae  = float(mean_absolute_error(y[test_idx], y_pred))
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        log.info(
            "  Fold %d/%d: train=%d test=%d | RMSE=%.3f%% MAE=%.3f%%",
            fold + 1, n_splits, len(train_idx), len(test_idx), rmse, mae,
        )

    # Final model trained on all available data
    final_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    metrics = {
        "target":    target_col,
        "features":  features,
        "n_obs":     len(X),
        "n_splits":  n_splits,
        "rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "rmse_std":  round(float(np.std(rmse_scores)),  4),
        "mae_mean":  round(float(np.mean(mae_scores)),  4),
        "mae_std":   round(float(np.std(mae_scores)),   4),
    }

    feat_importance = pd.DataFrame({
        "feature":    features,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    log.info(
        "XGBoost final: RMSE=%.3f±%.3f%% | MAE=%.3f±%.3f%%",
        metrics["rmse_mean"], metrics["rmse_std"],
        metrics["mae_mean"],  metrics["mae_std"],
    )
    return final_model, metrics, feat_importance


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Market signal generator
# ════════════════════════════════════════════════════════════════════════════

def generate_trading_signal(
    episode: Episode,
    pinn_model: LWRPINN,
    pinn_global_meta: dict,
    xgb_model,
    current_index_price: float,
    n_days: int = 10,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
) -> dict:
    """
    Step 4 — Generate a structured market alert for a live episode.

    Full chain:
        Episode → PINN → TTC → XGBoost → expected change → directional signal

    Parameters
    ----------
    episode             : episode detected in real time
    pinn_model          : pre-trained LWRPINN
    pinn_global_meta    : PINN checkpoint metadata
    xgb_model           : XGBRegressor trained with train_xgboost()
    current_index_price : current index price (e.g. BDI = 1250)
    n_days              : prediction horizon in days
    rho_threshold_frac  : must match the one used for training

    Returns
    -------
    dict: episode_id, gravity_score, ttc_days, predicted_return, signal, alert_text
    """
    # --- TTC via PINN (fallback = observed episode duration) ---
    ttc, ttc_source = float(episode.duration_days), "duration_fallback"
    if episode.n_points > 0:
        episode_data = pd.DataFrame({
            "x_km":   episode.x_norm * episode.channel_len_km,
            "t_days": episode.t_norm * episode.duration_days,
        })
        t_max = float(episode_data["t_days"].max()) * 3.0 + 2.0
        try:
            pinn_model.eval()
            profiles = extract_physical_profiles(
                model=pinn_model,
                episode_data=episode_data,
                global_meta=pinn_global_meta,
                t_max_days=t_max,
            )
            ttc, ttc_source = _compute_ttc(
                profiles, rho_threshold_frac,
                fallback_days=float(episode.duration_days),
            )
        except Exception as exc:
            log.warning("Signal: TTC error (%s) — fallback duration", exc)

    gravity = float(episode.trigger_score)

    # --- XGBoost prediction ---
    x_in = np.array([[gravity, ttc if ttc is not None else np.nan]])
    try:
        predicted_return = float(xgb_model.predict(x_in)[0])
    except Exception as exc:
        log.warning("Signal: XGBoost error (%s)", exc)
        predicted_return = np.nan

    # --- Signal classification ---
    if np.isnan(predicted_return):
        signal, direction = "NEUTRAL", "="
    elif predicted_return > 1.0:
        signal, direction = "BULLISH", "+"
    elif predicted_return < -1.0:
        signal, direction = "BEARISH", "-"
    else:
        signal, direction = "NEUTRAL", "="

    ttc_str  = f"{ttc:.1f}" if ttc is not None else "N/A"
    ret_str  = f"{predicted_return:+.2f}%" if not np.isnan(predicted_return) else "N/A"
    src_label = {"pinn": "PINN", "duration_fallback": "observed", "unavailable": "N/A"}.get(
        ttc_source, ttc_source
    )

    asset_info  = LOCATION_ASSET_MAP.get(episode.location, {})
    asset_name  = asset_info.get("asset_name",  "the freight index")
    asset_short = asset_info.get("asset_short", "IDX")
    consequence = asset_info.get("consequence", "reflecting pressure on freight prices.")

    alert_text = (
        f"\n{'='*68}\n"
        f"  [{direction}] MARITIME NOWCASTING ALERT — {episode.location.upper()}\n"
        f"{'='*68}\n"
        f"  Episode ID      : {episode.episode_id}\n"
        f"  Period          : {episode.start_date} -> {episode.end_date} "
        f"({episode.duration_days} days)\n"
        f"  Gravity Score   : {gravity:>12,.0f}\n"
        f"  TTC [{src_label:<13s}]: {ttc_str} days\n"
        f"  {asset_short:<14s}: {current_index_price:>+12.4f}\n"
        f"  Signal          : {signal} [{direction}]\n"
        f"  {'─'*60}\n"
        f"  NOWCASTING ALERT: the detected event (ID: {episode.episode_id})\n"
        f"  has a gravity score of {gravity:,.0f}. The physical PINN model\n"
        f"  estimates it will take {ttc_str} days to clear [{src_label}].\n"
        f"  Consequence: the XGBoost model expects a change of\n"
        f"  {ret_str} in {asset_name} prices,\n"
        f"  {consequence}\n"
        f"{'='*68}\n"
    )
    print(alert_text)

    return {
        "episode_id":       episode.episode_id,
        "location":         episode.location,
        "start_date":       episode.start_date,
        "gravity_score":    gravity,
        "ttc_days":         ttc,
        "ttc_source":       ttc_source,
        "predicted_return": predicted_return,
        "signal":           signal,
        "current_price":    current_index_price,
        "n_days_horizon":   n_days,
        "alert_text":       alert_text,
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Financial pipeline: PINN + XGBoost → market signal"
    )

    # Physical data
    parser.add_argument("--model",    type=Path, default=None,
                        help="PINN checkpoint (.pt) — not required in --gravity-only mode")
    parser.add_argument("--location", default="houston", choices=["houston", "la"])
    parser.add_argument("--start",    help="Start of the episode window (YYYY-MM-DD)")
    parser.add_argument("--end",      help="End of the episode window (YYYY-MM-DD)")
    parser.add_argument("--gravity-threshold", type=float, default=None,
                        help="Episode threshold (default: LOCATION_DEFAULTS)")
    parser.add_argument("--gravity-path", type=Path, default=None)
    parser.add_argument("--rho-threshold-frac", type=float, default=DEFAULT_RHO_THRESHOLD_FRAC,
                        help="Fraction of the peak defining the TTC (default: 0.3)")
    parser.add_argument("--gravity-only", action="store_true",
                        help="Fast mode: episodes from the gravity parquet only, "
                             "without loading AIS or running the PINN. "
                             "TTC = observed duration. Ideal for long periods (2021-2024).")
    parser.add_argument("--gap-tolerance", type=int, default=1,
                        help="Below-threshold days tolerated inside an episode "
                             "(default: 1 — handles weekends)")

    # Financial data — automatic routing per port
    parser.add_argument("--wti",   type=Path, default=None,
                        help="WTI crude oil CSV (required if location=houston). "
                             "Source: stooq.com/q/d/l/?s=cl.f&i=d")
    parser.add_argument("--sp500", type=Path, default=None,
                        help="S&P 500 CSV (required if location=la). "
                             "Source: stooq.com/q/d/l/?s=%%5Espx&i=d")
    parser.add_argument("--financial", type=Path, default=None,
                        help="Manual override: CSV of any index "
                             "(BDI, FBX, GSCSI…). Takes precedence over --wti/--sp500.")
    parser.add_argument("--date-col",  default="Date",  help="Date column in the CSV (default: Date)")
    parser.add_argument("--price-col", default="Close",
                        help="Price column in the CSV (default: Close)")
    parser.add_argument("--n-days",    type=int, nargs="+", default=DEFAULT_N_DAYS,
                        help="Forward horizons in days (default: 5 10 15 30)")

    # XGBoost
    parser.add_argument("--target", default="return_10d",
                        help="XGBoost target column (default: return_10d)")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="TimeSeriesSplit folds (default: 3)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output folder (CSV, XGBoost model)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the XGBoost model (.json)")

    args = parser.parse_args()

    # Force UTF-8 on the Windows terminal (avoids UnicodeEncodeError with ─ ══ etc.)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_d = date.fromisoformat(args.start) if args.start else None
    end_d   = date.fromisoformat(args.end)   if args.end   else None

    # ── STEP 1 — Episode extraction ───────────────────────────────────────
    if args.gravity_only:
        # Fast mode: gravity parquet only, no PINN or AIS
        log.info("--gravity-only mode: no PINN, TTC = observed duration")
        try:
            events_df = build_events_from_gravity(
                location=args.location,
                gravity_threshold=args.gravity_threshold,
                start_date=start_d,
                end_date=end_d,
                gravity_path=args.gravity_path,
                gap_tolerance_days=args.gap_tolerance,
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
        pinn_model = None
    else:
        # Mode complet : PINN + AIS
        if args.model is None:
            log.error("--model required in PINN mode (or use --gravity-only)")
            sys.exit(1)
        if not args.model.exists():
            log.error("Checkpoint not found: %s", args.model)
            sys.exit(1)
        pinn_model, global_meta = _load_checkpoint(args.model)
        events_df = build_events_dataset(
            model=pinn_model,
            global_meta=global_meta,
            location=args.location,
            gravity_threshold=args.gravity_threshold,
            start_date=start_d,
            end_date=end_d,
            gravity_path=args.gravity_path,
            rho_threshold_frac=args.rho_threshold_frac,
        )

    if events_df.empty:
        log.error("No episode detected — pipeline stopped.")
        sys.exit(1)

    events_path = args.output_dir / f"{args.location}_events.csv"
    events_df.to_csv(events_path, index=False)
    log.info("Events dataset saved -> %s", events_path)

    # ── STEP 2 — Load the target financial asset ──────────────────────────
    if args.financial is not None:
        # Manual override — bypasses per-port routing
        if not args.financial.exists():
            log.error("Financial file not found: %s", args.financial)
            sys.exit(1)
        financial_df = load_financial_data(args.financial, args.date_col, args.price_col)
        asset_info = LOCATION_ASSET_MAP.get(args.location, {
            "asset_name": args.price_col, "asset_short": args.price_col, "consequence": "",
        })
        log.info(
            "Manual override — index: %s | col: %s",
            args.financial.name, args.price_col,
        )
    else:
        try:
            financial_df, asset_info = select_financial_data(
                args.location, args.wti, args.sp500,
                args.date_col, args.price_col,
            )
        except (ValueError, FileNotFoundError) as exc:
            log.error("%s", exc)
            sys.exit(1)

    log.info(
        "=== STEP 2: Alignment | port=%s | asset=%s ===",
        args.location, asset_info.get("asset_name", args.price_col),
    )
    dataset = align_with_financials(
        events_df=events_df,
        financial_df=financial_df,
        price_col=args.price_col,
        n_days_list=args.n_days,
    )

    dataset_path = args.output_dir / f"{args.location}_dataset_aligned.csv"
    dataset.to_csv(dataset_path, index=False)
    log.info("Aligned dataset saved -> %s", dataset_path)

    # ── STEP 3a — Correlations ─────────────────────────────────────────────
    log.info("=== STEP 3a: Correlation analysis ===")
    corr_df = run_correlation_analysis(dataset)

    print("\n── Pearson / Spearman correlations ──")
    print(corr_df.to_string(index=False))

    corr_path = args.output_dir / f"{args.location}_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info("Correlations saved -> %s", corr_path)

    # ── STEP 3b — XGBoost ─────────────────────────────────────────────────
    if args.target not in dataset.columns:
        available_targets = [c for c in dataset.columns if c.startswith("return_")]
        log.error(
            "Target '%s' missing. Available: %s. "
            "Check --target or --n-days.",
            args.target, available_targets,
        )
        sys.exit(1)

    try:
        xgb_model, metrics, feat_imp = train_xgboost(
            dataset=dataset,
            target_col=args.target,
            n_splits=args.n_splits,
        )

        print(f"\n── XGBoost ({args.target}) ──")
        print(f"  Observations : {metrics['n_obs']}")
        print(f"  RMSE (CV)    : {metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} %")
        print(f"  MAE  (CV)    : {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} %")
        print(f"\n  Feature importance:")
        for _, row in feat_imp.iterrows():
            print(f"    {row['feature']:20s} {row['importance']:.4f}")

        metrics_path = args.output_dir / f"{args.location}_xgb_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        if args.save_model:
            try:
                xgb_path = args.output_dir / f"{args.location}_xgb_{args.target}.json"
                xgb_model.save_model(str(xgb_path))
                log.info("XGBoost model saved -> %s", xgb_path)
            except Exception as exc:
                log.warning("Could not save the XGBoost model: %s", exc)

        # ── PER-EPISODE TABLE ──────────────────────────────────────────────
        asset_info  = asset_info if "asset_info" in dir() else {}
        asset_short = asset_info.get("asset_short", args.price_col)
        features_used = metrics["features"]

        # Target horizon (e.g. return_10d → 10)
        n_target = int(args.target.replace("return_", "").replace("d", ""))

        ep_rows = []
        for _, row in dataset.sort_values("start_date").iterrows():
            x_ep = np.array([[row.get(f, np.nan) for f in features_used]])
            try:
                pred = float(xgb_model.predict(x_ep)[0])
            except Exception:
                pred = np.nan

            actual   = float(row[args.target]) if pd.notna(row.get(args.target)) else np.nan
            error    = pred - actual if (not np.isnan(pred) and not np.isnan(actual)) else np.nan
            ttc_val  = row.get("ttc_days")
            ttc_src  = row.get("ttc_source", "?")

            # Signal based on the prediction
            if np.isnan(pred):
                sig = "="
            elif pred > 1.0:
                sig = "+"
            elif pred < -1.0:
                sig = "-"
            else:
                sig = "="

            # Prediction quality (if the realised value is available)
            if not np.isnan(error):
                ok = "OK" if abs(error) < 1.5 else "KO"
            else:
                ok = "--"

            ep_rows.append({
                "episode_id":           row["episode_id"],
                "start_date":           str(row["start_date"]),
                "dur_d":                int(row["duration_days"]),
                "gravity":              int(row["gravity_score"]),
                "ttc_d":                f"{ttc_val:.0f}" if pd.notna(ttc_val) else "N/A",
                "ttc_src":              "PINN" if ttc_src == "pinn" else "obs",
                f"pred_{asset_short}%": f"{pred:+.2f}%" if not np.isnan(pred) else "N/A",
                f"actual_{asset_short}%": f"{actual:+.2f}%" if not np.isnan(actual) else "N/A",
                "error_%":             f"{error:+.2f}%" if not np.isnan(error) else "N/A",
                "sig":                  sig,
                "ok":                   ok,
            })

        ep_df = pd.DataFrame(ep_rows)

        # Global backtest metrics
        errs = [float(r["error_%"].replace("%","")) for r in ep_rows if r["error_%"] != "N/A"]
        mae_bt  = float(np.mean(np.abs(errs))) if errs else np.nan
        rmse_bt = float(np.sqrt(np.mean(np.array(errs)**2))) if errs else np.nan
        n_ok    = sum(1 for r in ep_rows if r["ok"] == "OK")

        print(f"\n{'='*80}")
        print(f"  NOWCASTING PER EPISODE — {args.location.upper()} | target: {asset_short} | horizon: {n_target}d")
        print(f"  ttc_d : estimated clearing time (obs=observed duration, PINN=physical model)")
        print(f"  pred  : change expected by XGBoost | actual : realised change")
        print(f"  ok    : |error| < 1.5% (indicative threshold)")
        print(f"{'='*80}")
        print(ep_df.to_string(index=False))
        print(f"{'─'*80}")
        if errs:
            print(f"  Backtest : MAE={mae_bt:.2f}%  RMSE={rmse_bt:.2f}%  "
                  f"Pred OK={n_ok}/{len(ep_rows)} episodes")
        print(f"{'='*80}")

        ep_path = args.output_dir / f"{args.location}_episodes_predictions.csv"
        ep_df.to_csv(ep_path, index=False)
        log.info("Per-episode predictions -> %s", ep_path)

    except ValueError as exc:
        log.warning("XGBoost ignore: %s", exc)

    print(f"\nResults -> {args.output_dir}/")


if __name__ == "__main__":
    main()
