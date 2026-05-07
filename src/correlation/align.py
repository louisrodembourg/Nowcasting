"""
Phase 4 — Temporal alignment: SCFI weekly → daily gravity score.

SCFI is published every Friday. The forward-fill strategy holds the Friday
value until the next publication — economically correct (index is known from
Friday onwards).

Usage:
    python src/correlation/align.py
"""
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

GRAVITY_PATH = Path("data/features/la_2019_gravity_score.parquet")
SCFI_PATH    = Path("data/financial/scfi_2019.parquet")


def align_gravity_scfi(
    gravity_path: Path | None = None,
    scfi_path: Path | None = None,
    method: str = "forward_fill",
) -> pl.DataFrame:
    """
    Merge daily gravity score with weekly SCFI via forward-fill (default)
    or linear interpolation.

    Returns a daily DataFrame with columns:
        date, gravity_score, deviation_score, blocked_capacity,
        utilization_rate_rho, is_characteristic,
        scfi, scfi_7d_delta, scfi_pct_change
    """
    g_path = Path(gravity_path) if gravity_path is not None else GRAVITY_PATH
    s_path = Path(scfi_path)    if scfi_path    is not None else SCFI_PATH

    df_gravity = pl.read_parquet(g_path).sort("date")
    df_scfi    = pl.read_parquet(s_path).sort("date")

    log.info("Gravity: %d rows  |  SCFI: %d weekly rows", len(df_gravity), len(df_scfi))

    # Build full daily spine for the gravity year
    year_start = df_gravity["date"].min()
    year_end   = df_gravity["date"].max()
    spine = pl.DataFrame({"date": pl.date_range(year_start, year_end, interval="1d", eager=True)})

    # Join SCFI onto daily spine, then fill
    df_daily = spine.join(df_scfi, on="date", how="left")

    if method == "forward_fill":
        # First backward-fill the leading nulls (days before first Friday)
        df_daily = df_daily.with_columns(
            pl.col("scfi").fill_null(strategy="backward")
        )
        # Then forward-fill the rest
        df_daily = df_daily.with_columns(
            pl.col("scfi").fill_null(strategy="forward")
        )
    elif method == "linear_interpolate":
        df_daily = df_daily.with_columns(
            pl.col("scfi").fill_null(strategy="backward")
        ).with_columns(
            pl.col("scfi").interpolate()
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'forward_fill' or 'linear_interpolate'.")

    # Derived columns
    scfi_vals = df_daily["scfi"].to_numpy().astype(float)
    scfi_7d_delta = np.concatenate([np.full(7, np.nan), scfi_vals[7:] - scfi_vals[:-7]])
    scfi_pct      = np.where(scfi_vals[:-1] != 0,
                             (scfi_vals[1:] - scfi_vals[:-1]) / scfi_vals[:-1] * 100,
                             0.0)
    scfi_pct = np.concatenate([[np.nan], scfi_pct])

    df_daily = df_daily.with_columns([
        pl.Series("scfi_7d_delta",  scfi_7d_delta.tolist(), dtype=pl.Float64),
        pl.Series("scfi_pct_change", scfi_pct.tolist(),     dtype=pl.Float64),
    ])

    # Join gravity features
    gravity_cols = [
        "date", "gravity_score", "deviation_score", "blocked_capacity",
        "utilization_rate_rho", "is_characteristic",
    ] + [c for c in df_gravity.columns if c.startswith("phi_")]
    df_out = df_daily.join(df_gravity.select(gravity_cols), on="date", how="left")

    # Drop rows with nulls in key columns
    n_before = len(df_out)
    df_out = df_out.drop_nulls(subset=["gravity_score", "scfi"])
    n_dropped = n_before - len(df_out)
    if n_dropped:
        log.warning("Dropped %d rows with nulls in gravity_score or scfi", n_dropped)

    log.info("Aligned DataFrame: %d rows  |  date range %s → %s",
             len(df_out), df_out["date"].min(), df_out["date"].max())
    return df_out.sort("date")


def compute_scfi_delta_at_lag(
    df: pl.DataFrame,
    lag_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (gravity[:-lag], scfi[lag:] - scfi[:-lag]) for cross-correlation
    at the given forward lag.
    """
    gravity = df["gravity_score"].to_numpy().astype(float)
    scfi    = df["scfi"].to_numpy().astype(float)
    if lag_days == 0:
        return gravity, np.zeros_like(gravity)
    return gravity[:-lag_days], scfi[lag_days:] - scfi[:-lag_days]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    df = align_gravity_scfi()
    print(df.select(["date", "gravity_score", "scfi", "scfi_7d_delta"]))
