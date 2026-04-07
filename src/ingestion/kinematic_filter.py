"""
Étape 2 (complète) + Étape 3 (gap detection) — Kinematic preprocessing.

Applied in-memory before HDBSCAN clustering. Takes a raw daily DataFrame and adds:
  - SOG_corr         : declared SOG corrected when aberrant (> 50 kt)
                       replaced by haversine-derived speed between consecutive messages
  - speed_computed_kt: raw haversine speed (NaN at first message per MMSI)
  - traj_id          : integer ID incremented at each gap > GAP_THRESHOLD_MIN
                       or at the first message of a new MMSI
                       → vessels with multiple static episodes get separate traj_ids

Usage:
    from src.ingestion.kinematic_filter import prepare_kinematics
    df_prepared = prepare_kinematics(pl.read_parquet(path))
"""
import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

EARTH_RADIUS_NM   = 3440.065  # nautical miles
SOG_ABERRANT_MIN  = 50.0      # kt — declared SOG above this triggers correction
SOG_MAX           = 50.0      # kt — hard cap after correction
GAP_THRESHOLD_MIN = 30        # minutes — gap above this starts a new trajectory


def prepare_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add SOG_corr, speed_computed_kt, and traj_id columns to a daily AIS DataFrame.
    Sorts by (MMSI, BaseDateTime) in place — required for shift operations.

    Returns the enriched DataFrame (original columns preserved).
    """
    df = df.sort(["MMSI", "BaseDateTime"])

    # Previous position and timestamp within each MMSI group
    df = df.with_columns([
        pl.col("LAT").shift(1).over("MMSI").alias("_lat_prev"),
        pl.col("LON").shift(1).over("MMSI").alias("_lon_prev"),
        pl.col("BaseDateTime").shift(1).over("MMSI").alias("_dt_prev"),
    ])

    # Delta time in seconds — null at first message per MMSI
    dt_sec_series = (df["BaseDateTime"] - df["_dt_prev"]).dt.total_seconds()
    null_mask     = dt_sec_series.is_null().to_numpy()
    dt_sec        = np.where(null_mask, np.nan,
                             dt_sec_series.fill_null(0).cast(pl.Float64).to_numpy())

    # Previous coordinates — NaN where null (first message per MMSI)
    lat1 = _series_to_float_numpy(df["_lat_prev"])
    lon1 = _series_to_float_numpy(df["_lon_prev"])
    lat2 = df["LAT"].to_numpy().astype(float)
    lon2 = df["LON"].to_numpy().astype(float)

    # Haversine speed in knots (NaN at first message per MMSI)
    dist_nm = _haversine_nm(lat1, lon1, lat2, lon2)
    dt_h    = np.where(dt_sec > 0, dt_sec / 3600.0, np.nan)
    speed   = np.where(~np.isnan(dt_h), dist_nm / dt_h, np.nan)

    # SOG correction: replace aberrant declared SOG with computed speed
    sog_declared = df["SOG"].to_numpy().astype(float)
    sog_corr = np.where(
        (sog_declared > SOG_ABERRANT_MIN) & (~np.isnan(speed)) & (speed <= SOG_MAX),
        speed,
        sog_declared,
    )
    sog_corr = np.minimum(sog_corr, SOG_MAX)

    # Trajectory IDs: increment at first message per MMSI or after a gap
    gap_mask = np.isnan(dt_sec) | (dt_sec > GAP_THRESHOLD_MIN * 60)
    traj_id  = np.cumsum(gap_mask.astype(np.int32))

    n_corrected = int(((sog_declared > SOG_ABERRANT_MIN) & ~np.isnan(speed)).sum())
    n_trajs     = int(gap_mask.sum())
    log.debug("SOG corrected: %d messages | trajectories: %d", n_corrected, n_trajs)

    return (
        df.drop(["_lat_prev", "_lon_prev", "_dt_prev"])
        .with_columns([
            pl.Series("SOG_corr",          sog_corr, dtype=pl.Float32),
            pl.Series("speed_computed_kt", speed,    dtype=pl.Float32),
            pl.Series("traj_id",           traj_id,  dtype=pl.Int32),
        ])
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _series_to_float_numpy(series: pl.Series) -> np.ndarray:
    """Convert a nullable Polars float series to numpy, replacing nulls with NaN."""
    null_mask = series.is_null().to_numpy()
    arr = series.fill_null(0.0).cast(pl.Float64).to_numpy().copy()
    arr[null_mask] = np.nan
    return arr


def _haversine_nm(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in nautical miles. Propagates NaN for null inputs."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat   = lat2_r - lat1_r
    dlon   = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
