"""
Étape 4: Daily HDBSCAN clustering of stationary vessels — Houston Ship Channel.

Pipeline for one day's Parquet:
  1. Kinematic preprocessing (SOG_corr, traj_id) via kinematic_filter
  2. Filter stationary vessels (SOG_corr < SOG_STATIC_THRESHOLD)
  3. Deduplicate to one position per (MMSI, traj_id) — each static episode separately
  4. Run HDBSCAN on (LAT, LON) using haversine metric
  5. Classify each cluster: 'docked' (aligned caps) vs 'waiting' (scattered caps)

Returns (cluster_df, prepared_df):
  - cluster_df  : one row per static episode (MMSI + traj_id) + cluster metadata
  - prepared_df : full day DataFrame with SOG_corr and traj_id columns added

Usage (standalone):
    python src/clustering/hdbscan_daily.py data/parquet/houston/houston_2017_07_01.parquet
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdbscan
import numpy as np
import polars as pl

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

SOG_STATIC_THRESHOLD     = 1.0   # knots — below this = stationary
HEADING_DOCKED_MAX_STD   = 25.0  # degrees — below this std = docked
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES      = 2


def cluster_day_from_df(
    prepared: pl.DataFrame,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Run HDBSCAN on an already-prepared DataFrame (output of prepare_kinematics).
    Returns (cluster_df, prepared_df) — same contract as cluster_day.
    """
    return _run_hdbscan(prepared, label="<DataFrame>")


def cluster_day(
    parquet_path: Path,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Run kinematic preprocessing + HDBSCAN on one daily Parquet file.

    Returns (cluster_df, prepared_df).
    Both are None if the file is missing or has too few static vessels.

    cluster_df columns:
        MMSI, traj_id, LAT, LON, Heading_mean, Heading_std,
        Draft, Length, Width, VesselType, nb_messages,
        cluster_label, membership_score, cluster_type
    """
    if not parquet_path.exists():
        log.warning("File not found: %s", parquet_path)
        return None, None

    raw_df     = pl.read_parquet(parquet_path)
    prepared   = prepare_kinematics(raw_df)

    return _run_hdbscan(prepared, label=parquet_path.name)


def _run_hdbscan(
    prepared: pl.DataFrame,
    label: str = "",
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Core HDBSCAN logic on a prepared DataFrame."""
    # --- Filter stationary vessels (using corrected SOG) --------------------
    static = prepared.filter(pl.col("SOG_corr") < SOG_STATIC_THRESHOLD)

    if len(static) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning(
            "%s: only %d static messages — skipping clustering",
            label, len(static),
        )
        return None, prepared

    # --- Deduplicate: one position per (MMSI, traj_id) ----------------------
    # Each continuous static episode is treated as a distinct data point.
    # Heading 511 = AIS "not available" — excluded before stats.

    # Build agg expression — only include columns that exist
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("Heading").filter(pl.col("Heading") < 360).mean().alias("Heading_mean")
            if "Heading" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Heading_mean"),
        pl.col("Heading").filter(pl.col("Heading") < 360).std().alias("Heading_std")
            if "Heading" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Heading_std"),
        pl.col("Draft").max().alias("Draft")
            if "Draft" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length")
            if "Length" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width")
            if "Width" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType")
            if "VesselType" in prepared.columns else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),
    ]

    agg = static.group_by(["MMSI", "traj_id"]).agg(agg_exprs)

    # Guard: HDBSCAN BallTree requires at least min_cluster_size points
    if len(agg) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning(
            "%s: only %d static episodes after dedup — skipping HDBSCAN",
            label, len(agg),
        )
        return None, prepared

    # --- HDBSCAN on (LAT, LON) haversine ------------------------------------
    coords_deg = agg.select(["LAT", "LON"]).to_numpy()
    coords_rad = np.radians(coords_deg)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="haversine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_rad)
    scores = clusterer.probabilities_

    agg = agg.with_columns([
        pl.Series("cluster_label",    labels, dtype=pl.Int32),
        pl.Series("membership_score", scores, dtype=pl.Float32),
    ])

    # --- Classify clusters: docked vs waiting --------------------------------
    cluster_types = _classify_clusters(agg)
    agg = agg.join(cluster_types, on="cluster_label", how="left")

    n_episodes = len(agg)
    n_vessels  = agg["MMSI"].n_unique()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info(
        "%s: %d static episodes (%d vessels) → %d clusters, %d noise (%.1f%%)",
        label, n_episodes, n_vessels,
        n_clusters, n_noise, 100 * n_noise / n_episodes,
    )

    return agg, prepared


def _classify_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """
    Label each cluster as 'docked' or 'waiting' based on heading std.
    Low std → vessels aligned with the berth → 'docked'.
    High std → wind/current scatter → 'waiting'.
    Null std (single-message episode) treated as max dispersion → 'waiting'.
    """
    return (
        df.filter(pl.col("cluster_label") >= 0)
        .group_by("cluster_label")
        .agg(
            pl.col("Heading_std")
            .fill_null(180.0)
            .mean()
            .alias("_mean_heading_std")
        )
        .with_columns(
            pl.when(pl.col("_mean_heading_std") < HEADING_DOCKED_MAX_STD)
            .then(pl.lit("docked"))
            .otherwise(pl.lit("waiting"))
            .alias("cluster_type")
        )
        .drop("_mean_heading_std")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDBSCAN clustering on a single daily Parquet — Houston"
    )
    parser.add_argument("parquet", help="Path to daily Parquet file")
    args = parser.parse_args()

    cluster_df, _ = cluster_day(Path(args.parquet))
    if cluster_df is not None:
        print(cluster_df)
        docked  = cluster_df.filter(pl.col("cluster_type") == "docked").height
        waiting = cluster_df.filter(pl.col("cluster_type") == "waiting").height
        noise   = cluster_df.filter(pl.col("cluster_label") == -1).height
        print(f"\ndocked={docked}  waiting={waiting}  noise={noise}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
