"""
Step 4: daily HDBSCAN clustering of stationary vessels.

Pipeline for one daily Parquet file:
  1. Kinematic preprocessing (SOG_corr, traj_id) via kinematic_filter
  2. Keep stationary vessels (SOG_corr < SOG_STATIC_THRESHOLD)
  3. Deduplicate to one position per (MMSI, traj_id) — each static episode separately
  4. Run HDBSCAN on (LAT, LON) with the haversine metric
  5. Classify each cluster: 'docked' (aligned headings) vs 'waiting' (scattered headings)

Returns (cluster_df, prepared_df):
  - cluster_df  : one row per static episode (MMSI + traj_id) + cluster metadata
  - prepared_df : full daily DataFrame with the SOG_corr and traj_id columns added

Usage (standalone):
    python src/clustering/hdbscan_daily.py data/parquet/houston/houston_2017_07_01.parquet
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdbscan
import numpy as np
import polars as pl
from scipy.stats import circmean, circstd

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# Clustering parameters and kinematic thresholds (defaults)
SOG_STATIC_THRESHOLD = 1.0   # knots — below = stationary vessel
HDBSCAN_MIN_CLUSTER_SIZE = 5  # minimum cluster size
HDBSCAN_MIN_SAMPLES = 2       # HDBSCAN min_samples


@dataclass
class ClusteringConfig:
    """
    HDBSCAN clustering parameters.
    Instantiate without arguments to get the default (baseline) behaviour.
    """

    sog_static_threshold: float = SOG_STATIC_THRESHOLD
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES
    # Max radius (nm) for a cluster to be classified as "docked".
    # None = no spatial constraint.
    # 0.3 nm ≈ 550 m — typical size of a quay terminal.
    max_docked_radius_nm: Optional[float] = None
    # Polygons delimiting the areas where the "waiting" status is possible.
    # Each polygon = list of (lat, lon) — implicitly closed.
    # Any cluster OUTSIDE ALL these polygons is forced to "docked".
    waiting_allowed_polygons: Optional[list[list[tuple[float, float]]]] = None


def cluster_day_from_df(
    prepared: pl.DataFrame,
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Run HDBSCAN on an already prepared DataFrame (output of prepare_kinematics).
    Returns (cluster_df, prepared_df) — same contract as cluster_day.
    """
    return _run_hdbscan(
        prepared, label="<DataFrame>", config=config or ClusteringConfig()
    )


def cluster_day(
    parquet_path: Path,
    config: Optional[ClusteringConfig] = None,
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

    raw_df = pl.read_parquet(parquet_path)
    prepared = prepare_kinematics(raw_df)

    return _run_hdbscan(
        prepared, label=parquet_path.name, config=config or ClusteringConfig()
    )


def _run_hdbscan(
    prepared: pl.DataFrame,
    label: str = "",
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Core HDBSCAN clustering logic on a prepared DataFrame."""
    cfg = config or ClusteringConfig()

    # --- Keep stationary vessels (using the corrected SOG) -------------------
    static = prepared.filter(pl.col("SOG_corr") < cfg.sog_static_threshold)

    if len(static) < cfg.hdbscan_min_cluster_size:
        log.warning(
            "%s : only %d static messages — skip clustering",
            label,
            len(static),
        )
        return None, prepared

    # --- Deduplicate: one position per (MMSI, traj_id) ----------------------
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        (
            pl.col("Heading")
            .filter(pl.col("Heading") < 360)
            .implode()
            .alias("_headings")
            if "Heading" in prepared.columns
            else pl.lit(None).alias("_headings")
        ),
        pl.col("Draft").max().alias("Draft")
        if "Draft" in prepared.columns
        else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length")
        if "Length" in prepared.columns
        else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width")
        if "Width" in prepared.columns
        else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType")
        if "VesselType" in prepared.columns
        else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),
    ]

    agg = static.group_by(["MMSI", "traj_id"]).agg(agg_exprs)

    headings_list = agg["_headings"].to_list()
    h_means, h_stds = [], []
    for raw in headings_list:
        h = [x for x in (raw or []) if x is not None]
        h_means.append(float(circmean(h, high=360, low=0)) if len(h) >= 1 else None)
        h_stds.append(float(circstd(h, high=360, low=0)) if len(h) >= 2 else 180.0)

    agg = agg.with_columns(
        [
            pl.Series("Heading_mean", h_means, dtype=pl.Float64),
            pl.Series("Heading_std", h_stds, dtype=pl.Float64),
        ]
    ).drop("_headings")

    if len(agg) < cfg.hdbscan_min_cluster_size:
        log.warning(
            "%s : only %d episodes after deduplication — skip HDBSCAN",
            label,
            len(agg),
        )
        return None, prepared

    # --- HDBSCAN on (LAT, LON) with haversine distance -----------------------
    coords_deg = agg.select(["LAT", "LON"]).to_numpy()
    coords_rad = np.radians(coords_deg)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="haversine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_rad)
    scores = clusterer.probabilities_

    agg = agg.with_columns(
        [
            pl.Series("cluster_label", labels, dtype=pl.Int32),
            pl.Series("membership_score", scores, dtype=pl.Float32),
        ]
    )

    # --- Classify clusters: docked vs waiting -------------------------------
    cluster_types = _classify_clusters(agg, cfg)
    agg = agg.join(cluster_types, on="cluster_label", how="left")

    n_episodes = len(agg)
    n_vessels = agg["MMSI"].n_unique()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(
        "%s : %d episodes (%d vessels) → %d clusters, %d noise (%.1f%%)",
        label,
        n_episodes,
        n_vessels,
        n_clusters,
        n_noise,
        100 * n_noise / n_episodes,
    )

    return agg, prepared


def _compute_cluster_radius_nm(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Cluster radius = max haversine distance from the centroid to the farthest point (nm).
    Returns 0 if the cluster has a single point.
    """
    if len(lats) <= 1:
        return 0.0
    lat_c = np.radians(np.mean(lats))
    lon_c = np.radians(np.mean(lons))
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat_c
    dlon = lon2 - lon_c
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_c) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(3440.065 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))).max())


def _point_in_polygon(
    lat: float, lon: float, polygon: list[tuple[float, float]]
) -> bool:
    """
    Ray-casting algorithm: is the point (lat, lon) inside the polygon?
    polygon = list of (lat, lon) — implicitly closed.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        # Horizontal ray to the right from (lat, lon)
        # Does the edge (lat_i,lon_i)-(lat_j,lon_j) cross this ray?
        edge_crosses_y = (lat_i > lat) != (lat_j > lat)
        if edge_crosses_y:
            # x-intersection of the ray with the edge
            x_intersect = lon_i + (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i)
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _load_geojson_polygons(path: Path) -> list[list[tuple[float, float]]]:
    """
    Load the polygons of a GeoJSON file (Polygon or MultiPolygon).
    Returns a list of polygons (each = list of (lat, lon)).
    """
    with open(path) as f:
        fc = json.load(f)
    zones = []
    for feat in fc["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            # coordinates[0] = outer ring; holes are ignored
            zones.append([(c[1], c[0]) for c in geom["coordinates"][0]])
        elif geom["type"] == "MultiPolygon":
            for polygon in geom["coordinates"]:
                zones.append([(c[1], c[0]) for c in polygon[0]])
    return zones


def load_waiting_zones(location: str) -> list[list[tuple[float, float]]]:
    """
    Load the waiting-zone polygons from data/features/{location}_waiting_zones.geojson.
    Returns a list of polygons (each = list of (lat, lon)).
    Raises FileNotFoundError if the file does not exist.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "features"
        / f"{location}_waiting_zones.geojson"
    )
    return _load_geojson_polygons(path)


def _classify_clusters(df: pl.DataFrame, config: ClusteringConfig) -> pl.DataFrame:
    """
    Label each cluster as 'docked' or 'waiting'.

    Classification rules (applied in order):
      0. If config.waiting_allowed_polygons is set:
           cluster INSIDE at least one polygon → forced "waiting"
           cluster OUTSIDE all polygons        → forced "docked"
      1. Otherwise, if config.max_docked_radius_nm is set,
           compact cluster → "docked", else "waiting"
      2. Without an explicit radius constraint the behaviour stays conservative:
           "docked" by default.
    """
    valid = df.filter(pl.col("cluster_label") >= 0)

    rows: list[dict] = []
    for label in valid["cluster_label"].unique().to_list():
        cluster = valid.filter(pl.col("cluster_label") == label)

        # Rule 0: waiting_allowed_polygons — waiting if at least one point of the
        # cluster falls inside a waiting zone.
        if config.waiting_allowed_polygons is not None:
            any_inside = False
            for lat_i, lon_i in zip(cluster["LAT"].to_numpy(), cluster["LON"].to_numpy()):
                if any(
                    _point_in_polygon(lat_i, lon_i, poly)
                    for poly in config.waiting_allowed_polygons
                ):
                    any_inside = True
                    break
            is_docked = not any_inside
        else:
            # Without a zonal filter, a cluster is only "docked" if it
            # is compact enough according to max_docked_radius_nm.
            if config.max_docked_radius_nm is None:
                is_docked = True
            else:
                lats = cluster["LAT"].to_numpy()
                lons = cluster["LON"].to_numpy()
                radius = _compute_cluster_radius_nm(lats, lons)
                is_docked = radius <= config.max_docked_radius_nm

        rows.append(
            {
                "cluster_label": label,
                "cluster_type": "docked" if is_docked else "waiting",
            }
        )

    if not rows:
        return pl.DataFrame(schema={"cluster_label": pl.Int32, "cluster_type": pl.Utf8})

    return pl.DataFrame(rows).with_columns(pl.col("cluster_label").cast(pl.Int32))


def main() -> None:
    """Entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="HDBSCAN clustering on a daily Parquet file"
    )
    parser.add_argument("parquet", help="Path to the daily Parquet file")
    parser.add_argument("--preview", type=int, default=10, metavar="N",
                        help="Number of rows to display (default: 10)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save cluster_df as CSV next to the input parquet")
    args = parser.parse_args()

    cluster_df, _ = cluster_day(Path(args.parquet))
    if cluster_df is None:
        print("No cluster found.")
        return

    docked = cluster_df.filter(pl.col("cluster_type") == "docked").height
    waiting = cluster_df.filter(pl.col("cluster_type") == "waiting").height
    noise = cluster_df.filter(pl.col("cluster_label") == -1).height
    print(f"\nTotal episodes : {len(cluster_df)}  |  docked={docked}  waiting={waiting}  noise={noise}")

    cols = ["MMSI", "traj_id", "LAT", "LON", "cluster_label", "cluster_type",
            "membership_score", "Heading_mean", "Heading_std", "Draft", "Length", "nb_messages"]
    cols = [c for c in cols if c in cluster_df.columns]
    preview = cluster_df.select(cols).head(args.preview)

    with pl.Config(tbl_rows=args.preview, tbl_cols=len(cols), tbl_width_chars=120):
        print(f"\n--- Preview ({args.preview}/{len(cluster_df)} rows) ---")
        print(preview)

    if args.save_csv:
        csv_path = Path(args.parquet).with_suffix(".clusters.csv")
        cluster_df.write_csv(csv_path)
        print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
