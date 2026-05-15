"""
Étape 4 : Clustering quotidien HDBSCAN des navires stationnaires — Chenal du Navire de Houston.

Pipeline pour un fichier Parquet quotidien :
  1. Prétraitement cinématique (SOG_corr, traj_id) via kinematic_filter
  2. Filtrer les navires stationnaires (SOG_corr < SOG_STATIC_THRESHOLD)
  3. Dédupliquer à une position par (MMSI, traj_id) — chaque épisode statique séparément
  4. Exécuter HDBSCAN sur (LAT, LON) en utilisant la métrique haversine
  5. Classer chaque cluster : 'à quai' (caps alignés) vs 'en attente' (caps dispersés)

Retourne (cluster_df, prepared_df) :
  - cluster_df  : une ligne par épisode statique (MMSI + traj_id) + métadonnées du cluster
  - prepared_df : DataFrame complet du jour avec colonnes SOG_corr et traj_id ajoutées

Usage (autonome) :
    python src/clustering/hdbscan_daily.py data/parquet/houston/houston_2017_07_01.parquet
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdbscan
import numpy as np
import polars as pl
from scipy.stats import circmean, circstd

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# Paramètres de clustering et seuils cinématiques (valeurs par défaut)
SOG_STATIC_THRESHOLD = 1.0  # nœuds — en dessous = navire stationnaire
HEADING_DOCKED_MAX_STD = 25.0  # degrés — écart-type en dessous = à quai
HDBSCAN_MIN_CLUSTER_SIZE = 3  # taille minimale du cluster HDBSCAN
HDBSCAN_MIN_SAMPLES = 2  # nombre minimal d'échantillons pour HDBSCAN


@dataclass
class ClusteringConfig:
    """
    Paramètres de clustering HDBSCAN.
    Instancier sans arguments pour obtenir le comportement par défaut (baseline).
    """

    sog_static_threshold: float = SOG_STATIC_THRESHOLD
    heading_docked_max_std: float = HEADING_DOCKED_MAX_STD
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES
    # Rayon max (en nm) pour qu'un cluster soit classé "docked".
    # None = pas de contrainte spatiale.
    # 0.3 nm ≈ 550 m — taille typique d'un terminal de quai.
    max_docked_radius_nm: Optional[float] = None
    # Polygones délimitant les zones où le statut "waiting" est possible.
    # Chaque polygone = liste de (lat, lon) — fermé implicitement.
    # Tout cluster en DEHORS de TOUS ces polygones est forcé à "docked".
    # Ainsi, il ne peut pas y avoir de waiting en dehors des zones définies.
    waiting_allowed_polygons: Optional[list[list[tuple[float, float]]]] = None
    # NOTE: the previous heading-based docked/waiting filter is deprecated.
    # Heading thresholds remain in the config for compatibility only.
    min_heading_data_ratio: float = 0.3


def cluster_day_from_df(
    prepared: pl.DataFrame,
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Exécute HDBSCAN sur un DataFrame déjà préparé (sortie de prepare_kinematics).
    Retourne (cluster_df, prepared_df) — même contrat que cluster_day.
    """
    return _run_hdbscan(
        prepared, label="<DataFrame>", config=config or ClusteringConfig()
    )


def cluster_day(
    parquet_path: Path,
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Exécute le prétraitement cinématique + HDBSCAN sur un fichier Parquet quotidien.

    Retourne (cluster_df, prepared_df).
    Les deux sont None si le fichier est manquant ou a trop peu de navires statiques.

    Colonnes de cluster_df :
        MMSI, traj_id, LAT, LON, Heading_mean, Heading_std,
        Draft, Length, Width, VesselType, nb_messages,
        cluster_label, membership_score, cluster_type
    """
    if not parquet_path.exists():
        log.warning("Fichier non trouvé : %s", parquet_path)
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
    """Logique centrale du clustering HDBSCAN sur un DataFrame préparé."""
    cfg = config or ClusteringConfig()

    # --- Filtre les navires stationnaires (en utilisant la SOG corrigée) -------
    static = prepared.filter(pl.col("SOG_corr") < cfg.sog_static_threshold)

    if len(static) < cfg.hdbscan_min_cluster_size:
        log.warning(
            "%s : seulement %d messages statiques — skip clustering",
            label,
            len(static),
        )
        return None, prepared

    # --- Déduplique : une position par (MMSI, traj_id) ----------------------
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
            "%s : seulement %d épisodes après dédupication — skip HDBSCAN",
            label,
            len(agg),
        )
        return None, prepared

    # --- HDBSCAN sur (LAT, LON) avec distance haversine ----------------------
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

    # --- Classifie les clusters : à quai vs en attente ----------------------
    cluster_types = _classify_clusters(agg, cfg)
    agg = agg.join(cluster_types, on="cluster_label", how="left")

    n_episodes = len(agg)
    n_vessels = agg["MMSI"].n_unique()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(
        "%s : %d épisodes (%d navires) → %d clusters, %d bruits (%.1f%%)",
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
    Rayon du cluster = distance haversine max du centroïde au point le plus éloigné (en nm).
    Retourne 0 si le cluster a un seul point.
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
    Ray-casting algorithm : point (lat, lon) dans le polygone ?
    polygon = liste de (lat, lon) — fermé implicitement.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        # Rayon horizontal vers la droite depuis (lat, lon)
        # L'arête (lat_i,lon_i)-(lat_j,lon_j) traverse-t-elle ce rayon ?
        edge_crosses_y = (lat_i > lat) != (lat_j > lat)
        if edge_crosses_y:
            # Intersection x du rayon avec l'arête
            x_intersect = lon_i + (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i)
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _load_geojson_polygons(path: Path) -> list[list[tuple[float, float]]]:
    """
    Charge les polygones d'un fichier GeoJSON.
    Retourne une liste de polygones (chacun = liste de (lat, lon)).
    """
    import json

    with open(path) as f:
        fc = json.load(f)
    zones = []
    for feat in fc["features"]:
        coords = feat["geometry"]["coordinates"][0]  # ring extérieur
        zones.append([(c[1], c[0]) for c in coords])  # (lon, lat) -> (lat, lon)
    return zones


def load_waiting_zones(location: str) -> list[list[tuple[float, float]]]:
    """
    Charge les polygones de zone d'attente depuis data/features/{location}_waiting_zones.geojson.
    Retourne une liste de polygones (chacun = liste de (lat, lon)).
    Lève FileNotFoundError si le fichier n'existe pas.
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
    Étiquette chaque cluster comme 'à quai' ou 'en attente'.

    Règles de classification (appliquées dans l'ordre) :
      0. Si config.waiting_allowed_polygons est défini :
           cluster DANS au moins un polygone → forcé "waiting"
           cluster HORS de tous les polygones → forcé "docked"
      1. Sinon : si config.max_docked_radius_nm est défini,
           cluster compact → "docked", sinon "waiting"
      2. Sans contrainte de rayon explicite, le comportement reste conservateur :
           par défaut "docked".
    """
    valid = df.filter(pl.col("cluster_label") >= 0)

    rows: list[dict] = []
    for label in valid["cluster_label"].unique().to_list():
        cluster = valid.filter(pl.col("cluster_label") == label)
        mean_hstd = float(cluster["Heading_std"].fill_null(180.0).mean())

        lat_med = float(cluster["LAT"].median())
        lon_med = float(cluster["LON"].median())

        # Règle 0 : waiting_allowed_polygons — waiting si au moins un point du
        # cluster tombe dans une zone d'attente.
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
            # Sans filtre zonal, on ne classe "docked" que si le cluster
            # est suffisamment compact selon max_docked_radius_nm.
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
    """Point d'entrée pour l'exécution autonome du script."""
    parser = argparse.ArgumentParser(
        description="Clustering HDBSCAN sur un fichier Parquet quotidien — Houston"
    )
    parser.add_argument("parquet", help="Chemin vers le fichier Parquet quotidien")
    args = parser.parse_args()

    cluster_df, _ = cluster_day(Path(args.parquet))
    if cluster_df is not None:
        print(cluster_df)

        docked = cluster_df.filter(pl.col("cluster_type") == "docked").height
        waiting = cluster_df.filter(pl.col("cluster_type") == "waiting").height
        noise = cluster_df.filter(pl.col("cluster_label") == -1).height
        print(f"\nà_quai={docked}  en_attente={waiting}  bruit={noise}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
