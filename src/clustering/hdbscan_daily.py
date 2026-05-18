"""
Étape 4 : Clustering quotidien HDBSCAN des navires stationnaires.

Pipeline pour un fichier Parquet quotidien :
  1. Prétraitement cinématique (SOG_corr, traj_id) via kinematic_filter
  2. Filtrer les navires stationnaires (SOG_corr < SOG_STATIC_THRESHOLD)
  3. Dédupliquer à une position par (MMSI, traj_id) — chaque épisode statique séparément
  4. Exécuter HDBSCAN sur (LAT, LON) en utilisant la métrique haversine
  5. Classer chaque épisode : 'docked' si dans le polygone docked GeoJSON, sinon 'waiting'

Retourne (cluster_df, prepared_df) :
  - cluster_df  : une ligne par épisode statique (MMSI + traj_id) + métadonnées du cluster
  - prepared_df : DataFrame complet du jour avec colonnes SOG_corr et traj_id ajoutées

Zones géographiques : data/zones/{location}_docked.geojson
  - Tout épisode dans le polygone docked → cluster_type = "docked"
  - Tout épisode en dehors              → cluster_type = "waiting"
  - Si aucun fichier de zones fourni    → tout = "waiting" (fallback)

Usage (autonome) :
    python src/clustering/hdbscan_daily.py data/parquet/houston/houston_2017_07_01.parquet
    python src/clustering/hdbscan_daily.py data/parquet/la/la_2019_01_01.parquet --location la
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
import shapely
from shapely.geometry import shape, base as shapely_base
from shapely.ops import unary_union

# Type alias pour les polygones shapely
ShapelyGeom = shapely_base.BaseGeometry

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# ── Paramètres par défaut ─────────────────────────────────────────────────────

SOG_STATIC_THRESHOLD     = 1.0   # nœuds — en dessous = navire stationnaire
HDBSCAN_MIN_CLUSTER_SIZE = 3     # taille minimale du cluster HDBSCAN
HDBSCAN_MIN_SAMPLES      = 2     # nombre minimal d'échantillons pour HDBSCAN

# Convention : data/zones/{location}_docked.geojson
ZONES_DIR = Path("data/zones")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ClusteringConfig:
    """
    Paramètres du pipeline HDBSCAN.

    zone_mode : "docked" (défaut) — polygone docked comme référence, reste = waiting
                "waiting"         — polygone waiting comme référence, reste = docked
    ref_polygon : polygone shapely de référence (docked ou waiting selon zone_mode).
                  None → tout = fallback (waiting si mode docked, docked si mode waiting).
    """
    sog_static_threshold:     float  = SOG_STATIC_THRESHOLD
    hdbscan_min_cluster_size: int    = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples:      int    = HDBSCAN_MIN_SAMPLES
    zone_mode:   str                        = "docked"   # "docked" ou "waiting"
    ref_polygon: Optional[ShapelyGeom]     = None


# ── Chargement des zones ──────────────────────────────────────────────────────

def load_docked_zones(location: str) -> ShapelyGeom:
    """
    Charge le polygone docked depuis data/zones/{location}_docked.geojson.
    Retourne un polygone shapely unifié (union de tous les polygones du fichier).
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    path = ZONES_DIR / f"{location}_docked.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de zones introuvable : {path}\n"
            f"Exporter depuis Google Earth → Enregistrer le lieu → GeoJSON"
        )
    with open(path) as f:
        fc = json.load(f)

    polys = []
    features = fc.get("features", [fc] if fc.get("type") == "Feature" else [])
    for feat in features:
        geom = feat.get("geometry", feat) if isinstance(feat, dict) else feat
        if geom:
            polys.append(shape(geom))

    if not polys:
        raise ValueError(f"Aucun polygone valide dans {path}")

    unified = unary_union(polys)
    log.info("Zones docked chargées : %d polygone(s), bounds=%s", len(polys), unified.bounds)
    return unified


def load_docked_zones_or_none(location: str) -> Optional[ShapelyGeom]:
    """Comme load_docked_zones() mais retourne None si le fichier est absent."""
    try:
        return load_docked_zones(location)
    except FileNotFoundError:
        log.info("Pas de zones docked pour '%s' — fallback tout=waiting", location)
        return None


def load_waiting_zones(location: str) -> ShapelyGeom:
    """
    Charge le polygone waiting depuis data/zones/{location}_waiting.geojson.
    Retourne un polygone shapely unifié.
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    path = ZONES_DIR / f"{location}_waiting.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Fichier de zones introuvable : {path}")
    with open(path) as f:
        fc = json.load(f)
    polys = []
    features = fc.get("features", [fc] if fc.get("type") == "Feature" else [])
    for feat in features:
        geom = feat.get("geometry", feat) if isinstance(feat, dict) else feat
        if geom:
            polys.append(shape(geom))
    if not polys:
        raise ValueError(f"Aucun polygone valide dans {path}")
    unified = unary_union(polys)
    log.info("Zones waiting chargées : %d polygone(s), bounds=%s", len(polys), unified.bounds)
    return unified


def load_waiting_zones_or_none(location: str) -> Optional[ShapelyGeom]:
    """Comme load_waiting_zones() mais retourne None si le fichier est absent."""
    try:
        return load_waiting_zones(location)
    except FileNotFoundError:
        log.info("Pas de zones waiting pour '%s' — fallback tout=docked", location)
        return None


# ── API publique ──────────────────────────────────────────────────────────────

def cluster_day_from_df(
    prepared: pl.DataFrame,
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Exécute HDBSCAN sur un DataFrame déjà préparé (sortie de prepare_kinematics).
    Retourne (cluster_df, prepared_df).
    """
    return _run_hdbscan(prepared, label="<DataFrame>", config=config or ClusteringConfig())


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

    raw_df   = pl.read_parquet(parquet_path)
    prepared = prepare_kinematics(raw_df)
    return _run_hdbscan(prepared, label=parquet_path.name, config=config or ClusteringConfig())


# ── Pipeline interne ──────────────────────────────────────────────────────────

def _run_hdbscan(
    prepared: pl.DataFrame,
    label: str = "",
    config: Optional[ClusteringConfig] = None,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    cfg = config or ClusteringConfig()

    static = prepared.filter(pl.col("SOG_corr") < cfg.sog_static_threshold)
    if len(static) < cfg.hdbscan_min_cluster_size:
        log.warning("%s : %d messages statiques — skip", label, len(static))
        return None, prepared

    # Agrégation : une position par épisode statique (MMSI, traj_id)
    cols = prepared.columns
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("Heading").filter(pl.col("Heading") < 360).mean().alias("Heading_mean")
            if "Heading" in cols else pl.lit(None).cast(pl.Float64).alias("Heading_mean"),
        pl.col("Heading").filter(pl.col("Heading") < 360).std().alias("Heading_std")
            if "Heading" in cols else pl.lit(None).cast(pl.Float64).alias("Heading_std"),
        pl.col("Draft").max().alias("Draft")
            if "Draft" in cols else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length")
            if "Length" in cols else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width")
            if "Width" in cols else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType")
            if "VesselType" in cols else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),
    ]
    agg = static.group_by(["MMSI", "traj_id"]).agg(agg_exprs)

    if len(agg) < cfg.hdbscan_min_cluster_size:
        log.warning("%s : %d épisodes après dédup — skip", label, len(agg))
        return None, prepared

    # HDBSCAN haversine
    coords_rad = np.radians(agg.select(["LAT", "LON"]).to_numpy())
    clusterer  = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="haversine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_rad)
    scores = clusterer.probabilities_

    agg = agg.with_columns([
        pl.Series("cluster_label",    labels, dtype=pl.Int32),
        pl.Series("membership_score", scores, dtype=pl.Float32),
    ])

    # Classification docked / waiting par zone géographique
    agg = _classify_by_zone(agg, cfg)

    n_ep      = len(agg)
    n_vessels = agg["MMSI"].n_unique()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info(
        "%s : %d épisodes (%d navires) → %d clusters, %d bruit (%.1f%%)",
        label, n_ep, n_vessels, n_clusters, n_noise, 100 * n_noise / n_ep,
    )
    return agg, prepared


def _classify_by_zone(agg: pl.DataFrame, config: "ClusteringConfig") -> pl.DataFrame:
    """
    Classifie chaque épisode selon zone_mode :
      - mode "docked"  : dans ref_polygon → "docked",  sinon → "waiting"
      - mode "waiting" : dans ref_polygon → "waiting", sinon → "docked"
    Utilise shapely.contains_xy (vectorisé).
    """
    lons = agg["LON"].to_numpy()
    lats = agg["LAT"].to_numpy()

    if config.ref_polygon is not None:
        in_ref = shapely.contains_xy(config.ref_polygon, lons, lats)
    else:
        in_ref = np.zeros(len(agg), dtype=bool)

    if config.zone_mode == "waiting":
        zone = np.where(in_ref, "waiting", "docked")
    else:
        zone = np.where(in_ref, "docked", "waiting")

    return agg.with_columns(pl.Series("cluster_type", zone, dtype=pl.Utf8))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clustering HDBSCAN sur un fichier Parquet quotidien"
    )
    parser.add_argument("parquet",     help="Chemin vers le fichier Parquet quotidien")
    parser.add_argument("--location",  default=None,
                        help="Nom du port (houston, la) pour charger les zones GeoJSON automatiquement")
    args = parser.parse_args()

    config = ClusteringConfig()
    if args.location:
        config.ref_polygon = load_docked_zones_or_none(args.location)

    cluster_df, _ = cluster_day(Path(args.parquet), config=config)
    if cluster_df is not None:
        print(cluster_df)
        docked  = cluster_df.filter(pl.col("cluster_type") == "docked").height
        waiting = cluster_df.filter(pl.col("cluster_type") == "waiting").height
        noise   = cluster_df.filter(pl.col("cluster_label") == -1).height
        print(f"\ndocked={docked}  waiting={waiting}  bruit={noise}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
