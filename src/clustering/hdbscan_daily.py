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
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdbscan
import numpy as np
import polars as pl
from scipy.stats import circmean, circstd

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# Paramètres de clustering et seuils cinématiques
SOG_STATIC_THRESHOLD     = 1.0   # nœuds — en dessous = navire stationnaire
HEADING_DOCKED_MAX_STD   = 25.0  # degrés — écart-type en dessous = à quai
HDBSCAN_MIN_CLUSTER_SIZE = 3     # taille minimale du cluster HDBSCAN
HDBSCAN_MIN_SAMPLES      = 2     # nombre minimal d'échantillons pour HDBSCAN


def cluster_day_from_df(
    prepared: pl.DataFrame,
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Exécute HDBSCAN sur un DataFrame déjà préparé (sortie de prepare_kinematics).
    Retourne (cluster_df, prepared_df) — même contrat que cluster_day.
    """
    return _run_hdbscan(prepared, label="<DataFrame>")


def cluster_day(
    parquet_path: Path,
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
    # Vérifie que le fichier existe
    if not parquet_path.exists():
        log.warning("Fichier non trouvé : %s", parquet_path)
        return None, None

    # Charge le fichier brut et applique le prétraitement cinématique
    raw_df     = pl.read_parquet(parquet_path)
    prepared   = prepare_kinematics(raw_df)

    # Exécute le pipeline HDBSCAN
    return _run_hdbscan(prepared, label=parquet_path.name)


def _run_hdbscan(
    prepared: pl.DataFrame,
    label: str = "",
) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Logique centrale du clustering HDBSCAN sur un DataFrame préparé."""
    # --- Filtre les navires stationnaires (en utilisant la SOG corrigée) -------
    static = prepared.filter(pl.col("SOG_corr") < SOG_STATIC_THRESHOLD)

    # Garde la main si pas assez de messages statiques
    if len(static) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning(
            "%s : seulement %d messages statiques — skip clustering",
            label, len(static),
        )
        return None, prepared

    # --- Déduplique : une position par (MMSI, traj_id) ----------------------
    # Chaque épisode statique continu est traité comme un point de données distinct.
    # Heading 511 = AIS « non disponible » — exclu avant calcul des statistiques.

    # Construit l'expression d'agrégation — inclut uniquement les colonnes existantes
    # Position médiane (latitude/longitude) + statistiques sur le cap (moyenne/écart-type)
    # + dimensions du navire (tirant d'eau, longueur, largeur) + type de navire + compte de messages
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        # Collect valid headings as list for circular stats (computed after group_by)
        (pl.col("Heading").filter(pl.col("Heading") < 360).implode().alias("_headings")
            if "Heading" in prepared.columns else pl.lit(None).alias("_headings")),
        # Caractéristiques du navire (tirant d'eau max, dimensions)
        pl.col("Draft").max().alias("Draft")
            if "Draft" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length")
            if "Length" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width")
            if "Width" in prepared.columns else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType")
            if "VesselType" in prepared.columns else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),  # Nombre de messages pour cet épisode
    ]

    # Agrège par (MMSI, traj_id) pour obtenir un point par épisode statique
    agg = static.group_by(["MMSI", "traj_id"]).agg(agg_exprs)

    # Circular mean and std on heading — extract to Python, compute, rejoin
    headings_list = agg["_headings"].to_list()
    h_means, h_stds = [], []
    for raw in headings_list:
        h = [x for x in (raw or []) if x is not None]
        h_means.append(float(circmean(h, high=360, low=0)) if len(h) >= 1 else None)
        h_stds.append(float(circstd(h,  high=360, low=0)) if len(h) >= 2 else 180.0)

    agg = agg.with_columns([
        pl.Series("Heading_mean", h_means, dtype=pl.Float64),
        pl.Series("Heading_std",  h_stds,  dtype=pl.Float64),
    ]).drop("_headings")

    # Garde la main : HDBSCAN BallTree nécessite au moins min_cluster_size points
    if len(agg) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning(
            "%s : seulement %d épisodes après dédupication — skip HDBSCAN",
            label, len(agg),
        )
        return None, prepared

    # --- HDBSCAN sur (LAT, LON) avec distance haversine ----------------------
    # Extrait les coordonnées en degrés, puis les convertit en radians
    coords_deg = agg.select(["LAT", "LON"]).to_numpy()
    coords_rad = np.radians(coords_deg)

    # Configure et exécute HDBSCAN avec la métrique géographique haversine
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="haversine",  # Distance géographique appropriée pour coordonnées lat/lon
        cluster_selection_method="eom",  # Excess of mass selection
    )
    labels = clusterer.fit_predict(coords_rad)
    scores = clusterer.probabilities_

    # Ajoute les labels de cluster et scores d'appartenance au DataFrame
    agg = agg.with_columns([
        pl.Series("cluster_label",    labels, dtype=pl.Int32),
        pl.Series("membership_score", scores, dtype=pl.Float32),
    ])

    # --- Classifie les clusters : à quai vs en attente ----------------------
    # Utilise l'écart-type des cap pour distinguer les navires amarrés (caps alignés)
    # des navires en attente (caps dispersés par vent/courant)
    cluster_types = _classify_clusters(agg)
    agg = agg.join(cluster_types, on="cluster_label", how="left")

    # Calcule et enregistre les statistiques de clustering
    n_episodes = len(agg)
    n_vessels  = agg["MMSI"].n_unique()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclut le label de bruit (-1)
    n_noise    = int((labels == -1).sum())
    log.info(
        "%s : %d épisodes (%d navires) → %d clusters, %d bruits (%.1f%%)",
        label, n_episodes, n_vessels,
        n_clusters, n_noise, 100 * n_noise / n_episodes,
    )

    # Retourne le DataFrame des clusters et le DataFrame préparé complet
    return agg, prepared


def _classify_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """
    Étiquette chaque cluster comme 'à quai' ou 'en attente' selon l'écart-type des cap.
    Écart-type faible → navires alignés avec le quai → 'à quai'.
    Écart-type élevé → dispersion par vent/courant → 'en attente'.
    Écart-type nul (épisode un message) traité comme dispersion max → 'en attente'.
    """
    # Filtre les clusters valides (exclut le bruit avec label -1)
    # Puis agrège par label de cluster et calcule l'écart-type moyen des cap
    # (remplace les NaN par 180° pour traiter les épisodes d'un seul message comme maximalement dispersés)
    # Finalement, classifie : écart-type < HEADING_DOCKED_MAX_STD = "à quai", sinon = "en attente"
    return (
        df.filter(pl.col("cluster_label") >= 0)
        .group_by("cluster_label")
        .agg(
            pl.col("Heading_std")
            .fill_null(180.0)  # Traite l'absence de cap comme maximale dispersion
            .mean()
            .alias("_mean_heading_std")
        )
        .with_columns(
            # Classification : seuil HEADING_DOCKED_MAX_STD (25°) pour distinguer quai/attente
            pl.when(pl.col("_mean_heading_std") < HEADING_DOCKED_MAX_STD)
            .then(pl.lit("docked"))  # Navires alignés = à quai
            .otherwise(pl.lit("waiting"))  # Navires dispersés = en attente
            .alias("cluster_type")
        )
        .drop("_mean_heading_std")  # Supprime la colonne temporaire après classification
    )
    


def main() -> None:
    """Point d'entrée pour l'exécution autonome du script."""
    parser = argparse.ArgumentParser(
        description="Clustering HDBSCAN sur un fichier Parquet quotidien — Houston"
    )
    parser.add_argument("parquet", help="Chemin vers le fichier Parquet quotidien")
    args = parser.parse_args()

    # Exécute le clustering sur le fichier spécifié
    cluster_df, _ = cluster_day(Path(args.parquet))
    if cluster_df is not None:
        print(cluster_df)
        
        # Affiche les statistiques par type de cluster
        docked  = cluster_df.filter(pl.col("cluster_type") == "docked").height
        waiting = cluster_df.filter(pl.col("cluster_type") == "waiting").height
        noise   = cluster_df.filter(pl.col("cluster_label") == -1).height
        print(f"\nà_quai={docked}  en_attente={waiting}  bruit={noise}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
