"""
Étape 2 (complète) + Étape 3 (gap detection) — Prétraitement cinématique.

Appliqué en mémoire avant le clustering HDBSCAN. Prend un DataFrame AIS quotidien brut et ajoute :
  - SOG_corr         : vitesse déclarée (SOG) corrigée si aberrante (> 50 nœuds)
                       remplacée par la vitesse calculée via haversine entre messages consécutifs
  - speed_computed_kt: vitesse haversine brute (NaN au premier message par MMSI)
  - traj_id          : identifiant entier incrémenté à chaque gap > GAP_THRESHOLD_MIN
                       ou au premier message d'un nouveau MMSI
                       → les navires avec plusieurs épisodes stationnaires obtiennent des traj_id distincts

Usage:
    from src.ingestion.kinematic_filter import prepare_kinematics
    df_prepared = prepare_kinematics(pl.read_parquet(path))
"""
import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Constantes de filtrage cinématique
EARTH_RADIUS_NM   = 3440.065  # rayon terrestre en milles nautiques
SOG_ABERRANT_MIN  = 50.0      # seuil (nœuds) déclenchant la correction de SOG environ 130 km/h
SOG_MAX           = 50.0      # plafond maximal appliqué après correction
GAP_THRESHOLD_MIN = 30        # minutes — rupture démarrant une nouvelle trajectoire


def prepare_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajoute les colonnes SOG_corr, speed_computed_kt et traj_id au DataFrame AIS quotidien.
    Trie par (MMSI, BaseDateTime) sur place — requis pour les opérations de décalage.

    Retourne le DataFrame enrichi (colonnes originales conservées).
    """
    # Tri par identifiant de navire et timestamp (préalable à la détection de gaps)
    df = df.sort(["MMSI", "BaseDateTime"])

    # Récupère la position précédente et le timestamp du message antérieur pour chaque MMSI
    df = df.with_columns([
        pl.col("LAT").shift(1).over("MMSI").alias("_lat_prev"),
        pl.col("LON").shift(1).over("MMSI").alias("_lon_prev"),
        pl.col("BaseDateTime").shift(1).over("MMSI").alias("_dt_prev"),
    ])

    # Calcule le delta temporel en secondes — NaN au premier message par MMSI
    dt_sec_series = (df["BaseDateTime"] - df["_dt_prev"]).dt.total_seconds()
    null_mask     = dt_sec_series.is_null().to_numpy()
    dt_sec        = np.where(null_mask, np.nan,
                             dt_sec_series.fill_null(0).cast(pl.Float64).to_numpy())

    # Récupère les coordonnées précédentes — NaN où null (premier message par MMSI)
    lat1 = _series_to_float_numpy(df["_lat_prev"])
    lon1 = _series_to_float_numpy(df["_lon_prev"])
    lat2 = df["LAT"].to_numpy().astype(float)
    lon2 = df["LON"].to_numpy().astype(float)

    # Calcule la distance haversine en milles nautiques (NaN au premier message par MMSI)
    dist_nm = _haversine_nm(lat1, lon1, lat2, lon2)
    dt_h    = np.where(dt_sec > 0, dt_sec / 3600.0, np.nan)
    speed   = np.where(~np.isnan(dt_h), dist_nm / dt_h, np.nan)

    # Correction de SOG : remplace la vitesse déclarée aberrante par la vitesse calculée
    sog_declared = df["SOG"].to_numpy().astype(float)
    sog_corr = np.where(
        (sog_declared > SOG_ABERRANT_MIN) & (~np.isnan(speed)) & (speed <= SOG_MAX),
        speed,
        sog_declared,
    )
    sog_corr = np.minimum(sog_corr, SOG_MAX)

    # Identifiants de trajectoire : incrémente au premier message par MMSI ou après un gap
    gap_mask = np.isnan(dt_sec) | (dt_sec > GAP_THRESHOLD_MIN * 60)
    traj_id  = np.cumsum(gap_mask.astype(np.int32))

    # Comptabilise les corrections et les trajectoires pour journalisation
    n_corrected = int(((sog_declared > SOG_ABERRANT_MIN) & ~np.isnan(speed)).sum())
    n_trajs     = int(gap_mask.sum())
    log.debug("SOG corrected: %d messages | trajectories: %d", n_corrected, n_trajs)

    # Retourne le DataFrame avec les 3 colonnes enrichies ajoutées
    return (
        df.drop(["_lat_prev", "_lon_prev", "_dt_prev"])
        .with_columns([
            pl.Series("SOG_corr",          sog_corr, dtype=pl.Float32),
            pl.Series("speed_computed_kt", speed,    dtype=pl.Float32),
            pl.Series("traj_id",           traj_id,  dtype=pl.Int32),
        ])
    )


# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------

def _series_to_float_numpy(series: pl.Series) -> np.ndarray:
    """Convertit une série Polars float nullable en numpy, remplaçant les valeurs nulles par NaN."""
    null_mask = series.is_null().to_numpy()
    arr = series.fill_null(0.0).cast(pl.Float64).to_numpy().copy()
    arr[null_mask] = np.nan
    return arr


def _haversine_nm(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Calcule la distance haversine vectorisée en milles nautiques.
    Propage les NaN pour les entrées nulles.
    """
    # Conversion en radians pour les formules trigonométriques
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat   = lat2_r - lat1_r
    dlon   = np.radians(lon2 - lon1)
    
    # Formula haversine : calcule le demi-angle de la grande cercle
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    
    # Distance en milles nautiques
    return 2.0 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
