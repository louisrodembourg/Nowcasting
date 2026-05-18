"""
Extraire le vecteur des 13 features quotidiennes pour l'entrée de la variété.

Features (une ligne par jour) :
  1.  vessel_count           — nombre distinct de MMSI dans la bbox ce jour-là
  2.  SOG_mean               — SOG moyen corrigé sur tous les messages
  3.  SOG_std                — écart-type de la SOG corrigée
  4.  SOG_median             — médiane de la SOG corrigée
  5.  utilization_rate_rho   — fraction de MMSI distincts avec ≥1 épisode statique
  6.  waiting_cluster_count  — nombre de clusters HDBSCAN en zone d'attente (hors bruit, hors docked)
  7.  hdbscan_noise_ratio    — épisodes bruits / total épisodes statiques
  8.  membership_score_mean  — probabilité moyenne d'appartenance HDBSCAN (hors bruit)
  9.  membership_score_std   — écart-type des probabilités d'appartenance
  10. draft_mean             — tirant d'eau moyen parmi les épisodes statiques (m)
  11. draft_std              — écart-type du tirant d'eau
  12. waiting_capacity       — Σ(Longueur × Largeur) des navires en zone d'attente uniquement (proxy m²)
  13. tanker_ratio           — fraction MMSI pétroliers parmi MMSI statiques (VesselType 80–89)

Usage (autonome, un jour):
    python src/clustering/features_daily.py --date 2017-07-01 --location houston
    python src/clustering/features_daily.py --date 2019-03-15 --location la
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from src.clustering.hdbscan_daily import (
    ClusteringConfig,
    cluster_day,
    load_docked_zones_or_none,
)
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

SOG_STATIC_THRESHOLD = 1.0


def _f(val, default: float = 0.0) -> float:
    """Convertit de manière sûre un scalaire Polars nullable en float."""
    return float(val) if val is not None else default


def compute_daily_features(
    day_data: Union[Path, pl.DataFrame],
    cluster_df: Optional[pl.DataFrame],
    d: date,
) -> Optional[dict]:
    """
    Calcule les 13 features quotidiennes pour la variété.

    Args:
        day_data:   Chemin Parquet brut OU DataFrame préparé (avec colonne SOG_corr).
        cluster_df: Sortie de cluster_day() — une ligne par épisode statique.
                    Passer None si HDBSCAN n'a produit aucun résultat.
        d:          La date (clé 'date' dans le dict de sortie).

    Retourne un dict avec 14 clés (date + 13 features), ou None si données manquantes.
    """
    if isinstance(day_data, Path):
        if not day_data.exists():
            log.warning("%s : Parquet non trouvé — %s", d, day_data)
            return None
        all_df  = pl.read_parquet(day_data)
        sog_col = "SOG"
    else:
        all_df  = day_data
        sog_col = "SOG_corr" if "SOG_corr" in all_df.columns else "SOG"

    # --- Trafic global -------------------------------------------------------
    vessel_count = all_df["MMSI"].n_unique()
    sog_mean     = _f(all_df[sog_col].mean())
    sog_std      = _f(all_df[sog_col].std())
    sog_median   = _f(all_df[sog_col].median())

    static_mmsi_count    = all_df.filter(pl.col(sog_col) < SOG_STATIC_THRESHOLD)["MMSI"].n_unique()
    utilization_rate_rho = static_mmsi_count / vessel_count if vessel_count > 0 else 0.0

    # --- Features HDBSCAN ---------------------------------------------------
    if cluster_df is None or len(cluster_df) == 0:
        log.warning("%s : aucune donnée cluster — features HDBSCAN = 0", d)
        return {
            "date":                  d.isoformat(),
            "vessel_count":          vessel_count,
            "SOG_mean":              round(sog_mean,   4),
            "SOG_std":               round(sog_std,    4),
            "SOG_median":            round(sog_median, 4),
            "utilization_rate_rho":  round(utilization_rate_rho, 4),
            "waiting_cluster_count": 0,
            "hdbscan_noise_ratio":   1.0,
            "membership_score_mean": 0.0,
            "membership_score_std":  0.0,
            "draft_mean":            0.0,
            "draft_std":             0.0,
            "waiting_capacity":      0.0,
            "tanker_ratio":          0.0,
        }

    labels     = cluster_df["cluster_label"].to_numpy()
    n_episodes = len(cluster_df)
    n_noise    = int((labels == -1).sum())
    non_noise  = cluster_df.filter(pl.col("cluster_label") >= 0)

    # waiting_cluster_count : only clusters in the waiting zone are congestion signal
    waiting_df = cluster_df.filter(pl.col("cluster_type") == "waiting")
    waiting_labels = waiting_df["cluster_label"].to_numpy()
    waiting_cluster_count = (
        len(set(waiting_labels) - {-1}) if len(waiting_labels) > 0 else 0
    )

    hdbscan_noise_ratio   = n_noise / n_episodes if n_episodes > 0 else 1.0
    membership_score_mean = _f(non_noise["membership_score"].mean())
    membership_score_std  = _f(non_noise["membership_score"].std())

    draft_series     = cluster_df.filter(pl.col("Draft") > 0)["Draft"]
    draft_mean       = _f(draft_series.mean())
    draft_std        = _f(draft_series.std())

    cap_series = (
        cluster_df
        .filter(
            (pl.col("cluster_type") == "waiting") &
            (pl.col("Length") > 0) &
            (pl.col("Width") > 0)
        )
        .select((pl.col("Length") * pl.col("Width")).alias("cap"))["cap"]
    )
    log.info("%s : %d épisodes, %d en attente avec dimensions valides (waiting_capacity)",
             d, n_episodes, len(cap_series))
    waiting_capacity = _f(cap_series.sum())

    n_static_mmsi = cluster_df["MMSI"].n_unique()
    tanker_mmsi   = cluster_df.filter(pl.col("VesselType").is_between(80, 89))["MMSI"].n_unique()
    tanker_ratio  = tanker_mmsi / n_static_mmsi if n_static_mmsi > 0 else 0.0

    return {
        "date":                  d.isoformat(),
        "vessel_count":          vessel_count,
        "SOG_mean":              round(sog_mean,   4),
        "SOG_std":               round(sog_std,    4),
        "SOG_median":            round(sog_median, 4),
        "utilization_rate_rho":  round(utilization_rate_rho, 4),
        "waiting_cluster_count": waiting_cluster_count,
        "hdbscan_noise_ratio":   round(hdbscan_noise_ratio,   4),
        "membership_score_mean": round(membership_score_mean, 4),
        "membership_score_std":  round(membership_score_std,  4),
        "draft_mean":            round(draft_mean, 2),
        "draft_std":             round(draft_std,  2),
        "waiting_capacity":      round(waiting_capacity, 1),
        "tanker_ratio":          round(tanker_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule les 13 features quotidiennes pour un jour"
    )
    parser.add_argument("--date",        required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--location",    default="houston",
                        choices=list(LOCATIONS.keys()),
                        help="Port cible (default: houston)")
    parser.add_argument("--parquet-dir", default=None,
                        help="Répertoire Parquet (défaut : data/parquet/{location})")
    args = parser.parse_args()

    loc_cfg     = LOCATIONS[args.location]
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else Path("data/parquet") / args.location
    d           = date.fromisoformat(args.date)
    parquet_path = parquet_dir / f"{loc_cfg['prefix']}_{d.strftime('%Y_%m_%d')}.parquet"

    # Charge les zones docked si disponibles
    docked_poly = load_docked_zones_or_none(args.location)
    config = ClusteringConfig(ref_polygon=docked_poly)
    if docked_poly is not None:
        log.info("Zones docked chargées pour '%s'", args.location)

    cluster_df, prepared_df = cluster_day(parquet_path, config=config)
    features = compute_daily_features(
        prepared_df if prepared_df is not None else parquet_path,
        cluster_df,
        d,
    )

    if features:
        for k, v in features.items():
            print(f"  {k:<26} {v}")
    else:
        print("Aucune feature calculée.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
