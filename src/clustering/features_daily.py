"""
Extraire le vecteur des 13 features quotidiennes pour l'entrée de la variété de Houston.

Features (une ligne par jour) :
  1.  vessel_count           — nombre distinct de MMSI dans la bbox ce jour-là
  2.  SOG_mean               — SOG moyen corrigé sur tous les messages
  3.  SOG_std                — écart-type de la SOG corrigée
  4.  SOG_median             — médiane de la SOG corrigée
  5.  utilization_rate_rho   — fraction de MMSI distincts avec ≥1 épisode statique
  6.  hdbscan_cluster_count  — nombre de clusters HDBSCAN (hors bruit)
  7.  hdbscan_noise_ratio    — épisodes bruits / total épisodes statiques
  8.  membership_score_mean  — probabilité moyenne d'appartenance HDBSCAN (hors bruit)
  9.  membership_score_std   — écart-type des probabilités d'appartenance
  10. draft_mean             — tirant d'eau moyen parmi les épisodes statiques (m)
  11. draft_std              — écart-type du tirant d'eau
  12. blocked_capacity       — Σ(Longueur × Largeur) pour épisodes statiques (proxy m²)
  13. tanker_ratio           — fraction MMSI pétroliers parmi MMSI statiques (VesselType 80–89)

Usage (autonome, un jour):
    python src/clustering/features_daily.py --date 2017-07-01
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from src.clustering.hdbscan_daily import cluster_day

log = logging.getLogger(__name__)

SOG_STATIC_THRESHOLD = 1.0
PARQUET_DIR = Path("data/parquet/houston")


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
        day_data:    Chemin vers le Parquet brut OU un DataFrame préparé (avec colonne SOG_corr).
                     Quand un DataFrame est passé (déjà préparé par prepare_kinematics),
                     la SOG corrigée est utilisée pour les stats de trafic. Sinon la SOG brute est utilisée.
        cluster_df:  Sortie de cluster_day() — une ligne par épisode statique + métadonnées du cluster.
                     Passer None si HDBSCAN n'a produit aucun résultat (features HDBSCAN défaut 0).
        d:           La date, utilisée comme clé 'date' dans le dict de sortie.

    Retourne un dict avec 14 clés (date + 13 features), ou None si les données manquent.
    """
    # Vérifie si day_data est un chemin ou un DataFrame déjà préparé
    if isinstance(day_data, Path):
        if not day_data.exists():
            log.warning("%s : Parquet non trouvé — %s", d, day_data)
            return None
        # Charge le fichier brut et utilise la SOG brute
        all_df = pl.read_parquet(day_data)
        sog_col = "SOG"
    else:
        # Utilise le DataFrame préparé, préférant la SOG corrigée si disponible
        all_df  = day_data
        sog_col = "SOG_corr" if "SOG_corr" in all_df.columns else "SOG"

    # --- Trafic global (tous les messages, tous les navires) ----------------
    vessel_count = all_df["MMSI"].n_unique()  # Nombre distinct de navires
    sog_mean     = _f(all_df[sog_col].mean())  # Vitesse moyenne
    sog_std      = _f(all_df[sog_col].std())   # Variabilité de vitesse
    sog_median   = _f(all_df[sog_col].median())  # Vitesse médiane

    # --- Fraction statique : MMSI distinct avec ≥1 épisode statique ---------
    static_mmsi_count = (
        all_df.filter(pl.col(sog_col) < SOG_STATIC_THRESHOLD)["MMSI"].n_unique()
    )
    # Ratio d'utilisation = navires statiques / navires totaux
    utilization_rate_rho = static_mmsi_count / vessel_count if vessel_count > 0 else 0.0

    # --- Features dérivées de HDBSCAN ------------------------------------------
    # Si pas de résultats de clustering, retourne les features par défaut (0)
    if cluster_df is None or len(cluster_df) == 0:
        log.warning("%s : aucune donnée de cluster — features HDBSCAN mises à 0", d)
        return {
            "date":                  d.isoformat(),
            "vessel_count":          vessel_count,
            "SOG_mean":              round(sog_mean,   4),
            "SOG_std":               round(sog_std,    4),
            "SOG_median":            round(sog_median, 4),
            "utilization_rate_rho":  round(utilization_rate_rho, 4),
            "hdbscan_cluster_count": 0,
            "hdbscan_noise_ratio":   1.0,
            "membership_score_mean": 0.0,
            "membership_score_std":  0.0,
            "draft_mean":            0.0,
            "draft_std":             0.0,
            "blocked_capacity":      0.0,
            "tanker_ratio":          0.0,
        }

    # Extrait les labels de cluster et calcule les statistiques de clustering
    labels     = cluster_df["cluster_label"].to_numpy()
    n_episodes = len(cluster_df)                              # épisodes statiques
    n_noise    = int((labels == -1).sum())                     # points bruits
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # clusters significatifs
    non_noise  = cluster_df.filter(pl.col("cluster_label") >= 0)  # exclu le bruit

    # Ratio de bruit et scores d'appartenance
    hdbscan_noise_ratio   = n_noise / n_episodes if n_episodes > 0 else 1.0
    membership_score_mean = _f(non_noise["membership_score"].mean())
    membership_score_std  = _f(non_noise["membership_score"].std())

    # Tirant d'eau : ignorer les valeurs zéro / inconnues
    draft_series   = cluster_df.filter(pl.col("Draft") > 0)["Draft"]
    draft_mean     = _f(draft_series.mean())
    draft_std      = _f(draft_series.std())

    # Capacité bloquée : Σ(Longueur × Largeur) — ignorer les navires avec dimensions inconnues
    cap_series = (
        cluster_df
        .filter((pl.col("Length") > 0) & (pl.col("Width") > 0))
        .select((pl.col("Length") * pl.col("Width")).alias("cap"))["cap"]
    )

    # Enregistre le nombre d'épisodes avec dimensions valides
    n_before = len(cluster_df)
    n_after = len(cap_series)
    log.info("%s : %d épisodes, %d avec dimensions valides pour blocked_capacity", d, n_before, n_after)

    blocked_capacity = _f(cap_series.sum())

    # Ratio pétroliérs : compte les MMSI uniques (pas les épisodes) avec VesselType 80–89
    n_static_mmsi  = cluster_df["MMSI"].n_unique()
    tanker_mmsi    = cluster_df.filter(pl.col("VesselType").is_between(80, 89))["MMSI"].n_unique()
    tanker_ratio   = tanker_mmsi / n_static_mmsi if n_static_mmsi > 0 else 0.0

    return {
        "date":                  d.isoformat(),
        "vessel_count":          vessel_count,
        "SOG_mean":              round(sog_mean,   4),
        "SOG_std":               round(sog_std,    4),
        "SOG_median":            round(sog_median, 4),
        "utilization_rate_rho":  round(utilization_rate_rho, 4),
        "hdbscan_cluster_count": n_clusters,
        "hdbscan_noise_ratio":   round(hdbscan_noise_ratio,   4),
        "membership_score_mean": round(membership_score_mean, 4),
        "membership_score_std":  round(membership_score_std,  4),
        "draft_mean":            round(draft_mean, 2),
        "draft_std":             round(draft_std,  2),
        "blocked_capacity":      round(blocked_capacity, 1),
        "tanker_ratio":          round(tanker_ratio, 4),
    }


def main() -> None:
    """Point d'entrée pour l'exécution autonome — calcul des features pour un jour donné."""
    parser = argparse.ArgumentParser(
        description="Calcule les 13 features quotidiennes pour un jour — Houston"
    )
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--parquet-dir", default=str(PARQUET_DIR))
    args = parser.parse_args()

    # Charge la date et construit le chemin du fichier Parquet
    d            = date.fromisoformat(args.date)
    parquet_path = Path(args.parquet_dir) / f"houston_{d.strftime('%Y_%m_%d')}.parquet"

    # Exécute le pipeline complet : clustering puis extraction des features
    cluster_df, prepared_df = cluster_day(parquet_path)
    features = compute_daily_features(
        prepared_df if prepared_df is not None else parquet_path,
        cluster_df,
        d,
    )

    # Affiche les résultats ou un message d'erreur
    if features:
        for k, v in features.items():
            print(f"  {k:<26} {v}")
    else:
        print("Aucune feature calculée.")


if __name__ == "__main__":
    # Configure le logging pour afficher les messages d'info avec timestamps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
