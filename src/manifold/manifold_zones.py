"""
Phase 2 — Manifold géospatial : LBO sur les zones géographiques.

Au lieu de traiter les jours, on traite les ZONES géographiques :
  - Chaque zone = un cluster HDBSCAN (ou un groupe de clusters proches)
  - Chaque jour = une dimension temporelle
  - Les vecteurs propres = les zones "constituantes" les plus importantes

Usage:
    python src/manifold/manifold_zones.py --start 2020-01-01 --end 2020-01-08 --location la
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from src.clustering.hdbscan_daily import cluster_day
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_N_EIGENVECTORS = 5


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalisation L2 par ligne (chaque zone / sa norme)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def build_weight_matrix(X_norm: np.ndarray, k: int) -> sparse.csr_matrix:
    """Graphe KNN + noyau gaussien."""
    n = len(X_norm)
    k_safe = min(k, n - 1)

    nbrs = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(X_norm)
    distances, indices = nbrs.kneighbors(X_norm)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    sigma = distances.mean() if distances.mean() > 0 else 1.0

    rows = np.repeat(np.arange(n), k_safe)
    cols = indices.ravel()
    vals = np.exp(-(distances.ravel() ** 2) / (sigma**2))

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2
    return W


def compute_eigenvectors(W: sparse.csr_matrix, n_eigenvectors: int):
    """Décomposition spectrale LBO."""
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-10
    A = sparse.diags(degree)

    n_eigs = min(n_eigenvectors + 1, W.shape[0] - 1)
    A_inv = sparse.diags(1.0 / degree)
    L = A_inv @ W

    eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which="LM")

    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def find_constituent_zones(
    eigenvectors: np.ndarray,
    W: sparse.csr_matrix,
    n_components: int = 3,
) -> np.ndarray:
    """Trouve les zones qui sont des extremums locaux dans les vecteurs propres."""
    phi = eigenvectors[:, 1 : n_components + 1]
    n = phi.shape[0]

    W_coo = W.tocoo()
    neighbours = [[] for _ in range(n)]
    for i, j in zip(W_coo.row, W_coo.col):
        if i != j:
            neighbours[i].append(j)

    is_constituent = np.zeros(n, dtype=bool)
    for i in range(n):
        nbr = neighbours[i]
        if not nbr:
            continue
        for c in range(phi.shape[1]):
            vals_nbr = phi[nbr, c]
            if phi[i, c] > vals_nbr.max() or phi[i, c] < vals_nbr.min():
                is_constituent[i] = True
                break

    return is_constituent


def build_zone_features_matrix(
    start: date,
    end: date,
    location: str = "la",
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Construit la matrice (zones × jours) pour le manifold.

    Chaque ligne = une zone géographique unique (cluster HDBSCAN)
    Chaque colonne = un jour

    Returns:
        zone_df: DataFrame avec info sur chaque zone (lat, lon, centroid, etc.)
        X: matrice (n_zones, n_days) des features (occupation rate)
    """
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]

    all_zones = {}  # zone_key -> zone_info
    zone_day_data = {}  # (zone_key, day) -> occupation

    dates = []
    d = start
    while d <= end:
        dates.append(d)
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

        if parquet_path.exists():
            cluster_df, _ = cluster_day(parquet_path)

            if cluster_df is not None and len(cluster_df) > 0:
                for row in cluster_df.iter_rows(named=True):
                    # Créer une clé de zone basée sur la position (arrondie)
                    lat_r = round(row["LAT"], 3)
                    lon_r = round(row["LON"], 3)
                    zone_key = f"{lat_r}_{lon_r}"

                    # _info sur la zone
                    if zone_key not in all_zones:
                        all_zones[zone_key] = {
                            "lat": row["LAT"],
                            "lon": row["LON"],
                            "cluster_label": row["cluster_label"],
                            "cluster_type": row.get("cluster_type", "unknown"),
                            "mmsi": [row["MMSI"]],
                        }
                    else:
                        all_zones[zone_key]["mmsi"].append(row["MMSI"])

                    # Occupation de la zone ce jour
                    key = (zone_key, d)
                    if key not in zone_day_data:
                        zone_day_data[key] = 0
                    zone_day_data[key] += 1

        d += timedelta(days=1)

    if len(all_zones) == 0:
        log.warning("Aucune zone trouvée entre %s et %s", start, end)
        return pl.DataFrame(), np.array([])

    # Créer le DataFrame des zones
    zone_rows = []
    for zone_key, info in all_zones.items():
        n_vessels = len(set(info["mmsi"]))
        zone_rows.append(
            {
                "zone_key": zone_key,
                "lat": info["lat"],
                "lon": info["lon"],
                "cluster_label": info["cluster_label"],
                "cluster_type": info["cluster_type"],
                "n_vessels": n_vessels,
            }
        )

    zone_df = pl.DataFrame(zone_rows)
    log.info("Trouvé %d zones géographiques uniques", len(zone_df))

    # Construire la matrice (zones × jours)
    n_zones = len(zone_df)
    n_days = len(dates)
    X = np.zeros((n_zones, n_days))

    zone_keys = zone_df["zone_key"].to_list()
    for j, day in enumerate(dates):
        for i, zkey in enumerate(zone_keys):
            key = (zkey, day)
            X[i, j] = zone_day_data.get(key, 0)

    # Normaliser par zone (occupation rate)
    max_per_zone = X.max(axis=1, keepdims=True)
    max_per_zone[max_per_zone == 0] = 1
    X = X / max_per_zone

    return zone_df, X


def run_zone_manifold(
    start: date,
    end: date,
    location: str = "la",
    k: int = DEFAULT_K,
    n_eigenvectors: int = DEFAULT_N_EIGENVECTORS,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """
    Pipeline principal : construit la matrice zones×jours et applique LBO.

    Returns:
        zone_df: info sur chaque zone
        X: matrice normalisée (zones × jours)
        constituents: masque booléen des zones constituantes
    """
    zone_df, X = build_zone_features_matrix(start, end, location)

    if len(X) == 0:
        return zone_df, X, np.array([])

    # LBO
    X_norm = normalize_features(X)
    W = build_weight_matrix(X_norm, k=k)
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)
    constituents = find_constituent_zones(eigenvectors, W)

    log.info("Zones constituantes trouvées: %d / %d", constituents.sum(), len(zone_df))
    log.info("Valeurs propres: %s", np.round(eigenvalues, 4))

    return zone_df, X, constituents


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifold géospatial sur les zones")
    parser.add_argument("--location", default="la")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    zone_df, X, constituents = run_zone_manifold(
        start, end, args.location, args.k, args.n_eigenvectors
    )

    if len(zone_df) > 0:
        zone_df = zone_df.with_columns(
            pl.Series("is_constituent", constituents.tolist())
        )

        constituents_only = zone_df.filter(pl.col("is_constituent"))
        print(f"\nConstituent zones found: {len(constituents_only)}")

        # Save results to parquet
        out_path = Path(f"data/features/{args.location}_constituent_zones.parquet")
        zone_df.write_parquet(out_path)
        print(f"Saved to: {out_path}")
        print(f"Total zones: {len(zone_df)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
