"""
Phase 2 corrigée — Manifold géospatial + Gravity Score quotidien.

Pipeline complet :
1. Identifier les zones constituantes (Manifold LBO sur zones × jours)
2. Calculer le gravity score quotidien sur ces zones

Usage :
    # Identification des zones constituantes (une seule fois)
    python src/manifold/manifold_pipeline.py --identify --start 2020-01-01 --end 2020-01-31 --location la

    # Scoring quotidien (après identification)
    python src/manifold/manifold_pipeline.py --score --date 2020-02-01 --location la

    # Scoring sur période
    python src/manifold/manifold_pipeline.py --score --start 2020-02-01 --end 2020-02-07 --location la
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
from sklearn.neighbors import NearestNeighbors

from src.clustering.hdbscan_daily import (
    ClusteringConfig,
    cluster_day,
    load_docked_zones_or_none,
)
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_N_EIGENVECTORS = 5


# ============================================================================
# PART 1 — IDENTIFICATION DES ZONES CONSTITUANTES
# ============================================================================


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalisation L2 par ligne."""
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
    eigenvectors: np.ndarray, W: sparse.csr_matrix, n_components: int = 3
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


def build_zone_matrix(
    start: date,
    end: date,
    location: str,
    config: Optional[ClusteringConfig] = None,
):
    """Construit la matrice (zones × jours) pour le manifold."""
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]

    all_zones = {}
    zone_day_data = {}

    dates = []
    d = start
    while d <= end:
        dates.append(d)
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

        if parquet_path.exists():
            cluster_df, _ = cluster_day(parquet_path, config=config)

            if cluster_df is not None and len(cluster_df) > 0:
                for row in cluster_df.iter_rows(named=True):
                    lat_r = round(row["LAT"], 3)
                    lon_r = round(row["LON"], 3)
                    zone_key = f"{lat_r}_{lon_r}"

                    if zone_key not in all_zones:
                        all_zones[zone_key] = {
                            "lat": row["LAT"],
                            "lon": row["LON"],
                            "cluster_label": row["cluster_label"],
                            "cluster_type": row.get("cluster_type", "unknown"),
                            "mmsi": [],
                        }
                    all_zones[zone_key]["mmsi"].append(row["MMSI"])

                    key = (zone_key, d)
                    zone_day_data[key] = zone_day_data.get(key, 0) + 1

        d += timedelta(days=1)

    if len(all_zones) == 0:
        return pl.DataFrame(), np.array([]), []

    # DataFrame des zones
    zone_rows = []
    for zone_key, info in all_zones.items():
        zone_rows.append(
            {
                "zone_key": zone_key,
                "lat": info["lat"],
                "lon": info["lon"],
                "cluster_label": info["cluster_label"],
                "cluster_type": info["cluster_type"],
                "n_vessels_total": len(set(info["mmsi"])),
            }
        )

    zone_df = pl.DataFrame(zone_rows)

    # Matrice (zones × jours)
    n_zones = len(zone_df)
    X = np.zeros((n_zones, len(dates)))

    zone_keys = zone_df["zone_key"].to_list()
    for j, day in enumerate(dates):
        for i, zkey in enumerate(zone_keys):
            key = (zkey, day)
            X[i, j] = zone_day_data.get(key, 0)

    # Normalisation occupation rate
    max_per_zone = X.max(axis=1, keepdims=True)
    max_per_zone[max_per_zone == 0] = 1
    X = X / max_per_zone

    return zone_df, X, dates


def identify_constituent_zones(
    start: date,
    end: date,
    location: str,
    k: int = DEFAULT_K,
    n_eigenvectors: int = DEFAULT_N_EIGENVECTORS,
) -> pl.DataFrame:
    """Point d'entrée : identification des zones constituantes."""
    config = ClusteringConfig(ref_polygon=load_docked_zones_or_none(location))
    log.info(f"Building zone matrix: {start} → {end}")
    zone_df, X, dates = build_zone_matrix(start, end, location, config=config)

    if len(X) == 0:
        raise ValueError("No zones found")

    log.info(f"Found {len(zone_df)} zones across {len(dates)} days")

    # LBO
    X_norm = normalize_features(X)
    W = build_weight_matrix(X_norm, k=k)
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)
    constituents = find_constituent_zones(eigenvectors, W)

    # Ajouter les colonnes
    zone_df = zone_df.with_columns(pl.Series("is_constituent", constituents.tolist()))

    # Ajouter les coordonnées manifold
    for i in range(1, min(n_eigenvectors + 1, eigenvectors.shape[1])):
        zone_df = zone_df.with_columns(
            pl.Series(f"phi_{i}", eigenvectors[:, i].tolist())
        )

    # Sauvegarder
    out_path = Path(f"data/features/{location}_constituent_zones.parquet")
    zone_df.write_parquet(out_path)
    log.info(f"Saved constituent zones to {out_path}")
    log.info(f"Found {constituents.sum()} constituent zones / {len(zone_df)}")
    log.info(f"Eigenvalues: {np.round(eigenvalues, 4)}")

    return zone_df


# ============================================================================
# PART 2 — GRAVITY SCORE QUOTIDIEN
# ============================================================================


def compute_daily_gravity(
    d: date,
    location: str,
    constituent_zones: pl.DataFrame,
    config: Optional[ClusteringConfig] = None,
) -> pl.DataFrame:
    """
    Calcule le gravity score quotidien sur les zones constituantes.

    Gravity score = Σ (navires dans zone constituante × capacité)
    """
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        log.warning(f"Parquet not found: {parquet_path}")
        return pl.DataFrame()

    if config is None:
        config = ClusteringConfig(ref_polygon=load_docked_zones_or_none(location))
    cluster_df, _ = cluster_day(parquet_path, config=config)

    if cluster_df is None or len(cluster_df) == 0:
        return pl.DataFrame()

    # Zones constituantes à surveiller
    const_keys = set(
        constituent_zones.filter(pl.col("is_constituent"))["zone_key"].to_list()
    )

    # Calculer le score par zone
    scores = []
    for row in cluster_df.iter_rows(named=True):
        lat_r = round(row["LAT"], 3)
        lon_r = round(row["LON"], 3)
        zone_key = f"{lat_r}_{lon_r}"

        is_const = zone_key in const_keys
        capacity = (row.get("Length", 0) or 0) * (row.get("Width", 0) or 0)

        scores.append(
            {
                "date": d.isoformat(),
                "zone_key": zone_key,
                "lat": row["LAT"],
                "lon": row["LON"],
                "mmsi": row["MMSI"],
                "cluster_label": row["cluster_label"],
                "cluster_type": row.get("cluster_type", "unknown"),
                "is_constituent": is_const,
                "waiting_capacity": capacity,
            }
        )

    return pl.DataFrame(scores)


def aggregate_gravity_score(daily_df: pl.DataFrame) -> dict:
    """Agège le gravity score global."""
    if len(daily_df) == 0:
        return {}

    const_only = daily_df.filter(pl.col("is_constituent"))

    return {
        "date": daily_df["date"][0],
        "total_vessels": daily_df["mmsi"].n_unique(),
        "constituent_vessels": const_only["mmsi"].n_unique()
        if len(const_only) > 0
        else 0,
        "total_waiting_capacity": daily_df["waiting_capacity"].sum(),
        "constituent_waiting_capacity": const_only["waiting_capacity"].sum()
        if len(const_only) > 0
        else 0,
        "gravity_score": const_only["waiting_capacity"].sum()
        if len(const_only) > 0
        else 0,
    }


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Manifold pipeline: identification + scoring"
    )
    parser.add_argument("--location", default="la")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--identify", action="store_true", help="Identify constituent zones"
    )
    mode.add_argument(
        "--score", action="store_true", help="Compute daily gravity score"
    )
    mode.add_argument(
        "--plot",
        action="store_true",
        help="Plot gravity score time series (requires --start and --end)",
    )

    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    args = parser.parse_args()

    OUTPUT_DIR = Path("outputs/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.identify:
        if not args.start or not args.end:
            parser.error("--start and --end required for --identify")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        zone_df = identify_constituent_zones(
            start, end, args.location, args.k, args.n_eigenvectors
        )

        const = zone_df.filter(pl.col("is_constituent"))
        print(f"\nConstituent zones: {len(const)} / {len(zone_df)}")
        const.write_parquet(f"data/features/{args.location}_constituent_zones.parquet")
        print(f"Saved to: data/features/{args.location}_constituent_zones.parquet")
        return

    if args.score:
        constituent_path = Path(
            f"data/features/{args.location}_constituent_zones.parquet"
        )
        if not constituent_path.exists():
            print(
                f"Run first: python src/manifold/manifold_pipeline.py --identify --start 2020-01-01 --end 2020-01-31 --location {args.location}"
            )
            return

        constituent_zones = pl.read_parquet(constituent_path)
        score_config = ClusteringConfig(ref_polygon=load_docked_zones_or_none(args.location))

        if args.date:
            daily_df = compute_daily_gravity(
                date.fromisoformat(args.date), args.location, constituent_zones,
                config=score_config,
            )
            if len(daily_df) > 0:
                agg = aggregate_gravity_score(daily_df)
                print(f"\n{daily_df['date'][0]}")
                print(f"  Total vessels: {agg['total_vessels']}")
                print(f"  Constituent vessels: {agg['constituent_vessels']}")
                print(f"  Gravity score: {agg['gravity_score']:.1f}")
            return

        if args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)

            all_scores = []
            d = start
            while d <= end:
                daily_df = compute_daily_gravity(d, args.location, constituent_zones,
                                                 config=score_config)
                if len(daily_df) > 0:
                    agg = aggregate_gravity_score(daily_df)
                    all_scores.append(agg)
                d += timedelta(days=1)

            if all_scores:
                scores_df = pl.DataFrame(all_scores)
                out_path = Path(f"data/features/{args.location}_gravity_daily.parquet")
                scores_df.write_parquet(out_path)
                print(f"\nSaved {len(scores_df)} days to {out_path}")

                # Summary
                print(f"\n=== Summary ===")
                print(f"  Avg gravity: {scores_df['gravity_score'].mean():.1f}")
                scores_np = scores_df["gravity_score"].to_numpy()
                max_idx = scores_np.argmax()
                max_date = scores_df["date"].to_list()[max_idx]
                print(f"  Max gravity: {scores_np.max():.1f} ({max_date})")

            # Générer le graphique interactif (Plotly HTML)
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            dates_str = scores_df["date"].to_list()
            gravity = scores_df["gravity_score"].to_numpy()
            vessels = scores_df["total_vessels"].to_numpy()
            const_vessels = scores_df["constituent_vessels"].to_numpy()

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Gravity score
            fig.add_trace(
                go.Scatter(
                    x=dates_str,
                    y=gravity,
                    name="Gravity Score",
                    mode="lines+markers",
                    line=dict(color="#1565C0", width=2),
                    marker=dict(size=6),
                )
            )

            # Vessels overlay
            fig.add_trace(
                go.Scatter(
                    x=dates_str,
                    y=vessels,
                    name="Total Vessels",
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dot"),
                    opacity=0.5,
                ),
                secondary_y=True,
            )

            fig.update_layout(
                title=dict(
                    text=f"{args.location.upper()} - Gravity Score ({args.start} to {args.end})",
                    x=0.5,
                ),
                xaxis_title="Date",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                height=450,
            )

            fig.update_yaxes(
                title_text="Gravity Score", color="#1565C0", secondary_y=False
            )
            fig.update_yaxes(
                title_text="Total Vessels",
                color="gray",
                secondary_y=True,
                showgrid=False,
            )

            plot_path = OUTPUT_DIR / f"{args.location}_gravity_score.html"
            fig.write_html(plot_path)
            print(f"\nSaved interactive plot: {plot_path}")
            return

        parser.error("--date, --start/--end or --plot required")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
