"""
Phase 2 — Manifold géospatial + Gravity Score.

Usage:
    python src/manifold/manifold_pipeline.py --both --start 2019-01-01 --end 2019-01-31 --location houston
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.clustering.hdbscan_daily import cluster_day
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_N_EIGENVECTORS = 5


# ============================================================================
# MANIFOLD LBO
# ============================================================================


def normalize_features(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def build_weight_matrix(X_norm, k):
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


def compute_eigenvectors(W, n_eigenvectors):
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-10
    A = sparse.diags(degree)
    n_eigs = min(n_eigenvectors + 1, W.shape[0] - 1)
    A_inv = sparse.diags(1.0 / degree)
    L = A_inv @ W
    eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which="LM")
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def find_constituent_zones(eigenvectors, W, n_components=3):
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


# ============================================================================
# ZONE MATRIX
# ============================================================================


def build_zone_matrix(start, end, location):
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
            cluster_df, _ = cluster_day(parquet_path)
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

    n_zones = len(zone_df)
    X = np.zeros((n_zones, len(dates)))

    zone_keys = zone_df["zone_key"].to_list()
    for j, day in enumerate(dates):
        for i, zkey in enumerate(zone_keys):
            key = (zkey, day)
            X[i, j] = zone_day_data.get(key, 0)

    max_per_zone = X.max(axis=1, keepdims=True)
    max_per_zone[max_per_zone == 0] = 1
    X = X / max_per_zone

    return zone_df, X, dates


def identify_constituent_zones(
    start, end, location, k=DEFAULT_K, n_eigenvectors=DEFAULT_N_EIGENVECTORS
):
    log.info(f"Building zone matrix: {start} -> {end}")
    zone_df, X, dates = build_zone_matrix(start, end, location)

    if len(X) == 0:
        raise ValueError("No zones found")

    log.info(f"Found {len(zone_df)} zones across {len(dates)} days")

    X_norm = normalize_features(X)
    W = build_weight_matrix(X_norm, k=k)
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)
    constituents = find_constituent_zones(eigenvectors, W)

    zone_df = zone_df.with_columns(pl.Series("is_constituent", constituents.tolist()))

    for i in range(1, min(n_eigenvectors + 1, eigenvectors.shape[1])):
        zone_df = zone_df.with_columns(
            pl.Series(f"phi_{i}", eigenvectors[:, i].tolist())
        )

    log.info(f"Found {constituents.sum()} constituent zones / {len(zone_df)}")
    log.info(f"Eigenvalues: {np.round(eigenvalues, 4)}")

    return zone_df


# ============================================================================
# GRAVITY SCORE
# ============================================================================


def compute_daily_gravity(d, location, constituent_zones):
    '''
    For a given day, compute gravity score based on the number of vessels in
    constituent zones, weighted by their severity (waiting vs docked) and capacity.
    Formula (very simple): gravity_score = sum(constituent_waiting_capacity)
    '''
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        return pl.DataFrame()

    cluster_df, _ = cluster_day(parquet_path)
    if cluster_df is None or len(cluster_df) == 0:
        return pl.DataFrame()

    const_keys = set(
        constituent_zones.filter(pl.col("is_constituent"))["zone_key"].to_list()
    )

    scores = []
    for row in cluster_df.iter_rows(named=True):
        lat_r = round(row["LAT"], 3)
        lon_r = round(row["LON"], 3)
        zone_key = f"{lat_r}_{lon_r}"

        is_const = zone_key in const_keys
        ctype = row.get("cluster_type", "unknown")
        capacity = (row.get("Length", 0) or 0) * (row.get("Width", 0) or 0)

        # Severity: waiting=1 (bloqué), docked=0 (normal)
        severity = 1.0 if ctype == "waiting" else 0.0

        scores.append(
            {
                "date": d.isoformat(),
                "zone_key": zone_key,
                "lat": row["LAT"],
                "lon": row["LON"],
                "mmsi": row["MMSI"],
                "cluster_type": ctype,
                "is_constituent": is_const,
                "severity": severity,
                "capacity": capacity,
            }
        )

    return pl.DataFrame(scores)


def aggregate_gravity_score(daily_df):
    '''Aggregate daily gravity score by summing the capacity of waiting vessels in constituent zones.
    Returns a dict with date, total vessels, constituent vessels, waiting vessels, total capacity, and gravity score.
    Formula: gravity_score = sum(constituent_waiting_capacity)'''
    if len(daily_df) == 0:
        return {}

    const_only = daily_df.filter(pl.col("is_constituent"))
    const_waiting = const_only.filter(pl.col("severity") > 0)

    if len(const_only) == 0:
        return {
            "date": daily_df["date"][0],
            "total_vessels": daily_df["mmsi"].n_unique(),
            "constituent_vessels": 0,
            "waiting_vessels": 0,
            "total_capacity": 0,
            "gravity_score": 0,
        }

    total_vessels = daily_df["mmsi"].n_unique()
    constituent_vessels = const_only["mmsi"].n_unique()
    waiting_vessels = const_waiting["mmsi"].n_unique() if len(const_waiting) > 0 else 0
    total_capacity = const_only["capacity"].sum()
    gravity_score = const_waiting["capacity"].sum() if len(const_waiting) > 0 else 0

    return {
        "date": daily_df["date"][0],
        "total_vessels": total_vessels,
        "constituent_vessels": constituent_vessels,
        "waiting_vessels": waiting_vessels,
        "total_capacity": int(total_capacity),
        "gravity_score": int(gravity_score),
    }


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Manifold pipeline")
    parser.add_argument("--location", default="la")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--identify", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--both", action="store_true")

    parser.add_argument("--start", help="Start date")
    parser.add_argument("--end", help="End date")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    args = parser.parse_args()

    OUTPUT_DIR = Path("outputs/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.identify and not args.score and not args.both:
        parser.print_help()
        return

    if args.identify or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        zone_df = identify_constituent_zones(
            start, end, args.location, args.k, args.n_eigenvectors
        )

        const = zone_df.filter(pl.col("is_constituent"))
        print(f"\nConstituent zones: {len(const)} / {len(zone_df)}")

        out_path = Path(f"data/features/{args.location}_constituent_zones.parquet")
        zone_df.write_parquet(out_path)
        print(f"Saved to: {out_path}")

    if args.score or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        constituent_path = Path(
            f"data/features/{args.location}_constituent_zones.parquet"
        )
        if not constituent_path.exists():
            print(f"Run first with --identify or --both")
            return

        constituent_zones = pl.read_parquet(constituent_path)

        all_scores = []
        d = start
        while d <= end:
            daily_df = compute_daily_gravity(d, args.location, constituent_zones)
            if len(daily_df) > 0:
                agg = aggregate_gravity_score(daily_df)
                all_scores.append(agg)
            d += timedelta(days=1)

        if not all_scores:
            print("No scores computed")
            return

        scores_df = pl.DataFrame(all_scores)
        out_path = Path(f"data/features/{args.location}_gravity_daily.parquet")
        scores_df.write_parquet(out_path)
        print(f"\nSaved {len(scores_df)} days to {out_path}")

        print(f"\n=== Summary ===")
        print(f"  Avg gravity: {scores_df['gravity_score'].mean():.1f}")
        scores_np = scores_df["gravity_score"].to_numpy()
        max_idx = scores_np.argmax()
        max_date = scores_df["date"].to_list()[max_idx]
        print(f"  Max gravity: {scores_np.max():.1f} ({max_date})")

        # Plot
        dates_str = scores_df["date"].to_list()
        gravity = scores_df["gravity_score"].to_numpy()
        vessels = scores_df["waiting_vessels"].to_numpy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

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

        fig.add_trace(
            go.Scatter(
                x=dates_str,
                y=vessels,
                name="Waiting Vessels",
                mode="lines",
                line=dict(color="#E65100", width=1, dash="dot"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(
                text=f"{args.location.upper()} - Gravity ({start} to {end})", x=0.5
            ),
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            height=450,
        )

        fig.update_yaxes(title_text="Gravity Score", color="#1565C0", secondary_y=False)
        fig.update_yaxes(
            title_text="Waiting Vessels",
            color="#E65100",
            secondary_y=True,
            showgrid=False,
        )

        plot_path = OUTPUT_DIR / f"{args.location}_gravity_score.html"
        fig.write_html(plot_path)
        print(f"\nSaved interactive plot: {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
