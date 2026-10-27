"""
Phase 2 — Suez Manifold Pipeline.

Adapted for Global Fishing Watch data format.
"""

import argparse
import logging
import sys
from datetime import date, timedelta 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parents[1]))

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_N_EIGENVECTORS = 5

SCRIPT_DIR = Path(
    "C:/Users/hadri/OneDrive/Documents/Documents/Enseignement/UTC/Cours/P26_TZ/Nowcasting_suez"
)
DATA_PATH = SCRIPT_DIR / "data/fusion.csv"
OUTPUT_DIR = Path(
    "C:/Users/hadri/OneDrive/Documents/Documents/Enseignement/UTC/Cours/P26_TZ/Nowcasting_suez/outputs/figures"
)
FEATURES_DIR = Path(
    "C:/Users/hadri/OneDrive/Documents/Documents/Enseignement/UTC/Cours/P26_TZ/Nowcasting_suez/data/features"
)


def load_suez_data():
    """Load Suez data from fusion.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")

    df = pl.read_csv(DATA_PATH)

    # Fix column names
    new_cols = {c: c.strip().lstrip(",").strip() for c in df.columns}
    df = df.rename(new_cols)

    # Extract date from Time Range
    def get_date(tr):
        if tr and len(tr) >= 10:
            return tr[:10]
        return None

    dates = [get_date(tr) for tr in df["Time Range"].to_list()]
    df = df.with_columns(
        [
            pl.Series("date_str", dates),
        ]
    )
    df = df.with_columns(
        [
            pl.col("date_str").cast(pl.Date).alias("date"),
        ]
    )

    # Clean coordinates
    df = df.with_columns(
        [
            pl.col("Lat").cast(pl.Float64).alias("lat"),
            pl.col("Lon").cast(pl.Float64).alias("lon"),
            (pl.col("Lat") * 100).round(1).alias("lat_r"),
            (pl.col("Lon") * 100).round(1).alias("lon_r"),
        ]
    )
    df = df.with_columns(
        [
            pl.concat_str(["lat_r", "lon_r"], separator="_").alias("zone_key"),
        ]
    )

    return df


def build_zone_matrix(start, end, df):
    """Build zone matrix from Suez data."""
    # Filter by date range - convert to string comparison
    start_str = start.isoformat()
    end_str = end.isoformat()
    df = df.filter((pl.col("date_str") >= start_str) & (pl.col("date_str") <= end_str))

    if len(df) == 0:
        return pl.DataFrame(), np.array([]), []

    # Create zone keys using lat/lon rounded (already computed in df)
    # Ensure lat_r exists
    if "lat_r" not in df.columns:
        df = df.with_columns(
            (pl.col("lat") * 100).round(1).alias("lat_r"),
            (pl.col("lon") * 100).round(1).alias("lon_r"),
            pl.concat_str(["lat_r", "lon_r"], separator="_").alias("zone_key"),
        )

    # Zone counts per day
    dates = []
    zone_data = {}
    d = start
    while d <= end:
        dates.append(d)
        day_df = df.filter(pl.col("date_str") == d.isoformat())

        for row in day_df.iter_rows(named=True):
            zkey = row["zone_key"]
            if zkey not in zone_data:
                zone_data[zkey] = {
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "vessels": [],
                    "hours": row.get("Vessel Presence Hours", 0) or 0,
                }
            zone_data[zkey]["vessels"].append(row.get("Vessel ID", ""))

        d += timedelta(days=1)

    # Build DataFrame
    zone_rows = []
    for zkey, info in zone_data.items():
        zone_rows.append(
            {
                "zone_key": zkey,
                "lat": info["lat"],
                "lon": info["lon"],
                "n_vessels": len(info["vessels"]),
                "vessel_hours": info["hours"],
            }
        )

    zone_df = pl.DataFrame(zone_rows)

    # Matrix (zones x days)
    n_zones = len(zone_df)
    X = np.zeros((n_zones, len(dates)))

    zone_keys = zone_df["zone_key"].to_list()
    for j, day in enumerate(dates):
        day_str = day.isoformat()
        day_df = df.filter(pl.col("date_str") == day_str)
        for i, zkey in enumerate(zone_keys):
            day_zone = day_df.filter(pl.col("zone_key") == zkey)
            X[i, j] = len(day_zone)

    # Normalize
    max_per_zone = X.max(axis=1, keepdims=True)
    max_per_zone[max_per_zone == 0] = 1
    X = X / max_per_zone

    return zone_df, X, dates


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


def identify_zones(start, end, k=DEFAULT_K, n_eigenvectors=DEFAULT_N_EIGENVECTORS):
    """Identify constituent zones."""
    log.info(f"Loading Suez data...")
    df = load_suez_data()
    log.info(f"Loaded {len(df)} records")

    log.info(f"Building zone matrix: {start} to {end}")
    zone_df, X, dates = build_zone_matrix(start, end, df)

    if len(X) == 0:
        raise ValueError("No zones found")

    log.info(f"Found {len(zone_df)} zones across {len(dates)} days")

    # LBO
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


def compute_daily_gravity(d, df, constituent_zones):
    """Compute daily gravity for Suez."""
    const_keys = set(
        constituent_zones.filter(pl.col("is_constituent"))["zone_key"].to_list()
    )

    day_df = df.filter(pl.col("date") == d)
    if len(day_df) == 0:
        return None

    # Score: all vessels in constitutive zones contribute
    scores = []
    for row in day_df.iter_rows(named=True):
        zkey = row["zone_key"]
        is_const = zkey in const_keys

        # Capacity = Vessel Presence Hours (proximity to congestion)
        capacity = row.get("Vessel Presence Hours", 1) or 1

        # Severity: all vessels contribute (no waiting vs docked distinction here)
        severity = 1.0 if is_const else 0.0

        scores.append(
            {
                "date": d.isoformat(),
                "zone_key": zkey,
                "lat": row["lat"],
                "lon": row["lon"],
                "vessel_id": row.get("Vessel ID", ""),
                "is_constituent": is_const,
                "severity": severity,
                "capacity": capacity,
            }
        )

    return pl.DataFrame(scores)


def aggregate_gravity(daily_df):
    if daily_df is None or len(daily_df) == 0:
        return {
            "date": d.isoformat(),
            "total_vessels": 0,
            "constituent_vessels": 0,
            "gravity_score": 0,
        }

    const_only = daily_df.filter(pl.col("is_constituent"))

    total_vessels = daily_df["vessel_id"].n_unique()
    constituent_vessels = const_only["vessel_id"].n_unique()
    gravity_score = const_only["capacity"].sum()

    return {
        "date": daily_df["date"][0],
        "total_vessels": total_vessels,
        "constituent_vessels": constituent_vessels,
        "gravity_score": int(gravity_score),
    }


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Suez Manifold Pipeline")
    parser.add_argument("--location", default="suez")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--identify", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--both", action="store_true")

    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.identify and not args.score and not args.both:
        parser.print_help()
        return

    if args.identify or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        zone_df = identify_zones(start, end, args.k, args.n_eigenvectors)

        const = zone_df.filter(pl.col("is_constituent"))
        print(f"\nConstituent zones: {len(const)} / {len(zone_df)}")

        # Save (in suez folder)
        out_path = Path(
            f"../Nowcasting_suez/data/features/{args.location}_constituent_zones.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        zone_df.write_parquet(out_path)
        print(f"Saved to: {out_path}")

    if args.score or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        const_path = Path(
            f"../Nowcasting_suez/data/features/{args.location}_constituent_zones.parquet"
        )
        if not const_path.exists():
            print(f"Run first with --identify or --both")
            return

        constituent_zones = pl.read_parquet(const_path)

        # Load data
        df = load_suez_data()

        all_scores = []
        d = start
        while d <= end:
            daily_df = compute_daily_gravity(d, df, constituent_zones)
            if daily_df is not None:
                agg = aggregate_gravity(daily_df)
                all_scores.append(agg)
            d += timedelta(days=1)

        if not all_scores:
            print("No scores computed")
            return

        scores_df = pl.DataFrame(all_scores)
        out_path = Path(
            f"../Nowcasting_suez/data/features/{args.location}_gravity_daily.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
        vessels = scores_df["constituent_vessels"].to_numpy()

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
                name="Vessels",
                mode="lines",
                line=dict(color="#E65100", width=1, dash="dot"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(text=f"SUEZ - Gravity ({start} to {end})", x=0.5),
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )

        plot_path = OUTPUT_DIR / f"suez_gravity_score.html"
        fig.write_html(plot_path)
        print(f"\nSaved interactive plot: {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
