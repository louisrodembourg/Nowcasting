"""
Phase 2 — Manifold géospatial + Gravity Score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE RECOMMANDÉ — Baseline fixe + Scoring (2 étapes séparées)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Étape 1 — Apprendre la topologie sur une période de référence calme :

    python src/manifold/manifold_pipeline.py --identify `
        --baseline-start 2017-01-01 --baseline-end 2017-01-30 `
        --start 2017-01-01 --end 2017-01-30 `
        --location houston

    → Sauvegarde data/features/houston_constituent_zones.parquet

  Étape 2 — Scorer la période de crise avec la topologie figée :

    python src/manifold/manifold_pipeline.py --score `
        --start 2020-01-01 --end 2020-12-31 `
        --location la

    → Sauvegarde data/features/houston_gravity_daily.parquet
      + outputs/figures/houston_gravity_score.html

  Raccourci (étapes 1+2 en une seule commande) :

    python src/manifold/manifold_pipeline.py --both `
        --baseline-start 2017-01-01 --baseline-end 2017-12-31 `
        --start 2017-01-01 --end 2017-12-31 `
        --location houston

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE ROLLING — Fenêtre glissante de 3 mois (expérimental)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━═════════════════════════════════════

  Sans normalisation baseline (dynamique par fenêtre) :

    python src/manifold/manifold_pipeline.py --rolling `
        --start 2017-01-01 --end 2017-12-31 `
        --location houston

  Avec normalisation baseline (valeurs > 1.0 = disruption) :

    python src/manifold/manifold_pipeline.py --rolling `
        --start 2017-01-01 --end 2017-12-31 `
        --baseline-start 2015-01-01 --baseline-end 2016-12-31 `
        --location houston

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONS COMMUNES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --location   houston | la | suez   (défaut : la)
  --k          voisins KNN pour le graphe LBO     (défaut : 5)
  --n-eigenvectors  vecteurs propres à conserver  (défaut : 5)
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

from src.clustering.hdbscan_daily import cluster_day, ClusteringConfig, load_waiting_zones
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_N_EIGENVECTORS = 5


def _load_config(location: str) -> ClusteringConfig:
    """Load ClusteringConfig with waiting-zone polygons if available."""
    try:
        polygons = load_waiting_zones(location)
        log.info("Loaded %d waiting zone polygons for %s", len(polygons), location)
        return ClusteringConfig(waiting_allowed_polygons=polygons)
    except FileNotFoundError:
        log.warning(
            "No waiting zones file for %s — clusters will all default to 'docked'", location
        )
        return ClusteringConfig()


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
    
    # Sigma local adaptatif (Zelnik-Manor & Perona) :
    # Pour chaque point i, sigma_local[i] = distance au k-ème voisin
    sigma_local = distances[:, -1]  # Dernière colonne = distance au k-ème voisin
    sigma_local = np.maximum(sigma_local, 1e-10)  # Éviter les zéros
    
    rows = np.repeat(np.arange(n), k_safe)
    cols = indices.ravel()
    
    # Noyau Gaussien avec sigmas locaux : exp(-(d^2) / (sigma_i * sigma_j))
    sigma_product = sigma_local[rows] * sigma_local[cols]
    vals = np.exp(-(distances.ravel() ** 2) / sigma_product)
    
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


def build_zone_matrix(start, end, location, config=None, zone_max_baseline=None):
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
                # Aggregate to HDBSCAN cluster centroids.
                # Noise points (label -1) have no stable location and are excluded.
                valid = cluster_df.filter(pl.col("cluster_label") >= 0)
                if len(valid) == 0:
                    d += timedelta(days=1)
                    continue

                centroids = valid.group_by("cluster_label").agg([
                    pl.col("LAT").median().alias("LAT"),
                    pl.col("LON").median().alias("LON"),
                    pl.col("cluster_type").first().alias("cluster_type"),
                    pl.col("MMSI").n_unique().alias("n_vessels"),
                ])

                for row in centroids.iter_rows(named=True):
                    lat_r = round(row["LAT"], 3)
                    lon_r = round(row["LON"], 3)
                    zone_key = f"{lat_r}_{lon_r}"

                    if zone_key not in all_zones:
                        all_zones[zone_key] = {
                            "lat": row["LAT"],
                            "lon": row["LON"],
                            "cluster_type": row["cluster_type"] or "unknown",
                            "n_vessels_seen": 0,
                        }
                    all_zones[zone_key]["n_vessels_seen"] += row["n_vessels"]

                    key = (zone_key, d)
                    zone_day_data[key] = zone_day_data.get(key, 0) + row["n_vessels"]

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
                "cluster_type": info["cluster_type"],
                "n_vessels_total": info["n_vessels_seen"],
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

    if zone_max_baseline is None:
        # Default: normalize each zone by its own max within this window.
        max_per_zone = X.max(axis=1, keepdims=True)
        max_per_zone[max_per_zone == 0] = 1
        X = X / max_per_zone
    else:
        # Baseline normalization: divide by the historical max for each zone.
        # Values > 1.0 are intentional — they signal congestion above baseline.
        # Zones absent from the baseline fall back to their within-window max.
        for i, zkey in enumerate(zone_keys):
            denom = zone_max_baseline.get(zkey, None)
            if denom is None or denom <= 0:
                denom = float(X[i].max()) or 1.0
            X[i, :] /= denom

    return zone_df, X, dates


def compute_baseline_maxima(
    start: date, end: date, location: str, config=None
) -> dict[str, float]:
    """
    Single pass over a baseline period to compute the maximum vessel count
    ever observed per zone. Returns {zone_key: max_vessels}.

    This dict is passed as zone_max_baseline to build_zone_matrix so that
    all rolling-window matrices are normalized against the same historical
    reference, making values > 1.0 a direct signal of disruption.
    """
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]

    zone_maxima: dict[str, float] = {}

    d = start
    while d <= end:
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"
        if parquet_path.exists():
            cluster_df, _ = cluster_day(parquet_path, config=config)
            if cluster_df is not None and len(cluster_df) > 0:
                valid = cluster_df.filter(pl.col("cluster_label") >= 0)
                if len(valid) > 0:
                    centroids = valid.group_by("cluster_label").agg([
                        pl.col("LAT").median().alias("LAT"),
                        pl.col("LON").median().alias("LON"),
                        pl.col("MMSI").n_unique().alias("n_vessels"),
                    ])
                    for row in centroids.iter_rows(named=True):
                        zkey = f"{round(row['LAT'], 3)}_{round(row['LON'], 3)}"
                        n = float(row["n_vessels"])
                        if zone_maxima.get(zkey, 0.0) < n:
                            zone_maxima[zkey] = n
        d += timedelta(days=1)

    log.info(
        "Baseline maxima computed over %s -> %s : %d zones",
        start, end, len(zone_maxima),
    )
    return zone_maxima


def identify_constituent_zones(
    start, end, location, k=DEFAULT_K, n_eigenvectors=DEFAULT_N_EIGENVECTORS,
    config=None, zone_max_baseline=None
):
    log.info(f"Building zone matrix: {start} -> {end}")
    zone_df, X, dates = build_zone_matrix(
        start, end, location, config=config, zone_max_baseline=zone_max_baseline
    )

    if len(X) == 0:
        raise ValueError("No zones found")

    log.info(f"Found {len(zone_df)} zones across {len(dates)} days")

    date_cols = {d.isoformat(): pl.Series(d.isoformat(), X[:, j].tolist()) for j, d in enumerate(dates)}
    zone_matrix_df = zone_df.select(["zone_key", "lat", "lon", "cluster_type"]).with_columns(
        list(date_cols.values())
    )
    matrix_path = Path(f"data/features/{location}_zone_matrix.parquet")
    zone_matrix_df.write_parquet(matrix_path)
    csv_path = Path(f"data/features/{location}_zone_matrix.csv")
    zone_matrix_df.write_csv(csv_path)
    log.info(f"Zone matrix saved: {matrix_path} + {csv_path}  ({len(zone_df)} zones x {len(dates)} days)")

    n_preview = min(8, len(zone_df))
    n_days_preview = min(7, len(dates))
    preview_dates = [d.isoformat()[5:] for d in dates[:n_days_preview]]
    header = f"{'zone_key':<22} {'type':<8} " + "  ".join(f"{d:>5}" for d in preview_dates)
    print(f"\n--- Zone matrix preview ({n_preview}/{len(zone_df)} zones, {n_days_preview}/{len(dates)} days) ---")
    print(header)
    print("-" * len(header))
    zone_keys = zone_df["zone_key"].to_list()
    types = zone_df["cluster_type"].to_list()
    for i in range(n_preview):
        vals = "  ".join(f"{X[i, j]:>5.3f}" for j in range(n_days_preview))
        print(f"{zone_keys[i]:<22} {(types[i] or '?'):<8} {vals}")
    print(f"  ... ({len(zone_df) - n_preview} more zones)\n")

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


def compute_daily_gravity(d, location, constituent_zones, config=None, spatial_tol_deg=0.005):
    '''
    For a given day, compute gravity score based on vessels in constituent zones.
    Formula: gravity_score = sum(capacity * temps_attente * manifold_deviation)

    Zone matching uses NearestNeighbors with spatial_tol_deg tolerance (~500 m at 0.005°)
    to handle centroid drift from tides and wind rather than requiring exact key matches.
    One row per HDBSCAN cluster (not per vessel episode).
    '''
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        return pl.DataFrame()

    cluster_df, prepared_df = cluster_day(parquet_path, config=config)
    if cluster_df is None or len(cluster_df) == 0:
        return pl.DataFrame()

    # Compute episode_duration_hours: span between first and last static AIS
    # message for each (MMSI, traj_id) episode, derived from raw timestamps.
    if prepared_df is not None and "BaseDateTime" in prepared_df.columns:
        static_df = prepared_df.filter(pl.col("SOG_corr") < 1.0)
        episode_durations = static_df.group_by(["MMSI", "traj_id"]).agg(
            (
                (pl.col("BaseDateTime").max() - pl.col("BaseDateTime").min())
                .dt.total_seconds()
                / 3600.0
            ).alias("episode_duration_hours")
        )
        cluster_df = cluster_df.join(episode_durations, on=["MMSI", "traj_id"], how="left")
        cluster_df = cluster_df.with_columns(
            pl.col("episode_duration_hours").fill_null(0.0)
        )
    else:
        cluster_df = cluster_df.with_columns(pl.lit(0.0).alias("episode_duration_hours"))

    # Build NearestNeighbors index from constituent zone centroids.
    # spatial_tol_deg ≈ 500 m at 0.005° — absorbs centroid drift from tides/wind
    # without requiring exact coordinate matches.
    const_df = constituent_zones.filter(pl.col("is_constituent"))
    if len(const_df) == 0:
        return pl.DataFrame()

    const_coords = const_df.select(["lat", "lon"]).to_numpy()
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(const_coords)
    const_phi1 = (
        const_df["phi_1"].to_numpy()
        if "phi_1" in const_df.columns
        else np.zeros(len(const_df))
    )
    const_keys = const_df["zone_key"].to_list()

    # Aggregate to HDBSCAN cluster centroids — one row per cluster.
    valid = cluster_df.filter(pl.col("cluster_label") >= 0)
    if len(valid) == 0:
        return pl.DataFrame()

    cluster_agg = valid.group_by("cluster_label").agg([
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("cluster_type").first().alias("cluster_type"),
        pl.col("MMSI").n_unique().alias("n_vessels"),
        (pl.col("Length").fill_null(0.0) * pl.col("Width").fill_null(0.0))
        .sum()
        .alias("total_capacity"),
        pl.col("episode_duration_hours").mean().alias("temps_attente"),
    ])

    scores = []
    for row in cluster_agg.iter_rows(named=True):
        cluster_coord = np.array([[row["LAT"], row["LON"]]])
        dists, idxs = nn.kneighbors(cluster_coord)
        dist = float(dists[0, 0])
        nearest_idx = int(idxs[0, 0])

        is_const = dist <= spatial_tol_deg
        matched_zone_key = (
            const_keys[nearest_idx]
            if is_const
            else f"{round(row['LAT'], 3)}_{round(row['LON'], 3)}"
        )
        ctype = row["cluster_type"] or "docked"
        capacity = row["total_capacity"] or 0.0
        temps_attente = row["temps_attente"] or 0.0
        manifold_deviation = float(abs(const_phi1[nearest_idx])) if is_const else 0.0

        if is_const and ctype == "waiting":
            severity = capacity * temps_attente * manifold_deviation
        else:
            severity = 0.0

        scores.append(
            {
                "date": d.isoformat(),
                "zone_key": matched_zone_key,
                "lat": row["LAT"],
                "lon": row["LON"],
                "n_vessels": row["n_vessels"],
                "cluster_type": ctype,
                "is_constituent": is_const,
                "severity": severity,
                "capacity": capacity,
                "temps_attente": temps_attente,
                "manifold_deviation": manifold_deviation,
                "dist_to_constituent": dist,
            }
        )

    return pl.DataFrame(scores)


def aggregate_gravity_score(daily_df):
    '''Aggregate daily gravity score by summing the continuous severity (capacity * temps_attente * manifold_deviation) 
    of waiting vessels in constituent zones.'''
    if len(daily_df) == 0:
        return {}

    const_only = daily_df.filter(pl.col("is_constituent"))
    const_waiting = const_only.filter(pl.col("severity") > 0)

    if len(const_only) == 0:
        return {
            "date": daily_df["date"][0],
            "total_vessels": int(daily_df["n_vessels"].sum()),
            "constituent_vessels": 0,
            "waiting_vessels": 0,
            "total_capacity": 0,
            "gravity_score": 0,
        }

    total_vessels = int(daily_df["n_vessels"].sum())
    constituent_vessels = int(const_only["n_vessels"].sum())
    waiting_vessels = int(const_waiting["n_vessels"].sum()) if len(const_waiting) > 0 else 0
    total_capacity = const_only["capacity"].sum()
    gravity_score = const_waiting["severity"].sum() if len(const_waiting) > 0 else 0

    return {
        "date": daily_df["date"][0],
        "total_vessels": total_vessels,
        "constituent_vessels": constituent_vessels,
        "waiting_vessels": waiting_vessels,
        "total_capacity": int(total_capacity),
        "gravity_score": float(gravity_score),
    }


# ============================================================================
# DAY-LEVEL MANIFOLD
# ============================================================================

DAILY_FEATURE_COLS = [
    "vessel_count", "SOG_mean", "SOG_std", "SOG_median",
    "utilization_rate_rho", "hdbscan_cluster_count", "hdbscan_noise_ratio",
    "membership_score_mean", "membership_score_std",
    "draft_mean", "draft_std", "blocked_capacity", "tanker_ratio",
]


def compute_day_manifold(
    location: str,
    start: date | None = None,
    end: date | None = None,
    k: int = DEFAULT_K,
    n_eigenvectors: int = DEFAULT_N_EIGENVECTORS,
) -> pl.DataFrame:
    """
    Apply LBO to the daily feature matrix: each day is a node, the 13 AIS
    features are its coordinates.  Saves {location}_manifold.parquet with
    columns: date, <13 features>, phi_1 .. phi_k, is_characteristic.

    is_characteristic flags days that are local extrema of phi_1 in the
    KNN graph — the same criterion used for zones.
    """
    features_path = Path(f"data/features/{location}_daily_features.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Daily features not found: {features_path}")

    df = pl.read_parquet(features_path)

    if df["date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("date").str.to_date())

    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)

    if len(df) < k + 2:
        raise ValueError(f"Not enough days ({len(df)}) for k={k}")

    # Fill nulls with column median so LBO is not distorted by missing days
    for col in DAILY_FEATURE_COLS:
        if col in df.columns:
            med = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(med if med is not None else 0.0))

    X = df.select(DAILY_FEATURE_COLS).to_numpy().astype(float)
    X_norm = normalize_features(X)

    W = build_weight_matrix(X_norm, k=k)
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)

    # is_characteristic: days that are local extrema of phi_1 in the KNN graph
    is_char = find_constituent_zones(eigenvectors, W, n_components=1)

    for i in range(1, min(n_eigenvectors + 1, eigenvectors.shape[1])):
        df = df.with_columns(pl.Series(f"phi_{i}", eigenvectors[:, i].tolist()))

    df = df.with_columns(pl.Series("is_characteristic", is_char.tolist()))

    log.info(
        "Day manifold: %d days, %d characteristic, eigenvalues=%s",
        len(df), int(is_char.sum()), np.round(eigenvalues[:5], 4),
    )
    return df


# ============================================================================
# ROLLING MANIFOLD
# ============================================================================


def get_month_range(year, month):
    """Retourne le premier et dernier jour d'un mois donné."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def rolling_manifold_pipeline(
    start,
    end,
    location,
    k=DEFAULT_K,
    n_eigenvectors=DEFAULT_N_EIGENVECTORS,
    baseline_start=None,
    baseline_end=None,
):
    """
    Rolling Manifold avec fenêtre glissante de 3 mois.
    Pour chaque mois M, identifie les zones constituantes sur M-3 à M-1,
    puis évalue le gravity_score quotidien sur M.

    baseline_start / baseline_end : période de référence (ex. 2018) utilisée
    pour calculer les maxima d'occupation par zone. La normalisation de chaque
    fenêtre glissante est alors faite contre ce référentiel statique, de sorte
    qu'une valeur > 1.0 dans X signale une congestion au-dessus de la baseline.
    Si non fournis, normalisation dynamique par fenêtre (comportement par défaut).
    """
    log.info(f"Rolling Manifold pipeline: {start} -> {end}")
    config = _load_config(location)

    # --- Normalisation baseline (optionnelle) --------------------------------
    zone_max_baseline = None
    if baseline_start is not None and baseline_end is not None:
        log.info(f"Computing baseline maxima: {baseline_start} -> {baseline_end}")
        zone_max_baseline = compute_baseline_maxima(
            baseline_start, baseline_end, location, config=config
        )
    
    all_scores = []
    
    # Déterminer tous les mois de calcul entre start et end
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    
    months = []
    while current <= end_month:
        months.append((current.year, current.month))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    
    for year, month in months:
        log.info(f"\n=== Processing month {year}-{month:02d} ===")
        
        # Mois de calcul M
        month_start, month_end = get_month_range(year, month)
        
        # Fenêtre historique : M-3 à M-1 (3 mois précédents)
        if month <= 3:
            hist_start_year = year - 1
            hist_start_month = 12 + month - 3
        else:
            hist_start_year = year
            hist_start_month = month - 3
        
        hist_start, _ = get_month_range(hist_start_year, hist_start_month)
        
        if month == 1:
            hist_end_year = year - 1
            hist_end_month = 12
        else:
            hist_end_year = year
            hist_end_month = month - 1
        
        _, hist_end = get_month_range(hist_end_year, hist_end_month)
        
        # S'assurer que la fenêtre historique est disponible
        if hist_start < start:
            log.warning(f"Insufficient historical data for {year}-{month:02d}, skipping")
            continue
        
        log.info(f"  Historical window: {hist_start} -> {hist_end}")
        log.info(f"  Scoring period: {month_start} -> {month_end}")
        
        # Calculer les zones constituantes sur la fenêtre historique
        try:
            constituent_zones = identify_constituent_zones(
                hist_start, hist_end, location, k, n_eigenvectors,
                config=config, zone_max_baseline=zone_max_baseline
            )
        except ValueError as e:
            log.warning(f"Could not identify constituent zones for {year}-{month:02d}: {e}")
            continue

        # Calculer les scores quotidiens pour le mois M
        d = max(month_start, start)
        month_end_bounded = min(month_end, end)

        while d <= month_end_bounded:
            daily_df = compute_daily_gravity(d, location, constituent_zones, config=config)
            if len(daily_df) > 0:
                agg = aggregate_gravity_score(daily_df)
                all_scores.append(agg)
            d += timedelta(days=1)
    
    return all_scores


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
    mode.add_argument("--rolling", action="store_true", help="Use rolling manifold approach")
    mode.add_argument("--day-manifold", action="store_true",
                      help="LBO on daily features (one node per day) → saves {location}_manifold.parquet")

    parser.add_argument("--start", help="Start date")
    parser.add_argument("--end", help="End date")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    parser.add_argument("--baseline-start", help="Start of baseline period (YYYY-MM-DD) for rolling normalization")
    parser.add_argument("--baseline-end", help="End of baseline period (YYYY-MM-DD) for rolling normalization")
    args = parser.parse_args()

    OUTPUT_DIR = Path("outputs/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.identify and not args.score and not args.both and not args.rolling \
            and not args.day_manifold:
        parser.print_help()
        return

    if args.day_manifold:
        start = date.fromisoformat(args.start) if args.start else None
        end   = date.fromisoformat(args.end)   if args.end   else None
        manifold_df = compute_day_manifold(
            args.location, start, end, args.k, args.n_eigenvectors
        )
        out_path = Path(f"data/features/{args.location}_manifold.parquet")
        manifold_df.write_parquet(out_path)
        n_char = int(manifold_df["is_characteristic"].sum())
        print(f"Saved {len(manifold_df)} days ({n_char} characteristic) -> {out_path}")
        return

    if args.rolling:
        if not args.start or not args.end:
            parser.error("--start and --end required for rolling manifold")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        
        baseline_start = date.fromisoformat(args.baseline_start) if args.baseline_start else None
        baseline_end = date.fromisoformat(args.baseline_end) if args.baseline_end else None

        all_scores = rolling_manifold_pipeline(
            start, end, args.location, args.k, args.n_eigenvectors,
            baseline_start=baseline_start, baseline_end=baseline_end,
        )
        
        if not all_scores:
            print("No scores computed")
            return
        
        scores_df = pl.DataFrame(all_scores)
        out_path = Path(f"data/features/{args.location}_gravity_rolling.parquet")
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
                name="Gravity Score (Rolling)",
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
                text=f"{args.location.upper()} - Rolling Gravity ({start} to {end})", x=0.5
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

        plot_path = OUTPUT_DIR / f"{args.location}_gravity_rolling.html"
        fig.write_html(plot_path)
        print(f"\nSaved interactive plot: {plot_path}")
        
        return

    config = _load_config(args.location)

    if args.identify or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        # Static baseline topology: use --baseline-start/--baseline-end when provided
        # so the manifold is fitted on a calm reference period and then frozen.
        # Falls back to --start/--end when no baseline is specified.
        id_start = date.fromisoformat(args.baseline_start) if args.baseline_start else start
        id_end = date.fromisoformat(args.baseline_end) if args.baseline_end else end
        if args.baseline_start:
            log.info("Identifying constituent zones on BASELINE: %s -> %s", id_start, id_end)

        zone_df = identify_constituent_zones(
            id_start, id_end, args.location, args.k, args.n_eigenvectors, config=config
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
            daily_df = compute_daily_gravity(d, args.location, constituent_zones, config=config)
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