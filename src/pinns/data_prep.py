"""
Phase 3 — Data preparation for PINN: project vessels onto channel axis,
bin spatially, build (x, t, ρ, v) tensors.
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import torch

from src.clustering.hdbscan_daily import cluster_day, ClusteringConfig, load_waiting_zones
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

# Absolute time reference: epoch = 2019-01-01 (earliest data available).
# Normalize by T_DAYS_MAX = 2000 to cover 2019-01-01 to ~2024-06-01 within [0, 1].
EPOCH_DATE = date(2019, 1, 1)
T_DAYS_MAX = 2000

# Channel axes as (lat, lon) waypoints defining the centerline.
# Distance is cumulative along the polyline.
CHANNEL_AXES = {
    "houston": [
        (29.7242780, -95.2156546),
        (29.7443919, -95.1924909),
        (29.7474086, -95.1646943),
        (29.7353411, -95.1403723),
        (29.7433863, -95.1102594),
        (29.7624910, -95.0859375),
        (29.7423807, -95.0488754),
        (29.7172372, -95.0187625),
        (29.6951056, -94.9967569),
        (29.6659248, -94.9770677),
        (29.6236480, -94.9643276),
        (29.5783316, -94.9342147),
        (29.5471019, -94.9133673),
        (29.5229174, -94.8878871),
        (29.4785641, -94.8705143),
        (29.4422606, -94.8450341),
        (29.3928267, -94.8137630),
        (29.3585114, -94.7894411),
        (29.3474070, -94.7743846),
        (29.3463974, -94.7361644),
        (29.3322626, -94.6944696),
        (29.3120665, -94.6736222),
        (29.2969169, -94.6446674),
        (29.2666108, -94.6052890),
        (29.2267520, -94.5628413),
    ],
    "la": [
        (33.7376958, -118.2255703),
        (33.7284130, -118.2204418),
        (33.7161181, -118.2147099),
        (33.7003077, -118.2074696),
        (33.6900168, -118.2023410),
        (33.6764611, -118.1935924),
        (33.6618988, -118.2008327),
        (33.6493196, -118.2082597),
        (33.6390226, -118.2320924),
        (33.6357575, -118.2625619),
        (33.6377668, -118.2942382),
        (33.6493196, -118.3271211),
        (33.6611219, -118.3651327),
        (33.6724205, -118.4088761),
        (33.6854748, -118.4493010),
        (33.6935073, -118.4710219),
    ]
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def project_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return t, dist, proj


def project_to_axis(lat, lon, waypoints):
    pts = np.array([(lat, lon)])
    seg_starts = np.array(waypoints[:-1])
    seg_ends = np.array(waypoints[1:])
    cum_dist = 0.0
    min_dist = float("inf")
    best_x = 0.0
    for i in range(len(seg_starts)):
        a, b = seg_starts[i], seg_ends[i]
        seg_len = haversine_km(a[0], a[1], b[0], b[1])
        t, cross_dist, proj = project_point_to_segment(pts[0], a, b)
        along = t * seg_len
        x = cum_dist + along
        if cross_dist < min_dist:
            min_dist = cross_dist
            best_x = x
        cum_dist += seg_len
    return best_x, min_dist


def compute_channel_length(waypoints):
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += haversine_km(
            waypoints[i][0],
            waypoints[i][1],
            waypoints[i + 1][0],
            waypoints[i + 1][1],
        )
    return total


def _project_vectorized(lats, lons, waypoints):
    """Project many points onto channel axis using fully vectorized numpy.
    Returns (x_along_km, cross_dist_km) for each point.
    """
    pts = np.column_stack([lats, lons])  # (N, 2)
    n = len(pts)
    seg_starts = np.array(waypoints[:-1])  # (S, 2)
    seg_ends = np.array(waypoints[1:])  # (S, 2)
    s = len(seg_starts)

    # Segment vectors
    vec = seg_ends - seg_starts  # (S, 2)
    vec_len_sq = np.sum(vec**2, axis=1)  # (S,)
    seg_lens = np.sqrt(vec_len_sq)

    # Haversine length for each segment
    haversine_lens = np.array(
        [
            haversine_km(
                seg_starts[i, 0], seg_starts[i, 1], seg_ends[i, 0], seg_ends[i, 1]
            )
            for i in range(s)
        ]
    )
    cum_dists = np.insert(np.cumsum(haversine_lens), 0, 0.0)

    # For each point and each segment, compute t and cross distance
    # P: (N, 2)  A: (S, 2)  →  AP: (N, S, 2)
    AP = pts[:, None, :] - seg_starts[None, :, :]  # (N, S, 2)
    AB = vec[None, :, :]  # (1, S, 2)

    # t = dot(AP, AB) / dot(AB, AB)
    dot_AP_AB = np.sum(AP * AB, axis=2)  # (N, S)
    dot_AB_AB = vec_len_sq[None, :]  # (1, S)
    # Avoid division by zero
    dot_AB_AB = np.where(dot_AB_AB < 1e-12, 1e-12, dot_AB_AB)
    t = np.clip(dot_AP_AB / dot_AB_AB, 0.0, 1.0)  # (N, S)

    # Projection point: A + t * AB  → (N, S, 2)
    proj = seg_starts[None, :, :] + t[:, :, None] * vec[None, :, :]

    # Cross distance: norm(P - proj) along last axis
    cross_dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)  # (N, S)

    # Along-channel distance for each segment
    along = cum_dists[None, :-1] + t * haversine_lens[None, :]  # (N, S)

    # Pick segment with minimum cross distance for each point
    best_seg = np.argmin(cross_dist, axis=1)  # (N,)
    x_vals = along[np.arange(n), best_seg]
    min_cross = cross_dist[np.arange(n), best_seg]

    return x_vals, min_cross


def compute_raw_sog_per_bin(parquet_path, waypoints, bin_edges, n_bins, channel_len):
    """Load raw parquet and compute mean SOG per spatial bin (vectorized)."""
    df = pl.read_parquet(parquet_path)
    if len(df) == 0:
        return np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)

    lats = df["LAT"].to_numpy()
    lons = df["LON"].to_numpy()
    sogs = df["SOG"].to_numpy()

    # Vectorized projection of all points at once
    x_vals, _ = _project_vectorized(lats, lons, waypoints)

    x_clipped = np.clip(x_vals, 0, channel_len - 1e-6)
    bin_idxs = np.searchsorted(bin_edges, x_clipped, side="right") - 1
    bin_idxs = np.clip(bin_idxs, 0, n_bins - 1)

    sog_sum   = np.bincount(bin_idxs, weights=sogs.astype(np.float64), minlength=n_bins)
    sog_count = np.bincount(bin_idxs,                                   minlength=n_bins)
    sog_grid  = np.where(sog_count > 0, sog_sum / sog_count, 0.0).astype(np.float32)

    return sog_grid, sog_count.astype(np.float32)


def build_rho_v_tensors(
    start,
    end,
    location,
    dx_km=2.0,
    constituent_path=None,
    use_waiting_only=True,
    use_all_v=False,
    use_raw_velocity=False,
):
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]
    waypoints = CHANNEL_AXES.get(location)
    if waypoints is None:
        raise ValueError(f"No channel axis for {location}")
    channel_len = compute_channel_length(waypoints)
    n_bins = max(1, int(np.ceil(channel_len / dx_km)))
    bin_edges = np.linspace(0, channel_len, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if constituent_path:
        const_zones = set(
            pl.read_parquet(constituent_path)
            .filter(pl.col("is_constituent"))["zone_key"]
            .to_list()
        )
    else:
        const_zones = None

    dates_list = []
    d = start
    while d <= end:
        dates_list.append(d)
        d += timedelta(days=1)

    n_days = len(dates_list)
    day_offsets = [(day - EPOCH_DATE).days for day in dates_list]
    rho_grid = np.zeros((n_bins, n_days), dtype=np.float32)
    v_grid = np.zeros((n_bins, n_days), dtype=np.float32)
    count_rho = np.zeros((n_bins, n_days), dtype=np.int32)
    count_v = np.zeros((n_bins, n_days), dtype=np.int32)

    # Load waiting-zone polygons so that _classify_clusters can correctly
    # distinguish waiting vessels (anchored, queuing) from docked ones.
    # Without this, the default ClusteringConfig marks everything as "docked".
    try:
        waiting_polygons = load_waiting_zones(location)
        cluster_config = ClusteringConfig(waiting_allowed_polygons=waiting_polygons)
    except (FileNotFoundError, Exception):
        cluster_config = ClusteringConfig()

    for j, day in enumerate(dates_list):
        parquet_path = parquet_dir / f"{prefix}_{day.strftime('%Y_%m_%d')}.parquet"
        if not parquet_path.exists():
            continue

        cluster_df, _ = cluster_day(parquet_path, config=cluster_config)
        if cluster_df is None or len(cluster_df) == 0:
            continue

        for row in cluster_df.iter_rows(named=True):
            lat_r = round(row["LAT"], 3)
            lon_r = round(row["LON"], 3)

            if const_zones is not None:
                zone_key = f"{lat_r}_{lon_r}"
                if zone_key not in const_zones:
                    continue

            ctype = row.get("cluster_type", "unknown")

            # Density: count waiting vessels only
            if ctype == "waiting":
                x, _ = project_to_axis(row["LAT"], row["LON"], waypoints)
                x = np.clip(x, 0, channel_len - 1e-6)
                bin_idx = int(np.searchsorted(bin_edges, x, side="right") - 1)
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                rho_grid[bin_idx, j] += 1.0
                count_rho[bin_idx, j] += 1

        # Velocity: compute from raw parquet (moving vessels) if requested
        if use_raw_velocity:
            sog_grid, sog_count = compute_raw_sog_per_bin(
                parquet_path, waypoints, bin_edges, n_bins, channel_len
            )
            for i in range(n_bins):
                if sog_count[i] > 0:
                    v_grid[i, j] = sog_grid[i]
                    count_v[i, j] = sog_count[i]

    if not use_raw_velocity:
        for j in range(n_days):
            for i in range(n_bins):
                if count_v[i, j] > 0:
                    v_grid[i, j] /= count_v[i, j]

    rho_density = rho_grid / dx_km

    x_flat, t_flat, rho_flat, v_flat = [], [], [], []
    for j in range(n_days):
        for i in range(n_bins):
            if count_rho[i, j] > 0:
                x_flat.append(bin_centers[i])
                days_since_epoch = (dates_list[j] - EPOCH_DATE).days
                t_flat.append(float(days_since_epoch))
                rho_flat.append(float(rho_density[i, j]))
                v_flat.append(float(v_grid[i, j]))

    x_arr = np.array(x_flat, dtype=np.float32)
    t_arr = np.array(t_flat, dtype=np.float32)
    rho_arr = np.array(rho_flat, dtype=np.float32)
    v_arr = np.array(v_flat, dtype=np.float32)

    rho_min, rho_max = rho_arr.min(), rho_arr.max()
    v_min, v_max = v_arr.min(), v_arr.max()

    rho_norm = (rho_arr - rho_min) / max(rho_max - rho_min, 1e-8)
    t_max = max(t_arr) if len(t_arr) > 0 else 0.0
    v_norm = v_arr / max(v_max, 1e-8)
    x_norm = x_arr / max(channel_len, 1e-8)
    t_norm = t_arr / T_DAYS_MAX

    X = np.stack([x_norm, t_norm], axis=1)
    y = np.stack([rho_norm, v_norm], axis=1)

    metadata = {
        "channel_len_km": channel_len,
        "n_bins": n_bins,
        "n_days": n_days,
        "n_points": len(X),
        "dx_km": dx_km,
        "location": location,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "epoch": EPOCH_DATE.isoformat(),
        "t_days_max": T_DAYS_MAX,
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "v_max": float(v_max),
        "rho_density_grid": rho_density,
        "v_grid": v_grid,
        "bin_centers": bin_centers.tolist(),
        "dates": [d.isoformat() for d in dates_list],
        "t_max": t_max,
        "x_max": channel_len,
    }

    log.info(
        "Built tensor: %d points across %d bins x %d days | "
        "rho=[%.2f, %.2f] v_max=%.2f | channel=%.1f km dx=%.1f km",
        len(X),
        n_bins,
        n_days,
        rho_min,
        rho_max,
        v_max,
        channel_len,
        dx_km,
    )

    return X, y, metadata


def get_tensors(
    device,
    start,
    end,
    location,
    dx_km=2.0,
    constituent_path=None,
    use_all_v=False,
    use_raw_velocity=True,
):
    X_np, y_np, meta = build_rho_v_tensors(
        start,
        end,
        location,
        dx_km=dx_km,
        constituent_path=constituent_path,
        use_all_v=use_all_v,
        use_raw_velocity=use_raw_velocity,
    )
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    return X_t, y_t, meta
