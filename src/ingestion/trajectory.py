"""
Étape 3 complète — Reconstruction et compression des trajectoires AIS.

Pour chaque MMSI+traj_id issu de kinematic_filter :
  1. Fragmentation : gap > GAP_THRESHOLD_MIN → déjà fait via traj_id (hérité)
  2. Interpolation intelligente :
       - Canal quasi-linéaire (|Δcap| < 30°) → interpolation linéaire
       - Changement de cap marqué               → Cubic Hermite spline
  3. Compression Douglas-Peucker (ε = 0.0001° ≈ 11 m)

Produit un DataFrame de trajectoires reconstruites à résolution uniforme,
utilisé par les PINNs pour disposer d'un champ de vitesse (x, t, ρ, v) continu.

Usage (standalone):
    python src/ingestion/trajectory.py data/parquet/houston/houston_2017_08_24.parquet
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from scipy.interpolate import CubicHermiteSpline

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# Thresholds
GAP_THRESHOLD_MIN      = 30     # minutes — inherited from kinematic_filter
HEADING_CHANGE_DEG     = 30.0   # degrees — above this → Hermite instead of linear
DOUGLAS_PEUCKER_EPS    = 1e-4   # degrees ≈ 11 m
INTERP_STEP_SECONDS    = 120    # target resolution after interpolation (2 min)


# ---------------------------------------------------------------------------
# Douglas-Peucker compression
# ---------------------------------------------------------------------------

def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker line simplification.
    points : (N, 2) array of [LAT, LON]
    Returns a boolean mask of retained points.
    """
    if len(points) <= 2:
        return np.ones(len(points), dtype=bool)

    mask = np.zeros(len(points), dtype=bool)
    mask[0] = True
    mask[-1] = True

    _rdp_recursive(points, 0, len(points) - 1, epsilon, mask)
    return mask


def _rdp_recursive(points: np.ndarray, start: int, end: int,
                   epsilon: float, mask: np.ndarray) -> None:
    if end <= start + 1:
        return

    # Perpendicular distance from points[start..end] to the line start→end
    p0, p1 = points[start], points[end]
    seg    = p1 - p0
    seg_len = np.linalg.norm(seg)

    if seg_len == 0:
        dists = np.linalg.norm(points[start:end+1] - p0, axis=1)
    else:
        t     = np.dot(points[start:end+1] - p0, seg) / (seg_len ** 2)
        proj  = p0 + np.outer(np.clip(t, 0, 1), seg)
        dists = np.linalg.norm(points[start:end+1] - proj, axis=1)

    idx_max = start + int(np.argmax(dists))
    if dists[idx_max - start] > epsilon:
        mask[idx_max] = True
        _rdp_recursive(points, start, idx_max, epsilon, mask)
        _rdp_recursive(points, idx_max, end, epsilon, mask)


# ---------------------------------------------------------------------------
# Per-trajectory interpolation
# ---------------------------------------------------------------------------

def _interpolate_trajectory(
    lats: np.ndarray,
    lons: np.ndarray,
    timestamps: np.ndarray,   # unix seconds
    headings: np.ndarray,
    sogs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a single trajectory to INTERP_STEP_SECONDS resolution.
    Uses Cubic Hermite if heading change > HEADING_CHANGE_DEG, else linear.
    Returns (lats, lons, times, headings, sogs) interpolated.
    """
    if len(lats) < 2:
        return lats, lons, timestamps, headings, sogs

    t_new = np.arange(timestamps[0], timestamps[-1] + 1, INTERP_STEP_SECONDS)
    if len(t_new) < 2:
        return lats, lons, timestamps, headings, sogs

    # Detect heading change
    valid_hdg = headings[headings < 360]
    heading_std = float(np.std(valid_hdg)) if len(valid_hdg) > 1 else 0.0
    use_hermite = heading_std > HEADING_CHANGE_DEG

    if use_hermite:
        # Tangent vectors from (dLAT/dt, dLON/dt) via finite differences
        dlat = np.gradient(lats, timestamps)
        dlon = np.gradient(lons, timestamps)
        cs_lat = CubicHermiteSpline(timestamps, lats, dlat)
        cs_lon = CubicHermiteSpline(timestamps, lons, dlon)
        lat_new = cs_lat(t_new)
        lon_new = cs_lon(t_new)
    else:
        lat_new = np.interp(t_new, timestamps, lats)
        lon_new = np.interp(t_new, timestamps, lons)

    sog_new = np.interp(t_new, timestamps, sogs)
    hdg_new = np.interp(t_new, timestamps, headings)

    return lat_new, lon_new, t_new, hdg_new, sog_new


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_trajectories(parquet_path: Path) -> pl.DataFrame:
    """
    Load a daily Parquet, apply kinematic preprocessing, then for each
    (MMSI, traj_id) : interpolate + Douglas-Peucker compress.

    Returns a DataFrame with columns:
        MMSI, traj_id, t_unix, LAT, LON, SOG_corr, Heading, use_hermite
    Sorted by (MMSI, traj_id, t_unix).
    """
    raw   = pl.read_parquet(parquet_path)
    df    = prepare_kinematics(raw)

    rows = []
    groups = df.group_by(["MMSI", "traj_id"])

    for (mmsi, traj_id), grp in groups:
        grp = grp.sort("BaseDateTime")
        if len(grp) < 2:
            continue

        lats  = grp["LAT"].to_numpy().astype(float)
        lons  = grp["LON"].to_numpy().astype(float)
        times = grp["BaseDateTime"].cast(pl.Int64).to_numpy() / 1e9   # → unix seconds
        hdgs  = grp["Heading"].fill_null(511).to_numpy().astype(float)
        sogs  = grp["SOG_corr"].to_numpy().astype(float)

        lat_i, lon_i, t_i, hdg_i, sog_i = _interpolate_trajectory(
            lats, lons, times, hdgs, sogs
        )

        # Douglas-Peucker compression
        points = np.column_stack([lat_i, lon_i])
        keep   = douglas_peucker(points, DOUGLAS_PEUCKER_EPS)

        valid_hdg = hdgs[hdgs < 360]
        heading_std = float(np.std(valid_hdg)) if len(valid_hdg) > 1 else 0.0

        for lat, lon, t, hdg, sog in zip(
            lat_i[keep], lon_i[keep], t_i[keep], hdg_i[keep], sog_i[keep]
        ):
            rows.append({
                "MMSI":        int(mmsi),
                "traj_id":     int(traj_id),
                "t_unix":      float(t),
                "LAT":         float(lat),
                "LON":         float(lon),
                "SOG_corr":    float(sog),
                "Heading":     float(hdg),
                "use_hermite": heading_std > HEADING_CHANGE_DEG,
            })

    if not rows:
        log.warning("%s: no trajectories built", parquet_path.name)
        return pl.DataFrame()

    result = pl.DataFrame(rows).sort(["MMSI", "traj_id", "t_unix"])
    n_traj = result.select(["MMSI", "traj_id"]).unique().height
    log.info("%s: %d trajectories, %d points (after compression)",
             parquet_path.name, n_traj, len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory interpolation + compression")
    parser.add_argument("parquet", help="Daily Parquet path")
    args = parser.parse_args()

    df = build_trajectories(Path(args.parquet))
    if len(df) > 0:
        print(df.head(20))
        print(f"\n{len(df)} points | {df.select(['MMSI','traj_id']).unique().height} trajectories")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
