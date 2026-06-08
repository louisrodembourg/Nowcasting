"""
Phase 3 — Features spatio-temporelles par zone constituante.

Produit : data/features/{loc}_zone_daily_features.parquet
  Schema : (date, zone_key, lat, lon, lon_norm, cluster_type,
            vessel_count, SOG_mean, waiting_capacity, rho_norm, v_norm)

Méthode :
  1. Pour chaque jour : charger parquet brut, filtrer SOG < 1 kt
  2. Déduplication par MMSI (médiane LAT/LON sur la journée → 1 pos/navire)
  3. Assignation KD-tree : chaque navire → zone constituante la plus proche (≤ max_dist_deg)
  4. Agrégat par zone : vessel_count, SOG_mean, waiting_capacity (Draft × Length)
  5. Normalisation globale : rho = vessel_count / cap_95, v = SOG_mean / sog_max

Usage:
    python src/pinns/features_per_zone.py --location houston
    python src/pinns/features_per_zone.py --location houston \\
        --start 2017-06-01 --end 2018-06-30 --max-dist 0.005
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# Rayon max d'assignation navire → zone (°) — ~500 m à la latitude de Houston
DEFAULT_MAX_DIST = 0.005
SOG_STATIONARY   = 1.0   # nœuds — seuil navire stationnaire


# ---------------------------------------------------------------------------
# Chargement des zones constituantes
# ---------------------------------------------------------------------------

def load_zones(zones_path: Path) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """
    Retourne (lats, lons, df_zones) pour les zones is_constituent=True,
    triées par longitude croissante.
    """
    zones = (
        pl.read_parquet(zones_path)
        .filter(pl.col("is_constituent"))
        .sort("lon")
    )
    lats = zones["lat"].to_numpy().astype(float)
    lons = zones["lon"].to_numpy().astype(float)
    return lats, lons, zones


# ---------------------------------------------------------------------------
# Traitement d'un seul fichier journalier
# ---------------------------------------------------------------------------

def process_day(
    parquet_path:  Path,
    zone_tree:     KDTree,
    zone_lats:     np.ndarray,
    zone_lons:     np.ndarray,
    zone_keys:     list[str],
    max_dist_deg:  float = DEFAULT_MAX_DIST,
) -> pl.DataFrame | None:
    """
    Charge un parquet journalier et retourne un DataFrame
    (zone_key, vessel_count, SOG_mean, waiting_capacity).

    Retourne None si le fichier est absent ou vide après filtrage.
    """
    if not parquet_path.exists():
        return None

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as exc:
        log.warning("Lecture échouée %s : %s", parquet_path.name, exc)
        return None

    # 1. Filtre stationnaire
    df = df.filter(pl.col("SOG") < SOG_STATIONARY)
    if len(df) == 0:
        return None

    # 2. Déduplication par MMSI (médiane position + SOG sur la journée)
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("SOG").mean().alias("SOG_mean"),
    ]
    if "Draft" in df.columns:
        agg_exprs.append(pl.col("Draft").mean().alias("Draft_mean"))
    if "Length" in df.columns:
        agg_exprs.append(pl.col("Length").mean().alias("Length_mean"))

    vessels = df.group_by("MMSI").agg(agg_exprs)
    if len(vessels) == 0:
        return None

    # 3. Assignation KD-tree
    pts = np.column_stack([
        vessels["LAT"].to_numpy().astype(float),
        vessels["LON"].to_numpy().astype(float),
    ])
    dists, idxs = zone_tree.query(pts, k=1)

    # Masque rayon max
    valid = dists <= max_dist_deg
    if valid.sum() == 0:
        return None

    assigned_zones = np.array(zone_keys)[idxs[valid]]
    sog_vals       = vessels["SOG_mean"].to_numpy().astype(float)[valid]

    # waiting_capacity = Draft × Length par navire (proxy tonnnage bloqué)
    if "Draft_mean" in vessels.columns and "Length_mean" in vessels.columns:
        draft  = vessels["Draft_mean"].to_numpy().astype(float)[valid]
        length = vessels["Length_mean"].to_numpy().astype(float)[valid]
        wc     = np.where(
            np.isnan(draft) | np.isnan(length), 0.0, draft * length
        )
    else:
        wc = np.zeros(valid.sum())

    # 4. Agrégat par zone
    tmp = pl.DataFrame({
        "zone_key":        assigned_zones,
        "SOG_mean":        sog_vals,
        "waiting_cap_raw": wc,
    })

    agg = (
        tmp.group_by("zone_key")
        .agg([
            pl.len().alias("vessel_count"),
            pl.col("SOG_mean").mean().alias("SOG_mean"),
            pl.col("waiting_cap_raw").sum().alias("waiting_capacity"),
        ])
    )
    return agg


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_zone_features(
    location:      str  = "houston",
    start:         date = date(2017, 6, 1),
    end:           date = date(2018, 6, 30),
    parquet_dir:   Path | None = None,
    zones_path:    Path | None = None,
    output_path:   Path | None = None,
    max_dist_deg:  float = DEFAULT_MAX_DIST,
) -> pl.DataFrame:
    """
    Construit la matrice ρ(zone, jour) sur la fenêtre [start, end].

    Retourne un DataFrame (date, zone_key, lat, lon, lon_norm, cluster_type,
    vessel_count, SOG_mean, waiting_capacity, rho_norm, v_norm).
    """
    base = Path(".")

    if parquet_dir is None:
        parquet_dir = base / "data" / "parquet" / location
    if zones_path is None:
        zones_path = base / "data" / "features" / f"{location}_constituent_zones.parquet"
    if output_path is None:
        output_path = base / "data" / "features" / f"{location}_zone_daily_features.parquet"

    if not zones_path.exists():
        raise FileNotFoundError(
            f"Zones constituantes introuvables : {zones_path}\n"
            "Lancez d'abord run_phase2.py."
        )

    zone_lats, zone_lons, df_zones = load_zones(zones_path)
    zone_keys  = df_zones["zone_key"].to_list()
    zone_types = df_zones["cluster_type"].to_list()
    lon_min    = float(zone_lons.min())
    lon_max    = float(zone_lons.max())

    zone_tree  = KDTree(np.column_stack([zone_lats, zone_lons]))
    log.info("KD-tree construit : %d zones", len(zone_keys))

    n_zones = len(zone_keys)
    all_days: list[pl.DataFrame] = []

    current = start
    n_days  = (end - start).days + 1
    processed = 0

    while current <= end:
        fname = f"{location}_{current.strftime('%Y_%m_%d')}.parquet"
        fpath = parquet_dir / fname

        day_agg = process_day(fpath, zone_tree, zone_lats, zone_lons,
                               zone_keys, max_dist_deg)

        if day_agg is not None:
            # Left join sur toutes les zones → zones sans navires = 0
            base_zones = pl.DataFrame({"zone_key": zone_keys})
            day_full = (
                base_zones
                .join(day_agg, on="zone_key", how="left")
                .with_columns([
                    pl.col("vessel_count").fill_null(0).cast(pl.Int32),
                    pl.col("SOG_mean").fill_null(0.0),
                    pl.col("waiting_capacity").fill_null(0.0),
                    pl.lit(current).alias("date"),
                ])
            )
        else:
            # Journée sans données : toutes les zones à 0
            day_full = pl.DataFrame({
                "zone_key":        zone_keys,
                "vessel_count":    np.zeros(n_zones, dtype=np.int32),
                "SOG_mean":        np.zeros(n_zones),
                "waiting_capacity":np.zeros(n_zones),
                "date":            [current] * n_zones,
            })

        all_days.append(day_full)
        processed += 1

        if processed % 30 == 0 or current == end:
            log.info("  %d/%d jours traités (%s)", processed, n_days, current)

        current += timedelta(days=1)

    log.info("Concaténation de %d jours × %d zones...", len(all_days), n_zones)
    df = pl.concat(all_days, how="diagonal_relaxed")

    # Ajouter lat/lon/cluster_type depuis les zones
    zone_meta = df_zones.select(["zone_key", "lat", "lon", "cluster_type"])
    df = df.join(zone_meta, on="zone_key", how="left")

    # Longitude normalisée [0, 1]
    df = df.with_columns(
        ((pl.col("lon") - lon_min) / max(lon_max - lon_min, 1e-8)).alias("lon_norm")
    )

    # Normalisation globale rho et v
    counts = df["vessel_count"].to_numpy().astype(float)
    counts_pos = counts[counts > 0]
    cap_95 = float(np.percentile(counts_pos, 95)) if len(counts_pos) > 0 else 1.0

    sog_vals = df["SOG_mean"].to_numpy().astype(float)
    sog_max  = float(sog_vals.max()) if sog_vals.max() > 0 else 1.0

    df = df.with_columns([
        (pl.col("vessel_count").cast(pl.Float64) / cap_95)
            .clip(0.0, 1.0).alias("rho_norm"),
        (pl.col("SOG_mean") / sog_max).alias("v_norm"),
    ])

    # Ordre final des colonnes
    df = df.select([
        "date", "zone_key", "lat", "lon", "lon_norm", "cluster_type",
        "vessel_count", "SOG_mean", "waiting_capacity",
        "rho_norm", "v_norm",
    ]).sort(["date", "lon"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    n_zone_days = df.filter(pl.col("vessel_count") > 0).height
    log.info(
        "Sauvegardé → %s  (%d lignes, %d zone-jours actifs, cap_95=%.1f navires)",
        output_path, len(df), n_zone_days, cap_95,
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Features per-zone spatio-temporelles"
    )
    parser.add_argument("--location",  default="houston",
                        choices=["houston", "la"])
    parser.add_argument("--start",     default="2017-06-01",
                        metavar="YYYY-MM-DD")
    parser.add_argument("--end",       default="2018-06-30",
                        metavar="YYYY-MM-DD")
    parser.add_argument("--max-dist",  type=float, default=DEFAULT_MAX_DIST,
                        help="Rayon max assignation navire→zone (degrés, défaut=0.005°≈500m)")
    parser.add_argument("--output",    default=None,
                        help="Chemin de sortie parquet (optionnel)")
    args = parser.parse_args()

    df = build_zone_features(
        location=args.location,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        max_dist_deg=args.max_dist,
        output_path=Path(args.output) if args.output else None,
    )

    print(f"\nShape : {df.shape}")
    print(f"Zone-jours actifs : {df.filter(pl.col('vessel_count') > 0).height}")
    print(f"\nAperçu (zones les + actives) :")
    top = (
        df.group_by("zone_key")
        .agg(pl.col("vessel_count").mean().alias("mean_count"))
        .sort("mean_count", descending=True)
        .head(5)
    )
    print(top)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
    )
    main()
