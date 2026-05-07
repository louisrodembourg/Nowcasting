"""
Phase 1 orchestrator — AIS → Daily feature matrix.

Steps for each day in [--start, --end]:
  1. Download daily ZIP from Marine Cadastre → filtered Parquet (skip if exists)
  2. HDBSCAN clustering on stationary vessels (SOG < 1 kt)
  3. Extract 13 daily features from traffic + cluster data

Writes the feature matrix to data/features/<location>_daily_features.parquet
(appends or merges with existing data if the file already exists).

Usage (run from Nowcasting/ root):
    # Houston (default) — download + process
    python run_phase1.py --start 2017-07-25 --end 2017-09-15

    # LA/Long Beach — process only (already downloaded)
    python run_phase1.py --location la --start 2019-01-01 --end 2019-12-31 --no-download

    # Force re-download
    python run_phase1.py --location houston --start 2017-08-01 --end 2017-08-31 --force
"""
import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.ingestion.download import download_day, LOCATIONS
from src.clustering.hdbscan_daily import cluster_day
from src.clustering.features_daily import compute_daily_features

log = logging.getLogger(__name__)


def run_pipeline(
    start: date,
    end: date,
    parquet_dir: Path,
    prefix: str,
    loc_cfg: dict,
    skip_download: bool = False,
    force: bool = False,
    delay: float = 3.0,
) -> pl.DataFrame:
    """
    Run the full Phase 1 pipeline for [start, end].
    Returns a DataFrame with 14 columns (date + 13 features), one row per day.
    """
    parquet_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    d = start
    while d <= end:
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

        # Step 1 — download
        if not skip_download:
            try:
                already_existed = parquet_path.exists() and not force
                result = download_day(
                    d, parquet_dir, force=force,
                    lat_min=loc_cfg["lat_min"], lat_max=loc_cfg["lat_max"],
                    lon_min=loc_cfg["lon_min"], lon_max=loc_cfg["lon_max"],
                    prefix=prefix,
                )
            except ConnectionAbortedError as exc:
                log.critical("Network abort: %s", exc)
                log.critical("Fix your connection then re-run from --start %s", d)
                break
            if result is None:
                log.warning("%s: download failed — skipping day", d)
                d += timedelta(days=1)
                continue
            # Pause between actual downloads to avoid NOAA rate-limiting
            if not already_existed:
                time.sleep(delay)

        if not parquet_path.exists():
            log.warning("%s: Parquet not found — skipping day", d)
            d += timedelta(days=1)
            continue

        # Step 2 — kinematic preprocessing + HDBSCAN
        cluster_df, prepared_df = cluster_day(parquet_path)

        # Step 3 — features (use prepared_df with SOG_corr when available)
        features = compute_daily_features(
            prepared_df if prepared_df is not None else parquet_path,
            cluster_df,
            d,
        )
        if features is not None:
            rows.append(features)

        d += timedelta(days=1)

    if not rows:
        log.error("No features produced for any day in [%s, %s]", start, end)
        return pl.DataFrame()

    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date()).sort("date")


def save_features(df_new: pl.DataFrame, path: Path) -> None:
    """
    Save feature matrix. If file exists, merge (keeping new rows for dates
    already present, appending rows for new dates).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_existing = pl.read_parquet(path)
        new_dates  = set(df_new["date"].to_list())
        df_kept    = df_existing.filter(~pl.col("date").is_in(list(new_dates)))
        df_merged  = pl.concat([df_kept, df_new]).sort("date")
        log.info(
            "Merged: %d existing + %d new = %d total days",
            len(df_existing), len(df_new), len(df_merged),
        )
        df_merged.write_parquet(path)
    else:
        df_new.write_parquet(path)
        log.info("Created %s (%d days)", path, len(df_new))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 — AIS → 13 daily features matrix"
    )
    parser.add_argument("--location",    default="houston", choices=list(LOCATIONS.keys()),
                        help="Target port (default: houston)")
    parser.add_argument("--start",       required=True, metavar="YYYY-MM-DD",
                        help="First day (inclusive)")
    parser.add_argument("--end",         required=True, metavar="YYYY-MM-DD",
                        help="Last day (inclusive)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download step (use existing Parquets)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-download even if Parquet already exists")
    parser.add_argument("--delay",       type=float, default=3.0, metavar="SEC",
                        help="Pause between downloads in seconds (default: 3)")
    args = parser.parse_args()

    loc_cfg     = LOCATIONS[args.location]
    parquet_dir = loc_cfg["out_dir"]
    prefix      = loc_cfg["prefix"]

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    # Auto-derive output path from year range — prevents accidental cross-year merges
    start_year, end_year = start.year, end.year
    period = str(start_year) if start_year == end_year else f"{start_year}_{end_year}"
    features_path = Path(f"data/features/{args.location}_{period}_daily_features.parquet")

    # Guard: refuse to append if existing file covers different years
    if features_path.exists():
        existing = pl.read_parquet(features_path)
        existing_years = set(existing["date"].dt.year().unique().to_list())
        requested_years = set(range(start_year, end_year + 1))
        if not existing_years.issubset(requested_years):
            log.error(
                "CONFLIT D'ANNÉES — %s contient %s mais tu demandes %s.\n"
                "  → Utilise un chemin différent ou supprime le fichier existant.",
                features_path, sorted(existing_years), sorted(requested_years),
            )
            return

    log.info("=== Phase 1 — %s ===", args.location.upper())
    log.info("Period   : %s → %s (%d days)", start, end, (end - start).days + 1)
    log.info("Parquets : %s", parquet_dir)
    log.info("Features : %s", features_path)
    log.info("Download : %s (delay=%.0fs)", "skip" if args.no_download else "yes", args.delay)

    df_features = run_pipeline(
        start, end,
        parquet_dir=parquet_dir,
        prefix=prefix,
        loc_cfg=loc_cfg,
        skip_download=args.no_download,
        force=args.force,
        delay=args.delay,
    )

    if len(df_features) == 0:
        log.error("Pipeline produced no output.")
        return

    save_features(df_features, features_path)

    log.info("\n--- Feature matrix summary ---")
    log.info("Shape      : %d days × %d features", *df_features.shape)
    log.info("Date range : %s → %s", df_features["date"].min(), df_features["date"].max())
    log.info("vessel_count  avg=%.0f  max=%d",
             df_features["vessel_count"].mean(), df_features["vessel_count"].max())
    log.info("clusters      avg=%.1f  max=%d",
             df_features["hdbscan_cluster_count"].mean(),
             df_features["hdbscan_cluster_count"].max())
    log.info("rho (static)  avg=%.3f",
             df_features["utilization_rate_rho"].mean())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
