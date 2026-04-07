"""
Phase 1 orchestrator — Houston Ship Channel AIS → Daily feature matrix.

Steps for each day in [--start, --end]:
  1. Download daily ZIP from Marine Cadastre → filtered Parquet (skip if exists)
  2. HDBSCAN clustering on stationary vessels (SOG < 1 kt)
  3. Extract 13 daily features from traffic + cluster data

Writes the feature matrix to data/features/houston_daily_features.parquet
(appends or merges with existing data if the file already exists).

Usage (run from Nowcasting/ root):
    # Full pipeline: download + process
    python run_phase1.py --start 2017-07-25 --end 2017-09-15

    # Process only (Parquets already downloaded)
    python run_phase1.py --start 2017-07-25 --end 2017-09-15 --no-download

    # Force re-download (e.g. after bbox change)
    python run_phase1.py --start 2017-08-01 --end 2017-08-31 --force
"""
import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.ingestion.download import download_day
from src.clustering.hdbscan_daily import cluster_day
from src.clustering.features_daily import compute_daily_features

log = logging.getLogger(__name__)

PARQUET_DIR   = Path("data/parquet/houston")
FEATURES_PATH = Path("data/features/houston_daily_features.parquet")


def run_pipeline(
    start: date,
    end: date,
    skip_download: bool = False,
    force: bool = False,
    delay: float = 3.0,
) -> pl.DataFrame:
    """
    Run the full Phase 1 pipeline for [start, end].
    Returns a DataFrame with 14 columns (date + 13 features), one row per day.
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    d = start
    while d <= end:
        parquet_path = PARQUET_DIR / f"houston_{d.strftime('%Y_%m_%d')}.parquet"

        # Step 1 — download
        if not skip_download:
            try:
                already_existed = parquet_path.exists() and not force
                result = download_day(d, PARQUET_DIR, force=force)
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
        # Drop dates that are being re-computed, then append new rows
        existing_dates = set(df_existing["date"].to_list())
        new_dates      = set(df_new["date"].to_list())
        df_kept = df_existing.filter(~pl.col("date").is_in(list(new_dates)))
        df_merged = pl.concat([df_kept, df_new]).sort("date")
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
        description="Phase 1 — Houston AIS → 13 daily features matrix"
    )
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

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    log.info("=== Phase 1 — Houston Ship Channel ===")
    log.info("Period : %s → %s (%d days)", start, end, (end - start).days + 1)
    log.info("Download: %s (delay=%.0fs)", "skip" if args.no_download else "yes", args.delay)

    df_features = run_pipeline(
        start, end, skip_download=args.no_download, force=args.force, delay=args.delay
    )

    if len(df_features) == 0:
        log.error("Pipeline produced no output.")
        return

    save_features(df_features, FEATURES_PATH)

    # Summary
    log.info("\n--- Feature matrix summary ---")
    log.info("Shape   : %d days × %d features", *df_features.shape)
    log.info("Date range: %s → %s", df_features["date"].min(), df_features["date"].max())
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
