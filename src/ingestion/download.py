"""
Download Marine Cadastre AIS daily files for Houston Ship Channel.

For each day in [--start, --end]:
  1. Download ZIP from NOAA coast server
  2. Extract CSV from ZIP (in memory)
  3. Filter by Houston bbox + valid MMSI + realistic SOG via DuckDB
  4. Save as Parquet (ZSTD compressed) — ~30–100x smaller than raw CSV
  5. Delete temp CSV

Usage (run from Nowcasting/ root):
    python src/ingestion/download.py --start 2017-07-25 --end 2017-09-15
    python src/ingestion/download.py --start 2015-01-01 --end 2016-12-31  # baseline
    python src/ingestion/download.py --start 2017-08-01 --end 2017-08-01 --force
"""
import argparse
import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import duckdb
import requests

log = logging.getLogger(__name__)

# Houston Ship Channel bounding box (Galveston Bay to turning basin)
LAT_MIN, LAT_MAX = 29.3, 29.85
LON_MIN, LON_MAX = -95.4, -94.7

# SOG sanity cap — anything above 50 kt is a sensor error for commercial vessels
SOG_MAX = 50.0

# NOAA Marine Cadastre URL pattern
NOAA_URL = (
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
    "/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
)

DEFAULT_OUT_DIR = Path("data/parquet/houston")


def download_day(d: date, out_dir: Path, force: bool = False) -> Path | None:
    """
    Download, filter, and convert one day to Parquet.
    Returns the output path on success, None on failure.
    Skips silently if the file already exists (unless force=True).
    """
    out_path = out_dir / f"houston_{d.strftime('%Y_%m_%d')}.parquet"

    if out_path.exists() and not force:
        log.info("%s: already exists — skipping", d)
        return out_path

    url = NOAA_URL.format(year=d.year, month=d.month, day=d.day)
    log.info("%s: GET %s", d, url)

    # resp.content reads the full body — keep it inside the network try/except
    # so read timeouts are caught the same as connection errors.
    csv_bytes = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_bytes = zf.read(zf.namelist()[0])
            break  # success
        except requests.HTTPError as exc:
            log.warning("%s: HTTP %s — day may not exist in archive", d, exc.response.status_code)
            return None  # no point retrying a 404
        except zipfile.BadZipFile:
            log.warning("%s: corrupt ZIP — skipping", d)
            return None
        except requests.RequestException as exc:
            # DNS failure is systemic — retrying other days is pointless
            if "NameResolutionError" in str(exc) or "Failed to resolve" in str(exc):
                raise ConnectionAbortedError(
                    f"DNS resolution failed for {url!r}. Check your network connection."
                ) from exc
            if attempt < 3:
                log.warning("%s: attempt %d failed (%s) — retrying in 10s", d, attempt, exc)
                time.sleep(10)
            else:
                log.warning("%s: all attempts failed — %s", d, exc)
                return None

    if csv_bytes is None:
        return None

    # DuckDB needs a file path, so write a minimal temp file
    tmp_csv = out_dir / f"_tmp_{d.strftime('%Y_%m_%d')}.csv"
    tmp_csv.write_bytes(csv_bytes)
    del csv_bytes  # free memory before DuckDB reads

    try:
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{tmp_csv}')
                WHERE LAT  BETWEEN {LAT_MIN} AND {LAT_MAX}
                  AND LON  BETWEEN {LON_MIN} AND {LON_MAX}
                  AND MMSI BETWEEN 200000000 AND 999999999
                  AND SOG  BETWEEN 0 AND {SOG_MAX}
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
        con.close()
        size_kb = out_path.stat().st_size // 1024
        log.info("%s: %d rows → %s (%d KB)", d, n_rows, out_path.name, size_kb)

    except Exception as exc:
        log.error("%s: DuckDB error — %s", d, exc)
        out_path.unlink(missing_ok=True)
        return None

    finally:
        tmp_csv.unlink(missing_ok=True)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Marine Cadastre AIS daily files → filtered Houston Parquet"
    )
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD",
                        help="First day to download (inclusive)")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD",
                        help="Last day to download (inclusive)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and overwrite existing Parquet files")
    args = parser.parse_args()

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s → %s into %s", start, end, out_dir)

    ok = failed = 0
    d = start
    while d <= end:
        result = download_day(d, out_dir, force=args.force)
        if result is not None:
            ok += 1
        else:
            failed += 1
        d += timedelta(days=1)

    log.info("Done — %d downloaded/existing, %d failed/skipped", ok, failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
