"""
Download Marine Cadastre AIS daily files for Houston Ship Channel or LA/Long Beach.

For each day in [--start, --end]:
  1. Download ZIP from NOAA coast server
  2. Extract CSV from ZIP (in memory)
  3. Filter by location bbox (envelope of GeoJSON zone) + valid MMSI + realistic SOG via DuckDB
  4. Save as Parquet (ZSTD compressed) — ~30–100x smaller than raw CSV
  5. Delete temp CSV

Download zones : data/zones/{location}_download.geojson
  The bounding box used for DuckDB filtering is derived from the polygon envelope.
  Falls back to hardcoded values if the GeoJSON is absent.

Usage (run from Nowcasting/ root):
    python src/ingestion/download.py --location houston --start 2017-07-25 --end 2017-09-15
    python src/ingestion/download.py --location la      --start 2019-01-01 --end 2019-12-31
    python src/ingestion/download.py --location houston --start 2017-08-01 --end 2017-08-01 --force
"""
import argparse
import io
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import duckdb
import requests

log = logging.getLogger(__name__)

# Project root (Nowcasting/) — works regardless of the CWD when the script is called
_ROOT = Path(__file__).resolve().parents[2]

DOWNLOAD_ZONES_DIR = _ROOT / "data/zones"


# ── Zones de téléchargement (GeoJSON) ────────────────────────────────────────

def bbox_from_geojson(path: Path) -> tuple[float, float, float, float]:
    """
    Calcule la bounding box (lat_min, lat_max, lon_min, lon_max)
    depuis un fichier GeoJSON — enveloppe de l'union de toutes les géométries.
    Lève FileNotFoundError si le fichier est absent.
    """
    with open(path) as f:
        fc = json.load(f)

    all_coords: list[list[float]] = []
    features = fc.get("features", [fc] if fc.get("type") == "Feature" else [])
    for feat in features:
        geom = feat.get("geometry", feat) if isinstance(feat, dict) else feat
        if not geom:
            continue
        gtype = geom.get("type", "")
        if gtype == "Polygon":
            for ring in geom["coordinates"]:
                all_coords.extend(ring)
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    all_coords.extend(ring)

    if not all_coords:
        raise ValueError(f"Aucune coordonnée trouvée dans {path}")

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    bbox = (min(lats), max(lats), min(lons), max(lons))
    log.info("bbox depuis %s : LAT[%.4f, %.4f]  LON[%.4f, %.4f]", path.name, *bbox)
    return bbox


def _load_bbox(
    location: str,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Charge la bbox depuis le GeoJSON ; fallback hardcodé si le fichier est absent."""
    path = DOWNLOAD_ZONES_DIR / f"{location}_download.geojson"
    try:
        return bbox_from_geojson(path)
    except FileNotFoundError:
        log.warning("Zone de téléchargement introuvable : %s — bbox hardcodée utilisée", path)
        return fallback


# ── Location configs ─────────────────────────────────────────────────────────

# Bbox dérivées des GeoJSON ; valeurs hardcodées = fallback si fichier absent
_houston_bbox = _load_bbox("houston", (28.95, 29.81, -95.31, -94.19))
_la_bbox      = _load_bbox("la",      (33.48, 33.79, -118.53, -117.96))

LOCATIONS = {
    "houston": {
        "lat_min":  _houston_bbox[0],
        "lat_max":  _houston_bbox[1],
        "lon_min":  _houston_bbox[2],
        "lon_max":  _houston_bbox[3],
        "geojson":  DOWNLOAD_ZONES_DIR / "houston_download.geojson",
        "out_dir":  _ROOT / "data/parquet/houston",
        "prefix":   "houston",
    },
    "la": {
        "lat_min":  _la_bbox[0],
        "lat_max":  _la_bbox[1],
        "lon_min":  _la_bbox[2],
        "lon_max":  _la_bbox[3],
        "geojson":  DOWNLOAD_ZONES_DIR / "la_download.geojson",
        "out_dir":  _ROOT / "data/parquet/la",
        "prefix":   "la",
    },
}

# SOG sanity cap — anything above 50 kt is a sensor error for commercial vessels
SOG_MAX = 50.0

# NOAA Marine Cadastre URL pattern
NOAA_URL = (
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
    "/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
)

# Keep old defaults for backwards compatibility
LAT_MIN, LAT_MAX = LOCATIONS["houston"]["lat_min"], LOCATIONS["houston"]["lat_max"]
LON_MIN, LON_MAX = LOCATIONS["houston"]["lon_min"], LOCATIONS["houston"]["lon_max"]
DEFAULT_OUT_DIR   = LOCATIONS["houston"]["out_dir"]


def download_day(
    d: date,
    out_dir: Path,
    force: bool = False,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    prefix: str = "houston",
) -> Path | None:
    """
    Download, filter, and convert one day to Parquet.
    Returns the output path on success, None on failure.
    Skips silently if the file already exists (unless force=True).
    """
    out_path = out_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

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
                FROM read_csv('{tmp_csv}',
                    delim=',', quote='"', escape='"', header=true,
                    ignore_errors=true, null_padding=true
                )
                WHERE LAT  BETWEEN {lat_min} AND {lat_max}
                  AND LON  BETWEEN {lon_min} AND {lon_max}
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
        description="Download Marine Cadastre AIS daily files → filtered Parquet"
    )
    parser.add_argument("--location", default="houston",
                        choices=list(LOCATIONS.keys()),
                        help="Target port (default: houston)")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD",
                        help="First day to download (inclusive)")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD",
                        help="Last day to download (inclusive)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: data/parquet/<location>)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and overwrite existing Parquet files")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between downloads in sequential mode (default: 1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download workers (default: 1 = sequential). "
                             "4–6 is a good balance for NOAA.")
    args = parser.parse_args()

    loc     = LOCATIONS[args.location]
    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir) if args.out_dir else loc["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Location : %s  bbox LAT[%.2f,%.2f] LON[%.2f,%.2f]",
             args.location, loc["lat_min"], loc["lat_max"], loc["lon_min"], loc["lon_max"])

    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)

    log.info("Downloading %d days (%s → %s) with %d worker(s) into %s",
             len(dates), start, end, args.workers, out_dir)

    kwargs = dict(
        out_dir=out_dir, force=args.force,
        lat_min=loc["lat_min"], lat_max=loc["lat_max"],
        lon_min=loc["lon_min"], lon_max=loc["lon_max"],
        prefix=loc["prefix"],
    )

    ok = failed = 0

    if args.workers <= 1:
        for i, day in enumerate(dates):
            result = download_day(day, **kwargs)
            if result is not None:
                ok += 1
            else:
                failed += 1
            if i < len(dates) - 1:
                time.sleep(args.delay)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_day, day, **kwargs): day for day in dates}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except ConnectionAbortedError as exc:
                    log.error("DNS failure — aborting: %s", exc)
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(1) from exc
                if result is not None:
                    ok += 1
                else:
                    failed += 1

    log.info("Done — %d downloaded/existing, %d failed/skipped", ok, failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
