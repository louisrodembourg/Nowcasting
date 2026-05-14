"""
Apply a KML polygon mask to existing LA Parquet files.

Workflow:
  1. Parse the polygon from a Google Earth KML export
  2. Load each Parquet file with Polars
  3. Keep only rows where (LON, LAT) falls inside the polygon (shapely vectorised)
  4. Write filtered result to --out-dir (default: data/parquet/la_polygon/)

Usage (from Nowcasting/ root):
    # Test on 3 days
    python src/ingestion/filter_polygon.py \
        --kml /path/to/my_zone.kml \
        --in-dir data/parquet/la \
        --sample 3

    # All files, custom output dir
    python src/ingestion/filter_polygon.py \
        --kml /path/to/my_zone.kml \
        --in-dir data/parquet/la \
        --out-dir data/parquet/la_polygon
"""
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl
import shapely
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


# ── KML parsing ──────────────────────────────────────────────────────────────

def parse_kml_polygon(kml_path: Path) -> Polygon:
    """
    Extract the first polygon from a Google Earth KML file.
    Returns a shapely Polygon (lon, lat) — KML stores coordinates as lon,lat,alt.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML namespace varies — strip it for simple XPath
    ns = ""
    tag = root.tag
    if tag.startswith("{"):
        ns = tag[: tag.index("}") + 1]

    coords_tag = f"{ns}coordinates"
    coords_el = root.find(f".//{coords_tag}")
    if coords_el is None or not coords_el.text:
        raise ValueError(f"No <coordinates> element found in {kml_path}")

    points = []
    for token in coords_el.text.strip().split():
        parts = token.split(",")
        lon, lat = float(parts[0]), float(parts[1])
        points.append((lon, lat))

    if len(points) < 3:
        raise ValueError(f"Polygon has only {len(points)} points — need at least 3")

    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)  # auto-fix self-intersections
    log.info("Polygon parsed: %d vertices, bounds=%s", len(points), poly.bounds)
    return poly


# ── Per-file filtering ────────────────────────────────────────────────────────

def filter_parquet(src: Path, dst: Path, polygon: Polygon) -> dict:
    """Filter one Parquet file by polygon and write result. Returns stats."""
    df = pl.read_parquet(src)
    n_before = len(df)

    # shapely.contains_xy is vectorised (shapely >= 2.0)
    lons = df["LON"].to_numpy()
    lats = df["LAT"].to_numpy()
    mask = shapely.contains_xy(polygon, lons, lats)

    df_filtered = df.filter(pl.Series(mask))
    n_after = len(df_filtered)

    df_filtered.write_parquet(dst, compression="zstd")
    pct = 100 * n_after / n_before if n_before else 0
    return {"file": src.name, "before": n_before, "after": n_after, "kept_pct": pct}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter LA Parquet files to a custom KML polygon"
    )
    parser.add_argument("--kml",     required=True,  help="Path to .kml file")
    parser.add_argument("--in-dir",  default="data/parquet/la",
                        help="Directory with source Parquet files")
    parser.add_argument("--out-dir", default="data/parquet/la_polygon",
                        help="Output directory for filtered Parquet files")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Process only N files (for quick testing)")
    args = parser.parse_args()

    kml_path = Path(args.kml)
    in_dir   = Path(args.in_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    polygon = parse_kml_polygon(kml_path)

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        log.error("No Parquet files found in %s", in_dir)
        return

    if args.sample:
        files = files[: args.sample]
        log.info("Sample mode: processing %d/%d files", len(files), len(sorted(in_dir.glob("*.parquet"))))

    log.info("Processing %d files → %s", len(files), out_dir)

    total_before = total_after = 0
    for src in files:
        dst = out_dir / src.name
        stats = filter_parquet(src, dst, polygon)
        total_before += stats["before"]
        total_after  += stats["after"]
        log.info("  %s  %d → %d rows (%.1f%%)",
                 stats["file"], stats["before"], stats["after"], stats["kept_pct"])

    overall_pct = 100 * total_after / total_before if total_before else 0
    log.info("Done — total %d → %d rows (%.1f%% kept)", total_before, total_after, overall_pct)
    log.info("Output: %s", out_dir.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
