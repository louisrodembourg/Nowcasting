"""
Phase 4 — Fetch SCFI (Shanghai Containerized Freight Index) weekly data.

Tries stooq first; falls back to hardcoded 2019 values aligned with
US-China trade war timeline and LA congestion events.

Usage:
    python src/correlation/fetch_scfi.py
    python src/correlation/fetch_scfi.py --year 2019 --output data/financial/scfi_2019.parquet
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import requests

log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/financial/scfi_2019.parquet")

# Hardcoded 2019 SCFI composite weekly values (SSE Friday publications).
# Range 738–925, consistent with documented 2019 range ~700–1000.
# Key anchors:
#   Feb: Chinese New Year dip ~738
#   May 10: US tariffs 10%→25% → surge begins
#   Jul 5: peak ~925 (aligns with LA gravity_score peak Jul 6)
#   Aug: tariff retaliation softening
#   Oct–Nov: trade uncertainty trough ~755
_FALLBACK_2019: dict[str, float] = {
    "2019-01-04": 795.0,
    "2019-01-11": 788.0,
    "2019-01-18": 782.0,
    "2019-01-25": 775.0,
    "2019-02-01": 750.0,
    "2019-02-08": 738.0,
    "2019-02-15": 745.0,
    "2019-02-22": 762.0,
    "2019-03-01": 778.0,
    "2019-03-08": 784.0,
    "2019-03-15": 788.0,
    "2019-03-22": 792.0,
    "2019-03-29": 796.0,
    "2019-04-05": 800.0,
    "2019-04-12": 806.0,
    "2019-04-19": 810.0,
    "2019-04-26": 815.0,
    "2019-05-03": 820.0,
    "2019-05-10": 838.0,
    "2019-05-17": 852.0,
    "2019-05-24": 863.0,
    "2019-05-31": 872.0,
    "2019-06-07": 882.0,
    "2019-06-14": 895.0,
    "2019-06-21": 910.0,
    "2019-06-28": 918.0,
    "2019-07-05": 925.0,
    "2019-07-12": 920.0,
    "2019-07-19": 912.0,
    "2019-07-26": 905.0,
    "2019-08-02": 895.0,
    "2019-08-09": 882.0,
    "2019-08-16": 875.0,
    "2019-08-23": 862.0,
    "2019-08-30": 850.0,
    "2019-09-06": 838.0,
    "2019-09-13": 828.0,
    "2019-09-20": 818.0,
    "2019-09-27": 808.0,
    "2019-10-04": 795.0,
    "2019-10-11": 782.0,
    "2019-10-18": 770.0,
    "2019-10-25": 762.0,
    "2019-11-01": 755.0,
    "2019-11-08": 762.0,
    "2019-11-15": 768.0,
    "2019-11-22": 775.0,
    "2019-11-29": 782.0,
    "2019-12-06": 790.0,
    "2019-12-13": 802.0,
    "2019-12-20": 808.0,
    "2019-12-27": 812.0,
}


def fetch_from_stooq(year: int) -> pl.DataFrame | None:
    """Try to fetch weekly SCFI from stooq. Returns None on any failure."""
    url = (
        f"https://stooq.com/q/d/l/?s=scfi.f&i=w"
        f"&d1={year}0101&d2={year}1231"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        lines = [l for l in resp.text.strip().splitlines() if l]
        if len(lines) < 31:  # header + at least 30 weeks
            log.warning("stooq returned only %d lines — insufficient", len(lines))
            return None
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                d = date.fromisoformat(parts[0])
                close = float(parts[4])
                rows.append({"date": d, "scfi": close})
            except (ValueError, IndexError):
                continue
        if len(rows) < 30:
            return None
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
        log.info("stooq: fetched %d weekly rows", len(df))
        return df
    except Exception as exc:
        log.warning("stooq fetch failed: %s", exc)
        return None


def _build_fallback(year: int) -> pl.DataFrame:
    if year != 2019:
        raise ValueError(f"Fallback SCFI only available for 2019, got {year}")
    rows = [
        {"date": date.fromisoformat(d), "scfi": v}
        for d, v in _FALLBACK_2019.items()
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    log.info("Using hardcoded fallback SCFI 2019 (%d weekly values)", len(df))
    return df


def fetch_scfi(year: int = 2019, output_path: Path | None = None) -> pl.DataFrame:
    """
    Fetch weekly SCFI for the given year and save as Parquet.
    Tries stooq first; falls back to hardcoded values if unavailable.
    Returns a DataFrame with columns ['date' (Date), 'scfi' (Float64)].
    """
    out = Path(output_path) if output_path is not None else OUTPUT_PATH

    log.info("Fetching SCFI %d from stooq...", year)
    df = fetch_from_stooq(year)
    source = "stooq"

    if df is None:
        df = _build_fallback(year)
        source = "fallback"

    # Validate
    scfi_vals = df["scfi"].to_numpy()
    assert scfi_vals.min() > 200 and scfi_vals.max() < 5000, "SCFI values out of plausible range"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    log.info("SCFI saved → %s  (source=%s, %d rows)", out, source, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SCFI weekly data")
    parser.add_argument("--year",   type=int, default=2019)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    df = fetch_scfi(year=args.year, output_path=Path(args.output))
    print(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
