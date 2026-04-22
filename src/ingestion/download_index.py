"""
Download XLE historical data from Yahoo Finance
→ clean → save as Parquet (ZSTD)

Usage:
    python src/ingestion/download_index.py --start 2017-01-01 --end 2020-12-31
"""

import argparse
import logging
from datetime import date
from pathlib import Path
import string

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("data/parquet/market")
TICKER = "FBX"


def download_xle(
    start: date,
    end: date,
    out_dir: Path,
    force: bool = False,
) -> Path | None:
    """
    Download XLE data → Parquet
    """
    out_path = out_dir / f"fbx_{start}_{end}.parquet"

    if out_path.exists() and not force:
        log.info("File exists — skipping")
        return out_path

    log.info("Downloading %s from %s to %s", TICKER, start, end)

    try:
        df = yf.download(
            TICKER,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
            group_by="ticker"
        )

        # 🔥 sélection propre du ticker (évite MultiIndex galère)
        df = df[TICKER]

        df = df.reset_index()

        # normalisation
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # vérification sécurité
        expected_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]

        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing} | got: {df.columns.tolist()}")

        df = df[expected_cols]

        # feature utile
        df["return"] = df["adj_close"].pct_change()

        # Add returns (useful for correlation later)
        df["return"] = df["adj_close"].pct_change()

        # Save parquet
        df.to_parquet(out_path, compression="zstd")

        size_kb = out_path.stat().st_size // 1024
        log.info("Saved %s (%d KB, %d rows)", out_path.name, size_kb, len(df))

        return out_path

    except Exception as exc:
        log.error("Download failed — %s", exc)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download XLE (Yahoo Finance) → Parquet"
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = download_xle(start, end, out_dir, force=args.force)

    if result:
        log.info("Done")
    else:
        log.warning("Failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()