"""
Phase 4 — Chargement des données financières pour la corrélation LA congestion.

Source unique : CSV téléchargé manuellement.
Placer le fichier dans data/financial/scfi_raw.csv avec colonnes : date, scfi

Sources recommandées pour LA (routes TransPacifique) :
  - FBX01 (Freightos, Far East → NA West Coast) : https://fbx.freightos.com
  - WCI (Drewry World Container Index)           : https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry
  - SCFI composite                               : https://www.sse.net.cn/index/scfi

Usage:
    python src/correlation/fetch_scfi.py
    python src/correlation/fetch_scfi.py --year 2019 --csv data/financial/fbx01_2019.csv
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/financial/scfi_2019.parquet")
SCFI_CSV    = Path("data/financial/scfi_raw.csv")


def load_scfi_from_csv(csv_path: Path, year: int) -> pl.DataFrame:
    """
    Charge un indice financier depuis un CSV téléchargé manuellement.
    Colonnes attendues : date (YYYY-MM-DD), scfi (float).
    Lève FileNotFoundError si le fichier est absent.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier financier introuvable : {csv_path}\n"
            f"Télécharge le CSV manuellement et place-le dans {csv_path}\n"
            f"Sources recommandées pour LA (TransPac) :\n"
            f"  - FBX01 : https://fbx.freightos.com\n"
            f"  - WCI   : https://www.drewry.co.uk\n"
            f"  - SCFI  : https://www.sse.net.cn/index/scfi"
        )

    df = pl.read_csv(csv_path)

    rename_map = {}
    for col in df.columns:
        if col.lower() in ("date", "week", "period"):
            rename_map[col] = "date"
        elif col.lower() in ("scfi", "fbx01", "fbx", "wci", "close", "value", "index", "composite"):
            rename_map[col] = "scfi"
    df = df.rename(rename_map)

    if "date" not in df.columns or "scfi" not in df.columns:
        raise ValueError(
            f"CSV mal formé — colonnes attendues : date, scfi. Trouvé : {df.columns}"
        )

    df = (
        df.with_columns([
            pl.col("date").str.to_date(strict=False),
            pl.col("scfi").cast(pl.Float64),
        ])
        .drop_nulls(subset=["date", "scfi"])
        .filter(pl.col("date").dt.year() == year)
        .sort("date")
        .select(["date", "scfi"])
    )

    if len(df) < 10:
        raise ValueError(
            f"CSV trop court : seulement {len(df)} lignes pour {year}"
        )

    log.info("Indice financier chargé : %d lignes pour %d | range [%.1f, %.1f]",
             len(df), year, df["scfi"].min(), df["scfi"].max())
    return df


def fetch_scfi(
    year: int = 2019,
    output_path: Path | None = None,
    csv_path: Path | None = None,
) -> pl.DataFrame:
    """Charge l'indice financier depuis CSV et sauvegarde en Parquet."""
    out     = Path(output_path) if output_path is not None else OUTPUT_PATH
    csv_src = Path(csv_path)    if csv_path    is not None else SCFI_CSV

    df = load_scfi_from_csv(csv_src, year)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    log.info("Sauvegardé → %s  (%d lignes)", out, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Chargement indice financier Phase 4")
    parser.add_argument("--year",   type=int, default=2019)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--csv",    default=str(SCFI_CSV))
    args = parser.parse_args()

    df = fetch_scfi(
        year=args.year,
        output_path=Path(args.output),
        csv_path=Path(args.csv),
    )
    print(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
