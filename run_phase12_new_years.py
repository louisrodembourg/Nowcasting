"""
Orchestrateur Phase 1 + Phase 2 pour les années sans features.
Détecte automatiquement quelles années ont des parquets mais pas de features.

Usage:
    python run_phase12_new_years.py --location la
    python run_phase12_new_years.py --location la --years 2017 2018
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from run_phase1 import run_pipeline, save_features, build_config_for_location
from src.manifold.lbo import run_lbo
from src.manifold.gravity_score import run_gravity_score
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)


def detect_missing_years(location: str, parquet_dir: Path, prefix: str) -> list[int]:
    """Retourne les années qui ont des parquets mais pas de features annuelles."""
    years_with_parquet = set()
    for f in parquet_dir.glob(f"{prefix}_*.parquet"):
        parts = f.stem.split("_")
        try:
            years_with_parquet.add(int(parts[-3]))
        except (IndexError, ValueError):
            pass

    missing = []
    for year in sorted(years_with_parquet):
        feat_path = Path(f"data/features/{location}_{year}_daily_features.parquet")
        if not feat_path.exists():
            missing.append(year)

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1+2 pour les nouvelles années")
    parser.add_argument("--location", default="la", choices=list(LOCATIONS.keys()))
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Années à traiter (défaut: détection automatique)")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Ne pas lancer la phase 2 (LBO + gravity)")
    args = parser.parse_args()

    loc     = args.location
    loc_cfg = LOCATIONS[loc]
    parquet_dir = loc_cfg["out_dir"]
    prefix      = loc_cfg["prefix"]

    years = args.years or detect_missing_years(loc, parquet_dir, prefix)
    if not years:
        log.info("Aucune année manquante détectée — rien à faire.")
        return

    log.info("Années à traiter : %s", years)

    config = build_config_for_location(loc)

    for year in years:
        from datetime import date
        start = date(year, 1, 1)
        end   = date(year, 12, 31)

        # ── Phase 1 ────────────────────────────────────────────────────────
        log.info("\n=== PHASE 1 — %s %d (%s → %s) ===", loc.upper(), year, start, end)
        df_features = run_pipeline(
            start, end,
            parquet_dir=parquet_dir,
            prefix=prefix,
            loc_cfg=loc_cfg,
            skip_download=True,
            config=config,
        )

        if len(df_features) == 0:
            log.error("%d : aucune feature produite — année ignorée", year)
            continue

        # Sauvegarde per-year
        year_feat_path = Path(f"data/features/{loc}_{year}_daily_features.parquet")
        df_features.write_parquet(year_feat_path)
        log.info("%d : %d jours → %s", year, len(df_features), year_feat_path)

        # Merge dans le fichier global
        global_feat_path = Path(f"data/features/{loc}_daily_features.parquet")
        save_features(df_features, global_feat_path)

        if args.skip_phase2:
            continue

        # ── Phase 2 — LBO ──────────────────────────────────────────────────
        log.info("\n=== PHASE 2 LBO — %s %d ===", loc.upper(), year)
        year_manifold_path = Path(f"data/features/{loc}_{year}_manifold.parquet")
        run_lbo(
            features_path=year_feat_path,
            output_path=year_manifold_path,
            location=loc,
        )
        log.info("%d : manifold → %s", year, year_manifold_path)

        # ── Phase 2 — Gravity Score ────────────────────────────────────────
        log.info("\n=== PHASE 2 GRAVITY — %s %d ===", loc.upper(), year)
        year_gravity_path = Path(f"data/features/{loc}_{year}_gravity_score.parquet")
        run_gravity_score(
            manifold_path=year_manifold_path,
            output_path=year_gravity_path,
        )
        log.info("%d : gravity → %s", year, year_gravity_path)

        df_grav = pl.read_parquet(year_gravity_path)
        top5 = (df_grav.sort("gravity_score", descending=True)
                .head(5)
                .select(["date", "gravity_score", "waiting_cluster_count"]))
        log.info("Top 5 jours gravity %d :\n%s", year, top5)

    log.info("\n=== TERMINÉ — années traitées : %s ===", years)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
