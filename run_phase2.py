"""
Phase 2 orchestrator — Manifold géospatial + Gravity Score.

Utilise le pipeline manifold GÉOSPATIAL (LBO sur les zones HDBSCAN)
pour identifier les zones constituantes et calculer le gravity score quotidien.
Les sorties sont directement utilisées par le PINN (Phase 3).

Sorties (mode standard) :
    data/features/{loc}_constituent_zones.parquet  — zones avec lat/lon/phi_k
    data/features/{loc}_gravity_daily.parquet      — gravity_score par jour

Sorties (mode --year-by-year) :
    data/features/{loc}_constituent_zones.parquet  — identifié sur la baseline
    data/features/{loc}_{year}_gravity_daily.parquet  — un fichier par année
    data/features/{loc}_gravity_daily.parquet         — concaténation complète

Usage (run from Nowcasting/ root):
    # Mode standard (une fenêtre) :
    python run_phase2.py --location la \\
        --baseline-start 2019-01-01 --baseline-end 2019-12-31 \\
        --start 2020-01-01 --end 2020-12-31

    # Mode year-by-year (2017→2024, un fichier par an) :
    python run_phase2.py --location la --year-by-year \\
        --baseline-start 2019-01-01 --baseline-end 2019-12-31 \\
        --start 2017-01-01 --end 2024-12-31

    # Scoring uniquement (zones déjà identifiées) :
    python run_phase2.py --location la --score-only \\
        --start 2020-01-01 --end 2020-12-31

    # Year-by-year sans re-identifier les zones :
    python run_phase2.py --location la --score-only --year-by-year \\
        --start 2017-01-01 --end 2024-12-31
"""
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.ingestion.download import LOCATIONS
from src.manifold.manifold_pipeline import (
    _load_config,
    identify_constituent_zones,
    compute_daily_gravity,
    aggregate_gravity_score,
    DEFAULT_K,
    DEFAULT_N_EIGENVECTORS,
)

log = logging.getLogger(__name__)

BASELINE_WINDOWS = {
    "houston": ("2015-01-01", "2016-12-31"),
    "la":      ("2019-01-01", "2019-12-31"),
}

EVENT_WINDOWS = {
    "houston": ("2017-06-01", "2017-10-31"),
    "la":      ("2020-01-01", "2020-12-31"),
}


def _score_period(
    score_start: date,
    score_end: date,
    loc: str,
    zone_df: pl.DataFrame,
    config,
) -> list[dict]:
    """Calcule les gravity scores quotidiens sur une fenêtre et retourne la liste de dicts."""
    all_scores = []
    d = score_start
    total = (score_end - score_start).days + 1
    done = 0
    while d <= score_end:
        daily_df = compute_daily_gravity(d, loc, zone_df, config=config)
        if len(daily_df) > 0:
            agg = aggregate_gravity_score(daily_df)
            all_scores.append(agg)
        d += timedelta(days=1)
        done += 1
        if done % 30 == 0:
            log.info("  %d / %d jours traités...", done, total)
    return all_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Manifold géospatial + Gravity Score")
    parser.add_argument("--location",       default="houston", choices=list(LOCATIONS.keys()))
    parser.add_argument("--k",              type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    parser.add_argument("--baseline-start", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--baseline-end",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--start",          default=None, metavar="YYYY-MM-DD",
                        help="Début de la période de scoring")
    parser.add_argument("--end",            default=None, metavar="YYYY-MM-DD",
                        help="Fin de la période de scoring")
    parser.add_argument("--score-only",     action="store_true",
                        help="Sauter l'identification des zones (utiliser le parquet existant)")
    parser.add_argument("--year-by-year",   action="store_true",
                        help="Sauvegarder un fichier par année + concaténation complète")
    args = parser.parse_args()

    loc = args.location
    bl_start_s, bl_end_s = BASELINE_WINDOWS.get(loc, (None, None))
    ev_start_s, ev_end_s = EVENT_WINDOWS.get(loc, (None, None))

    if not args.score_only:
        if not (args.baseline_start or bl_start_s):
            parser.error("--baseline-start requis")
        if not (args.baseline_end or bl_end_s):
            parser.error("--baseline-end requis")
    if not (args.start or ev_start_s):
        parser.error("--start requis")
    if not (args.end or ev_end_s):
        parser.error("--end requis")

    score_start = date.fromisoformat(args.start or ev_start_s)
    score_end   = date.fromisoformat(args.end   or ev_end_s)
    zones_path  = Path(f"data/features/{loc}_constituent_zones.parquet")

    log.info("=== Phase 2 — %s (manifold géospatial) ===", loc.upper())
    log.info("Scoring  : %s → %s", score_start, score_end)

    config = _load_config(loc)

    # ── Étape 1 : Identification des zones constituantes ──────────────────────
    if not args.score_only:
        baseline_start = date.fromisoformat(args.baseline_start or bl_start_s)
        baseline_end   = date.fromisoformat(args.baseline_end   or bl_end_s)
        log.info("Baseline : %s → %s", baseline_start, baseline_end)
        log.info("--- Étape 1 : Identification des zones constituantes (LBO géospatial) ---")

        zone_df = identify_constituent_zones(
            baseline_start, baseline_end, loc,
            k=args.k, n_eigenvectors=args.n_eigenvectors,
            config=config,
        )
        n_const = int(zone_df.filter(pl.col("is_constituent"))["is_constituent"].sum())
        log.info("Zones constituantes : %d / %d", n_const, len(zone_df))

        zones_path.parent.mkdir(parents=True, exist_ok=True)
        zone_df.write_parquet(zones_path)
        log.info("Zones sauvegardées → %s", zones_path)
    else:
        if not zones_path.exists():
            log.error("--score-only : zones introuvables : %s — relancez sans --score-only",
                      zones_path)
            return
        log.info("--- Étape 1 : skip (--score-only) ---")
        zone_df = pl.read_parquet(zones_path)

    # ── Étape 2 : Calcul du gravity score ─────────────────────────────────────
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    if args.year_by_year:
        # Un fichier par année + concaténation finale
        years = range(score_start.year, score_end.year + 1)
        all_dfs: list[pl.DataFrame] = []

        for year in years:
            y_start = max(score_start, date(year, 1, 1))
            y_end   = min(score_end,   date(year, 12, 31))
            log.info("--- Année %d : %s → %s ---", year, y_start, y_end)

            scores = _score_period(y_start, y_end, loc, zone_df, config)

            if not scores:
                log.warning("  Aucun score pour %d — fichiers Parquet manquants ?", year)
                continue

            year_df   = pl.DataFrame(scores)
            year_path = features_dir / f"{loc}_{year}_gravity_daily.parquet"
            year_df.write_parquet(year_path)
            log.info("  %d jours → %s", len(year_df), year_path)

            _print_summary(year_df, f"{loc.upper()} {year}")
            all_dfs.append(year_df)

        if not all_dfs:
            log.error("Aucun score calculé sur la période complète.")
            return

        full_df = pl.concat(all_dfs).sort("date")
        full_path = features_dir / f"{loc}_gravity_daily.parquet"
        full_df.write_parquet(full_path)
        log.info("Concaténation complète → %s  (%d jours)", full_path, len(full_df))

    else:
        # Mode standard : une seule fenêtre → un seul fichier
        log.info("--- Étape 2 : Gravity score (%s → %s) ---", score_start, score_end)
        scores = _score_period(score_start, score_end, loc, zone_df, config)

        if not scores:
            log.error("Aucun score calculé — vérifier les fichiers Parquet dans data/parquet/%s/",
                      loc)
            return

        scores_df  = pl.DataFrame(scores)
        out_path   = features_dir / f"{loc}_gravity_daily.parquet"
        scores_df.write_parquet(out_path)
        log.info("Gravity score → %s  (%d jours)", out_path, len(scores_df))
        _print_summary(scores_df, loc.upper())


def _print_summary(df: pl.DataFrame, label: str) -> None:
    scores_np = df["gravity_score"].to_numpy()
    max_idx   = int(scores_np.argmax())
    print(f"\n  {label} : moy={scores_np.mean():.0f}  "
          f"max={scores_np.max():.0f} ({df['date'].to_list()[max_idx]})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
