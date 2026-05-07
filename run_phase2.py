"""
Phase 2 orchestrator — Manifold Learning + Gravity Score.

Le chemin des features est dérivé automatiquement depuis --year (ou --period
pour les périodes multi-années). Tous les fichiers intermédiaires portent
le même suffixe temporel — impossible de mélanger des années par erreur.

Usage (run from Nowcasting/ root):
    python run_phase2.py --location la --year 2019
    python run_phase2.py --location la --year 2019 --event-start 2019-06-01 --event-end 2019-09-30
    python run_phase2.py --location la --period 2023_2024 --event-start 2023-11-01 --event-end 2024-06-30
    python run_phase2.py --location houston --year 2017 --event-start 2017-08-17 --event-end 2017-09-10
"""
import argparse
import logging
from pathlib import Path

from src.ingestion.download import LOCATIONS
from src.manifold.lbo import run_lbo, DEFAULT_K, DEFAULT_N_EIGENVECTORS
from src.manifold.gravity_score import run_gravity_score

log = logging.getLogger(__name__)

# Fenêtres événements par défaut (location, period)
EVENT_WINDOWS = {
    ("houston", "2017"):      ("2017-08-17", "2017-09-10"),
    ("la",      "2019"):      ("2019-06-01", "2019-09-30"),
    ("la",      "2020"):      ("2020-03-15", "2020-06-30"),  # COVID-19 shutdown
    ("la",      "2021"):      ("2021-09-01", "2021-12-31"),  # port backlog crisis
    ("la",      "2022"):      ("2022-01-01", "2022-04-30"),  # fin backlog COVID
    ("la",      "2023"):      ("2023-11-01", "2023-12-31"),
    ("la",      "2023_2024"): ("2023-11-01", "2024-06-30"),
    ("la",      "2024"):      ("2024-01-01", "2024-06-30"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Manifold + Gravity Score")
    parser.add_argument("--location",       default="houston", choices=list(LOCATIONS.keys()))
    parser.add_argument("--year",           type=int, default=None,
                        help="Année des données (ex: 2019). Dérive tous les chemins.")
    parser.add_argument("--period",         default=None,
                        help="Période multi-années (ex: 2023_2024). Priorité sur --year.")
    parser.add_argument("--k",              type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    parser.add_argument("--event-start",    default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--event-end",      default=None, metavar="YYYY-MM-DD")
    args = parser.parse_args()

    loc    = args.location
    period = args.period or (str(args.year) if args.year else None)

    if period is None:
        log.error(
            "Précise --year YYYY ou --period YYYY_YYYY.\n"
            "  Exemples : --year 2019  |  --period 2023_2024"
        )
        return

    features_path = Path(f"data/features/{loc}_{period}_daily_features.parquet")
    manifold_path = Path(f"data/features/{loc}_{period}_manifold.parquet")
    gravity_path  = Path(f"data/features/{loc}_{period}_gravity_score.parquet")

    default_start, default_end = EVENT_WINDOWS.get((loc, period), (None, None))
    event_start = args.event_start or default_start
    event_end   = args.event_end   or default_end

    log.info("=== Phase 2 — %s [%s] ===", loc.upper(), period)
    log.info("Features  : %s", features_path)
    log.info("Manifold  : %s", manifold_path)
    log.info("Gravity   : %s", gravity_path)
    log.info("Event     : %s → %s", event_start, event_end)

    if not features_path.exists():
        log.error(
            "Features introuvables : %s\n"
            "  → Lance d'abord : python run_phase1.py --location %s "
            "--start %s-01-01 --end %s-12-31 --no-download",
            features_path, loc, period, period,
        )
        return

    log.info("--- Step 1-4 : LBO + eigenvectors ---")
    run_lbo(
        k=args.k,
        n_eigenvectors=args.n_eigenvectors,
        features_path=features_path,
        output_path=manifold_path,
        save_reference=(loc == "la" and period == "2019"),
    )

    log.info("--- Step 5 : Gravity Score ---")
    df = run_gravity_score(
        manifold_path=manifold_path,
        output_path=gravity_path,
        harvey_start=event_start,
        harvey_end=event_end,
    )

    import polars as pl
    print(f"\n--- Gravity Score {loc.upper()} [{period}] — top 15 days ---")
    print(df.select(["date", "gravity_score", "deviation_score",
                     "blocked_capacity", "is_characteristic"])
          .sort("gravity_score", descending=True)
          .head(15))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
