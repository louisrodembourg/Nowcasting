"""
Évaluation du pipeline — métriques + visualisations.

Usage:
    python run_evaluate.py --location la --year 2019
    python run_evaluate.py --location la --year 2020
"""
import argparse
import logging
from pathlib import Path

from src.evaluation.snapshot import compute_snapshot, print_metrics
from src.evaluation.visualize_congestion import generate_all

log = logging.getLogger(__name__)

EVENT_WINDOWS = {
    ("la",      2019): ("2019-06-01", "2019-09-30"),
    ("la",      2020): ("2020-03-15", "2020-06-30"),
    ("la",      2021): ("2021-09-01", "2021-12-31"),
    ("la",      2022): ("2022-01-01", "2022-04-30"),
    ("la",      2023): ("2023-11-01", "2023-12-31"),
    ("la",      2024): ("2024-01-01", "2024-06-30"),
    ("houston", 2017): ("2017-08-17", "2017-09-10"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du pipeline manifold")
    parser.add_argument("--location", default="la")
    parser.add_argument("--year",     type=int, default=2019)
    args = parser.parse_args()

    loc  = args.location
    year = args.year

    gravity_path = Path(f"data/features/{loc}_{year}_gravity_score.parquet")

    if not gravity_path.exists():
        log.error("Gravity score introuvable : %s — lancer Phase 2 d'abord", gravity_path)
        return

    event_start, event_end = EVENT_WINDOWS.get((loc, year), (None, None))

    log.info("=== Évaluation — %s %d ===", loc.upper(), year)

    metrics = compute_snapshot(
        gravity_path=gravity_path,
        location=loc,
        year=year,
        event_start=event_start,
        event_end=event_end,
    )
    print_metrics(metrics)

    log.info("--- Génération des visualisations ---")
    outs = generate_all(gravity_path, loc, year)
    print("\nFichiers générés :")
    for name, path in outs.items():
        print(f"  {name:12s} → {path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
