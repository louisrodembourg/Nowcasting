"""
Phase 2 orchestrator — Manifold Learning + Gravity Score.

Usage (run from Nowcasting/ root):
    python run_phase2.py                              # houston (default)
    python run_phase2.py --location la
    python run_phase2.py --location la --k 7 --n-eigenvectors 8
"""
import argparse
import logging
from pathlib import Path

from src.ingestion.download import LOCATIONS
from src.manifold.lbo import run_lbo, DEFAULT_K, DEFAULT_N_EIGENVECTORS
from src.manifold.gravity_score import run_gravity_score

log = logging.getLogger(__name__)

# Event windows per location (used for gravity score baseline split)
EVENT_WINDOWS = {
    "houston": ("2017-08-17", "2017-09-10"),  # Hurricane Harvey
    "la":      ("2019-06-01", "2019-09-30"),  # placeholder — update as needed
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Manifold + Gravity Score")
    parser.add_argument("--location",      default="houston", choices=list(LOCATIONS.keys()),
                        help="Target port (default: houston)")
    parser.add_argument("--k",             type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors",type=int, default=DEFAULT_N_EIGENVECTORS)
    parser.add_argument("--event-start",   default=None, metavar="YYYY-MM-DD",
                        help="Start of disruption window (overrides default)")
    parser.add_argument("--event-end",     default=None, metavar="YYYY-MM-DD",
                        help="End of disruption window (overrides default)")
    args = parser.parse_args()

    loc = args.location
    features_path = Path(f"data/features/{loc}_daily_features.parquet")
    manifold_path = Path(f"data/features/{loc}_manifold.parquet")
    gravity_path  = Path(f"data/features/{loc}_gravity_score.parquet")

    default_start, default_end = EVENT_WINDOWS.get(loc, (None, None))
    event_start = args.event_start or default_start
    event_end   = args.event_end   or default_end

    log.info("=== Phase 2 — %s ===", loc.upper())
    log.info("Features  : %s", features_path)
    log.info("Manifold  : %s", manifold_path)
    log.info("Gravity   : %s", gravity_path)
    log.info("Event     : %s → %s", event_start, event_end)

    if not features_path.exists():
        log.error("Features file not found: %s — run Phase 1 first", features_path)
        return

    log.info("--- Step 1-4 : LBO + eigenvectors ---")
    df_manifold = run_lbo(
        k=args.k,
        n_eigenvectors=args.n_eigenvectors,
        features_path=features_path,
        output_path=manifold_path,
    )

    log.info("--- Step 5 : Gravity Score ---")
    df = run_gravity_score(
        manifold_path=manifold_path,
        output_path=gravity_path,
        harvey_start=event_start,
        harvey_end=event_end,
    )

    import polars as pl
    print(f"\n--- Gravity Score {loc.upper()} — top 15 days ---")
    print(df.select(["date", "gravity_score", "deviation_score",
                     "blocked_capacity", "is_characteristic"])
          .sort("gravity_score", descending=True)
          .head(15))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
