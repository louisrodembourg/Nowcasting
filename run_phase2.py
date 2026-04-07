"""
Phase 2 orchestrator — Manifold Learning + Gravity Score.

Usage (run from Nowcasting/ root):
    python run_phase2.py
    python run_phase2.py --k 7 --n-eigenvectors 8
"""
import argparse
import logging

from src.manifold.lbo import run_lbo, DEFAULT_K, DEFAULT_N_EIGENVECTORS
from src.manifold.gravity_score import run_gravity_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Manifold + Gravity Score")
    parser.add_argument("--k",              type=int, default=DEFAULT_K)
    parser.add_argument("--n-eigenvectors", type=int, default=DEFAULT_N_EIGENVECTORS)
    args = parser.parse_args()

    log.info("=== Phase 2 — Manifold Learning ===")

    log.info("--- Step 1-4 : LBO + eigenvectors ---")
    run_lbo(k=args.k, n_eigenvectors=args.n_eigenvectors)

    log.info("--- Step 5 : Gravity Score ---")
    df = run_gravity_score()

    import polars as pl
    print("\n--- Gravity Score — top 15 days ---")
    print(df.select(["date", "gravity_score", "deviation_score",
                     "blocked_capacity", "is_characteristic"])
          .sort("gravity_score", descending=True)
          .head(15))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
