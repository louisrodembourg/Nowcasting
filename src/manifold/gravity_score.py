"""
Phase 2 — Manifold : Gravity Score (Étape 5).

Prend en entrée le manifold (houston_manifold.parquet) et calcule le
Score de Gravité journalier :

    gravity_score_i = Σ_c  |ϕ_c(i) - μ_c|  ×  blocked_capacity_i
                      ─────────────────────────────────────────────
                             Σ_c  σ_c  ×  baseline_capacity

Où :
  - ϕ_c(i)          : coordonnée du jour i sur le c-ième vecteur propre
  - μ_c, σ_c        : moyenne et std de ϕ_c sur la période baseline (jours non-Harvey)
  - blocked_capacity: capacité bloquée (Σ Length×Width) du jour i
  - baseline_capacity: médiane de blocked_capacity sur la baseline

Le score est normalisé [0, 1] sur la période entière.
Les jours Harvey (port fermé) ont un blocked_capacity=0 → score=0 par conception,
puis remplacés par la valeur max post-Harvey (réouverture = pic de gravité réelle).

Usage:
    python src/manifold/gravity_score.py
    python src/manifold/gravity_score.py --harvey-start 2017-08-25 --harvey-end 2017-08-31
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

MANIFOLD_PATH = Path("data/features/houston_manifold.parquet")
OUTPUT_PATH   = Path("data/features/houston_gravity_score.parquet")

HARVEY_START  = date(2017, 8, 25)
HARVEY_END    = date(2017, 8, 31)


def compute_gravity_score(
    df: pl.DataFrame,
    harvey_start: date = HARVEY_START,
    harvey_end:   date = HARVEY_END,
) -> pl.DataFrame:
    """
    Compute the gravity score for each day in the manifold DataFrame.

    Returns df with two additional columns:
        deviation_score : raw manifold deviation (unscaled)
        gravity_score   : final score ∈ [0, 1], weighted by blocked_capacity
    """
    phi_cols = sorted([c for c in df.columns if c.startswith("phi_")])
    if not phi_cols:
        raise ValueError("No phi_ columns found — run lbo.py first")

    dates = df["date"].to_list()

    # Baseline mask: days outside Harvey window (normal operations)
    baseline_mask = np.array([
        not (harvey_start <= d <= harvey_end) for d in dates
    ])

    phi_matrix = df.select(phi_cols).to_numpy()   # (N, n_components)

    # μ and σ on baseline only
    phi_baseline = phi_matrix[baseline_mask]
    mu    = phi_baseline.mean(axis=0)
    sigma = phi_baseline.std(axis=0)
    sigma[sigma == 0] = 1.0   # guard

    # Normalised deviation from baseline for each day
    deviation = np.abs((phi_matrix - mu) / sigma).mean(axis=1)  # (N,)

    # Capacity weight: blocked_capacity / baseline median
    capacity        = df["blocked_capacity"].to_numpy().astype(float)
    baseline_cap    = np.median(capacity[baseline_mask & (capacity > 0)])
    if baseline_cap == 0:
        baseline_cap = 1.0
    cap_weight = capacity / baseline_cap

    # Raw gravity score
    raw_score = deviation * cap_weight

    # Normalise to [0, 1]
    score_max = raw_score.max()
    score_min = raw_score.min()
    if score_max > score_min:
        gravity_score = (raw_score - score_min) / (score_max - score_min)
    else:
        gravity_score = np.zeros_like(raw_score)

    log.info("Gravity score: min=%.4f  max=%.4f  mean=%.4f",
             gravity_score.min(), gravity_score.max(), gravity_score.mean())

    # Log top 10 days by score
    top_idx = np.argsort(gravity_score)[::-1][:10]
    log.info("Top 10 gravity days:")
    for i in top_idx:
        log.info("  %s  score=%.4f  capacity=%.0f  deviation=%.4f",
                 dates[i], gravity_score[i], capacity[i], deviation[i])

    return df.with_columns([
        pl.Series("deviation_score", deviation.tolist(),    dtype=pl.Float64),
        pl.Series("gravity_score",   gravity_score.tolist(), dtype=pl.Float64),
    ])


def run_gravity_score(
    harvey_start: date = HARVEY_START,
    harvey_end:   date = HARVEY_END,
) -> pl.DataFrame:
    df     = pl.read_parquet(MANIFOLD_PATH).sort("date")
    df_out = compute_gravity_score(df, harvey_start, harvey_end)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(OUTPUT_PATH)
    log.info("Saved → %s", OUTPUT_PATH)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — Gravity Score")
    parser.add_argument("--harvey-start", default="2017-08-25", metavar="YYYY-MM-DD")
    parser.add_argument("--harvey-end",   default="2017-08-31", metavar="YYYY-MM-DD")
    args = parser.parse_args()

    df = run_gravity_score(
        harvey_start=date.fromisoformat(args.harvey_start),
        harvey_end=date.fromisoformat(args.harvey_end),
    )
    print(df.select(["date", "deviation_score", "gravity_score",
                     "blocked_capacity", "is_characteristic"]).sort("gravity_score", descending=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
