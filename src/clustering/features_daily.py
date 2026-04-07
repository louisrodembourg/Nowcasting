"""
Extract the 13 daily feature vector for the Houston manifold input.

Features (one row per day):
  1.  vessel_count           — distinct MMSI in bbox that day
  2.  SOG_mean               — mean corrected SOG across all messages
  3.  SOG_std                — std of corrected SOG
  4.  SOG_median             — median corrected SOG
  5.  utilization_rate_rho   — fraction of distinct MMSI with ≥1 static episode
  6.  hdbscan_cluster_count  — number of non-noise HDBSCAN clusters
  7.  hdbscan_noise_ratio    — noise episodes / total static episodes
  8.  membership_score_mean  — mean HDBSCAN membership probability (non-noise)
  9.  membership_score_std   — std of membership probabilities
  10. draft_mean             — mean max-draft among static episodes (m)
  11. draft_std              — std of draft
  12. blocked_capacity       — Σ(Length × Width) for static episodes (m² proxy)
  13. tanker_ratio           — tanker MMSI fraction among static MMSI (VesselType 80–89)

Usage (standalone, one day):
    python src/clustering/features_daily.py --date 2017-07-01
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from src.clustering.hdbscan_daily import cluster_day

log = logging.getLogger(__name__)

SOG_STATIC_THRESHOLD = 1.0
PARQUET_DIR = Path("data/parquet/houston")


def _f(val, default: float = 0.0) -> float:
    """Safely cast a nullable Polars scalar to float."""
    return float(val) if val is not None else default


def compute_daily_features(
    day_data: Union[Path, pl.DataFrame],
    cluster_df: Optional[pl.DataFrame],
    d: date,
) -> Optional[dict]:
    """
    Compute 13 daily features for the manifold.

    Args:
        day_data:    Path to raw Parquet OR a prepared DataFrame (with SOG_corr column).
                     When a DataFrame is passed (already prepared by prepare_kinematics),
                     corrected SOG is used for traffic stats. Otherwise raw SOG is used.
        cluster_df:  Output of cluster_day() — one row per static episode + cluster metadata.
                     Pass None if HDBSCAN produced no result (HDBSCAN features default to 0).
        d:           The date, used as the 'date' key in the output dict.

    Returns a dict with 14 keys (date + 13 features), or None if data is missing.
    """
    if isinstance(day_data, Path):
        if not day_data.exists():
            log.warning("%s: Parquet not found — %s", d, day_data)
            return None
        all_df = pl.read_parquet(day_data)
        sog_col = "SOG"
    else:
        all_df  = day_data
        sog_col = "SOG_corr" if "SOG_corr" in all_df.columns else "SOG"

    # --- Overall traffic (all messages, all vessels) -------------------------
    vessel_count = all_df["MMSI"].n_unique()
    sog_mean     = _f(all_df[sog_col].mean())
    sog_std      = _f(all_df[sog_col].std())
    sog_median   = _f(all_df[sog_col].median())

    # --- Static fraction: distinct MMSI with ≥1 static episode --------------
    static_mmsi_count = (
        all_df.filter(pl.col(sog_col) < SOG_STATIC_THRESHOLD)["MMSI"].n_unique()
    )
    utilization_rate_rho = static_mmsi_count / vessel_count if vessel_count > 0 else 0.0

    # --- HDBSCAN-derived features --------------------------------------------
    if cluster_df is None or len(cluster_df) == 0:
        log.warning("%s: no cluster data — HDBSCAN features set to 0", d)
        return {
            "date":                  d.isoformat(),
            "vessel_count":          vessel_count,
            "SOG_mean":              round(sog_mean,   4),
            "SOG_std":               round(sog_std,    4),
            "SOG_median":            round(sog_median, 4),
            "utilization_rate_rho":  round(utilization_rate_rho, 4),
            "hdbscan_cluster_count": 0,
            "hdbscan_noise_ratio":   1.0,
            "membership_score_mean": 0.0,
            "membership_score_std":  0.0,
            "draft_mean":            0.0,
            "draft_std":             0.0,
            "blocked_capacity":      0.0,
            "tanker_ratio":          0.0,
        }

    labels     = cluster_df["cluster_label"].to_numpy()
    n_episodes = len(cluster_df)                              # static episodes
    n_noise    = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    non_noise  = cluster_df.filter(pl.col("cluster_label") >= 0)

    hdbscan_noise_ratio   = n_noise / n_episodes if n_episodes > 0 else 1.0
    membership_score_mean = _f(non_noise["membership_score"].mean())
    membership_score_std  = _f(non_noise["membership_score"].std())

    # Draft: ignore zero / unknown values
    draft_series   = cluster_df.filter(pl.col("Draft") > 0)["Draft"]
    draft_mean     = _f(draft_series.mean())
    draft_std      = _f(draft_series.std())

    # Blocked capacity: Σ(Length × Width) — ignore vessels with unknown dimensions
    cap_series = (
        cluster_df
        .filter((pl.col("Length") > 0) & (pl.col("Width") > 0))
        .select((pl.col("Length") * pl.col("Width")).alias("cap"))["cap"]
    )
    blocked_capacity = _f(cap_series.sum())

    # Tanker ratio: count unique MMSI (not episodes) with VesselType 80–89
    n_static_mmsi  = cluster_df["MMSI"].n_unique()
    tanker_mmsi    = cluster_df.filter(pl.col("VesselType").is_between(80, 89))["MMSI"].n_unique()
    tanker_ratio   = tanker_mmsi / n_static_mmsi if n_static_mmsi > 0 else 0.0

    return {
        "date":                  d.isoformat(),
        "vessel_count":          vessel_count,
        "SOG_mean":              round(sog_mean,   4),
        "SOG_std":               round(sog_std,    4),
        "SOG_median":            round(sog_median, 4),
        "utilization_rate_rho":  round(utilization_rate_rho, 4),
        "hdbscan_cluster_count": n_clusters,
        "hdbscan_noise_ratio":   round(hdbscan_noise_ratio,   4),
        "membership_score_mean": round(membership_score_mean, 4),
        "membership_score_std":  round(membership_score_std,  4),
        "draft_mean":            round(draft_mean, 2),
        "draft_std":             round(draft_std,  2),
        "blocked_capacity":      round(blocked_capacity, 1),
        "tanker_ratio":          round(tanker_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 13 daily features for one day — Houston"
    )
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--parquet-dir", default=str(PARQUET_DIR))
    args = parser.parse_args()

    d            = date.fromisoformat(args.date)
    parquet_path = Path(args.parquet_dir) / f"houston_{d.strftime('%Y_%m_%d')}.parquet"

    cluster_df, prepared_df = cluster_day(parquet_path)
    features = compute_daily_features(
        prepared_df if prepared_df is not None else parquet_path,
        cluster_df,
        d,
    )

    if features:
        for k, v in features.items():
            print(f"  {k:<26} {v}")
    else:
        print("No features computed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
