"""
Phase 4 orchestrator — Correlation gravity_score ↔ SCFI.

Usage (run from Nowcasting/ root):
    python run_phase4.py --location la --year 2019
    python run_phase4.py --location la --year 2019 --skip-fetch
    python run_phase4.py --location la --year 2019 --max-lag 14
    python run_phase4.py --location la --year 2019 --figures-only
"""
import argparse
import logging
from pathlib import Path

from src.ingestion.download import LOCATIONS
from src.correlation.fetch_scfi import fetch_scfi
from src.correlation.align import align_gravity_scfi
from src.correlation.cross_correlation import (
    compute_cross_correlation,
    run_granger_causality,
    summarize_analysis,
)
from src.correlation.visualize_corr import (
    plot_timeseries,
    plot_cross_correlation,
    plot_scatter,
)

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — Correlation gravity ↔ SCFI")
    parser.add_argument("--location",         default="la", choices=list(LOCATIONS.keys()))
    parser.add_argument("--year",             type=int, default=None,
                        help="Année des données (ex: 2019). Dérive tous les chemins.")
    parser.add_argument("--period",           default=None,
                        help="Période multi-années (ex: 2023_2024). Priorité sur --year.")
    parser.add_argument("--max-lag",          type=int, default=21,
                        help="Maximum lag in days for cross-correlation (default: 21)")
    parser.add_argument("--granger-max-lag",  type=int, default=8,
                        help="Maximum lag in weeks for Granger causality (default: 8)")
    parser.add_argument("--align-method",     default="forward_fill",
                        choices=["forward_fill", "linear_interpolate"])
    parser.add_argument("--skip-fetch",       action="store_true",
                        help="Skip SCFI download (reuse existing parquet)")
    parser.add_argument("--figures-only",     action="store_true",
                        help="Regenerate figures from existing intermediate parquets")
    args = parser.parse_args()

    loc    = args.location
    period = args.period or (str(args.year) if args.year else None)

    if period is None:
        log.error(
            "Précise --year YYYY ou --period YYYY_YYYY.\n"
            "  Exemples : --year 2019  |  --period 2023_2024"
        )
        return

    gravity_path = Path(f"data/features/{loc}_{period}_gravity_score.parquet")
    scfi_path    = Path(f"data/financial/scfi_{period}.parquet")
    corr_path    = Path(f"data/features/{loc}_{period}_scfi_correlation.parquet")
    granger_path = Path(f"data/features/{loc}_{period}_scfi_granger.parquet")
    out_dir      = Path("outputs/figures")

    if not gravity_path.exists():
        log.error("Gravity score not found: %s — run Phase 2 first", gravity_path)
        return

    log.info("=== Phase 4 — %s ===", loc.upper())

    # ── Step 1 — Fetch SCFI ──────────────────────────────────────────────────
    if not args.figures_only:
        if args.skip_fetch and scfi_path.exists():
            log.info("--- Step 1 : SCFI (skipped — using %s) ---", scfi_path)
        else:
            log.info("--- Step 1 : Fetch SCFI [%s] ---", period)
            scfi_year = int(period.split("_")[0])
            fetch_scfi(year=scfi_year, output_path=scfi_path)

        # ── Step 2 — Align ───────────────────────────────────────────────────
        log.info("--- Step 2 : Temporal alignment (%s) ---", args.align_method)
        df_aligned = align_gravity_scfi(gravity_path, scfi_path, method=args.align_method)

        # ── Step 3 — Cross-correlation + Granger ─────────────────────────────
        log.info("--- Step 3 : Cross-correlation (lags 0–%d days) ---", args.max_lag)
        gravity_np = df_aligned["gravity_score"].to_numpy()
        scfi_np    = df_aligned["scfi"].to_numpy()

        cross_corr_df = compute_cross_correlation(gravity_np, scfi_np, max_lag=args.max_lag)
        granger_df    = run_granger_causality(gravity_np, scfi_np, max_lag_weeks=args.granger_max_lag)
        summary       = summarize_analysis(cross_corr_df, granger_df)

        corr_path.parent.mkdir(parents=True, exist_ok=True)
        cross_corr_df.write_parquet(corr_path)
        granger_df.write_parquet(granger_path)

        print("\n--- Cross-Correlation (Pearson) ---")
        print(cross_corr_df.select(["lag_days", "pearson_r", "pearson_p", "spearman_r", "n_obs"]))
        print("\n--- Granger Causality ---")
        print(granger_df)
        print(f"\nOptimal lag : {summary['optimal_lag']} days  (r={summary['max_pearson_r']:.4f})")
        print(f"Best Granger: {summary['best_granger_lag']} days  (p={summary['min_granger_p']:.4f})")

    else:
        log.info("--- Figures only mode ---")
        if not corr_path.exists() or not granger_path.exists():
            log.error("Intermediate parquets not found — run without --figures-only first")
            return
        import polars as pl
        df_aligned    = align_gravity_scfi(gravity_path, scfi_path, method=args.align_method)
        cross_corr_df = pl.read_parquet(corr_path)
        granger_df    = pl.read_parquet(granger_path)
        summary       = summarize_analysis(cross_corr_df, granger_df)

    # ── Step 4 — Visualizations ───────────────────────────────────────────────
    log.info("--- Step 4 : Figures ---")
    plot_timeseries(df_aligned, out_dir / f"{loc}_scfi_timeseries.png")
    plot_cross_correlation(cross_corr_df, granger_df, out_dir / f"{loc}_scfi_cross_correlation.png")
    plot_scatter(df_aligned, summary["optimal_lag"], out_dir / f"{loc}_scfi_scatter.png")

    log.info("=== Phase 4 complete ===")
    log.info("Figures → %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
