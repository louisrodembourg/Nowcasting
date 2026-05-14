"""
Phase 3 walk-forward — train multi-year, test next year (TTC).

Trains the PINN on a multi-year window, then predicts TTC on a future year
using a gravity-score percentile to select the peak date.

Usage (run from Nowcasting/ root):
    python run_phase3_walkforward.py --location la --train-start 2019-01-01 \
        --train-end 2020-12-31 --test-year 2021 --percentile 0.9
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from src.pinns.train import train, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_LAMBDA_KIN
from src.pinns.predict import compute_time_to_clear
from src.pinns.visualize_pinn import (
    plot_density_timeseries,
    plot_spacetime_heatmap,
    plot_loss_history,
)

log = logging.getLogger(__name__)


def _get_peak_date(
    gravity_df: pl.DataFrame,
    percentile: float,
) -> tuple[date | None, float | None, float | None]:
    if gravity_df is None or len(gravity_df) == 0:
        return None, None, None
    scores = gravity_df["gravity_score"].to_numpy().astype(float)
    if scores.size == 0:
        return None, None, None
    threshold = float(np.quantile(scores, percentile))
    candidates = (
        gravity_df
        .filter(pl.col("gravity_score") >= threshold)
        .sort("gravity_score", descending=True)
    )
    if len(candidates) == 0:
        return None, None, threshold
    peak_date = candidates["date"][0]
    peak_score = float(candidates["gravity_score"][0])
    return peak_date, peak_score, threshold


def _compute_obs_rho(df_features: pl.DataFrame) -> pl.DataFrame:
    if "blocked_capacity" in df_features.columns and df_features["blocked_capacity"].max() > 0:
        full_cap = df_features["blocked_capacity"].to_numpy().astype(float)
        cap_95   = float(np.percentile(full_cap[full_cap > 0], 95))
        rho_obs  = np.clip(
            df_features["blocked_capacity"].to_numpy().astype(float) / cap_95,
            0.0,
            1.0,
        )
    else:
        rho_obs = df_features["utilization_rate_rho"].to_numpy().astype(float)
    return pl.DataFrame({
        "date": df_features["date"],
        "rho_obs": rho_obs,
    })


def _baseline_metrics(
    df_result: pl.DataFrame,
    df_features: pl.DataFrame,
    ma_window: int,
) -> tuple[float | None, float | None, int]:
    if len(df_result) == 0 or len(df_features) == 0:
        return None, None, 0

    df_obs = _compute_obs_rho(df_features).sort("date")
    df_obs = df_obs.with_columns(
        pl.col("rho_obs").rolling_mean(ma_window, min_periods=1).alias("rho_ma")
    )
    df_join = df_result.select(["date", "rho_pred"]).join(df_obs, on="date", how="inner")
    if len(df_join) == 0:
        return None, None, 0

    pred = df_join["rho_pred"].to_numpy().astype(float)
    obs = df_join["rho_obs"].to_numpy().astype(float)
    base = df_join["rho_ma"].to_numpy().astype(float)

    mse_pinn = float(np.mean((pred - obs) ** 2))
    mse_ma = float(np.mean((base - obs) ** 2))
    return mse_pinn, mse_ma, len(df_join)


def _ttc_days(df_result: pl.DataFrame) -> int | None:
    if "time_to_clear_days" not in df_result.columns:
        return None
    vals = df_result["time_to_clear_days"].drop_nulls()
    if len(vals) == 0:
        return None
    return int(vals[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — walk-forward TTC")
    parser.add_argument("--location", default="la")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--test-year", type=int, default=2021)
    parser.add_argument("--percentile", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-bc", type=float, default=0.1)
    parser.add_argument("--lambda-kin", type=float, default=DEFAULT_LAMBDA_KIN)
    parser.add_argument("--rho-threshold", type=float, default=0.85)
    parser.add_argument("--n-consecutive", type=int, default=3)
    parser.add_argument("--n-eval-days", type=int, default=90)
    parser.add_argument("--gravity-weight", type=float, default=0.3)
    parser.add_argument("--ma-window", type=int, default=14)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()

    loc = args.location
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)

    train_features_path = Path(f"data/features/{loc}_daily_features.parquet")
    train_gravity_path = Path(f"data/features/{loc}_gravity_daily.parquet")

    test_features_path = Path(f"data/features/{loc}_{args.test_year}_daily_features.parquet")
    test_gravity_path = Path(f"data/features/{loc}_{args.test_year}_gravity_score.parquet")

    zones_path = Path(f"data/features/{loc}_constituent_zones.parquet")

    if not train_features_path.exists():
        raise FileNotFoundError(f"Missing train features: {train_features_path}")
    if not train_gravity_path.exists():
        raise FileNotFoundError(f"Missing train gravity: {train_gravity_path}")
    if not test_features_path.exists():
        raise FileNotFoundError(f"Missing test features: {test_features_path}")
    if not test_gravity_path.exists():
        raise FileNotFoundError(f"Missing test gravity: {test_gravity_path}")

    df_test_features = pl.read_parquet(test_features_path).sort("date")
    df_test_gravity = pl.read_parquet(test_gravity_path).sort("date")

    peak_date, peak_score, threshold = _get_peak_date(df_test_gravity, args.percentile)
    if peak_date is None:
        raise RuntimeError("No gravity peak found in test year")

    model_tag = f"{train_start.isoformat()}_{train_end.isoformat()}"
    model_path = Path(f"outputs/models/{loc}_lwr_pinn_{model_tag}.pt")

    out_dir = Path("outputs/pinn_walkforward")
    fig_dir = Path("outputs/figures/pinn_walkforward")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ttc_path = out_dir / f"{loc}_{args.test_year}_time_to_clear.parquet"

    log.info("=== Walk-forward %s ===", loc.upper())
    log.info("Train window : %s → %s", train_start, train_end)
    log.info("Test year    : %s", args.test_year)
    log.info("Gravity peak : %s (score=%.4f, %.0fth pct=%.4f)",
             peak_date, peak_score, args.percentile * 100, threshold)

    if not args.skip_train:
        train(
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_kin=args.lambda_kin,
            train_start=train_start,
            train_end=train_end,
            features_path=train_features_path,
            zones_path=zones_path if zones_path.exists() else None,
            gravity_path=train_gravity_path,
            model_path=model_path,
        )
    elif not model_path.exists():
        raise FileNotFoundError(f"Model missing: {model_path}")

    df_result = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
        n_eval_days=args.n_eval_days,
        features_path=test_features_path,
        model_path=model_path,
        output_path=ttc_path,
        harvey_peak=peak_date,
        gravity_path=test_gravity_path,
        gravity_weight=args.gravity_weight,
        time_mode="absolute",
    )

    mse_pinn, mse_ma, n_eval = _baseline_metrics(
        df_result,
        df_test_features,
        args.ma_window,
    )

    summary = pl.DataFrame([
        {
            "location": loc,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_year": args.test_year,
            "peak_date": peak_date.isoformat(),
            "gravity_peak": peak_score,
            "gravity_threshold": threshold,
            "percentile": args.percentile,
            "ttc_days": _ttc_days(df_result),
            "mse_pinn": mse_pinn,
            "mse_ma": mse_ma,
            "ma_window": args.ma_window,
            "n_eval_points": n_eval,
            "model_path": str(model_path),
            "ttc_path": str(ttc_path),
        }
    ])

    summary_path = out_dir / f"{loc}_{args.test_year}_walkforward_summary.csv"
    summary.write_csv(summary_path)
    log.info("Summary written: %s", summary_path)

    if not args.skip_viz:
        location_tag = f"{loc}_{args.test_year}_walkforward"
        plot_density_timeseries(
            df_result,
            df_test_features,
            df_test_gravity,
            peak_date=peak_date,
            location=location_tag,
            output_dir=fig_dir,
            ma_window=args.ma_window,
        )
        plot_spacetime_heatmap(
            model_path=model_path,
            train_start=train_start,
            train_end=train_end,
            peak_date=peak_date,
            location=location_tag,
            output_dir=fig_dir,
        )
        plot_loss_history(
            model_path=model_path,
            location=location_tag,
            output_dir=fig_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
