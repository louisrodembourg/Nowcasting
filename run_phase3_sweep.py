"""
Phase 3 lambda sweep — PINN LWR.

Sweeps lambda_pde / lambda_kin / lambda_bc for a given year and reports
TTC + baseline MSE (moving average) to compare configurations.

Usage (run from Nowcasting/ root):
    python run_phase3_sweep.py --location la --year 2020 \
        --pde-list 0.1,0.3,1.0 --kin-list 0.05,0.2,0.5 --bc-list 0.1
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
import torch

from src.pinns.train import train, DEFAULT_EPOCHS, DEFAULT_LR
from src.pinns.predict import compute_time_to_clear
from src.pinns.visualize_pinn import (
    plot_density_timeseries,
    plot_spacetime_heatmap,
    plot_loss_history,
)

log = logging.getLogger(__name__)


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip().replace("..", ".")
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def _tag_float(val: float) -> str:
    s = f"{val:.3g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


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
    parser = argparse.ArgumentParser(description="Phase 3 — PINN lambda sweep")
    parser.add_argument("--location", default="la")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--percentile", type=float, default=0.9)
    parser.add_argument("--pde-list", default="0.1,0.3,1.0")
    parser.add_argument("--kin-list", default="0.05,0.2,0.5")
    parser.add_argument("--bc-list", default="0.1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rho-threshold", type=float, default=0.85)
    parser.add_argument("--n-consecutive", type=int, default=3)
    parser.add_argument("--n-eval-days", type=int, default=90)
    parser.add_argument("--gravity-weight", type=float, default=0.3)
    parser.add_argument("--ma-window", type=int, default=14)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--make-figs", action="store_true")
    args = parser.parse_args()

    loc = args.location
    year = args.year

    pde_list = _parse_float_list(args.pde_list)
    kin_list = _parse_float_list(args.kin_list)
    bc_list = _parse_float_list(args.bc_list)

    features_path = Path(f"data/features/{loc}_{year}_daily_features.parquet")
    gravity_path = Path(f"data/features/{loc}_{year}_gravity_score.parquet")
    zones_path = Path(f"data/features/{loc}_constituent_zones.parquet")

    if not features_path.exists() or not gravity_path.exists():
        raise FileNotFoundError("Missing features or gravity file for sweep")

    df_features = pl.read_parquet(features_path).sort("date")
    df_gravity = pl.read_parquet(gravity_path).sort("date")

    train_start = df_features["date"].min()
    train_end = df_features["date"].max()

    peak_date, peak_score, threshold = _get_peak_date(df_gravity, args.percentile)
    if peak_date is None:
        raise RuntimeError("No gravity peak found for sweep")

    out_dir = Path("outputs/pinn_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("outputs/figures/pinn_sweep")
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for lam_pde in pde_list:
        for lam_kin in kin_list:
            for lam_bc in bc_list:
                tag = f"pde{_tag_float(lam_pde)}_kin{_tag_float(lam_kin)}_bc{_tag_float(lam_bc)}"
                model_path = Path(f"outputs/models/{loc}_{year}_{tag}_lwr_pinn.pt")
                ttc_path = out_dir / f"{loc}_{year}_{tag}_time_to_clear.parquet"

                log.info("=== %s %s | %s ===", loc.upper(), year, tag)
                if not args.skip_train:
                    train(
                        epochs=args.epochs,
                        lr=args.lr,
                        lambda_pde=lam_pde,
                        lambda_bc=lam_bc,
                        lambda_kin=lam_kin,
                        train_start=train_start,
                        train_end=train_end,
                        features_path=features_path,
                        zones_path=zones_path if zones_path.exists() else None,
                        gravity_path=gravity_path,
                        model_path=model_path,
                    )
                elif not model_path.exists():
                    log.warning("Model missing: %s — skipping", model_path)
                    continue

                df_result = compute_time_to_clear(
                    rho_threshold=args.rho_threshold,
                    n_consecutive=args.n_consecutive,
                    n_eval_days=args.n_eval_days,
                    features_path=features_path,
                    model_path=model_path,
                    output_path=ttc_path,
                    harvey_peak=peak_date,
                    gravity_path=gravity_path,
                    gravity_weight=args.gravity_weight,
                )

                if args.make_figs:
                    location_tag = f"{loc}_{year}_{tag}"
                    plot_density_timeseries(
                        df_result,
                        df_features,
                        df_gravity,
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

                mse_pinn, mse_ma, n_eval = _baseline_metrics(
                    df_result,
                    df_features,
                    args.ma_window,
                )

                best_loss = None
                if model_path.exists():
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                    best_loss = float(ckpt.get("best_loss")) if "best_loss" in ckpt else None

                summary_rows.append({
                    "location": loc,
                    "year": year,
                    "peak_date": peak_date.isoformat(),
                    "gravity_peak": peak_score,
                    "gravity_threshold": threshold,
                    "percentile": args.percentile,
                    "lambda_pde": lam_pde,
                    "lambda_bc": lam_bc,
                    "lambda_kin": lam_kin,
                    "best_loss": best_loss,
                    "ttc_days": _ttc_days(df_result),
                    "mse_pinn": mse_pinn,
                    "mse_ma": mse_ma,
                    "ma_window": args.ma_window,
                    "n_eval_points": n_eval,
                    "model_path": str(model_path),
                    "ttc_path": str(ttc_path),
                })

    summary_df = pl.DataFrame(summary_rows)
    out_csv = out_dir / f"{loc}_{year}_lambda_sweep.csv"
    out_parquet = out_dir / f"{loc}_{year}_lambda_sweep.parquet"
    summary_df.write_csv(out_csv)
    summary_df.write_parquet(out_parquet)
    log.info("Sweep summary written: %s", out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
