"""
Phase 3 — Evaluate PINN forecasting performance.

Rolling evaluation:
  1. Train on [train_start, train_end]
  2. For each day in [eval_start, eval_end], predict ρ(x, t) and compare with actual
  3. Metrics: RMSE, MAE, R² per day, persistence baseline comparison

Usage:
    python src/pinns/evaluate.py --location houston --train-start 2019-01-01 --train-end 2019-01-21 --eval-days 10
    python src/pinns/evaluate.py --location houston --train-start 2019-01-01 --train-end 2019-03-31 --eval-days 30
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pinns.train import train as run_train
from src.pinns.data_prep import (
    build_rho_v_tensors,
    CHANNEL_AXES,
    compute_channel_length,
)
from src.pinns.lwr_pinn import LWRPINN, get_device

log = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs/figures")


def evaluate(
    location="houston",
    train_start=date(2019, 1, 1),
    train_end=date(2019, 1, 21),
    eval_days=10,
    epochs=1000,
    dx_km=2.0,
):
    if isinstance(train_start, str):
        train_start = date.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = date.fromisoformat(train_end)

    eval_start = train_end + timedelta(days=1)
    eval_end = eval_start + timedelta(days=eval_days - 1)

    log.info(
        "Train: %s -> %s  |  Evaluate: %s -> %s",
        train_start,
        train_end,
        eval_start,
        eval_end,
    )

    # ── Train model ──
    model, history, train_meta = run_train(
        location=location,
        start=train_start,
        end=train_end,
        epochs=epochs,
        dx_km=dx_km,
        model_name=f"eval_{location}_{train_start.isoformat()}_{train_end.isoformat()}",
    )
    device = get_device()
    model.eval()

    # ── Get evaluation data ──
    X_eval, y_eval, eval_meta = build_rho_v_tensors(
        eval_start,
        eval_end,
        location,
        dx_km=dx_km,
        use_raw_velocity=True,
    )

    if len(X_eval) == 0:
        log.error("No evaluation data available")
        return

    rho_max = eval_meta["rho_max"]
    v_max = eval_meta["v_max"]
    t_days_max = eval_meta.get("t_days_max", eval_meta.get("t_max", 1))
    x_max = eval_meta["x_max"]
    n_train_days = (train_end - train_start).days + 1

    # ── Predict on eval points ──
    x_norm = torch.tensor(X_eval[:, 0], dtype=torch.float32, device=device).reshape(
        -1, 1
    )
    t_norm = torch.tensor(X_eval[:, 1], dtype=torch.float32, device=device).reshape(
        -1, 1
    )

    with torch.no_grad():
        rho_pred_norm, v_pred_norm = model(x_norm, t_norm)
    rho_pred = rho_pred_norm.cpu().numpy().ravel() * rho_max
    rho_true = y_eval[:, 0] * rho_max

    # ── Metrics ──
    residuals = rho_pred - rho_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((rho_true - rho_true.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    # ── Baselines ──
    persist_rmse = np.nan
    persist_mae = np.nan
    mean_rmse = np.nan
    mean_mae = np.nan
    mean_train = rho_true.mean()
    if len(rho_true) > 1:
        mean_resid = rho_true - mean_train
        mean_rmse = float(np.sqrt(np.mean(mean_resid**2)))
        mean_mae = float(np.mean(np.abs(mean_resid)))

    log.info(
        "RMSE=%.3f  MAE=%.3f  R2=%.3f  |  Mean-baseline RMSE=%.3f  MAE=%.3f",
        rmse,
        mae,
        r2,
        mean_rmse,
        mean_mae,
    )

    # ── Per-day metrics ──
    n_bins = eval_meta["n_bins"]
    n_days = eval_meta["n_days"]
    day_metrics = []
    unique_days = np.unique((X_eval[:, 1] * t_days_max).astype(int))
    for day_offset in sorted(unique_days):
        mask = (X_eval[:, 1] * t_days_max).astype(int) == day_offset
        if mask.sum() == 0:
            continue
        r_daily_true = rho_true[mask]
        r_daily_pred = rho_pred[mask]
        day_rmse = float(np.sqrt(np.mean((r_daily_pred - r_daily_true) ** 2)))
        day_mae = float(np.mean(np.abs(r_daily_pred - r_daily_true)))
        day_metrics.append(
            {
                "day_offset": int(day_offset),
                "date": eval_meta["dates"][day_offset]
                if day_offset < len(eval_meta["dates"])
                else str(day_offset),
                "rmse": day_rmse,
                "mae": day_mae,
                "n_points": int(mask.sum()),
            }
        )

    metrics_df = pl.DataFrame(day_metrics)

    # ── Plots ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual scatter
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=rho_true,
            y=rho_pred,
            mode="markers",
            marker=dict(size=4, opacity=0.5, color="#1565C0"),
            name="Predictions",
        )
    )
    max_val = max(rho_true.max(), rho_pred.max())
    fig1.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Perfect fit",
        )
    )
    fig1.update_layout(
        title=dict(text=f"{location.upper()} — Predicted vs Actual ρ", x=0.5),
        xaxis_title="Actual ρ (vessels/km)",
        yaxis_title="Predicted ρ (vessels/km)",
        template="plotly_white",
        height=500,
        annotations=[
            dict(
                x=max_val * 0.7,
                y=max_val * 0.15,
                text=f"RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}",
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
    fig1.write_html(OUTPUT_DIR / f"{location}_pinn_eval_scatter.html")

    # 2. Daily RMSE
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=metrics_df["date"].to_list(),
            y=metrics_df["rmse"].to_list(),
            name="Daily RMSE",
            marker_color="#1565C0",
        )
    )
    fig2.add_hline(
        y=rmse,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Overall RMSE={rmse:.2f}",
    )
    fig2.update_layout(
        title=dict(text=f"{location.upper()} — Daily RMSE (holdout)", x=0.5),
        xaxis_title="Date",
        yaxis_title="RMSE",
        template="plotly_white",
        height=400,
    )
    fig2.write_html(OUTPUT_DIR / f"{location}_pinn_eval_daily_rmse.html")

    # 3. Time series: first bin
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    dates_str = eval_meta["dates"]
    mid_bin = n_bins // 2
    mask_mid = (X_eval[:, 0] * n_bins).astype(int) == mid_bin
    if mask_mid.sum() > 0:
        ts_true = rho_true[mask_mid]
        ts_pred = rho_pred[mask_mid]
        fig3.add_trace(
            go.Scatter(
                x=dates_str[: len(ts_true)],
                y=ts_true,
                mode="lines+markers",
                name="Actual ρ",
                line=dict(color="#1565C0", width=2),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=dates_str[: len(ts_pred)],
                y=ts_pred,
                mode="lines+markers",
                name="Predicted ρ",
                line=dict(color="#E65100", width=2, dash="dot"),
            )
        )
        fig3.update_layout(
            title=dict(text=f"{location.upper()} — ρ(x≈mid, t) holdout", x=0.5),
            xaxis_title="Date",
            yaxis_title="ρ (vessels/km)",
            template="plotly_white",
            height=400,
        )
        fig3.write_html(OUTPUT_DIR / f"{location}_pinn_eval_timeseries.html")

    # ── Print summary ──
    print(f"\n=== Evaluation: {location} ===")
    print(f"  Train: {train_start} -> {train_end}")
    print(f"  Eval:  {eval_start} -> {eval_end}")
    print(f"  Points: {len(X_eval)} ({n_bins} bins x {n_days} days)")
    print(f"  -----------------------------------------")
    print(f"  RMSE:             {rmse:.3f} vessels/km")
    print(f"  MAE:              {mae:.3f} vessels/km")
    print(f"  R2:               {r2:.3f}")
    print(f"  Mean-baseline RMSE: {mean_rmse:.3f} vessels/km")
    print(f"  Mean-baseline MAE:  {mean_mae:.3f} vessels/km")
    print(f"  -----------------------------------------")
    print(f"  Plots:")
    print(f"    {OUTPUT_DIR / f'{location}_pinn_eval_scatter.html'}")
    print(f"    {OUTPUT_DIR / f'{location}_pinn_eval_daily_rmse.html'}")
    print(f"    {OUTPUT_DIR / f'{location}_pinn_eval_timeseries.html'}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_baseline_rmse": mean_rmse,
        "mean_baseline_mae": mean_mae,
        "n_train_days": (train_end - train_start).days + 1,
        "n_eval_days": n_days,
        "n_points": len(X_eval),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Evaluate PINN")
    parser.add_argument("--location", default="houston")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2019-01-21")
    parser.add_argument("--eval-days", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--dx-km", type=float, default=2.0)
    args = parser.parse_args()

    evaluate(
        location=args.location,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        eval_days=args.eval_days,
        epochs=args.epochs,
        dx_km=args.dx_km,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
