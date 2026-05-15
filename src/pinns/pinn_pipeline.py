"""
Phase 3 — PINN Pipeline: train + predict in one command.

Usage:
    python src/pinns/pinn_pipeline.py --train --location houston --start 2019-01-01 --end 2019-03-31
    python src/pinns/pinn_pipeline.py --predict --model outputs/models/foo.pt
    python src/pinns/pinn_pipeline.py --both --location houston --start 2019-01-01 --end 2019-03-31
    python src/pinns/pinn_pipeline.py --data-only --location houston --start 2019-01-01 --end 2019-01-07
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pinns.train import train as run_train
from src.pinns.predict import predict_time_to_clear
from src.pinns.data_prep import (
    build_rho_v_tensors,
    CHANNEL_AXES,
    compute_channel_length,
)

log = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs/figures")


def plot_data_preview(X, y, meta):
    dates = meta["dates"]
    channel_len = meta["channel_len_km"]
    bin_centers = meta["bin_centers"]
    n_days = meta["n_days"]
    n_bins = meta["n_bins"]

    rho_grid = np.zeros((n_bins, n_days))
    v_grid = np.zeros((n_bins, n_days))
    for k in range(len(X)):
        xi = int(X[k, 0] * n_bins)
        if xi >= n_bins:
            xi = n_bins - 1
        tj = int(X[k, 1] * n_days)
        if tj >= n_days:
            tj = n_days - 1
        rho_grid[xi, tj] = y[k, 0]
        v_grid[xi, tj] = y[k, 1]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Normalized density ρ(x,t)", "Normalized velocity v(x,t)"),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=dates,
            y=[f"{c:.1f}" for c in bin_centers],
            colorscale="Blues",
            zsmooth="best",
            colorbar=dict(title="ρ norm"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=v_grid,
            x=dates,
            y=[f"{c:.1f}" for c in bin_centers],
            colorscale="Reds",
            zsmooth="best",
            colorbar=dict(title="v norm"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="x (km)", row=1, col=1)
    fig.update_yaxes(title_text="x (km)", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{meta['location'].upper()} — Data preview: {meta['start']} to {meta['end']}",
            x=0.5,
        ),
        height=700,
        template="plotly_white",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{meta['location']}_pinn_data_preview.html"
    fig.write_html(path)
    log.info("Saved data preview → %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — PINN Pipeline")
    parser.add_argument("--location", default="houston")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--predict", action="store_true")
    mode.add_argument("--both", action="store_true")
    mode.add_argument("--data-only", action="store_true")

    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", help="Path to .pt checkpoint (for --predict)")
    parser.add_argument(
        "--model-name", default=None, help="Output model name (default: auto)"
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dx-km", type=float, default=2.0)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-kin", type=float, default=0.05)
    parser.add_argument("--constituent-path", default=None)

    parser.add_argument(
        "--peak-date", default=None, help="Peak congestion date for TTC"
    )
    parser.add_argument("--rho-threshold-frac", type=float, default=0.3)
    parser.add_argument("--n-eval-days", type=int, default=90)

    args = parser.parse_args()

    if not args.train and not args.predict and not args.both and not args.data_only:
        parser.print_help()
        return

    # ── Data-only mode ────────────────────────────────────────────────────
    if args.data_only:
        if not args.start or not args.end:
            parser.error("--start and --end required for --data-only")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        X, y, meta = build_rho_v_tensors(
            start,
            end,
            args.location,
            dx_km=args.dx_km,
            constituent_path=args.constituent_path,
            use_raw_velocity=True,
        )
        print(f"\n=== Data summary ===")
        print(f"  Points:  {len(X)}")
        print(f"  Bins:    {meta['n_bins']}")
        print(f"  Days:    {meta['n_days']}")
        print(f"  Channel: {meta['channel_len_km']:.1f} km")
        print(f"  rho range: [{meta['rho_min']:.2f}, {meta['rho_max']:.2f}]")
        print(f"  v max:   {meta['v_max']:.2f}")
        plot_data_preview(X, y, meta)
        return

    # ── Train mode ────────────────────────────────────────────────────────
    if args.train or args.both:
        if not args.start or not args.end:
            parser.error("--start and --end required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

        model, history, meta = run_train(
            location=args.location,
            start=start,
            end=end,
            epochs=args.epochs,
            lr=args.lr,
            dx_km=args.dx_km,
            lambda_pde=args.lambda_pde,
            lambda_kin=args.lambda_kin,
            constituent_path=args.constituent_path,
            model_name=args.model_name,
        )

        # Save loss curve as numpy (avoids Plotly MemoryError on some systems)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(
            OUTPUT_DIR / f"{args.location}_pinn_training_loss.npy", np.array(history)
        )
        log.info(
            "Saved loss curve -> %s",
            OUTPUT_DIR / f"{args.location}_pinn_training_loss.npy",
        )

        model_path = Path(
            f"outputs/models/{args.model_name or f'pinn_{args.location}_{args.start}_{args.end}'}.pt"
        )
        print(f"\n  Training done -> {model_path}")

    # ── Predict mode ─────────────────────────────────────────────────────
    if args.predict or args.both:
        if args.both:
            # Auto-detect model from training
            model_name = (
                args.model_name or f"pinn_{args.location}_{args.start}_{args.end}"
            )
            args.model = str(Path(f"outputs/models/{model_name}.pt"))
            # Default peak_date = first day after training (out-of-sample)
            if not args.peak_date:
                train_end = date.fromisoformat(args.end)
                args.peak_date = (train_end + timedelta(days=1)).isoformat()
                log.info("Auto peak_date (out-of-sample): %s", args.peak_date)
            print(f"\n  Predict window: from {args.peak_date} (after training end)")

        if not args.model:
            parser.error("--model required for --predict")

        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return

        df = predict_time_to_clear(
            model_path=str(model_path),
            peak_date=args.peak_date,
            rho_threshold_frac=args.rho_threshold_frac,
            n_eval_days=args.n_eval_days,
        )

        print(f"  TTC data saved -> data/features/{Path(args.model).stem}_ttc.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
