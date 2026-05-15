"""
Phase 3 — Time to Clear prediction using trained PINN.

Loads a trained PINN and estimates TTC:
  TTC = min Δt such that max_x ρ(x, t_peak + Δt) < threshold

Usage:
    python src/pinns/predict.py --model outputs/models/pinn_houston_2019-01-01_2019-03-31.pt
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

from src.pinns.lwr_pinn import LWRPINN, get_device
from src.pinns.data_prep import CHANNEL_AXES, compute_channel_length

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/figures")
RHO_THRESHOLD_FRAC = 0.3
N_EVAL_DAYS = 90


def load_model(path):
    device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = ckpt.get("meta", {})
    log.info("Loaded model: %s (best_loss=%.6f)", path, ckpt["best_loss"])
    return model, ckpt, meta


def predict_time_to_clear(
    model_path,
    peak_date=None,
    rho_threshold_frac=RHO_THRESHOLD_FRAC,
    n_eval_days=N_EVAL_DAYS,
):
    model, ckpt, meta = load_model(model_path)
    location = ckpt.get("location", "houston")
    waypoints = CHANNEL_AXES.get(location)
    channel_len = compute_channel_length(waypoints)
    train_start = date.fromisoformat(ckpt["start"])
    train_end = date.fromisoformat(ckpt["end"])
    n_train_days = (train_end - train_start).days + 1
    dx_km = meta.get("dx_km", 2.0)
    n_bins = meta.get("n_bins", int(np.ceil(channel_len / dx_km)))

    if peak_date is None:
        peak_date = train_start + timedelta(days=n_train_days // 2)
    if isinstance(peak_date, str):
        peak_date = date.fromisoformat(peak_date)

    rho_max = meta.get("rho_max", 1.0)
    threshold = rho_max * rho_threshold_frac
    log.info(
        "rho_max=%.2f  threshold=%.2f  channel_len=%.1f km  n_bins=%d",
        rho_max,
        threshold,
        channel_len,
        n_bins,
    )

    eval_dates = [
        train_start + timedelta(days=d)
        for d in range(
            int(np.ceil((peak_date - train_start).days)),
            int(np.ceil((peak_date - train_start).days)) + n_eval_days,
        )
    ]
    if len(eval_dates) == 0:
        eval_dates = [peak_date + timedelta(days=i) for i in range(n_eval_days)]

    x_positions = np.linspace(0, channel_len, n_bins)

    epoch = date.fromisoformat(meta.get("epoch", ckpt["start"]))
    t_days_max = meta.get("t_days_max", max(n_train_days - 1, 1))

    with torch.no_grad():
        rows = []
        peak_rho_tot = 0.0
        first = True
        for d in eval_dates:
            t_norm = (d - epoch).days / max(t_days_max, 1)
            t_norm = np.clip(t_norm, 0.0, 1.0)
            t_t = torch.full((n_bins, 1), t_norm, dtype=torch.float32)
            x_t = torch.tensor(
                (x_positions / channel_len).reshape(-1, 1), dtype=torch.float32
            )
            rho_pred, v_pred = model(x_t, t_t)
            rho_vals = rho_pred.cpu().numpy().ravel()
            v_vals = v_pred.cpu().numpy().ravel()

            rho_denorm = rho_vals * rho_max
            v_denorm = v_vals * meta.get("v_max", 10.0)
            rho_total = float(rho_denorm.sum())
            rho_max_val = float(rho_denorm.max())

            if first:
                peak_rho_tot = rho_total
                first = False

            is_cleared = rho_max_val < threshold

            rows.append(
                {
                    "date": d.isoformat(),
                    "t_norm": float(t_norm),
                    "rho_total": rho_total,
                    "rho_max": rho_max_val,
                    "rho_threshold": threshold,
                    "is_cleared": is_cleared,
                    "congestion_pct": float(rho_total / max(peak_rho_tot, 1) * 100),
                }
            )

    df = pl.DataFrame(rows)
    ttc_days = None
    for i, r in enumerate(rows):
        if r["is_cleared"]:
            ttc_days = i
            break

    if ttc_days is not None:
        log.info("Time to Clear: %d days after first eval date", ttc_days)
    else:
        log.warning("TTC not reached in %d eval days", n_eval_days)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["date"].to_list(),
            y=df["rho_max"].to_list(),
            name="ρ_max (predicted)",
            mode="lines+markers",
            line=dict(color="#1565C0", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"].to_list(),
            y=df["congestion_pct"].to_list(),
            name="Congestion %",
            mode="lines",
            line=dict(color="#E65100", width=1, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold:.1f})",
    )

    if ttc_days is not None:
        ttc_date = rows[ttc_days]["date"]
        fig.add_vline(x=ttc_date, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=ttc_date,
            y=max(df["rho_max"].to_list()) * 0.95,
            text=f"TTC = {ttc_days}d",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-30,
        )

    fig.update_layout(
        title=dict(
            text=f"{location.upper()} — Time to Clear (peak {peak_date})", x=0.5
        ),
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    fig.update_yaxes(
        title_text="ρ_max (vessels/km)", color="#1565C0", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Congestion %", color="#E65100", secondary_y=True, showgrid=False
    )

    plot_path = OUTPUT_DIR / f"{location}_time_to_clear.html"
    fig.write_html(plot_path)
    log.info("Saved plot → %s", plot_path)

    model_name = Path(model_path).stem
    out_parquet = Path(f"data/features/{model_name}_ttc.parquet")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_parquet)
    log.info("Saved TTC data → %s", out_parquet)

    print(f"\n=== Time to Clear ===")
    print(f"  Peak rho_max: {rows[0]['rho_max']:.1f}" if rows else "  No data")
    print(f"  Threshold:  {threshold:.1f}")
    if ttc_days is not None:
        print(f"  TTC:        {ttc_days} days ({rows[ttc_days]['date']})")
    else:
        print(f"  TTC:        Not reached in {n_eval_days} days")
    print(f"  Plot:       {plot_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Time to Clear")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--peak-date", default=None, help="Peak congestion date (YYYY-MM-DD)"
    )
    parser.add_argument("--rho-threshold-frac", type=float, default=RHO_THRESHOLD_FRAC)
    parser.add_argument("--n-eval-days", type=int, default=N_EVAL_DAYS)
    args = parser.parse_args()

    predict_time_to_clear(
        model_path=args.model,
        peak_date=args.peak_date,
        rho_threshold_frac=args.rho_threshold_frac,
        n_eval_days=args.n_eval_days,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
