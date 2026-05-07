"""
Evaluation — Snapshot de métriques du pipeline.

Sauvegarde un JSON dans data/evaluation/<location>_<year>_metrics.json
et un CSV top20 pour lookup manuel dans les indices financiers.

Usage:
    python src/evaluation/snapshot.py --location la --year 2019
"""
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

EVAL_DIR = Path("data/evaluation")


def compute_snapshot(
    gravity_path: Path,
    location: str = "la",
    year: int = 2019,
    event_start: str | None = None,
    event_end: str | None = None,
) -> dict:
    """
    Calcule les métriques clés du pipeline et les sauvegarde.
    Retourne le dict de métriques.
    """
    df = pl.read_parquet(gravity_path).sort("date")
    gs = df["gravity_score"].to_numpy()
    dates = df["date"].to_list()

    metrics = {
        "location": location,
        "year":     year,
        "n_days":   len(df),
        "mean":     float(np.mean(gs)),
        "std":      float(np.std(gs)),
        "max":      float(np.max(gs)),
        "p50":      float(np.percentile(gs, 50)),
        "p90":      float(np.percentile(gs, 90)),
        "p95":      float(np.percentile(gs, 95)),
        "p99":      float(np.percentile(gs, 99)),
        "peak_date":  str(dates[int(np.argmax(gs))]),
        "peak_score": float(np.max(gs)),
        "n_characteristic": int(df["is_characteristic"].sum()),
        "pct_characteristic": float(df["is_characteristic"].mean() * 100),
    }

    if event_start and event_end:
        e_start = date.fromisoformat(event_start)
        e_end   = date.fromisoformat(event_end)
        event_mask    = np.array([e_start <= d <= e_end for d in dates])
        baseline_mask = ~event_mask

        if event_mask.sum() > 0 and baseline_mask.sum() > 0:
            event_mean    = float(np.mean(gs[event_mask]))
            baseline_mean = float(np.mean(gs[baseline_mask]))
            baseline_std  = float(np.std(gs[baseline_mask]))
            snr = (event_mean - baseline_mean) / (baseline_std + 1e-9)
            metrics.update({
                "event_start":     event_start,
                "event_end":       event_end,
                "event_mean":      event_mean,
                "baseline_mean":   baseline_mean,
                "baseline_std":    baseline_std,
                "signal_to_noise": float(snr),
            })

    top20_idx = np.argsort(gs)[::-1][:20]
    blocked   = df["blocked_capacity"].to_numpy()
    rho_arr   = df["utilization_rate_rho"].to_numpy()
    top20 = [
        {
            "rank":             int(i + 1),
            "date":             str(dates[int(idx)]),
            "gravity_score":    round(float(gs[idx]), 4),
            "blocked_capacity": round(float(blocked[idx]), 0),
            "rho":              round(float(rho_arr[idx]), 4),
        }
        for i, idx in enumerate(top20_idx)
    ]
    metrics["top20"] = top20

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{location}_{year}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Snapshot sauvegardé → %s", out_path)

    csv_path = EVAL_DIR / f"{location}_{year}_top20.csv"
    pl.DataFrame(top20).write_csv(csv_path)
    log.info("Top 20 → %s", csv_path)

    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {metrics['location'].upper()} — {metrics['year']}")
    print(f"{'='*55}")
    print(f"  Jours analysés     : {metrics['n_days']}")
    print(f"  Gravity score mean : {metrics['mean']:.4f}")
    print(f"  Gravity score std  : {metrics['std']:.4f}")
    print(f"  Peak date          : {metrics['peak_date']}")
    print(f"  Peak score         : {metrics['peak_score']:.4f}")
    print(f"  P90 / P95 / P99    : {metrics['p90']:.4f} / {metrics['p95']:.4f} / {metrics['p99']:.4f}")
    print(f"  Points car.        : {metrics['n_characteristic']} ({metrics['pct_characteristic']:.1f}%)")
    if "signal_to_noise" in metrics:
        print(f"  SNR événement      : {metrics['signal_to_noise']:.2f}σ")
        print(f"  Mean événement     : {metrics['event_mean']:.4f}")
        print(f"  Mean baseline      : {metrics['baseline_mean']:.4f}")
    print(f"\n  Top 5 jours :")
    for row in metrics["top20"][:5]:
        print(f"    {row['rank']:2d}. {row['date']}  score={row['gravity_score']:.4f}"
              f"  rho={row['rho']:.3f}  cap={row['blocked_capacity']:,.0f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--location",    default="la")
    parser.add_argument("--year",        type=int, default=2019)
    parser.add_argument("--event-start", default=None)
    parser.add_argument("--event-end",   default=None)
    args = parser.parse_args()

    path = Path(f"data/features/{args.location}_{args.year}_gravity_score.parquet")
    m = compute_snapshot(path, args.location, args.year, args.event_start, args.event_end)
    print_metrics(m)
