"""
Evaluation — Visualisations de congestion pour comparaison manuelle avec indices financiers.

Génère 3 outputs :
  1. <loc>_<year>_weekly_gravity.png  — gravity score hebdomadaire (même fréquence que FBX/WCI)
  2. <loc>_<year>_dashboard.png       — 4 panels journaliers : gravity, ρ, capacité, clusters
  3. <loc>_<year>_top20.csv           — top 20 dates pour lookup dans les indices en ligne

Usage:
    python src/evaluation/visualize_congestion.py --location la --year 2019
"""
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

FIGS_DIR = Path("outputs/figures")
EVAL_DIR = Path("data/evaluation")

KNOWN_EVENTS = {
    ("la", 2019): [
        (date(2019, 5, 10),  "Tarifs US\n10%→25%"),
        (date(2019, 8,  1),  "Tarifs\n$300B"),
        (date(2019, 8, 23),  "Riposte\nChine"),
        (date(2019, 12, 13), "Phase 1\naccord"),
    ],
    ("la", 2020): [
        (date(2020, 3, 15),  "COVID\nshutdown"),
        (date(2020, 5, 25),  "Reprise\nactivité"),
    ],
    ("la", 2021): [
        (date(2021, 9,  1),  "Début\nbacklog"),
        (date(2021, 10, 20), "Pic\nancrage"),
        (date(2021, 12, 31), "Fin\nbacklog"),
    ],
    ("la", 2023): [
        (date(2023, 10,  7), "Hamas\nattaque"),
        (date(2023, 11, 19), "Houthis\n1er détournement"),
        (date(2023, 12, 15), "Houthis\nescalade"),
        (date(2024,  1, 12), "Frappes US/UK\nYémen"),
        (date(2024,  2, 18), "Pic\nMer Rouge"),
        (date(2024,  5,  1), "Négo.\ncessez-feu"),
    ],
    ("la", 2024): [
        (date(2024,  1, 12), "Frappes US/UK\nYémen"),
        (date(2024,  2, 18), "Pic\nRedSea"),
        (date(2024,  5,  1), "Négociations\ncessez-feu"),
    ],
    ("houston", 2017): [
        (date(2017, 8, 25),  "Harvey\natterrissage"),
        (date(2017, 8, 29),  "Réouverture\npartielle"),
        (date(2017, 9,  5),  "Réouverture\ncomplète"),
    ],
}


def _month_axis(ax, dates):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


def plot_weekly_gravity(
    df: pl.DataFrame,
    location: str,
    year: int,
    output_path: Path | None = None,
) -> Path:
    """Gravity score agrégé par semaine — prêt pour comparaison avec FBX/WCI."""
    df_w = (
        df.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([
            pl.col("gravity_score").mean().alias("gs_mean"),
            pl.col("gravity_score").max().alias("gs_max"),
            pl.col("blocked_capacity").mean().alias("cap_mean"),
        ])
        .sort("week")
    )

    weeks  = df_w["week"].to_list()
    gs_m   = df_w["gs_mean"].to_numpy()
    gs_max = df_w["gs_max"].to_numpy()
    cap    = df_w["cap_mean"].to_numpy()
    events = KNOWN_EVENTS.get((location, year), [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1.fill_between(weeks, gs_m, alpha=0.35, color="#1565C0", label="Gravity score (mean hebdo)")
    ax1.plot(weeks, gs_m,   color="#1565C0", linewidth=1.5)
    ax1.plot(weeks, gs_max, color="#0D47A1", linewidth=0.8, linestyle="--", alpha=0.6, label="Max semaine")
    ax1.set_ylabel("Gravity score", fontsize=11)
    ax1.set_ylim(bottom=0)

    p90 = np.percentile(gs_m, 90)
    baseline_mean = float(np.mean(gs_m[gs_m < p90]))
    ax1.axhline(baseline_mean, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
                label=f"Baseline ({baseline_mean:.3f})")

    for ev_date, label in events:
        ax1.axvline(ev_date, color="#B71C1C", linewidth=0.9, linestyle="--", alpha=0.7)
        ax1.text(ev_date, ax1.get_ylim()[1] * 0.88, label,
                 fontsize=6.5, color="#B71C1C", ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(
        f"{location.upper()} — Gravity Score Hebdomadaire {year}\n"
        "À comparer avec : FBX01 (freightos.com) / WCI (drewry.co.uk) / SCFI (sse.net.cn)",
        fontsize=11, fontweight="bold",
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2.bar(weeks, cap / 1e6, width=6, color="#E65100", alpha=0.65, label="Capacité bloquée (M m²)")
    ax2.set_ylabel("Capacité bloquée (M m²)", fontsize=11)
    for ev_date, _ in events:
        ax2.axvline(ev_date, color="#B71C1C", linewidth=0.9, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    _month_axis(ax2, weeks)

    note = (
        "↑ Pour comparer : capturer cet écran + ouvrir freightos.com/fbx ou macrotrends.net/SCFI\n"
        "→ Aligner les axes temporels et chercher les pics simultanés"
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#555", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="none", alpha=0.8))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_path or (FIGS_DIR / f"{location}_{year}_weekly_gravity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Weekly gravity → %s", out)
    return out


def plot_congestion_dashboard(
    df: pl.DataFrame,
    location: str,
    year: int,
    output_path: Path | None = None,
) -> Path:
    """Dashboard 4 panels journaliers : gravity + ρ + blocked_capacity + cluster_count."""
    dates  = df["date"].to_list()
    gs     = df["gravity_score"].to_numpy()
    rho    = df["utilization_rate_rho"].to_numpy()
    cap    = df["blocked_capacity"].to_numpy() / 1e6
    clust  = df["hdbscan_cluster_count"].to_numpy()
    char   = df["is_characteristic"].to_numpy()
    events = KNOWN_EVENTS.get((location, year), [])

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    panels = [
        (axes[0], gs,    "#1565C0", "Gravity score",            True),
        (axes[1], rho,   "#2E7D32", "Utilisation ρ (statique)", False),
        (axes[2], cap,   "#E65100", "Capacité bloquée (M m²)",  False),
        (axes[3], clust, "#6A1B9A", "Clusters HDBSCAN / jour",  False),
    ]

    for ax, vals, color, ylabel, is_gravity in panels:
        ax.fill_between(dates, vals, alpha=0.3, color=color)
        ax.plot(dates, vals, color=color, linewidth=0.9, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_gravity:
            char_dates = [d for d, c in zip(dates, char) if c]
            char_gs    = [gs[i] for i, c in enumerate(char) if c]
            ax.scatter(char_dates, char_gs, color="#B71C1C", s=18, zorder=5,
                       label="Points caractéristiques", alpha=0.7)
            ax.legend(loc="upper left", fontsize=8)

        for ev_date, label in events:
            ax.axvline(ev_date, color="#B71C1C", linewidth=0.8, linestyle="--", alpha=0.6)

    for ev_date, label in events:
        axes[0].text(ev_date, gs.max() * 0.90, label,
                     fontsize=6.5, color="#B71C1C", ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    _month_axis(axes[3], dates)
    axes[0].set_title(
        f"{location.upper()} — Dashboard Congestion {year}\n"
        "Points rouges = jours caractéristiques (extremums locaux du manifold)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(h_pad=1.5)
    out = output_path or (FIGS_DIR / f"{location}_{year}_dashboard.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard → %s", out)
    return out


def export_top20_csv(
    df: pl.DataFrame,
    location: str,
    year: int,
    output_path: Path | None = None,
) -> Path:
    """Exporte le top 20 des jours de congestion en CSV pour lookup dans les indices."""
    gs        = df["gravity_score"].to_numpy()
    top20_idx = np.argsort(gs)[::-1][:20]
    dates_arr = df["date"].to_list()
    rho_arr   = df["utilization_rate_rho"].to_numpy()
    cap_arr   = df["blocked_capacity"].to_numpy()
    clust_arr = df["hdbscan_cluster_count"].to_numpy()
    char_arr  = df["is_characteristic"].to_numpy()

    rows = []
    for rank, idx in enumerate(top20_idx, 1):
        i = int(idx)
        rows.append({
            "rank":              rank,
            "date":              str(dates_arr[i]),
            "gravity_score":     round(float(gs[i]), 4),
            "rho":               round(float(rho_arr[i]), 4),
            "blocked_cap_M":     round(float(cap_arr[i]) / 1e6, 3),
            "cluster_count":     int(clust_arr[i]),
            "is_characteristic": bool(char_arr[i]),
            "lookup_hint":       f"Chercher {str(dates_arr[i])} dans SCFI/FBX/WCI",
        })

    out = output_path or (EVAL_DIR / f"{location}_{year}_top20.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)
    log.info("Top 20 CSV → %s", out)
    return out


def generate_all(
    gravity_path: Path,
    location: str,
    year: int,
) -> dict[str, Path]:
    """Génère les 3 outputs pour une année."""
    df = pl.read_parquet(gravity_path).sort("date")
    return {
        "weekly":    plot_weekly_gravity(df, location, year),
        "dashboard": plot_congestion_dashboard(df, location, year),
        "top20_csv": export_top20_csv(df, location, year),
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="la")
    parser.add_argument("--year",     type=int, default=2019)
    args = parser.parse_args()

    path = Path(f"data/features/{args.location}_{args.year}_gravity_score.parquet")
    outs = generate_all(path, args.location, args.year)
    for k, p in outs.items():
        print(f"  {k:12s} → {p}")
