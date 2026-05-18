"""
Visualisation — Phase 1 & Phase 2 (multi-location).

Phase 1 — clusters HDBSCAN:
    python src/utils/visualize.py --date 2017-08-25
    python src/utils/visualize.py --location la --date 2019-06-05
    python src/utils/visualize.py --start 2017-07-01 --end 2017-09-30

Phase 2 — gravity score (time series + manifold 2D):
    python src/utils/visualize.py --gravity
    python src/utils/visualize.py --gravity --location la

Outputs → outputs/figures/
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
from folium.plugins import HeatMapWithTime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

from src.clustering.hdbscan_daily import cluster_day
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/figures")
COLORS  = {"docked": "#2196F3", "waiting": "#FF9800", "noise": "#9E9E9E"}

# Per-location display config
LOC_CONFIG = {
    "houston": {
        "center":       [29.60, -95.05],
        "zoom":         11,
        "event_start":  date(2017, 8, 25),
        "event_end":    date(2017, 8, 31),
        "event_label":  "Hurricane Harvey (25–31 août 2017)",
        "event_marker": [29.76, -95.37],
        "title_suffix": "Hurricane Harvey, juil–sept 2017",
    },
    "la": {
        "center":       [33.745, -118.22],
        "zoom":         11,
        "event_start":  date(2019, 6, 1),
        "event_end":    date(2019, 9, 30),
        "event_label":  "Tensions US-Chine (juin–sept 2019)",
        "event_marker": [33.745, -118.22],
        "title_suffix": "Port LA/Long Beach 2019",
    },
}


def _get_cfg(location: str) -> tuple[dict, dict, Path, Path, Path]:
    """Return (loc_cfg, display_cfg, parquet_dir, gravity_path, manifold_path)."""
    loc_cfg  = LOCATIONS[location]
    disp_cfg = LOC_CONFIG.get(location, LOC_CONFIG["houston"])
    parquet_dir  = loc_cfg["out_dir"]
    prefix       = loc_cfg["prefix"]
    gravity_path = Path(f"data/features/{location}_gravity_score.parquet")
    manifold_path= Path(f"data/features/{location}_manifold.parquet")
    return loc_cfg, disp_cfg, parquet_dir, gravity_path, manifold_path


# ---------------------------------------------------------------------------
# Single day
# ---------------------------------------------------------------------------

def map_single_day(d: date, location: str = "houston") -> Path:
    _, disp, parquet_dir, _, _ = _get_cfg(location)
    prefix       = LOCATIONS[location]["prefix"]
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    df             = pl.read_parquet(parquet_path)
    cluster_df, _  = cluster_day(parquet_path)

    if cluster_df is not None and "cluster_type" not in cluster_df.columns:
        cluster_df = cluster_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("cluster_type"))

    m = folium.Map(location=disp["center"], zoom_start=disp["zoom"], tiles="CartoDB positron")

    moving = df.filter(pl.col("SOG") >= 1.0)
    if len(moving) > 0:
        sample = moving.sample(min(len(moving), 2000), seed=42)
        moving_group = folium.FeatureGroup(name="En mouvement (SOG ≥ 1 kt)", show=False)
        for row in sample.iter_rows(named=True):
            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=2, color="#BDBDBD", fill=True, fill_opacity=0.35,
                tooltip=f"MMSI {row['MMSI']} | {row['SOG']:.1f} kt",
            ).add_to(moving_group)
        moving_group.add_to(m)

    if cluster_df is not None and len(cluster_df) > 0:
        static_group = folium.FeatureGroup(name="Stationnaires (HDBSCAN)")
        for row in cluster_df.iter_rows(named=True):
            ctype  = row.get("cluster_type") or "noise"
            color  = COLORS.get(ctype, COLORS["noise"])
            score  = row["membership_score"]
            draft  = row["Draft"] or 0.0
            length = row["Length"] or 0.0
            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=5 + score * 3,
                color=color, fill=True,
                fill_opacity=0.5 + score * 0.4,
                popup=folium.Popup(
                    f"<b>MMSI {row['MMSI']}</b><br>"
                    f"Cluster {row['cluster_label']} — <i>{ctype}</i><br>"
                    f"Membership : {score:.2f}<br>"
                    f"Draft : {draft:.1f} m &nbsp; Length : {length:.0f} m",
                    max_width=220,
                ),
                tooltip=f"C{row['cluster_label']} {ctype} | score={score:.2f}",
            ).add_to(static_group)
        static_group.add_to(m)

    n_vessels  = df["MMSI"].n_unique()
    n_static   = cluster_df.height if cluster_df is not None else 0
    n_clusters = 0
    if cluster_df is not None:
        n_clusters = len(set(cluster_df["cluster_label"].to_list())) - (
            1 if -1 in cluster_df["cluster_label"].to_list() else 0
        )

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,.15);">
      <b>{location.upper()} — {d}</b><br><br>
      <span style="color:{COLORS['docked']}">●</span> Docked &nbsp;
      <span style="color:{COLORS['waiting']}">●</span> Waiting &nbsp;
      <span style="color:{COLORS['noise']}">●</span> Noise<br><br>
      Navires distincts : <b>{n_vessels}</b><br>
      Stationnaires     : <b>{n_static}</b><br>
      Clusters          : <b>{n_clusters}</b>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    out_path = OUT_DIR / f"{location}_clusters_{d.strftime('%Y_%m_%d')}.html"
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Période — heatmap animée
# ---------------------------------------------------------------------------

def map_period(start: date, end: date, location: str = "houston",
               max_points_per_day: int = 300) -> Path:
    _, disp, parquet_dir, _, _ = _get_cfg(location)
    prefix = LOCATIONS[location]["prefix"]

    m = folium.Map(location=disp["center"], zoom_start=10, tiles="CartoDB positron")

    heat_data    = []
    dates_index  = []
    days_missing = 0

    d = start
    while d <= end:
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"
        if parquet_path.exists():
            df     = pl.read_parquet(parquet_path)
            static = df.filter(pl.col("SOG") < 1.0)
            static = static.group_by("MMSI").agg([
                pl.col("LAT").median(),
                pl.col("LON").median(),
            ])
            if len(static) > max_points_per_day:
                static = static.sample(max_points_per_day, seed=42)
            points = static.select(["LAT", "LON"]).to_numpy().tolist()
        else:
            points = []
            days_missing += 1

        heat_data.append(points)
        dates_index.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    if days_missing > 0:
        log.warning("%d jour(s) sans parquet", days_missing)

    HeatMapWithTime(
        heat_data, index=dates_index,
        auto_play=False, radius=14, min_opacity=0.3, max_opacity=0.85,
        gradient={0.2: "#2196F3", 0.55: "#FF9800", 1.0: "#F44336"},
        name="Densité navires stationnaires",
    ).add_to(m)

    # Marqueur événement si dans la période
    ev_start = disp["event_start"]
    if start <= ev_start <= end:
        folium.Marker(
            location=disp["event_marker"],
            popup=folium.Popup(f"<b>{disp['event_label']}</b>", max_width=200),
            icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
            tooltip=disp["event_label"],
        ).add_to(m)

    folium.LayerControl().add_to(m)

    out_path = OUT_DIR / f"{location}_heatmap_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.html"
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Phase 2 — Gravity score + manifold 2D
# ---------------------------------------------------------------------------

def plot_gravity_score(location: str = "houston") -> tuple[Path, Path]:
    _, disp, _, gravity_path, _ = _get_cfg(location)

    if not gravity_path.exists():
        raise FileNotFoundError(f"Gravity score introuvable : {gravity_path} — run run_phase2.py --location {location}")

    df       = pl.read_parquet(gravity_path).sort("date")
    dates    = df["date"].to_list()
    gravity  = df["gravity_score"].to_numpy()
    capacity = df["waiting_capacity"].to_numpy()
    is_char  = df["is_characteristic"].to_numpy()
    phi_cols = sorted([c for c in df.columns if c.startswith("phi_")])

    import matplotlib
    matplotlib.use("Agg")

    ev_start = disp["event_start"]
    ev_end   = disp["event_end"]
    ev_label = disp["event_label"]

    # ── Figure 1 : Time series ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{location.upper()} — Phase 2 : Gravity Score\n({disp['title_suffix']})",
                 fontsize=13, fontweight="bold")

    for ax in axes:
        ax.axvspan(ev_start, ev_end, color="#FFCDD2", alpha=0.6, label=ev_label)

    ax0 = axes[0]
    ax0.plot(dates, gravity, color="#1565C0", linewidth=1.8, zorder=3)
    ax0.fill_between(dates, gravity, alpha=0.15, color="#1565C0")
    char_dates = [d for d, c in zip(dates, is_char) if c]
    char_vals  = [g for g, c in zip(gravity, is_char) if c]
    ax0.scatter(char_dates, char_vals, color="#F44336", zorder=5, s=40, label="Point caractéristique")
    ax0.set_ylabel("Gravity Score [0–1]", fontsize=10)
    ax0.set_ylim(-0.05, 1.1)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(axis="y", alpha=0.3)

    ax1 = axes[1]
    ax1.bar(dates, capacity / 1e6, color="#FB8C00", alpha=0.7, width=0.8)
    ax1.set_ylabel("Capacité bloquée\n(Σ L×l, ×10⁶ m²)", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[2]
    vessel_count = df["vessel_count"].to_numpy()
    ax2.plot(dates, vessel_count, color="#388E3C", linewidth=1.5)
    ax2.fill_between(dates, vessel_count, alpha=0.15, color="#388E3C")
    ax2.set_ylabel("Navires distincts", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    ts_path = OUT_DIR / f"{location}_gravity_timeseries.png"
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Sauvegardé : %s", ts_path)

    # ── Figure 2 : Manifold 2D scatter ───────────────────────────────────────
    if len(phi_cols) < 2:
        log.warning("Pas assez de vecteurs propres pour le scatter 2D")
        return ts_path, ts_path

    phi1 = df[phi_cols[0]].to_numpy()
    phi2 = df[phi_cols[1]].to_numpy()

    fig2, ax = plt.subplots(figsize=(9, 7))
    fig2.suptitle(f"{location.upper()} — Espace Manifold (ϕ₁ vs ϕ₂)\ncoloré par Gravity Score",
                  fontsize=12, fontweight="bold")

    sc = ax.scatter(phi1, phi2, c=gravity, cmap="RdYlBu_r", s=60, zorder=3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Gravity Score")

    for d, p1, p2 in zip(dates, phi1, phi2):
        if ev_start <= d <= ev_end:
            ax.scatter(p1, p2, s=120, facecolors="none",
                       edgecolors="#B71C1C", linewidths=2, zorder=4)
            ax.annotate(d.strftime("%d/%m"), (p1, p2),
                        fontsize=7, color="#B71C1C",
                        xytext=(4, 4), textcoords="offset points")

    for d, p1, p2, c in zip(dates, phi1, phi2, is_char):
        if c and not (ev_start <= d <= ev_end):
            ax.annotate(d.strftime("%d/%m"), (p1, p2),
                        fontsize=7, color="#1A237E",
                        xytext=(4, 4), textcoords="offset points")

    ev_patch = mpatches.Patch(edgecolor="#B71C1C", facecolor="none",
                               linewidth=2, label=ev_label)
    ax.legend(handles=[ev_patch], fontsize=9)

    eig_col = phi_cols[0].replace("phi", "eigenvalue")
    eig_val = f"{df[eig_col][0]:.4f}" if eig_col in df.columns else "?"
    ax.set_xlabel(f"ϕ₁ (λ={eig_val})", fontsize=10)
    ax.set_ylabel("ϕ₂", fontsize=10)
    ax.grid(alpha=0.2)

    scatter_path = OUT_DIR / f"{location}_manifold_2d.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Sauvegardé : %s", scatter_path)

    return ts_path, scatter_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation Phase 1 & 2 (multi-location)")
    parser.add_argument("--location", default="houston", choices=list(LOCATIONS.keys()),
                        help="Port cible (default: houston)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--date",    metavar="YYYY-MM-DD", help="Carte clusters pour un seul jour")
    mode.add_argument("--start",   metavar="YYYY-MM-DD", help="Période — début (utiliser avec --end)")
    mode.add_argument("--gravity", action="store_true",  help="Phase 2 : gravity score + manifold 2D")
    parser.add_argument("--end",      metavar="YYYY-MM-DD", help="Fin de période (obligatoire avec --start)")
    parser.add_argument("--each-day", action="store_true",
                        help="Avec --start/--end : génère une carte de clusters par jour")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.gravity:
        ts_path, scatter_path = plot_gravity_score(location=args.location)
        print(f"\nTime series : {ts_path.resolve()}")
        print(f"Manifold 2D : {scatter_path.resolve()}")
        return

    if args.date:
        out = map_single_day(date.fromisoformat(args.date), location=args.location)
        print(f"\nOuvrir dans le navigateur :\n  {out.resolve()}")
    else:
        if not args.end:
            parser.error("--end requis avec --start")
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)

        out = map_period(start, end, location=args.location)
        print(f"\nHeatmap : {out.resolve()}")

        if args.each_day:
            prefix = LOCATIONS[args.location]["prefix"]
            parquet_dir = LOCATIONS[args.location]["out_dir"]
            d = start
            while d <= end:
                parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"
                if parquet_path.exists():
                    try:
                        out_day = map_single_day(d, location=args.location)
                        print(f"  {d} : {out_day.name}")
                    except Exception as exc:
                        log.warning("%s : %s", d, exc)
                d += timedelta(days=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
