"""
Phase 2 — NMF pour zones constituantes + Gravity Score.

Alternative au manifold LBO : NMF (Non-negative Matrix Factorization) sur la matrice
(zones × jours) pour identifier les zones constituantes et calculer le gravity score.

Différence clé vs manifold_pipeline.py :
  - NMF remplace LBO pour la décomposition spectrale
  - gravity_score pondère uniquement la capacité des zones "waiting" (pas "docked")
  - cluster_type est résolu via les zones GeoJSON docked (data/zones/{location}_docked.geojson)

Usage:
    python src/manifold/nmf_pipeline.py --both  --start 2019-01-01 --end 2019-01-31 --location la
    python src/manifold/nmf_pipeline.py --both  --start 2017-07-01 --end 2017-09-15 --location houston
    python src/manifold/nmf_pipeline.py --score --start 2019-02-01 --end 2019-02-28 --location la
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
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.neighbors import NearestNeighbors

from src.clustering.hdbscan_daily import (
    ClusteringConfig,
    cluster_day,
    load_docked_zones_or_none,
)
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS = 5
DEFAULT_K            = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(location: str) -> ClusteringConfig:
    """Charge la zone docked GeoJSON et construit le ClusteringConfig."""
    docked_polys = load_docked_zones_or_none(location)
    if docked_polys is None:
        log.warning("Pas de zone docked pour '%s' — tout sera classé 'waiting'", location)
    return ClusteringConfig(docked_polygons=docked_polys)


# ============================================================================
# ZONE MATRIX
# ============================================================================

def build_zone_matrix(
    start: date,
    end: date,
    location: str,
    config: ClusteringConfig,
) -> tuple[pl.DataFrame, np.ndarray, list[date]]:
    """
    Construit la matrice d'occupation (zones × jours).
    Chaque zone = position médiane arrondie à 3 décimales d'un épisode statique.
    Chaque cellule = nombre d'épisodes dans cette zone ce jour-là.
    Normalisé par le max de chaque zone (occupation rate ∈ [0, 1]).
    """
    loc_cfg     = LOCATIONS[location]
    prefix      = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]

    all_zones:     dict[str, dict]               = {}
    zone_day_data: dict[tuple[str, date], int]   = {}

    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

        if parquet_path.exists():
            cluster_df, _ = cluster_day(parquet_path, config=config)
            if cluster_df is not None and len(cluster_df) > 0:
                for row in cluster_df.iter_rows(named=True):
                    lat_r    = round(row["LAT"], 3)
                    lon_r    = round(row["LON"], 3)
                    zone_key = f"{lat_r}_{lon_r}"

                    if zone_key not in all_zones:
                        all_zones[zone_key] = {
                            "lat":          row["LAT"],
                            "lon":          row["LON"],
                            "cluster_label": row["cluster_label"],
                            "cluster_type": row.get("cluster_type", "waiting"),
                            "mmsi":         [],
                        }
                    all_zones[zone_key]["mmsi"].append(row["MMSI"])

                    key = (zone_key, d)
                    zone_day_data[key] = zone_day_data.get(key, 0) + 1

        d += timedelta(days=1)

    if not all_zones:
        return pl.DataFrame(), np.array([]), []

    zone_rows = [
        {
            "zone_key":       zk,
            "lat":            info["lat"],
            "lon":            info["lon"],
            "cluster_label":  info["cluster_label"],
            "cluster_type":   info["cluster_type"],
            "n_vessels_total": len(set(info["mmsi"])),
        }
        for zk, info in all_zones.items()
    ]
    zone_df   = pl.DataFrame(zone_rows)
    zone_keys = zone_df["zone_key"].to_list()

    X = np.zeros((len(zone_df), len(dates)))
    for j, day in enumerate(dates):
        for i, zk in enumerate(zone_keys):
            X[i, j] = zone_day_data.get((zk, day), 0)

    max_per_zone = X.max(axis=1, keepdims=True)
    max_per_zone[max_per_zone == 0] = 1
    X = X / max_per_zone

    return zone_df, X, dates


# ============================================================================
# NMF — IDENTIFICATION DES ZONES CONSTITUANTES
# ============================================================================

def _find_constituent_from_components(components: np.ndarray, k: int) -> np.ndarray:
    """
    Détecte les zones qui sont des extremums locaux dans l'espace NMF.
    Un point i est constituant si sa valeur sur au moins une composante NMF
    est strictement supérieure (ou inférieure) à tous ses k voisins.
    """
    n      = components.shape[0]
    k_safe = min(k, n - 1)

    nbrs = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(components)
    _, indices = nbrs.kneighbors(components)

    is_constituent = np.zeros(n, dtype=bool)
    for i in range(n):
        nbr = indices[i, 1:]
        for c in range(components.shape[1]):
            vals_nbr = components[nbr, c]
            if components[i, c] > vals_nbr.max() or components[i, c] < vals_nbr.min():
                is_constituent[i] = True
                break

    return is_constituent


def identify_constituent_zones(
    start: date,
    end: date,
    location: str,
    config: ClusteringConfig,
    n_components: int = DEFAULT_N_COMPONENTS,
    k: int = DEFAULT_K,
) -> tuple[pl.DataFrame, float]:
    """
    Identifie les zones constituantes via NMF sur la matrice (zones × jours).
    Retourne (zone_df enrichi, reconstruction_error).
    """
    log.info("Construction matrice zones : %s → %s", start, end)
    zone_df, X, dates = build_zone_matrix(start, end, location, config)

    if len(X) == 0:
        raise ValueError("Aucune zone trouvée — vérifier les fichiers Parquet")

    log.info("%d zones × %d jours", *X.shape)

    n_comp = min(n_components, min(X.shape) - 1)
    if n_comp < 1:
        raise ValueError(f"Pas assez de données pour NMF (zones={X.shape[0]}, jours={X.shape[1]})")

    model = SklearnNMF(n_components=n_comp, init="random", random_state=42, max_iter=1000)
    W     = model.fit_transform(X + 1e-10)   # NMF requiert des valeurs non-négatives
    recon_err = model.reconstruction_err_
    log.info("NMF reconstruction error : %.4f", recon_err)

    constituents = _find_constituent_from_components(W, k=k)
    zone_df = zone_df.with_columns(pl.Series("is_constituent", constituents.tolist()))
    for i in range(n_comp):
        zone_df = zone_df.with_columns(pl.Series(f"nmf_{i + 1}", W[:, i].tolist()))

    log.info("%d zones constituantes / %d", int(constituents.sum()), len(zone_df))
    return zone_df, recon_err


# ============================================================================
# GRAVITY SCORE QUOTIDIEN
# ============================================================================

def compute_daily_gravity(
    d: date,
    location: str,
    constituent_zones: pl.DataFrame,
    config: ClusteringConfig,
) -> pl.DataFrame:
    """
    Calcule le gravity score quotidien sur les zones constituantes.
    severity = 1.0 pour les épisodes "waiting" (congestion), 0.0 pour "docked" (normal).
    """
    loc_cfg      = LOCATIONS[location]
    parquet_path = loc_cfg["out_dir"] / f"{loc_cfg['prefix']}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        return pl.DataFrame()

    cluster_df, _ = cluster_day(parquet_path, config=config)
    if cluster_df is None or len(cluster_df) == 0:
        return pl.DataFrame()

    const_keys = set(
        constituent_zones.filter(pl.col("is_constituent"))["zone_key"].to_list()
    )

    scores = []
    for row in cluster_df.iter_rows(named=True):
        lat_r    = round(row["LAT"], 3)
        lon_r    = round(row["LON"], 3)
        zone_key = f"{lat_r}_{lon_r}"
        ctype    = row.get("cluster_type", "waiting")
        capacity = (row.get("Length", 0) or 0) * (row.get("Width", 0) or 0)

        scores.append({
            "date":         d.isoformat(),
            "zone_key":     zone_key,
            "lat":          row["LAT"],
            "lon":          row["LON"],
            "mmsi":         row["MMSI"],
            "cluster_type": ctype,
            "is_constituent": zone_key in const_keys,
            "severity":     1.0 if ctype == "waiting" else 0.0,
            "capacity":     capacity,
        })

    return pl.DataFrame(scores)


def aggregate_gravity_score(daily_df: pl.DataFrame) -> dict:
    """
    Agrège le gravity score global d'un jour.
    gravity_score = Σ capacité des épisodes waiting dans les zones constituantes.
    """
    if len(daily_df) == 0:
        return {}

    const_only   = daily_df.filter(pl.col("is_constituent"))
    const_waiting = const_only.filter(pl.col("severity") > 0)

    return {
        "date":               daily_df["date"][0],
        "total_vessels":      daily_df["mmsi"].n_unique(),
        "constituent_vessels": const_only["mmsi"].n_unique() if len(const_only) > 0 else 0,
        "waiting_vessels":    const_waiting["mmsi"].n_unique() if len(const_waiting) > 0 else 0,
        "total_capacity":     int(const_only["capacity"].sum()) if len(const_only) > 0 else 0,
        "gravity_score":      int(const_waiting["capacity"].sum()) if len(const_waiting) > 0 else 0,
    }


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NMF pipeline : zones constituantes + gravity score")
    parser.add_argument("--location",     default="la",
                        choices=list(LOCATIONS.keys()))
    parser.add_argument("--start",        help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end",          help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--k",            type=int, default=DEFAULT_K)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identify", action="store_true", help="Identifier les zones constituantes (NMF)")
    mode.add_argument("--score",    action="store_true", help="Calculer le gravity score quotidien")
    mode.add_argument("--both",     action="store_true", help="Identifier puis scorer sur la même période")

    args = parser.parse_args()

    if not args.start or not args.end:
        parser.error("--start et --end sont requis")

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    OUTPUT_DIR = Path("outputs/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Config commune (zone docked chargée une seule fois)
    config = _make_config(args.location)

    # ── Identification ────────────────────────────────────────────────────────
    if args.identify or args.both:
        zone_df, recon_err = identify_constituent_zones(
            start, end, args.location, config, args.n_components, args.k
        )
        const = zone_df.filter(pl.col("is_constituent"))
        print(f"\nZones constituantes : {len(const)} / {len(zone_df)}")
        print(f"NMF reconstruction error : {recon_err:.4f}")

        out_path = Path(f"data/features/{args.location}_nmf_constituent_zones.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        zone_df.write_parquet(out_path)
        print(f"Sauvegardé : {out_path}")

    # ── Gravity score ─────────────────────────────────────────────────────────
    if args.score or args.both:
        constituent_path = Path(f"data/features/{args.location}_nmf_constituent_zones.parquet")
        if not constituent_path.exists():
            print("Lancer d'abord --identify ou --both pour générer les zones constituantes.")
            return

        constituent_zones = pl.read_parquet(constituent_path)

        all_scores = []
        d = start
        while d <= end:
            daily_df = compute_daily_gravity(d, args.location, constituent_zones, config)
            if len(daily_df) > 0:
                agg = aggregate_gravity_score(daily_df)
                if agg:
                    all_scores.append(agg)
            d += timedelta(days=1)

        if not all_scores:
            print("Aucun score calculé — vérifier les fichiers Parquet.")
            return

        scores_df = pl.DataFrame(all_scores)
        out_path  = Path(f"data/features/{args.location}_nmf_gravity_daily.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.write_parquet(out_path)
        print(f"\n{len(scores_df)} jours sauvegardés → {out_path}")

        scores_np = scores_df["gravity_score"].to_numpy()
        max_idx   = int(scores_np.argmax())
        print(f"\n=== NMF Gravity Score ===")
        print(f"  Moyenne : {scores_np.mean():.1f}")
        print(f"  Maximum : {scores_np.max():.1f}  ({scores_df['date'].to_list()[max_idx]})")

        # ── Graphique Plotly ──────────────────────────────────────────────────
        dates_str = scores_df["date"].to_list()
        gravity   = scores_df["gravity_score"].to_numpy()
        waiting_v = scores_df["waiting_vessels"].to_numpy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=dates_str, y=gravity,
            name="Gravity Score (NMF)",
            mode="lines+markers",
            line=dict(color="#1565C0", width=2),
            marker=dict(size=6),
        ))

        fig.add_trace(go.Scatter(
            x=dates_str, y=waiting_v,
            name="Navires en attente (zones const.)",
            mode="lines",
            line=dict(color="#E65100", width=1, dash="dot"),
        ), secondary_y=True)

        fig.update_layout(
            title=dict(
                text=f"{args.location.upper()} — NMF Gravity Score ({args.start} → {args.end})",
                x=0.5,
            ),
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=450,
        )
        fig.update_yaxes(title_text="Gravity Score",        color="#1565C0", secondary_y=False)
        fig.update_yaxes(title_text="Navires en attente",   color="#E65100", secondary_y=True, showgrid=False)

        plot_path = OUTPUT_DIR / f"{args.location}_nmf_gravity_score.html"
        fig.write_html(plot_path)
        print(f"Graphique → {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
