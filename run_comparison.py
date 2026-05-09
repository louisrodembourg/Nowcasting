"""
Comparaison de 4 configurations de clustering HDBSCAN sur un jour donné.

Génère 4 cartes Folium dans outputs/figures/ :
  - comparison_baseline_<location>_YYYY_MM_DD.html
  - comparison_radius_0.3nm_<location>_YYYY_MM_DD.html
  - comparison_heading_15deg_<location>_YYYY_MM_DD.html
  - comparison_min_size_5_<location>_YYYY_MM_DD.html

Usage:
    python run_comparison.py --date 2021-10-15 --location la
    python run_comparison.py --date 2017-08-25 --location houston
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import polars as pl

from src.clustering.hdbscan_daily import ClusteringConfig, cluster_day
from src.clustering.visualize_clusters import (
    COLORS,
    _add_satellite_layer,
    _build_geojson_features,
)
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Les 5 configurations à comparer
CONFIGS: dict[str, ClusteringConfig] = {
    "baseline":    ClusteringConfig(),
    "radius_0.3nm": ClusteringConfig(max_docked_radius_nm=0.3),
    "heading_15deg": ClusteringConfig(heading_docked_max_std=15.0),
    "min_size_5":  ClusteringConfig(hdbscan_min_cluster_size=5, hdbscan_min_samples=3),
    "combined":    ClusteringConfig(
                       max_docked_radius_nm=0.3,
                       heading_docked_max_std=15.0,
                       hdbscan_min_cluster_size=5,
                       hdbscan_min_samples=3,
                   ),
}

CONFIG_LABELS: dict[str, str] = {
    "baseline":      "Baseline (actuel — heading_std < 25°)",
    "radius_0.3nm":  "Rayon max 0.3 nm (contrainte spatiale)",
    "heading_15deg": "Heading std < 15° (seuil plus strict)",
    "min_size_5":    "min_cluster_size = 5 (clusters plus grands)",
    "combined":      "Combined : rayon 0.3 nm + heading < 15 deg + min_size 5",
}


def _build_stats_html(
    cluster_df: pl.DataFrame,
    total_vessels: int,
    d: date,
    location: str,
    config_label: str,
) -> str:
    docked_count  = cluster_df.filter(pl.col("cluster_type") == "docked")["cluster_label"].n_unique()
    waiting_count = cluster_df.filter(pl.col("cluster_type") == "waiting")["cluster_label"].n_unique()
    noise_count   = int((cluster_df["cluster_label"] == -1).sum())

    return f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,.18);max-width:300px;">
      <b style="font-size:14px;">{location.upper()} — {d.strftime("%d %b %Y")}</b>
      <div style="margin:4px 0 8px;color:#555;font-size:11px;font-style:italic;">{config_label}</div>
      <hr style="margin:6px 0;border:none;border-top:1px solid #eee">
      <div style="display:flex;gap:12px;margin-bottom:6px;">
        <span><span style="color:{COLORS["docked"]};font-size:16px;">■</span> Terminaux</span>
        <span><b>{docked_count}</b></span>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:6px;">
        <span><span style="color:{COLORS["waiting"]};font-size:16px;">■</span> Congestion</span>
        <span><b>{waiting_count}</b></span>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:6px;">
        <span><span style="color:{COLORS["noise"]};font-size:16px;">●</span> Bruit</span>
        <span><b>{noise_count}</b></span>
      </div>
      <hr style="margin:6px 0;border:none;border-top:1px solid #eee">
      <div><b>Navires totaux :</b> {total_vessels}</div>
      <div style="font-size:10px;color:#666;margin-top:4px;">🛰️ Satellite en overlay (haut droite)</div>
    </div>
    """


def generate_comparison_map(
    d: date,
    location: str,
    config_name: str,
    config: ClusteringConfig,
) -> Path:
    """Génère et sauvegarde la carte Folium pour une configuration donnée."""
    loc_cfg    = LOCATIONS[location]
    prefix     = loc_cfg["prefix"]
    parquet_dir  = Path("data/parquet") / location
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    vessel_df     = pl.read_parquet(parquet_path)
    cluster_df, _ = cluster_day(parquet_path, config=config)
    total_vessels = vessel_df["MMSI"].n_unique()

    if cluster_df is None:
        log.warning("Pas de clusters pour %s / config=%s", d, config_name)
        cluster_df = pl.DataFrame(
            schema={
                "MMSI": pl.Int64, "traj_id": pl.Int32,
                "cluster_label": pl.Int32, "cluster_type": pl.Utf8,
                "LAT": pl.Float64, "LON": pl.Float64,
                "Heading_std": pl.Float64, "membership_score": pl.Float32,
                "nb_messages": pl.UInt32,
            }
        )

    features     = _build_geojson_features(cluster_df, vessel_df)
    config_label = CONFIG_LABELS[config_name]
    stats_html   = _build_stats_html(cluster_df, total_vessels, d, location, config_label)

    disp_cfg = {
        "houston": {"center": [29.60, -95.05], "zoom": 12},
        "la":      {"center": [33.745, -118.22], "zoom": 11},
    }.get(location, {"center": [29.60, -95.05], "zoom": 12})

    m = folium.Map(
        location=disp_cfg["center"],
        zoom_start=disp_cfg["zoom"],
        tiles="CartoDB positron",
    )
    _add_satellite_layer(m)

    # Titre config en bandeau
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
                background:white;padding:8px 20px;border-radius:8px;
                border:2px solid #1565C0;font-family:sans-serif;font-size:13px;
                font-weight:bold;box-shadow:2px 2px 6px rgba(0,0,0,.15);">
        {config_label}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Polygones sémantiques (docked + waiting)
    polygon_group = folium.FeatureGroup(name="Zones sémantiques (HDBSCAN)", show=True)
    for feat in features:
        props = feat["properties"]
        if props["cluster_label"] == -1:
            continue
        if feat["geometry"]["type"] == "Polygon":
            coords = feat["geometry"]["coordinates"]
            folium.Polygon(
                locations=[[p[1], p[0]] for p in coords[0]],
                popup=folium.Popup(props["popup"], max_width=250),
                tooltip=(
                    f"Cluster {props['cluster_label']} — "
                    f"{props['cluster_type']} ({props['n_vessels']} navires)"
                ),
                **props["style"],
            ).add_to(polygon_group)
        elif feat["geometry"]["type"] == "LineString":
            coords = feat["geometry"]["coordinates"]
            folium.PolyLine(
                locations=[[p[1], p[0]] for p in coords],
                popup=folium.Popup(props["popup"], max_width=250),
                tooltip=f"Cluster {props['cluster_label']} — ligne",
                **props["style"],
            ).add_to(polygon_group)
    polygon_group.add_to(m)

    # Points bruit
    noise_group = folium.FeatureGroup(name="Bruit HDBSCAN (-1)", show=False)
    for feat in features:
        props = feat["properties"]
        if props["cluster_label"] != -1:
            continue
        geom = feat["geometry"]
        lon, lat = geom["coordinates"][1], geom["coordinates"][0]
        folium.CircleMarker(
            location=[lat, lon],
            radius=props["style"]["radius"],
            color=props["style"]["color"],
            fill=True,
            fill_opacity=props["style"]["fillOpacity"],
        ).add_to(noise_group)
    noise_group.add_to(m)

    m.get_root().html.add_child(folium.Element(stats_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = OUT_DIR / f"comparison_{config_name}_{location}_{d.strftime('%Y_%m_%d')}.html"
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 4 configurations HDBSCAN — génère 4 cartes Folium"
    )
    parser.add_argument(
        "--date", default="2021-10-15", metavar="YYYY-MM-DD",
        help="Date à analyser (default: 2021-10-15)",
    )
    parser.add_argument(
        "--location", default="la", choices=list(LOCATIONS.keys()),
        help="Port cible (default: la)",
    )
    args = parser.parse_args()

    d = date.fromisoformat(args.date)
    print(f"\nComparaison HDBSCAN — {args.location.upper()} {d.strftime('%d %b %Y')}")
    print("=" * 60)

    generated = []
    for config_name, config in CONFIGS.items():
        try:
            out = generate_comparison_map(d, args.location, config_name, config)
            print(f"  ✓  {CONFIG_LABELS[config_name]}")
            print(f"       → {out.name}")
            generated.append(out)
        except Exception as exc:
            print(f"  ✗  {CONFIG_LABELS[config_name]} : {exc}")

    print(f"\n{len(generated)}/4 cartes générées dans outputs/figures/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
