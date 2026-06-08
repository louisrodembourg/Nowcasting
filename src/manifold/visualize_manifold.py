"""
Visualisation des points caractéristiques du manifold sur une carte.

Les points caractéristiques sont les jours qui correspondent à des extremums locaux
dans l'espace des vecteurs propres — ils représentent les "régimes de trafic" distincts.

Usage:
    python src/manifold/visualize_manifold.py --date 2017-08-25
    python src/manifold/visualize_manifold.py --start 2017-07-01 --end 2017-09-30
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import numpy as np
import polars as pl

from src.manifold.notuse_lbo import run_lbo
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/figures")


def visualize_manifold_day(
    d: date,
    manifold_df: pl.DataFrame,
    location: str = "houston",
) -> Path:
    """Affiche un point caractéristique sur une carte avec les clusters HDBSCAN du jour."""
    from src.clustering.hdbscan_daily import cluster_day

    loc_cfg = LOCATIONS[location]
    disp_cfg = {
        "houston": {"center": [29.60, -95.05], "zoom": 12},
        "la": {"center": [33.745, -118.22], "zoom": 11},
    }.get(location, {"center": [29.60, -95.05], "zoom": 12})

    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    m = folium.Map(
        location=disp_cfg["center"],
        zoom_start=disp_cfg["zoom"],
        tiles="CartoDB positron",
    )

    # Ajoute couche satellite
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=True,
        show=False,
    ).add_to(m)

    # Charge les données du jour
    if parquet_path.exists():
        cluster_df, _ = cluster_day(parquet_path)

        # Affiche les clusters stationnaires
        if cluster_df is not None and len(cluster_df) > 0:
            colors = {"docked": "#1565C0", "waiting": "#E65100"}
            for row in cluster_df.iter_rows(named=True):
                ctype = row.get("cluster_type", "noise")
                if ctype == "docked":
                    color = colors["docked"]
                elif ctype == "waiting":
                    color = colors["waiting"]
                else:
                    color = "#9E9E9E"

                folium.CircleMarker(
                    location=[row["LAT"], row["LON"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"Cluster {row['cluster_label']} — {ctype}",
                ).add_to(m)

    manifold_df = manifold_df.with_columns(pl.col("date").cast(pl.Utf8))

    # Affiche le point caractéristique
    row = manifold_df.filter(pl.col("date") == d.isoformat())
    if len(row) > 0:
        phi1 = row["phi_1"][0]
        phi2 = row["phi_2"][0] if "phi_2" in row.columns else 0
        is_char = row["is_characteristic"][0]
        
        # Récupère les coordonnées réelles des clusters du jour
        if cluster_df is not None and len(cluster_df) > 0:
            # Marque le centre de gravité de tous les clusters
            lats = cluster_df["LAT"].to_numpy()
            lons = cluster_df["LON"].to_numpy()
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))
            marker_location = [center_lat, center_lon]
        else:
            marker_location = disp_cfg["center"]

        # Icône selon si caractéristique ou non
        icon_color = "red" if is_char else "blue"
        icon_icon = "star" if is_char else "circle"

        folium.Marker(
            location=marker_location,
            popup=folium.Popup(
                f"""
                <div style="font-family:sans-serif;width:200px;">
                    <b>{d.strftime("%d %b %Y")}</b>
                    <hr>
                    <b>φ₁</b>: {phi1:.4f}<br>
                    <b>φ₂</b>: {phi2:.4f}<br>
                    <b>Position</b>: {marker_location[0]:.4f}, {marker_location[1]:.4f}<br>
                    <b>Caractéristique</b>: {"✅ OUI" if is_char else "❌ NON"}
                </div>
            """,
                max_width=250,
            ),
            icon=folium.Icon(color=icon_color, icon=icon_icon, prefix="glyphicon"),
        ).add_to(m)

    # Légende
    legend_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;">
        <b>{location.upper()} — {d.strftime("%d %b %Y")}</b><br><br>
        <span style="color:#1565C0">●</span> Docked<br>
        <span style="color:#E65100">●</span> Waiting<br>
        <span style="color:red">★</span> Point caractéristique
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = OUT_DIR / f"{location}_manifold_{d.strftime('%Y_%m_%d')}.html"
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


def visualize_manifold_map(
    start: date,
    end: date,
    location: str = "la",
) -> Path:
    """Affiche tous les jours avec leurs points caractéristiques sur une carte."""
    from src.clustering.hdbscan_daily import cluster_day

    loc_cfg = LOCATIONS[location]
    disp_cfg = {
        "houston": {"center": [29.60, -95.05], "zoom": 11},
        "la": {"center": [33.745, -118.22], "zoom": 10},
    }.get(location, {"center": [29.60, -95.05], "zoom": 11})

    m = folium.Map(
        location=disp_cfg["center"],
        zoom_start=disp_cfg["zoom"],
        tiles="CartoDB positron",
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=True,
        show=False,
    ).add_to(m)

    # Grouper par date
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)

    # Créer des FeatureGroups pour chaque type
    char_group = folium.FeatureGroup(name="Points caractéristiques", show=True)
    normal_group = folium.FeatureGroup(name="Jours normaux", show=False)

    prefix = loc_cfg["prefix"]
    parquet_dir = loc_cfg["out_dir"]

    manifold_path = Path(f"data/features/{location}_manifold.parquet")
    if manifold_path.exists():
        manifold_df = pl.read_parquet(manifold_path)
    else:
        log.warning("Manifold non trouvé, création...")
        manifold_df = run_lbo(
            features_path=Path(f"data/features/{location}_daily_features.parquet")
        )

    # Traite chaque jour
    char_count = 0
    manifold_df = manifold_df.with_columns(pl.col("date").cast(pl.Utf8))

    for day in dates:
        parquet_path = parquet_dir / f"{prefix}_{day.strftime('%Y_%m_%d')}.parquet"

        row = manifold_df.filter(pl.col("date") == day.isoformat())
        if len(row) == 0:
            continue

        is_char = row["is_characteristic"][0]
        phi1 = row["phi_1"][0]
        phi2 = row["phi_2"][0] if "phi_2" in row.columns else 0

        # Charge les clusters HDBSCAN du jour pour obtenir les coordonnées réelles
        if not parquet_path.exists():
            log.debug("%s : parquet inexistant, utilise centre par défaut", day)
            locations = [disp_cfg["center"]]
        else:
            try:
                cluster_df, _ = cluster_day(parquet_path)
                if cluster_df is None or len(cluster_df) == 0:
                    # Si pas de clusters, utilise le centre de la carte
                    locations = [disp_cfg["center"]]
                else:
                    # Récupère les coordonnées des clusters stationnaires
                    locations = cluster_df.select(["LAT", "LON"]).to_numpy().tolist()
            except Exception as e:
                log.debug("%s : erreur chargement clusters — %s", day, e)
                locations = [disp_cfg["center"]]
        
        # Utilise la première coordonnée disponible, ou le centre par défaut
        center = locations[0] if locations else disp_cfg["center"]

        if is_char:
            char_count += 1
            # Affiche des marqueurs pour tous les clusters du jour caractéristique
            for loc in locations:
                folium.Marker(
                    location=loc,
                    popup=folium.Popup(
                        f"""
                        <div style="font-family:sans-serif;width:180px;">
                            <b>{day.strftime("%d %b %Y")}</b> ⭐
                            <hr>
                            φ₁: {phi1:.4f}<br>
                            φ₂: {phi2:.4f}<br>
                            Lat: {loc[0]:.4f}<br>
                            Lon: {loc[1]:.4f}
                        </div>
                    """,
                        max_width=200,
                    ),
                    icon=folium.Icon(color="red", icon="star", prefix="glyphicon"),
                ).add_to(char_group)
        else:
            # Pour les jours normaux, affiche seulement le premier cluster
            folium.CircleMarker(
                location=center,
                radius=3,
                color="blue",
                fill=True,
                fill_opacity=0.5,
                popup=f"{day.strftime('%d %b %Y')}: φ₁={phi1:.3f}",
            ).add_to(normal_group)

    char_group.add_to(m)
    normal_group.add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;">
        <b>{location.upper()} — {start} → {end}</b><br><br>
        <span style="color:red">★</span> Points caractéristiques: <b>{char_count}</b><br>
        <span style="color:blue">●</span> Jours normaux (masqué)<br>
        <br><i>Layer control en haut à droite</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    out_path = (
        OUT_DIR
        / f"{location}_manifold_map_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.html"
    )
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisation des points caractéristiques du manifold"
    )
    parser.add_argument(
        "--location",
        default="la",
        choices=list(LOCATIONS.keys()),
    )
    parser.add_argument("--date", help="Un jour spécifique")
    parser.add_argument("--start", help="Début période")
    parser.add_argument("--end", help="Fin période")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        manifold_path = Path(f"data/features/{args.location}_manifold.parquet")
        if not manifold_path.exists():
            log.warning("Manifold inexistant — création...")
            run_lbo(
                features_path=Path(
                    f"data/features/{args.location}_daily_features.parquet"
                ),
                output_path=manifold_path,
            )
        manifold_df = pl.read_parquet(manifold_path)
        d = date.fromisoformat(args.date)
        out = visualize_manifold_day(d, manifold_df, location=args.location)
        print(f"\n{out.resolve()}")
        return

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        out = visualize_manifold_map(start, end, location=args.location)
        print(f"\n{out.resolve()}")
        return

    parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
