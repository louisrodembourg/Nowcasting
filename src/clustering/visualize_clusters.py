"""
Visualisation géométrique des clusters HDBSCAN — Phase 1.

Génère des polygones sémantiques :
  - Rotated Bounding Boxes  -> clusters 'docked'  (navires alignés au quai)
  - Convex Hulls            -> clusters 'waiting' (zone de mouillage dispersée)
  - Points gris             -> bruit

Un seul fichier HTML avec time slider pour visualiser un jour ou une période.

Usage :
    python src/clustering/visualize_clusters.py --date 2017-08-25 --location houston
    python src/clustering/visualize_clusters.py --start 2017-08-01 --end 2017-09-30 --location houston
    python src/clustering/visualize_clusters.py --start 2019-01-01 --end 2019-08-31 --location la --no-noise
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
from folium import plugins
import numpy as np
import polars as pl
import shapely.geometry as sg
from shapely.geometry import Point, MultiPoint
from scipy.spatial import ConvexHull

from src.clustering.hdbscan_daily import cluster_day, ClusteringConfig, load_waiting_zones
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/figures")

COLORS = {
    "docked":  "#1565C0",
    "waiting": "#E65100",
    "noise":   "#9E9E9E",
}


# ============================================================================
# Config helper
# ============================================================================


def _load_config(location: str) -> ClusteringConfig:
    try:
        polygons = load_waiting_zones(location)
        log.info("Loaded %d waiting zone polygons for %s", len(polygons), location)
        return ClusteringConfig(waiting_allowed_polygons=polygons)
    except FileNotFoundError:
        log.info("No waiting zones file for %s", location)
        return ClusteringConfig()


# ============================================================================
# Geometry helpers
# ============================================================================


def _rotated_min_bounding_rectangle(points: np.ndarray) -> sg.Polygon:
    """Smallest rotated rectangle (MBR) around points, using rotating calipers."""
    if len(points) < 2:
        cx, cy = float(points[0, 0]), float(points[0, 1])
        return sg.box(cx - 0.001, cy - 0.001, cx + 0.001, cy + 0.001)
    try:
        hull_idx = ConvexHull(points).vertices
        hull_pts = points[hull_idx]
    except Exception:
        hull_pts = points

    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    min_area = float("inf")
    best_rect = None

    for i in range(len(hull_pts) - 1):
        edge = hull_pts[i + 1] - hull_pts[i]
        length = np.linalg.norm(edge)
        if length < 1e-10:
            continue
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        proj = np.dot(points, np.column_stack([direction, normal]))
        min_x, max_x = proj[:, 0].min(), proj[:, 0].max()
        min_y, max_y = proj[:, 1].min(), proj[:, 1].max()
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            corners = np.array([
                direction * min_x + normal * min_y,
                direction * max_x + normal * min_y,
                direction * max_x + normal * max_y,
                direction * min_x + normal * max_y,
            ])
            best_rect = sg.Polygon(corners)

    return best_rect if best_rect is not None else sg.box(
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()),
    )


def _safe_convex_hull(points: np.ndarray) -> sg.base.BaseGeometry:
    """Convex hull robuste — gère les cas dégénérés (1 ou 2 points)."""
    if len(points) == 1:
        return Point(points[0]).buffer(0.0005)
    if len(points) == 2:
        return sg.LineString(points).buffer(0.0003)
    try:
        return MultiPoint([Point(p) for p in points]).convex_hull
    except Exception:
        return sg.box(
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        )


def _cluster_to_geometry(
    cluster_df: pl.DataFrame, label: int
) -> sg.base.BaseGeometry | None:
    """
    Polygone sémantique pour un cluster :
      docked  -> rotated MBR  (rectangle orienté — terminal compact)
      waiting -> convex hull  (zone de mouillage dispersée)
    """
    rows = cluster_df.filter(pl.col("cluster_label") == label)
    if len(rows) == 0:
        return None
    points = np.column_stack([rows["LON"].to_numpy(), rows["LAT"].to_numpy()])
    ctype = (rows["cluster_type"][0] or "waiting") if "cluster_type" in rows.columns else "waiting"
    return (
        _rotated_min_bounding_rectangle(points)
        if ctype == "docked"
        else _safe_convex_hull(points)
    )


# ============================================================================
# Satellite tile helper
# ============================================================================


def _add_satellite_layer(m: folium.Map) -> None:
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services"
            "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite ESRI",
        overlay=True,
        show=False,
    ).add_to(m)


# ============================================================================
# Feature builder (one day)
# ============================================================================


def build_features_and_stats(
    d: date, location: str, config: ClusteringConfig
) -> tuple[list[dict], str, int, int]:
    """
    Retourne (features_geojson, stats_html, total_vessels, waiting_vessels) pour un jour.
    Chaque feature a properties.times = [date_str] pour le time slider.
    """
    loc_cfg = LOCATIONS[location]
    parquet_path = loc_cfg["out_dir"] / f"{loc_cfg['prefix']}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    vessel_df = pl.read_parquet(parquet_path)
    cluster_df, _ = cluster_day(parquet_path, config=config)

    if cluster_df is None:
        log.warning("Pas de clusters pour %s", d)
        cluster_df = pl.DataFrame(schema=vessel_df.schema)

    if "cluster_type" not in cluster_df.columns:
        cluster_df = cluster_df.with_columns(pl.lit("noise").alias("cluster_type"))

    date_str = d.strftime("%Y-%m-%d")
    features: list[dict] = []

    # --- Clusters valides : polygones sémantiques ----------------------------
    valid_labels = (
        cluster_df.filter(pl.col("cluster_label") >= 0)["cluster_label"]
        .unique().sort().to_list()
    )
    for label in valid_labels:
        geom = _cluster_to_geometry(cluster_df, label)
        if geom is None:
            continue

        rows = cluster_df.filter(pl.col("cluster_label") == label)
        ctype = (rows["cluster_type"][0] or "waiting") if len(rows) > 0 else "waiting"
        n_vessels = rows["MMSI"].n_unique()
        avg_score = (
            float(rows["membership_score"].mean())
            if "membership_score" in rows.columns else 0.0
        )
        heading_std = (
            float(rows["Heading_std"].drop_nulls().mean())
            if "Heading_std" in rows.columns and rows["Heading_std"].drop_nulls().len() > 0
            else None
        )

        popup_html = (
            f'<div style="font-family:sans-serif;width:220px;">'
            f'<b>Cluster {label}</b> — '
            f'<span style="color:{COLORS[ctype]};font-weight:bold">{ctype.upper()}</span>'
            f'<hr style="margin:4px 0">'
            f'<table style="font-size:11px;width:100%">'
            f'<tr><td><b>Navires</b></td><td>{n_vessels}</td></tr>'
            f'<tr><td><b>Membership score</b></td><td>{avg_score:.3f}</td></tr>'
            + (f'<tr><td><b>Heading std</b></td><td>{heading_std:.1f}°</td></tr>'
               if heading_std is not None else "")
            + "</table></div>"
        )

        features.append({
            "type": "Feature",
            "geometry": sg.mapping(geom),
            "properties": {
                "times": [date_str],
                "cluster_label": int(label),
                "cluster_type": ctype,
                "n_vessels": n_vessels,
                "popup": popup_html,
                "style": {
                    "color": COLORS[ctype],
                    "fillColor": COLORS[ctype],
                    "fillOpacity": 0.30 if ctype == "waiting" else 0.22,
                    "weight": 2,
                    "opacity": 0.85,
                },
            },
        })

    # --- Points bruit --------------------------------------------------------
    noise_rows = cluster_df.filter(pl.col("cluster_label") == -1)
    for row in noise_rows.iter_rows(named=True):
        popup_html = (
            f'<div style="font-family:sans-serif;width:160px;">'
            f'<b>Bruit — MMSI {row["MMSI"]}</b>'
            f'<hr style="margin:4px 0">'
            f'<span style="font-size:11px;">Draft : {row.get("Draft", "N/A")}</span>'
            f"</div>"
        )
        features.append({
            "type": "Feature",
            "geometry": sg.mapping(Point(row["LON"], row["LAT"])),
            "properties": {
                "times": [date_str],
                "cluster_label": -1,
                "cluster_type": "noise",
                "popup": popup_html,
                "style": {
                    "radius": 2,
                    "color": COLORS["noise"],
                    "fillColor": COLORS["noise"],
                    "fillOpacity": 0.35,
                },
            },
        })

    # --- Stats ---------------------------------------------------------------
    docked_count  = sum(1 for f in features if f["properties"]["cluster_type"] == "docked")
    waiting_count = sum(1 for f in features if f["properties"]["cluster_type"] == "waiting")
    noise_count   = sum(1 for f in features if f["properties"]["cluster_label"] == -1)
    total_vessels = vessel_df["MMSI"].n_unique()
    waiting_vessels = sum(
        f["properties"].get("n_vessels", 0)
        for f in features if f["properties"]["cluster_type"] == "waiting"
    )

    stats_html = (
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        f'background:white;padding:14px 18px;border-radius:10px;'
        f'border:1px solid #ddd;font-family:sans-serif;font-size:12px;'
        f'box-shadow:2px 2px 8px rgba(0,0,0,.18);max-width:260px;">'
        f'<b style="font-size:14px;">{location.upper()} — {d.strftime("%d %b %Y")}</b>'
        f'<hr style="margin:6px 0;border:none;border-top:1px solid #eee">'
        f'<div><span style="color:{COLORS["docked"]}">&#9632;</span>'
        f' Terminaux : <b>{docked_count}</b></div>'
        f'<div><span style="color:{COLORS["waiting"]}">&#9632;</span>'
        f' Congestion : <b>{waiting_count}</b></div>'
        f'<div><span style="color:{COLORS["noise"]}">&#9632;</span>'
        f' Bruit : <b>{noise_count}</b></div>'
        f'<hr style="margin:6px 0;border:none;border-top:1px solid #eee">'
        f'<div><b>Navires totaux :</b> {total_vessels}</div>'
        f"</div>"
    )

    return features, stats_html, total_vessels, waiting_vessels


# ============================================================================
# Single-day map
# ============================================================================


def visualize_day(
    d: date, location: str, config: ClusteringConfig, out_path: Path
) -> None:
    loc_cfg = LOCATIONS[location]
    features, stats_html, _, _ = build_features_and_stats(d, location, config)

    center_lat = (loc_cfg.get("lat_min", 29.0) + loc_cfg.get("lat_max", 30.0)) / 2
    center_lon = (loc_cfg.get("lon_min", -95.0) + loc_cfg.get("lon_max", -94.0)) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    _add_satellite_layer(m)

    cluster_group = folium.FeatureGroup(name="Zones HDBSCAN", show=True)
    noise_group   = folium.FeatureGroup(name="Bruit (-1)", show=False)

    for feat in features:
        props = feat["properties"]
        ctype = props["cluster_type"]
        style = props["style"]
        geom  = feat["geometry"]
        popup = folium.Popup(props["popup"], max_width=250)

        if ctype == "noise":
            folium.CircleMarker(
                location=[geom["coordinates"][1], geom["coordinates"][0]],
                radius=style["radius"],
                color=style["color"],
                fill=True,
                fill_color=style["fillColor"],
                fill_opacity=style["fillOpacity"],
                popup=popup,
            ).add_to(noise_group)
        else:
            coords = geom["coordinates"]
            if geom["type"] == "Polygon":
                folium.Polygon(
                    locations=[[p[1], p[0]] for p in coords[0]],
                    popup=popup,
                    tooltip=(
                        f"Cluster {props['cluster_label']} — "
                        f"{ctype} ({props['n_vessels']} navires)"
                    ),
                    color=style["color"],
                    fill_color=style["fillColor"],
                    fill_opacity=style["fillOpacity"],
                    weight=style["weight"],
                    opacity=style["opacity"],
                ).add_to(cluster_group)
            elif geom["type"] == "LineString":
                folium.PolyLine(
                    locations=[[p[1], p[0]] for p in coords],
                    popup=popup,
                    color=style["color"],
                    weight=style["weight"],
                ).add_to(cluster_group)

    cluster_group.add_to(m)
    noise_group.add_to(m)

    if config.waiting_allowed_polygons:
        wz_group = folium.FeatureGroup(name="Zones d'attente", show=True)
        for poly in config.waiting_allowed_polygons:
            folium.Polygon(
                locations=poly, color="#FF6F00", weight=2,
                fill=True, fill_color="#FF6F00", fill_opacity=0.08,
            ).add_to(wz_group)
        wz_group.add_to(m)

    m.get_root().html.add_child(folium.Element(stats_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved: {out_path}")


# ============================================================================
# Period map with time slider
# ============================================================================


def visualize_period(
    start: date,
    end: date,
    location: str,
    config: ClusteringConfig,
    out_path: Path,
    show_noise: bool = True,
) -> None:
    """Un seul HTML avec time slider — un frame par jour."""
    loc_cfg = LOCATIONS[location]
    center_lat = (loc_cfg.get("lat_min", 29.0) + loc_cfg.get("lat_max", 30.0)) / 2
    center_lon = (loc_cfg.get("lon_min", -95.0) + loc_cfg.get("lon_max", -94.0)) / 2

    all_features: list[dict] = []
    congestion_data: dict[str, dict] = {}
    total_days = (end - start).days + 1

    for i in range(total_days):
        d = start + timedelta(days=i)
        try:
            features, _, total_vessels, waiting_vessels = build_features_and_stats(
                d, location, config
            )
            date_str = d.strftime("%Y-%m-%d")
            waiting_clusters = sum(
                1 for f in features if f["properties"]["cluster_type"] == "waiting"
            )
            congestion_data[date_str] = {
                "waiting_clusters": waiting_clusters,
                "waiting_vessels": waiting_vessels,
                "total_vessels": total_vessels,
            }
            for feat in features:
                if not show_noise and feat["properties"]["cluster_type"] == "noise":
                    continue
                all_features.append(feat)
            log.info("Processed %s (%d/%d)", d, i + 1, total_days)
        except FileNotFoundError:
            log.warning("Parquet manquant pour %s — ignoré", d)
        except Exception as exc:
            log.warning("%s : %s", d, exc)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    _add_satellite_layer(m)

    if all_features:
        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": all_features},
            period="P1D",
            duration="P1D",
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options="YYYY-MM-DD",
            time_slider_drag_update=True,
        ).add_to(m)

    if config.waiting_allowed_polygons:
        for poly in config.waiting_allowed_polygons:
            folium.Polygon(
                locations=poly, color="#FF6F00", weight=2,
                fill=True, fill_color="#FF6F00", fill_opacity=0.08,
            ).add_to(m)

    # --- Légende fixe --------------------------------------------------------
    legend_html = (
        f'<div style="position:fixed;top:10px;right:10px;z-index:9999;'
        f'background:white;padding:14px 18px;border-radius:10px;'
        f'border:1px solid #ddd;font-family:sans-serif;font-size:12px;'
        f'box-shadow:2px 2px 8px rgba(0,0,0,.18);min-width:190px;">'
        f'<b style="font-size:14px;">{location.upper()} — {start.year}</b>'
        f'<hr style="margin:6px 0;border:none;border-top:1px solid #eee">'
        f'<div style="margin-bottom:4px;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{COLORS["docked"]};border-radius:2px;"></span>'
        f'&nbsp;Terminaux (docked)</div>'
        f'<div style="margin-bottom:4px;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{COLORS["waiting"]};border-radius:2px;"></span>'
        f'&nbsp;Congestion (waiting)</div>'
        f'<div>'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{COLORS["noise"]};border-radius:2px;"></span>'
        f'&nbsp;Bruit</div>'
        f"</div>"
    )

    # --- Gauge de congestion dynamique ---------------------------------------
    gauge_html = (
        '<div id="congestion-gauge" style="position:fixed;top:10px;left:60px;z-index:9999;'
        'background:white;padding:12px 16px;border-radius:10px;'
        'border:1px solid #ddd;font-family:sans-serif;font-size:12px;'
        'box-shadow:2px 2px 8px rgba(0,0,0,.18);min-width:230px;">'
        '<b>Niveau de congestion</b>'
        '<div id="gauge-text" style="margin:6px 0;color:#666;">—</div>'
        '<div style="width:100%;background:#eee;border-radius:4px;height:14px;">'
        '<div id="gauge-bar" style="height:100%;background:#E65100;'
        'border-radius:4px;width:0%;transition:width 0.3s;"></div>'
        '</div></div>'
    )

    max_waiting = max(
        (v["waiting_vessels"] for v in congestion_data.values()), default=1
    ) or 1

    js_script = f"""
    <script>
    var congestionData = {json.dumps(congestion_data)};
    var maxWaiting = {max_waiting};

    function updateGauge(dateStr) {{
        var data = congestionData[dateStr];
        if (!data) return;
        var pct = Math.round(data.waiting_vessels / maxWaiting * 100);
        document.getElementById('gauge-text').innerHTML =
            '<b>' + dateStr + '</b><br>' +
            data.waiting_vessels + ' navires en attente (' +
            data.waiting_clusters + ' clusters)';
        document.getElementById('gauge-bar').style.width = pct + '%';
    }}

    function tryBindTimeDimension(attempts) {{
        var map = null;
        for (var key in window) {{
            if (key.startsWith('map_') && window[key] && window[key].timeDimension) {{
                map = window[key]; break;
            }}
        }}
        if (!map) {{
            if (attempts > 0) setTimeout(function() {{ tryBindTimeDimension(attempts - 1); }}, 300);
            return;
        }}
        map.timeDimension.on('timeload', function() {{
            var t = map.timeDimension.getCurrentTime();
            if (t) updateGauge(new Date(t).toISOString().slice(0, 10));
        }});
        var t0 = map.timeDimension.getCurrentTime();
        if (t0) updateGauge(new Date(t0).toISOString().slice(0, 10));
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        tryBindTimeDimension(20);
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(gauge_html))
    m.get_root().html.add_child(folium.Element(js_script))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Saved: {out_path}")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisation géométrique des clusters HDBSCAN"
    )
    parser.add_argument(
        "--location", default="houston", choices=list(LOCATIONS.keys())
    )
    parser.add_argument("--out", default=None, metavar="PATH")
    parser.add_argument(
        "--no-noise", action="store_true", help="Masquer les points de bruit"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--date",  metavar="YYYY-MM-DD", help="Carte pour un seul jour")
    mode.add_argument(
        "--start", metavar="YYYY-MM-DD", help="Debut de periode (avec --end)"
    )
    parser.add_argument(
        "--end", metavar="YYYY-MM-DD", help="Fin de periode (requis avec --start)"
    )
    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--end est requis avec --start")

    config = _load_config(args.location)

    if args.date:
        d = date.fromisoformat(args.date)
        out = (
            Path(args.out)
            if args.out
            else OUT_DIR / f"{args.location}_clusters_{d.isoformat()}.html"
        )
        visualize_day(d, args.location, config, out)
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        out = (
            Path(args.out)
            if args.out
            else OUT_DIR / f"{args.location}_clusters_{args.start}_{args.end}.html"
        )
        visualize_period(start, end, args.location, config, out, show_noise=not args.no_noise)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
