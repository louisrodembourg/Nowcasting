"""
Visualisation géometrique des clusters HDBSCAN — Phase 1.

Génère des polygones sémantiques :
  - Rotated Bounding Boxes  → clusters 'docked' (navires alignés au quai)
  - Convex Hulls           → clusters 'waiting' (zone de mouillage dispersée)
  - Points gris            → bruit (navires en mouvement)

Satellite imagery tiles en fond pour validation visuelle.

Usage:
    python src/clustering/visualize_clusters.py --date 2017-08-25
    python src/clustering/visualize_clusters.py --date 2017-08-25 --location la
    python src/clustering/visualize_clusters.py --start 2019-01-01 --end 2019-12-31 --location la --no-docked --no-noise
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
from folium import plugins
import numpy as np
import polars as pl
import shapely.geometry as sg
import shapely.ops as so
from shapely import Point, MultiPoint
from scipy.spatial import ConvexHull

from src.clustering.hdbscan_daily import (
    cluster_day,
    ClusteringConfig,
    load_waiting_zones,
)
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "docked": "#1565C0",  # Blue — terminals alignés
    "waiting": "#E65100",  # Deep orange — zone congestion
    "noise": "#9E9E9E",  # Gray — en mouvement
}

STYLE_FUNCTIONS = {
    "docked": lambda f: {
        "fillColor": COLORS["docked"],
        "color": COLORS["docked"],
        "fillOpacity": 0.25,
        "weight": 2,
        "opacity": 0.8,
    },
    "waiting": lambda f: {
        "fillColor": COLORS["waiting"],
        "color": COLORS["waiting"],
        "fillOpacity": 0.30,
        "weight": 2,
        "opacity": 0.9,
    },
}


# ----------------------------------------------------------------------------
# Geometrical helpers
# ----------------------------------------------------------------------------


def _rotated_min_bounding_rectangle(points: np.ndarray) -> sg.Polygon:
    """
    Calcule le plus petit rectangle orienté (MBR) contenant les points.
    Utilise le rotating calipers sur l'enveloppe convexe pour O(n) au lieu de O(n⁴).
    Returns a shapely Polygon (lon, lat) coordinates.
    """
    if len(points) < 2:
        return (
            sg.box(
                *points[:, 0].min(),
                *points[:, 1].min(),
                *points[:, 0].max(),
                *points[:, 1].max(),
            )
            .boundary.geoms[0]
            .buffer(0)
        )
    try:
        hull_pts = points[ConvexHull(points).vertices]
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
            corners = np.column_stack(
                [
                    direction * min_x + normal * min_y,
                    direction * max_x + normal * min_y,
                    direction * max_x + normal * max_y,
                    direction * min_x + normal * max_y,
                ]
            ).T
            best_rect = sg.Polygon(corners)

    if best_rect is None:
        return sg.box(
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        )
    return best_rect


def _safe_convex_hull(points: np.ndarray) -> sg.Polygon:
    """Convex hull robuste (gère les cas degenerés)."""
    if len(points) < 3:
        if len(points) == 1:
            return Point(points[0]).buffer(0.0005)  # ~50m buffer
        if len(points) == 2:
            p1, p2 = Point(points[0]), Point(points[1])
            return sg.LineString([p1, p2]).buffer(0.0003)
    try:
        return MultiPoint([Point(p) for p in points]).convex_hull
    except Exception:
        return sg.box(*points.min(axis=0), *points.max(axis=0))


def _cluster_to_geometry(cluster_df: pl.DataFrame, cluster_label: int) -> sg.Polygon:
    """
    Construit le polygone sémantique pour un cluster :
      - docked  → rotated MBR (rectangle orienté)
      - waiting → convex hull
    """
    rows = cluster_df.filter(pl.col("cluster_label") == cluster_label)
    if len(rows) == 0:
        return None

    lats = rows["LAT"].to_numpy()
    lons = rows["LON"].to_numpy()
    points = np.column_stack([lons, lats])  # (lon, lat) pour shapely

    ctype = rows["cluster_type"][0] if "cluster_type" in rows.columns else "waiting"
    if ctype == "docked":
        return _rotated_min_bounding_rectangle(points)
    else:
        return _safe_convex_hull(points)


# ----------------------------------------------------------------------------
# Satellite tile layer helper
# ----------------------------------------------------------------------------


def _add_satellite_layer(m: folium.Map) -> None:
    """Ajoute une couche satellite ESRI en overlay, par défaut masquée."""
    satellite = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite ESRI",
        overlay=True,
        show=False,
    )
    satellite.add_to(m)


# ----------------------------------------------------------------------------
# Build polygon GeoJSON features
# ----------------------------------------------------------------------------


def _build_geojson_features(
    cluster_df: pl.DataFrame,
    vessel_df: pl.DataFrame,
) -> list[dict]:
    """
    Construit la liste des features GeoJSON pour les clusters et le bruit.
    Chaque cluster est un polygone sémantique (MBR ou hull).
    Chaque point bruit est un simple Marker.
    """
    features = []
    unique_labels = set(cluster_df["cluster_label"].to_list())

    for label in sorted(unique_labels):
        if label == -1:
            continue

        poly = _cluster_to_geometry(cluster_df, label)
        if poly is None:
            continue

        rows = cluster_df.filter(pl.col("cluster_label") == label)
        ctype = rows["cluster_type"][0] if "cluster_type" in rows.columns else "noise"
        n_vessels = rows["MMSI"].n_unique()
        n_messages = rows["nb_messages"].sum() if "nb_messages" in rows.columns else 0
        avg_score = (
            rows["membership_score"].mean() if "membership_score" in rows.columns else 0
        )
        heading_std = (
            rows["Heading_std"].mean()
            if "Heading_std" in rows.columns
            and rows["Heading_std"].is_null().sum() != len(rows)
            else None
        )

        # Satellite image URL (zoom 17, centered on polygon centroid)
        cx, cy = poly.centroid.x, poly.centroid.y
        sat_url = (
            f"https://static-maps.yandex.ru/1.x/?ll={cx},{cy}&z=15&l=sat"
            if -180 <= cx <= 180 and -85 <= cy <= 85
            else None
        )

        popup_html = f"""
        <div style="font-family:sans-serif;width:220px;">
          <b>Cluster {label}</b> — <span style="color:{COLORS[ctype]};font-weight:bold">{ctype.upper()}</span>
          <hr style="margin:4px 0">
          <table style="font-size:11px;width:100%">
            <tr><td><b>Navires</b></td><td>{n_vessels}</td></tr>
            <tr><td><b>Messages AIS</b></td><td>{n_messages}</td></tr>
            <tr><td><b>Membership score</b></td><td>{avg_score:.3f}</td></tr>
            {f"<tr><td><b>Heading std</b></td><td>{heading_std:.1f}°</td></tr>" if heading_std else ""}
          </table>
          {f'<img src="{sat_url}" width="210" style="margin-top:6px;border-radius:4px;">' if sat_url else ""}
        </div>
        """

        feature = {
            "type": "Feature",
            "geometry": sg.mapping(poly),
            "properties": {
                "cluster_label": int(label),
                "cluster_type": ctype,
                "n_vessels": n_vessels,
                "popup": popup_html,
                "style": STYLE_FUNCTIONS[ctype](None),
            },
        }
        features.append(feature)

    # Bruit : points simples
    noise = cluster_df.filter(pl.col("cluster_label") == -1)
    for row in noise.iter_rows(named=True):
        pt = sg.mapping(Point(row["LON"], row["LAT"]))
        popup_html = f"""
        <div style="font-family:sans-serif;width:180px;">
          <b>Bruit — MMSI {row["MMSI"]}</b>
          <hr style="margin:4px 0">
          <table style="font-size:11px;">
            <tr><td>Membership</td><td>{row.get("membership_score", 0):.3f}</td></tr>
            <tr><td>Draft</td><td>{row.get("Draft", "N/A")}</td></tr>
          </table>
        </div>
        """
        features.append(
            {
                "type": "Feature",
                "geometry": pt,
                "properties": {
                    "cluster_label": -1,
                    "cluster_type": "noise",
                    "popup": popup_html,
                    "style": {
                        "radius": 1,
                        "color": COLORS["noise"],
                        "fillColor": COLORS["noise"],
                        "fillOpacity": 0.3,
                    },
                },
            }
        )

    return features


# ----------------------------------------------------------------------------
# Main map builder
# ----------------------------------------------------------------------------


def build_features_and_stats(
    d: date,
    location: str = "houston",
    config: Optional[ClusteringConfig] = None,
) -> tuple[list, str, int]:
    """
    Construit les features GeoJSON et les statistiques pour un jour.
    Retourne (features, stats_html, total_vessels)
    """
    loc_cfg = LOCATIONS[location]
    prefix = loc_cfg["prefix"]
    parquet_dir = Path(__file__).resolve().parents[2] / "data" / "parquet" / location
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    vessel_df = pl.read_parquet(parquet_path)
    cluster_df, _ = cluster_day(parquet_path, config=config)

    if cluster_df is None:
        log.warning("Pas de clusters pour %s", d)
        cluster_df = vessel_df.head(0)

    if "cluster_type" not in cluster_df.columns:
        cluster_df = cluster_df.with_columns(pl.lit("noise").alias("cluster_type"))

    features = _build_geojson_features(cluster_df, vessel_df)

    # Statistiques
    docked_count = sum(
        1 for f in features if f["properties"]["cluster_type"] == "docked"
    )
    waiting_count = sum(
        1 for f in features if f["properties"]["cluster_type"] == "waiting"
    )
    noise_count = sum(1 for f in features if f["properties"]["cluster_label"] == -1)
    total_vessels = vessel_df["MMSI"].n_unique()

    stats_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,.18);max-width:280px;">
      <b style="font-size:14px;">{location.upper()} — {d.strftime("%d %b %Y")}</b>
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
      <div style="font-size:10px;color:#666;margin-top:4px;">🛰️ Satellite en overlay (icône en haut à droite)</div>
    </div>
    """

    return features, stats_html, total_vessels


def visualize_day(
    d: date,
    location: str = "houston",
    config: Optional[ClusteringConfig] = None,
) -> Path:
    """
    Génère la carte Folium pour un jour.
    """
    disp_cfg = {
        "houston": {"center": [29.60, -95.05], "zoom": 12},
        "la": {"center": [33.745, -118.22], "zoom": 11},
    }.get(location, {"center": [29.60, -95.05], "zoom": 12})

    features, stats_html, total_vessels = build_features_and_stats(
        d, location, config=config
    )

    # Map
    m = folium.Map(
        location=disp_cfg["center"],
        zoom_start=disp_cfg["zoom"],
        tiles="CartoDB positron",
    )
    _add_satellite_layer(m)

    # --- Polygones sémantiques (docked + waiting) ---
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
                tooltip=f"Cluster {props['cluster_label']} — {props['cluster_type']} ({props['n_vessels']} navires)",
                **props["style"],
            ).add_to(polygon_group)

        elif feat["geometry"]["type"] == "LineString":
            coords = feat["geometry"]["coordinates"]
            folium.PolyLine(
                locations=[[p[1], p[0]] for p in coords],
                popup=folium.Popup(props["popup"], max_width=250),
                tooltip=f"Cluster {props['cluster_label']} — {props['cluster_type']} (ligne)",
                **props["style"],
            ).add_to(polygon_group)

    polygon_group.add_to(m)

    # --- Points bruit ---
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
            popup=folium.Popup(props["popup"], max_width=200),
        ).add_to(noise_group)
    noise_group.add_to(m)

    # --- Navires en mouvement (échantillon) ---
    loc_cfg_v = LOCATIONS[location]
    prefix = loc_cfg_v["prefix"]
    parquet_dir = Path(__file__).resolve().parents[2] / "data" / "parquet" / location
    parquet_path = parquet_dir / f"{prefix}_{d.strftime('%Y_%m_%d')}.parquet"
    vessel_df = pl.read_parquet(parquet_path)
    moving = vessel_df.filter(pl.col("SOG") >= 1.0)
    if len(moving) > 0:
        moving_group = folium.FeatureGroup(
            name=f"En mouvement (SOG≥1 kt) — sample 500", show=False
        )
        sample = moving.sample(min(len(moving), 500), seed=42)
        for row in sample.iter_rows(named=True):
            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=1.5,
                color="#BDBDBD",
                fill=True,
                fill_opacity=0.2,
                tooltip=f"MMSI {row['MMSI']} | {row['SOG']:.1f} kt",
            ).add_to(moving_group)
        moving_group.add_to(m)

    m.get_root().html.add_child(folium.Element(stats_html))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = OUT_DIR / f"{location}_semantic_{d.strftime('%Y_%m_%d')}.html"
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


def visualize_period(
    start: date,
    end: date,
    location: str = "houston",
    no_docked: bool = False,
    no_waiting: bool = False,
    no_noise: bool = False,
    config: Optional[ClusteringConfig] = None,
) -> Path:
    """
    Génère une carte Folium avec time slider pour la période.
    """
    disp_cfg = {
        "houston": {"center": [29.60, -95.05], "zoom": 12},
        "la": {"center": [33.745, -118.22], "zoom": 11},
    }.get(location, {"center": [29.60, -95.05], "zoom": 12})

    # Collect features with timestamps by type
    docked_features = []
    waiting_features = []
    noise_features = []
    congestion_data = {}
    total_days = (end - start).days + 1
    for i, d in enumerate(range((end - start).days + 1)):
        d_date = start + timedelta(days=i)
        try:
            features, _, total_vessels = build_features_and_stats(
                d_date, location, config=config
            )
            waiting_count = sum(
                1 for f in features if f["properties"]["cluster_type"] == "waiting"
            )
            waiting_vessels = sum(
                f["properties"].get("n_vessels", 0)
                for f in features
                if f["properties"]["cluster_type"] == "waiting"
            )
            congestion_data[d_date.strftime("%Y-%m-%d")] = {
                "waiting_clusters": waiting_count,
                "waiting_vessels": waiting_vessels,
                "total_vessels": total_vessels,
            }
            for feat in features:
                ctype = feat["properties"]["cluster_type"]
                feat["properties"]["times"] = [d_date.strftime("%Y-%m-%d")]
                if ctype == "docked":
                    docked_features.append(feat)
                elif ctype == "waiting":
                    waiting_features.append(feat)
                elif ctype == "noise":
                    noise_features.append(feat)
            log.info(f"Processed {d_date} ({i + 1}/{total_days})")
        except Exception as exc:
            log.warning("%s : %s", d_date, exc)

    # Map
    m = folium.Map(
        location=disp_cfg["center"],
        zoom_start=disp_cfg["zoom"],
        tiles="CartoDB positron",
    )
    _add_satellite_layer(m)

    # Add a single TimestampedGeoJson layer for all features
    all_features = docked_features + waiting_features + noise_features
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

    # Add legend with checkboxes to toggle feature types
    legend_title = f"{location.upper()} — {start.strftime('%Y')}"
    legend_content = """
    <div style="margin-bottom:6px;">
      <label><input type="checkbox" id="toggle-docked" checked> Terminaux</label>
    </div>
    <div style="margin-bottom:6px;">
      <label><input type="checkbox" id="toggle-waiting" checked> Congestion</label>
    </div>
    <div style="margin-bottom:6px;">
      <label><input type="checkbox" id="toggle-noise" checked> Bruit</label>
    </div>
    """

    legend_html = f"""
    <div style="position:fixed;top:10px;right:10px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,.18);max-width:260px;">
      <b style="font-size:14px;">{legend_title}</b>
      <hr style="margin:6px 0;border:none;border-top:1px solid #eee">
      {legend_content}
      <div style="font-size:10px;color:#666;margin-top:4px;">🛰️ Slider unique + filtres</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add congestion gauge
    gauge_html = f"""
    <div id="congestion-gauge" style="position:fixed;top:10px;left:10px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,.18);max-width:220px;">
      <b>Niveau de congestion</b>
      <div id="gauge-text">Chargement...</div>
      <div style="margin-top:10px;">
        <div style="width:100%;background:#eee;border-radius:5px;height:20px;">
          <div id="gauge-bar" style="height:100%;background:#E65100;border-radius:5px;width:0%;"></div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(gauge_html))

    # Add JS to update the gauge and checkboxes dynamically
    js_script = f"""
    <script>
    var congestionData = {json.dumps(congestion_data)};
    var maxWaiting = Math.max(...Object.values(congestionData).map(d => d.waiting_vessels || d.waiting_clusters));

    function updateGauge(date) {{
        if (congestionData[date]) {{
            var data = congestionData[date];
            var waitingVessels = data.waiting_vessels || 0;
            var level = maxWaiting > 0 ? (waitingVessels / maxWaiting * 100) : 0;
            document.getElementById('gauge-text').innerHTML =
                date + '<br>' + waitingVessels + ' navires en attente<br>' + data.waiting_clusters + ' clusters en attente<br>' + data.total_vessels + ' navires totaux';
            document.getElementById('gauge-bar').style.width = level + '%';
        }}
    }}

    function getMapInstance() {{
        return Object.keys(window).reduce(function(found, key) {{
            if (found) return found;
            if (key.startsWith('map_') && window[key] && window[key].timeDimension) {{
                return window[key];
            }}
            return null;
        }}, null);
    }}

    function getFeatureType(layer) {{
        return layer && layer.feature && layer.feature.properties ? layer.feature.properties.cluster_type : null;
    }}

    function setLayerVisibility(layer, visible) {{
        if (!layer || !layer.feature || !layer.feature.properties) return;
        var type = layer.feature.properties.cluster_type;
        if (!type) return;
        if (layer.setStyle) {{
            if (type === 'noise') {{
                layer.setStyle({{opacity: visible ? 0.7 : 0, fillOpacity: visible ? 0.3 : 0, radius: visible ? 1 : 0}});
                if (typeof layer.setRadius === 'function') {{
                    layer.setRadius(visible ? 1 : 0);
                }}
            }} else {{
                layer.setStyle({{opacity: visible ? 0.8 : 0, fillOpacity: visible ? 0.3 : 0}});
            }}
        }} else if (typeof layer.setOpacity === 'function') {{
            layer.setOpacity(visible ? 1 : 0);
        }}
    }}

    function traverseLayer(layer, callback) {{
        if (!layer) return;
        callback(layer);
        if (layer._layers) {{
            Object.values(layer._layers).forEach(function(sub) {{
                traverseLayer(sub, callback);
            }});
        }}
        if (typeof layer.getLayers === 'function') {{
            layer.getLayers().forEach(function(sub) {{
                traverseLayer(sub, callback);
            }});
        }}
    }}

    function applyFilters() {{
        var showDocked = document.getElementById('toggle-docked').checked;
        var showWaiting = document.getElementById('toggle-waiting').checked;
        var showNoise = document.getElementById('toggle-noise').checked;
        var map = getMapInstance();
        if (!map) return;
        map.eachLayer(function(layer) {{
            traverseLayer(layer, function(sub) {{
                var type = getFeatureType(sub);
                if (type === 'docked') setLayerVisibility(sub, showDocked);
                if (type === 'waiting') setLayerVisibility(sub, showWaiting);
                if (type === 'noise') setLayerVisibility(sub, showNoise);
            }});
        }});
    }}

    function extractCurrentDate() {{
        var map = getMapInstance();
        if (!map || !map.timeDimension || typeof map.timeDimension.getCurrentTime !== 'function') return '{start.strftime("%Y-%m-%d")}';
        var current = map.timeDimension.getCurrentTime();
        if (!current) return '{start.strftime("%Y-%m-%d")}';
        var cur = new Date(current);
        return cur.toISOString().slice(0, 10);
    }}

    function bindEvents() {{
        var map = getMapInstance();
        if (!map || !map.timeDimension || typeof map.timeDimension.on !== 'function') {{
            setTimeout(bindEvents, 250);
            return;
        }}

        var updateAndFilter = function() {{
            var date = extractCurrentDate();
            updateGauge(date);
            setTimeout(applyFilters, 50);
        }};

        map.timeDimension.on('timeload', updateAndFilter);
        map.timeDimension.on('timechange', updateAndFilter);
        map.timeDimension.on('timeloading', updateAndFilter);

        updateAndFilter();
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        var dockedToggle = document.getElementById('toggle-docked');
        var waitingToggle = document.getElementById('toggle-waiting');
        var noiseToggle = document.getElementById('toggle-noise');
        if (dockedToggle) dockedToggle.addEventListener('change', applyFilters);
        if (waitingToggle) waitingToggle.addEventListener('change', applyFilters);
        if (noiseToggle) noiseToggle.addEventListener('change', applyFilters);
        bindEvents();
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_script))

    out_path = (
        OUT_DIR
        / f"{location}_semantic_{start.strftime('%Y_%m_%d')}_to_{end.strftime('%Y_%m_%d')}.html"
    )
    m.save(str(out_path))
    log.info("Sauvegardé : %s", out_path)
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisation géometrique des clusters HDBSCAN avec polygones sémantiques"
    )
    parser.add_argument(
        "--location",
        default="houston",
        choices=list(LOCATIONS.keys()),
        help="Port cible (default: houston)",
    )
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="Carte pour un seul jour")
    parser.add_argument(
        "--start", metavar="YYYY-MM-DD", help="Début de période (utiliser avec --end)"
    )
    parser.add_argument(
        "--end", metavar="YYYY-MM-DD", help="Fin de période (obligatoire avec --start)"
    )
    parser.add_argument(
        "--no-docked",
        action="store_true",
        help="Masquer les clusters 'docked'",
    )
    parser.add_argument(
        "--no-waiting",
        action="store_true",
        help="Masquer les clusters 'waiting'",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Masquer les points de bruit",
    )
    args = parser.parse_args()

    # Charge automatiquement les zones d'attente si le fichier existe
    config = None
    try:
        zones = load_waiting_zones(args.location)
        config = ClusteringConfig(waiting_allowed_polygons=zones)
        log.info("Zones d'attente chargees automatiquement : %d polygones", len(zones))
    except FileNotFoundError:
        log.info("Pas de fichier de zones d'attente pour %s — clustering standard", args.location)


    if args.date:
        out = visualize_day(
            date.fromisoformat(args.date), location=args.location, config=config
        )
        print(f"\nOuvrir dans le navigateur :\n  {out.resolve()}")
        return

    if args.start:
        if not args.end:
            parser.error("--end requis avec --start")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        out = visualize_period(
            start,
            end,
            location=args.location,
            no_docked=args.no_docked,
            no_waiting=args.no_waiting,
            no_noise=args.no_noise,
            config=config,
        )
        print(f"\nOuvrir dans le navigateur :\n  {out.resolve()}")
        return

    parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
