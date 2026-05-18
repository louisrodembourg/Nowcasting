"""
Carte animée annuelle — HDBSCAN + classification docked/waiting par zone GeoJSON.

Pour chaque jour de l'année :
  1. Chargement parquet + prétraitement cinématique
  2. HDBSCAN sur épisodes statiques (SOG_corr < 1 kt)
  3. Classification : dans le polygone docked → "docked", sinon → "waiting"
  4. Collecte des features GeoJSON avec timestamp

Zones : data/zones/{location}_docked.geojson
  Seule la zone docked est définie — tout le reste est automatiquement "waiting".

Output : une seule carte HTML avec slider jour par jour + stats live.

Usage (depuis Nowcasting/) :
    # Port de LA (défaut)
    python run_yearly_map.py --year 2019
    python run_yearly_map.py --year 2019 --location la

    # Houston Ship Channel
    python run_yearly_map.py --year 2017 --location houston
    python run_yearly_map.py --year 2020 --location houston

    # Comparer les deux modes de zone
    python run_yearly_map.py --year 2019 --location la --zone-mode docked   # zone docked → reste waiting
    python run_yearly_map.py --year 2019 --location la --zone-mode waiting  # zone waiting → reste docked

    # Surcharge manuelle
    python run_yearly_map.py --year 2019 --parquet-dir data/parquet/la --out outputs/figures/la_2019_animated.html
"""
import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
from folium import plugins
import hdbscan
import numpy as np
import polars as pl
import shapely
import shapely.geometry as sg
from scipy.spatial import ConvexHull
from shapely.geometry import Point, MultiPoint

from src.clustering.hdbscan_daily import load_docked_zones_or_none, load_waiting_zones_or_none
from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# ── Paramètres ────────────────────────────────────────────────────────────────

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES      = 2
SOG_STATIC_THRESHOLD     = 1.0

COLORS = {
    "docked":  "#1565C0",
    "waiting": "#E65100",
    "noise":   "#BDBDBD",
}

# Configs par location
LOCATION_CONFIGS = {
    "la": {
        "prefix":      "la",
        "center":      [33.72, -118.18],
        "zoom":        11,
        "title":       "Port de LA",
        "parquet_dir": "data/parquet/la",
        "harvey":       False,
    },
    "houston": {
        "prefix":      "houston",
        "center":      [29.75, -95.00],
        "zoom":        11,
        "title":       "Houston Ship Channel",
        "parquet_dir": "data/parquet/houston",
        "harvey":      True,
    },
}

# Période Harvey (pour badge conditionnel)
HARVEY_START = date(2017, 8, 25)
HARVEY_END   = date(2017, 9, 2)


# ── Géométrie ─────────────────────────────────────────────────────────────────

def _safe_hull(lons, lats):
    pts = np.column_stack([lons, lats])
    if len(pts) == 1:
        return Point(pts[0]).buffer(0.0005)
    if len(pts) == 2:
        return sg.LineString(pts).buffer(0.0003)
    try:
        return MultiPoint([Point(p) for p in pts]).convex_hull
    except Exception:
        return sg.box(*pts.min(axis=0), *pts.max(axis=0))


def _rotated_mbr(lons, lats):
    pts = np.column_stack([lons, lats])
    if len(pts) < 2:
        return _safe_hull(lons, lats)
    try:
        hull_pts = pts[ConvexHull(pts).vertices]
    except Exception:
        return _safe_hull(lons, lats)
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    best, best_area = None, float("inf")
    for i in range(len(hull_pts) - 1):
        edge = hull_pts[i + 1] - hull_pts[i]
        norm = np.linalg.norm(edge)
        if norm < 1e-10:
            continue
        d    = edge / norm
        n    = np.array([-d[1], d[0]])
        proj = pts @ np.column_stack([d, n])
        minx, maxx = proj[:, 0].min(), proj[:, 0].max()
        miny, maxy = proj[:, 1].min(), proj[:, 1].max()
        area = (maxx - minx) * (maxy - miny)
        if area < best_area:
            best_area = area
            best = sg.Polygon([d*minx+n*miny, d*maxx+n*miny,
                               d*maxx+n*maxy, d*minx+n*maxy])
    return best if best is not None else _safe_hull(lons, lats)


def cluster_geom_and_zone(cluster_df, label):
    """Retourne (géométrie shapely, zone majoritaire) pour un cluster."""
    rows = cluster_df.filter(pl.col("cluster_label") == label)
    if len(rows) == 0:
        return None, None
    lons  = rows["LON"].to_numpy()
    lats  = rows["LAT"].to_numpy()
    zones = rows["zone"].value_counts().sort("count", descending=True)
    zone  = zones["zone"][0] if len(zones) else "waiting"
    geom  = _rotated_mbr(lons, lats) if zone == "docked" else _safe_hull(lons, lats)
    return geom, zone


# ── Pipeline par jour ─────────────────────────────────────────────────────────

def process_day(parquet_path: Path, ref_poly, zone_mode: str = "docked"):
    """
    Charge un fichier parquet, exécute HDBSCAN + classification zone.
    zone_mode "docked"  : ref_poly = zone docked, reste = waiting.
    zone_mode "waiting" : ref_poly = zone waiting, reste = docked.
    Retourne (cluster_df, stats_dict) ou (None, stats_dict) si skip.
    """
    raw      = pl.read_parquet(parquet_path)
    prepared = prepare_kinematics(raw)

    vessel_count = raw["MMSI"].n_unique()
    sog_col = "SOG_corr" if "SOG_corr" in prepared.columns else "SOG"

    static = prepared.filter(pl.col(sog_col) < SOG_STATIC_THRESHOLD)
    if len(static) < HDBSCAN_MIN_CLUSTER_SIZE:
        return None, {"vessel_count": vessel_count, "n_docked_v": 0, "n_waiting_v": 0, "n_clusters": 0}

    cols = static.columns
    agg  = static.group_by(["MMSI", "traj_id"]).agg([
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("Draft").max().alias("Draft") if "Draft" in cols else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length") if "Length" in cols else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width") if "Width" in cols else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType") if "VesselType" in cols else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),
    ])

    if len(agg) < HDBSCAN_MIN_CLUSTER_SIZE:
        return None, {"vessel_count": vessel_count, "n_docked_v": 0, "n_waiting_v": 0, "n_clusters": 0}

    coords_rad = np.radians(agg.select(["LAT", "LON"]).to_numpy())
    clusterer  = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="haversine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_rad)
    scores = clusterer.probabilities_

    agg = agg.with_columns([
        pl.Series("cluster_label",    labels, dtype=pl.Int32),
        pl.Series("membership_score", scores, dtype=pl.Float32),
    ])

    # Classification selon zone_mode
    lons = agg["LON"].to_numpy()
    lats = agg["LAT"].to_numpy()
    if ref_poly is not None:
        in_ref = shapely.contains_xy(ref_poly, lons, lats)
    else:
        in_ref = np.zeros(len(agg), dtype=bool)
    if zone_mode == "waiting":
        zone = np.where(in_ref, "waiting", "docked")
    else:
        zone = np.where(in_ref, "docked", "waiting")
    agg  = agg.with_columns(pl.Series("zone", zone, dtype=pl.Utf8))

    # Stats
    valid      = agg.filter(pl.col("cluster_label") >= 0)
    n_clusters = valid["cluster_label"].n_unique()
    n_docked_v = valid.filter(pl.col("zone") == "docked")["MMSI"].n_unique()
    n_waiting_v = valid.filter(pl.col("zone") == "waiting")["MMSI"].n_unique()

    stats = {
        "vessel_count":  vessel_count,
        "n_clusters":    n_clusters,
        "n_docked_v":    n_docked_v,
        "n_waiting_v":   n_waiting_v,
        "n_noise":       int((labels == -1).sum()),
    }
    return agg, stats


# ── Conversion GeoJSON ────────────────────────────────────────────────────────

def cluster_df_to_geojson_features(cluster_df, d: date) -> tuple[list, list, list]:
    """
    Convertit cluster_df en listes de features GeoJSON horodatées.
    Retourne (docked_features, waiting_features, noise_features).
    """
    ts = d.isoformat()
    docked_feats, waiting_feats, noise_feats = [], [], []

    unique_labels = sorted(set(cluster_df["cluster_label"].to_list()))

    for label in unique_labels:
        if label == -1:
            # Points bruit
            for row in cluster_df.filter(pl.col("cluster_label") == -1).iter_rows(named=True):
                noise_feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row["LON"], row["LAT"]]},
                    "properties": {"times": [ts], "zone": "noise",
                                   "popup": f"Bruit MMSI {row['MMSI']}",
                                   "style": {"color": COLORS["noise"], "radius": 2}},
                })
            continue

        geom, zone = cluster_geom_and_zone(cluster_df, label)
        if geom is None:
            continue

        rows      = cluster_df.filter(pl.col("cluster_label") == label)
        n_vessels = rows["MMSI"].n_unique()
        n_ep      = len(rows)
        color     = COLORS.get(zone, COLORS["waiting"])

        if geom.geom_type not in ("Polygon", "LineString", "MultiPolygon"):
            continue

        if geom.geom_type == "Polygon":
            coords = [list(geom.exterior.coords)]
        elif geom.geom_type == "MultiPolygon":
            coords = [list(g.exterior.coords) for g in geom.geoms]
        else:
            coords = [list(geom.coords)]

        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon" if geom.geom_type != "LineString" else "LineString",
                "coordinates": coords if geom.geom_type != "LineString" else list(geom.coords),
            },
            "properties": {
                "times":     [ts],
                "zone":      zone,
                "cluster":   label,
                "n_vessels": n_vessels,
                "n_ep":      n_ep,
                "style": {
                    "color":       color,
                    "fillColor":   color,
                    "fillOpacity": 0.25,
                    "weight":      2,
                    "opacity":     0.9,
                },
            },
        }

        if zone == "docked":
            docked_feats.append(feat)
        else:
            waiting_feats.append(feat)

    return docked_feats, waiting_feats, noise_feats


# ── Construction de la carte ──────────────────────────────────────────────────

def build_yearly_map(
    year: int,
    parquet_dir: Path,
    ref_poly,
    out_path: Path,
    cfg: dict,
    zone_mode: str = "docked",
) -> None:
    prefix = cfg["prefix"]
    files  = sorted(parquet_dir.glob(f"{prefix}_{year}_*.parquet"))
    if not files:
        log.error("Aucun fichier parquet %s_%d_*.parquet dans %s", prefix, year, parquet_dir)
        return
    log.info("%d fichiers pour %d", len(files), year)

    all_docked:  list[dict] = []
    all_waiting: list[dict] = []
    all_noise:   list[dict] = []
    daily_stats: dict[str, dict] = {}

    for i, f in enumerate(files):
        stem  = f.stem.split("_")
        d     = date(int(stem[-3]), int(stem[-2]), int(stem[-1]))
        ds    = d.isoformat()

        cluster_df, stats = process_day(f, ref_poly, zone_mode)
        daily_stats[ds]   = stats

        if cluster_df is not None:
            df, wf, nf = cluster_df_to_geojson_features(cluster_df, d)
            all_docked.extend(df)
            all_waiting.extend(wf)
            all_noise.extend(nf)

        if (i + 1) % 30 == 0 or i == len(files) - 1:
            log.info("  %d/%d jours traités", i + 1, len(files))

    log.info("Construction de la carte (%d feat. docked, %d waiting, %d bruit) ...",
             len(all_docked), len(all_waiting), len(all_noise))

    # ── Carte ────────────────────────────────────────────────────────────────
    m = folium.Map(location=cfg["center"], zoom_start=cfg["zoom"], tiles="CartoDB positron")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite ESRI", overlay=True, show=False,
    ).add_to(m)


    # TimestampedGeoJson par type
    def _make_timestamped(features, name, show):
        if not features:
            return
        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="P1D", duration="P1D",
            auto_play=False, loop=False,
            max_speed=5, loop_button=True,
            date_options="YYYY-MM-DD",
            time_slider_drag_update=True,
            add_last_point=False,
        ).add_to(m)

    # On met tout dans un seul TimestampedGeoJson (un seul slider)
    all_features = all_docked + all_waiting
    if all_features:
        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": all_features},
            period="P1D", duration="P1D",
            auto_play=False, loop=False,
            max_speed=5, loop_button=True,
            date_options="YYYY-MM-DD",
            time_slider_drag_update=True,
            add_last_point=False,
        ).add_to(m)

    # ── Légende + panneau stats ───────────────────────────────────────────────
    max_waiting = max((v.get("n_waiting_v", 0) for v in daily_stats.values()), default=1) or 1

    title        = cfg["title"]
    show_harvey  = cfg.get("harvey", False) and year == 2017
    harvey_note  = (
        '<div style="margin-top:6px;padding:4px 8px;background:#FFF3E0;border-radius:4px;'
        'font-size:10px;color:#E65100;border:1px solid #FFB74D;">⚠ Hurricane Harvey : 25 août – 2 sept 2017</div>'
        if show_harvey else ""
    )

    legend_html = f"""
    <div style="position:fixed;top:10px;right:10px;z-index:9999;background:white;
                padding:14px 18px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px;box-shadow:2px 2px 8px rgba(0,0,0,.2);
                min-width:200px;">
      <b style="font-size:14px;">{title} — {year}</b>
      <hr style="margin:6px 0">
      <div style="margin:3px 0"><span style="color:{COLORS['docked']};font-size:18px;">■</span> Terminaux (docked)</div>
      <div style="margin:3px 0"><span style="color:{COLORS['waiting']};font-size:18px;">■</span> Anchorage (waiting)</div>
      <hr style="margin:8px 0">
      <div style="font-size:10px;color:#888">Classification géographique (KML)</div>
      {harvey_note}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Panneau stats dynamique (mis à jour par JS avec le slider)
    harvey_badge_html = (
        '<div id="harvey-badge" style="display:none;margin-top:4px;padding:3px 8px;background:#FFEBEE;'
        'border-radius:4px;font-size:10px;color:#C62828;border:1px solid #EF9A9A;">⚠ Pendant Hurricane Harvey</div>'
        if show_harvey else ""
    )

    stats_panel_html = f"""
    <div id="stats-panel" style="position:fixed;bottom:60px;left:10px;z-index:9999;background:white;
                padding:14px 18px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px;box-shadow:2px 2px 8px rgba(0,0,0,.2);
                min-width:230px;">
      <b id="stats-date">—</b>
      {harvey_badge_html}
      <hr style="margin:6px 0">
      <table style="width:100%;font-size:11px;border-collapse:collapse;">
        <tr><td>Navires total</td><td id="s-total" style="text-align:right;font-weight:bold">—</td></tr>
        <tr><td style="color:#1565C0;">▸ À quai (docked)</td><td id="s-docked" style="text-align:right;font-weight:bold;color:#1565C0">—</td></tr>
        <tr><td style="color:#E65100;">▸ En attente (waiting)</td><td id="s-waiting" style="text-align:right;font-weight:bold;color:#E65100">—</td></tr>
        <tr><td>Clusters HDBSCAN</td><td id="s-clusters" style="text-align:right">—</td></tr>
      </table>
      <hr style="margin:8px 0">
      <div style="font-size:10px;color:#777;margin-bottom:4px">Pression anchorage</div>
      <div style="width:100%;background:#eee;border-radius:4px;height:12px;">
        <div id="gauge-bar" style="height:100%;background:#E65100;border-radius:4px;width:0%;transition:width 0.3s;"></div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_panel_html))

    # Nom exact de la variable map générée par folium — évite la recherche via Object.values(window)
    map_var = m.get_name()

    harvey_js = (
        f"var harveyStart = '{HARVEY_START.isoformat()}';\n"
        f"    var harveyEnd   = '{HARVEY_END.isoformat()}';\n"
        f"    function isHarvey(iso) {{ return iso >= harveyStart && iso <= harveyEnd; }}"
        if show_harvey else
        "var harveyStart = null; var harveyEnd = null;\n"
        "    function isHarvey(iso) { return false; }"
    )

    js = f"""
    <script>
    var dailyStats = {json.dumps(daily_stats)};
    var maxWaiting = {max_waiting};
    {harvey_js}

    function isoFromTime(t) {{
        if (!t) return null;
        return new Date(t).toISOString().slice(0, 10);
    }}

    // ── Panneau stats ──────────────────────────────────────────────────────
    function updatePanel(iso) {{
        var data = dailyStats[iso];
        if (!data) return;
        document.getElementById('stats-date').textContent  = iso;
        document.getElementById('s-total').textContent     = data.vessel_count || '—';
        document.getElementById('s-docked').textContent    = data.n_docked_v   || 0;
        document.getElementById('s-waiting').textContent   = data.n_waiting_v  || 0;
        document.getElementById('s-clusters').textContent  = data.n_clusters   || 0;
        var pct = maxWaiting > 0 ? ((data.n_waiting_v || 0) / maxWaiting * 100) : 0;
        document.getElementById('gauge-bar').style.width   = pct + '%';
        var badge = document.getElementById('harvey-badge');
        if (badge) badge.style.display = isHarvey(iso) ? 'block' : 'none';
    }}

    // ── Tooltips — bindés via layeradd (TimeDimension swaps son layer interne) ──
    var ZONE_LABELS = {{
        docked:  'Terminaux (docked)',
        waiting: 'Anchorage (waiting)',
        other:   'Hors zone'
    }};

    function bindFeatureLayer(layer) {{
        if (!layer || !layer.feature || !layer.feature.properties) return;
        if (layer._hdbBound) return;
        var p = layer.feature.properties;
        if (!p.zone || p.zone === 'noise') return;
        layer._hdbBound = true;

        var c      = (p.style && p.style.color) ? p.style.color : '#666';
        var zLabel = ZONE_LABELS[p.zone] || p.zone;

        var tip =
            '<div style="font-family:sans-serif;padding:3px 6px;line-height:1.6">' +
            '<b style="color:' + c + '">' + zLabel + '</b><br>' +
            '<b>' + p.n_vessels + '</b> navires · <b>' + p.n_ep + '</b> épisodes' +
            '</div>';

        var popup =
            '<div style="font-family:sans-serif;min-width:190px">' +
            '<b style="color:' + c + ';font-size:13px">' + zLabel + '</b>' +
            '<hr style="margin:5px 0">' +
            '<table style="font-size:11px;width:100%">' +
            '<tr><td>Cluster</td><td style="text-align:right"><b>' + p.cluster + '</b></td></tr>' +
            '<tr><td>Navires</td><td style="text-align:right"><b>' + p.n_vessels + '</b></td></tr>' +
            '<tr><td>Épisodes statiques</td><td style="text-align:right"><b>' + p.n_ep + '</b></td></tr>' +
            '</table></div>';

        layer.bindTooltip(tip, {{sticky: true, opacity: 0.95, direction: 'top'}});
        layer.bindPopup(popup, {{maxWidth: 240}});
    }}

    // Descend dans un layer group pour binder chaque feature individuelle
    function bindGroupLayers(group) {{
        if (typeof group.eachLayer === 'function') {{
            group.eachLayer(function(child) {{
                bindFeatureLayer(child);      // tente sur le child direct
                bindGroupLayers(child);       // descend si c'est un groupe
            }});
        }}
    }}

    // ── Init ──────────────────────────────────────────────────────────────
    function init() {{
        var map = {map_var};
        if (!map || !map.timeDimension) {{ setTimeout(init, 200); return; }}

        // layeradd fire quand TimeDimension swaps son inner GeoJSON layer
        // → c'est le bon moment pour binder les tooltips sur les nouvelles features
        map.on('layeradd', function(e) {{
            bindGroupLayers(e.layer);
        }});

        // Stats panel : mise à jour à chaque pas de temps
        var updateStep = function() {{
            var iso = isoFromTime(map.timeDimension.getCurrentTime());
            updatePanel(iso);
        }};
        map.timeDimension.on('timeload',   updateStep);
        map.timeDimension.on('timechange', updateStep);
        updateStep();
    }}

    window.addEventListener('load', function() {{ setTimeout(init, 400); }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    log.info("Carte sauvegardée : %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carte animée annuelle HDBSCAN + zone docked GeoJSON (LA ou Houston)"
    )
    parser.add_argument("--location",      choices=["la", "houston"], default="la",
                        help="Port cible : la (défaut) ou houston")
    parser.add_argument("--year",          type=int, default=None,
                        help="Année à tracer (défaut: 2019 pour LA, 2017 pour Houston)")
    parser.add_argument("--parquet-dir",   default=None,
                        help="Répertoire Parquet (surcharge la valeur par défaut)")
    parser.add_argument("--zone-mode",     choices=["docked", "waiting"], default="docked",
                        help="Mode de classification : 'docked' (défaut) = zone docked comme référence, "
                             "reste=waiting ; 'waiting' = zone waiting comme référence, reste=docked")
    parser.add_argument("--docked-geojson", default=None,
                        help="Fichier GeoJSON zone docked (surcharge data/zones/{location}_docked.geojson)")
    parser.add_argument("--waiting-geojson", default=None,
                        help="Fichier GeoJSON zone waiting (surcharge data/zones/{location}_waiting.geojson)")
    parser.add_argument("--out",           default=None,
                        help="Chemin HTML de sortie")
    args = parser.parse_args()

    cfg  = LOCATION_CONFIGS[args.location]
    year = args.year or (2019 if args.location == "la" else 2017)

    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else Path(cfg["parquet_dir"])
    out_path    = Path(args.out) if args.out else Path(
        f"outputs/figures/{args.location}_{year}_mode_{args.zone_mode}_animated.html"
    )

    # Chargement de la zone de référence selon zone_mode
    if args.zone_mode == "waiting":
        if args.waiting_geojson:
            import json as _json
            from shapely.geometry import shape as _shape
            from shapely.ops import unary_union as _uu
            with open(args.waiting_geojson) as f:
                fc = _json.load(f)
            feats = fc.get("features", [fc] if fc.get("type") == "Feature" else [])
            ref_poly = _uu([_shape(ft.get("geometry", ft)) for ft in feats])
        else:
            ref_poly = load_waiting_zones_or_none(args.location)
        if ref_poly is None:
            log.warning("Pas de zone waiting trouvée — tous les clusters seront classés 'docked'")
    else:
        if args.docked_geojson:
            import json as _json
            from shapely.geometry import shape as _shape
            from shapely.ops import unary_union as _uu
            with open(args.docked_geojson) as f:
                fc = _json.load(f)
            feats = fc.get("features", [fc] if fc.get("type") == "Feature" else [])
            ref_poly = _uu([_shape(ft.get("geometry", ft)) for ft in feats])
        else:
            ref_poly = load_docked_zones_or_none(args.location)
        if ref_poly is None:
            log.warning("Pas de zone docked trouvée — tous les clusters seront classés 'waiting'")

    log.info("zone_mode=%s  ref_poly=%s", args.zone_mode, "chargé" if ref_poly is not None else "None")

    build_yearly_map(
        year        = year,
        parquet_dir = parquet_dir,
        ref_poly    = ref_poly,
        out_path    = out_path,
        cfg         = cfg,
        zone_mode   = args.zone_mode,
    )
    print(f"\nOuvrir dans le navigateur : {out_path.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
