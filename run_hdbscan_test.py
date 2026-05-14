"""
Pipeline HDBSCAN sur les parquets filtrés par polygone (la_polygon/).
Classification docked / waiting par zones géographiques KML — pas par heading.

Zones :
  docked.klm  → terminaux intérieurs (navires à quai)
  waiting.klm → zone d'anchorage San Pedro Bay (navires en attente)

Pour chaque fichier Parquet :
  1. Prétraitement cinématique
  2. HDBSCAN sur les épisodes statiques (SOG_corr < 1 kt)
  3. Classification de chaque épisode : docked / waiting / other
     selon son appartenance aux zones KML
  4. Carte Folium : zones KML en overlay + clusters colorés par type
  5. CSV récapitulatif des features quotidiennes

Usage (depuis Nowcasting/) :
    python run_hdbscan_test.py
    python run_hdbscan_test.py --parquet-dir data/parquet/la_polygon --out-dir outputs/hdbscan_test
"""
import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import hdbscan
import numpy as np
import polars as pl
import shapely
import shapely.geometry as sg
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, MultiPoint

from src.ingestion.kinematic_filter import prepare_kinematics

log = logging.getLogger(__name__)

# ── Paramètres HDBSCAN ────────────────────────────────────────────────────────

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES      = 2
SOG_STATIC_THRESHOLD     = 1.0   # nœuds

# ── Couleurs ──────────────────────────────────────────────────────────────────

COLORS = {
    "docked":  "#1565C0",  # bleu  — terminaux
    "waiting": "#E65100",  # orange — anchorage
    "other":   "#757575",  # gris   — hors zones définies
    "noise":   "#BDBDBD",  # gris clair — bruit HDBSCAN
}


# ── Parsing KML/KLM ───────────────────────────────────────────────────────────

def load_zone_polygon(path: Path) -> Polygon:
    """
    Charge un fichier KML/KLM Google Earth et retourne le premier polygone trouvé.
    Coordonnées KML : lon,lat,alt → shapely Polygon(lon, lat).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns   = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""

    coords_el = root.find(f".//{ns}coordinates")
    if coords_el is None or not coords_el.text:
        raise ValueError(f"Pas de <coordinates> dans {path}")

    points = []
    for token in coords_el.text.strip().split():
        parts = token.split(",")
        points.append((float(parts[0]), float(parts[1])))

    if len(points) < 3:
        raise ValueError(f"{path.name}: seulement {len(points)} points")

    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    log.info("Zone '%s' chargée : %d sommets, bbox=%s", path.stem, len(points), poly.bounds)
    return poly


# ── Agrégation épisodes statiques ─────────────────────────────────────────────

def aggregate_static(static: pl.DataFrame) -> pl.DataFrame:
    """Un point représentatif par épisode statique (MMSI, traj_id)."""
    cols = static.columns
    agg_exprs = [
        pl.col("LAT").median().alias("LAT"),
        pl.col("LON").median().alias("LON"),
        pl.col("Heading").filter(pl.col("Heading") < 360).mean().alias("Heading_mean")
            if "Heading" in cols else pl.lit(None).cast(pl.Float64).alias("Heading_mean"),
        pl.col("Heading").filter(pl.col("Heading") < 360).std().alias("Heading_std")
            if "Heading" in cols else pl.lit(None).cast(pl.Float64).alias("Heading_std"),
        pl.col("Draft").max().alias("Draft")
            if "Draft"  in cols else pl.lit(None).cast(pl.Float64).alias("Draft"),
        pl.col("Length").max().alias("Length")
            if "Length" in cols else pl.lit(None).cast(pl.Float64).alias("Length"),
        pl.col("Width").max().alias("Width")
            if "Width"  in cols else pl.lit(None).cast(pl.Float64).alias("Width"),
        pl.col("VesselType").max().alias("VesselType")
            if "VesselType" in cols else pl.lit(0).cast(pl.Int64).alias("VesselType"),
        pl.len().alias("nb_messages"),
    ]
    return static.group_by(["MMSI", "traj_id"]).agg(agg_exprs)


# ── Classification géographique ───────────────────────────────────────────────

def classify_by_zone(
    agg: pl.DataFrame,
    docked_poly: Polygon,
    waiting_poly: Polygon,
) -> pl.DataFrame:
    """
    Attribue à chaque épisode statique la zone géographique KML.
    Priorité : docked > waiting > other.
    Utilise shapely.contains_xy (vectorisé, shapely >= 2.0).
    """
    lons = agg["LON"].to_numpy()
    lats = agg["LAT"].to_numpy()

    in_docked  = shapely.contains_xy(docked_poly,  lons, lats)
    in_waiting = shapely.contains_xy(waiting_poly, lons, lats)

    # Zones géographiquement distinctes — docked = terminaux intérieurs, waiting = anchorage extérieur
    zone = np.where(in_docked, "docked", np.where(in_waiting, "waiting", "other"))

    return agg.with_columns(pl.Series("zone", zone, dtype=pl.Utf8))


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(
    prepared: pl.DataFrame,
    docked_poly: Polygon,
    waiting_poly: Polygon,
    label: str = "",
) -> pl.DataFrame | None:
    """
    Exécute le pipeline complet sur un jour préparé :
      filtre statique → agrégation → HDBSCAN → classification par zone.
    Retourne cluster_df ou None si trop peu de données.
    """
    static = prepared.filter(pl.col("SOG_corr") < SOG_STATIC_THRESHOLD)

    if len(static) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning("%s : %d messages statiques — skip", label, len(static))
        return None

    agg = aggregate_static(static)

    if len(agg) < HDBSCAN_MIN_CLUSTER_SIZE:
        log.warning("%s : %d épisodes — skip HDBSCAN", label, len(agg))
        return None

    # HDBSCAN sur (LAT, LON) en radians avec métrique haversine
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

    # Classification géographique
    agg = classify_by_zone(agg, docked_poly, waiting_poly)

    # Stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    # Comptage par zone sur les épisodes NON-bruit uniquement
    valid_agg  = agg.filter(pl.col("cluster_label") >= 0)
    n_docked   = int((valid_agg["zone"] == "docked").sum())
    n_waiting  = int((valid_agg["zone"] == "waiting").sum())
    n_other    = int((valid_agg["zone"] == "other").sum())
    log.info(
        "%s : %d épisodes → %d clusters | docked=%d waiting=%d other=%d | bruit=%d (%.0f%%)",
        label, len(agg), n_clusters,
        n_docked, n_waiting, n_other,
        n_noise, 100 * n_noise / len(agg),
    )
    return agg


# ── Géométrie ─────────────────────────────────────────────────────────────────

def _safe_hull(lons: np.ndarray, lats: np.ndarray) -> sg.base.BaseGeometry:
    pts = np.column_stack([lons, lats])
    if len(pts) == 1:
        return Point(pts[0]).buffer(0.0005)
    if len(pts) == 2:
        return sg.LineString(pts).buffer(0.0003)
    try:
        return MultiPoint([Point(p) for p in pts]).convex_hull
    except Exception:
        return sg.box(*pts.min(axis=0), *pts.max(axis=0))


def _rotated_mbr(lons: np.ndarray, lats: np.ndarray) -> sg.base.BaseGeometry:
    """Smallest rotated bounding rectangle via rotating calipers."""
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
        d = edge / norm
        n = np.array([-d[1], d[0]])
        proj = pts @ np.column_stack([d, n])
        minx, maxx = proj[:, 0].min(), proj[:, 0].max()
        miny, maxy = proj[:, 1].min(), proj[:, 1].max()
        area = (maxx - minx) * (maxy - miny)
        if area < best_area:
            best_area = area
            best = sg.Polygon([d*minx+n*miny, d*maxx+n*miny, d*maxx+n*maxy, d*minx+n*maxy])
    return best if best is not None else _safe_hull(lons, lats)


def episode_geometry(cluster_df: pl.DataFrame, label: int) -> sg.base.BaseGeometry | None:
    rows = cluster_df.filter(pl.col("cluster_label") == label)
    if len(rows) == 0:
        return None
    lons = rows["LON"].to_numpy()
    lats = rows["LAT"].to_numpy()
    # Zone majoritaire dans ce cluster
    zone_counts = rows["zone"].value_counts().sort("count", descending=True)
    majority_zone = zone_counts["zone"][0] if len(zone_counts) > 0 else "other"
    # docked → rectangle orienté (navires alignés au quai)
    # waiting / other → convex hull
    geom = _rotated_mbr(lons, lats) if majority_zone == "docked" else _safe_hull(lons, lats)
    return geom, majority_zone


# ── Carte Folium ──────────────────────────────────────────────────────────────

def _add_zone_overlay(m: folium.Map, poly: Polygon, color: str, name: str) -> None:
    """Dessine le polygone de zone KML en overlay semi-transparent."""
    coords = [[p[1], p[0]] for p in poly.exterior.coords]
    folium.Polygon(
        locations=coords,
        color=color,
        fill_color=color,
        fill_opacity=0.08,
        weight=2.5,
        opacity=0.9,
        dash_array="8 4",
        tooltip=f"Zone {name}",
    ).add_to(m)


def build_map(
    cluster_df: pl.DataFrame,
    docked_poly: Polygon,
    waiting_poly: Polygon,
    d: date,
    out_path: Path,
) -> None:
    m = folium.Map(location=[33.735, -118.22], zoom_start=12, tiles="CartoDB positron")

    # Satellite overlay
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite ESRI", overlay=True, show=False,
    ).add_to(m)

    # ── Zones KML ──────────────────────────────────────────
    zone_group = folium.FeatureGroup(name="Zones géographiques (KML)", show=True)
    _add_zone_overlay(zone_group, docked_poly,  COLORS["docked"],  "Terminaux (docked)")
    _add_zone_overlay(zone_group, waiting_poly, COLORS["waiting"], "Anchorage (waiting)")
    zone_group.add_to(m)

    if cluster_df is None:
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(str(out_path))
        return

    # ── Clusters HDBSCAN colorés par zone ──────────────────
    fg_docked  = folium.FeatureGroup(name=f"Docked  ({int((cluster_df['zone']=='docked').sum())} ép.)",  show=True)
    fg_waiting = folium.FeatureGroup(name=f"Waiting ({int((cluster_df['zone']=='waiting').sum())} ép.)", show=True)
    fg_other   = folium.FeatureGroup(name="Hors zones", show=False)
    fg_noise   = folium.FeatureGroup(name="Bruit HDBSCAN", show=False)

    unique_labels = sorted(set(cluster_df["cluster_label"].to_list()))

    for label in unique_labels:
        if label == -1:
            # Bruit : petits cercles
            for row in cluster_df.filter(pl.col("cluster_label") == -1).iter_rows(named=True):
                folium.CircleMarker(
                    location=[row["LAT"], row["LON"]],
                    radius=2, color=COLORS["noise"], fill=True,
                    fill_opacity=0.4, weight=0,
                    tooltip=f"Bruit — MMSI {row['MMSI']}",
                ).add_to(fg_noise)
            continue

        result = episode_geometry(cluster_df, label)
        if result is None:
            continue
        geom, zone = result
        color = COLORS.get(zone, COLORS["other"])

        rows      = cluster_df.filter(pl.col("cluster_label") == label)
        n_vessels = rows["MMSI"].n_unique()
        n_ep      = len(rows)
        avg_score = rows["membership_score"].mean()
        popup_html = f"""
        <div style="font-family:sans-serif;width:210px;">
          <b style="color:{color};">Cluster {label} — {zone.upper()}</b>
          <hr style="margin:4px 0">
          <table style="font-size:11px;width:100%">
            <tr><td><b>Navires</b></td><td>{n_vessels}</td></tr>
            <tr><td><b>Épisodes</b></td><td>{n_ep}</td></tr>
            <tr><td><b>Membership</b></td><td>{avg_score:.3f}</td></tr>
          </table>
        </div>
        """

        kwargs = dict(
            color=color, fill_color=color,
            fill_opacity=0.25, weight=2, opacity=0.9,
            tooltip=f"Cluster {label} — {zone} ({n_vessels} navires)",
            popup=folium.Popup(popup_html, max_width=220),
        )

        if geom.geom_type == "Polygon":
            coords = [[p[1], p[0]] for p in geom.exterior.coords]
            el = folium.Polygon(locations=coords, **kwargs)
        elif geom.geom_type == "LineString":
            coords = [[p[1], p[0]] for p in geom.coords]
            el = folium.PolyLine(locations=coords, **{k: v for k, v in kwargs.items() if k not in ("fill_color", "fill_opacity")})
        else:
            continue

        target = {"docked": fg_docked, "waiting": fg_waiting}.get(zone, fg_other)
        el.add_to(target)

    fg_docked.add_to(m)
    fg_waiting.add_to(m)
    fg_other.add_to(m)
    fg_noise.add_to(m)

    # ── Légende ────────────────────────────────────────────
    valid      = cluster_df.filter(pl.col("cluster_label") >= 0)
    n_cl_total = valid["cluster_label"].n_unique()
    n_docked_cl  = valid.filter(pl.col("zone") == "docked")["cluster_label"].n_unique()
    n_waiting_cl = valid.filter(pl.col("zone") == "waiting")["cluster_label"].n_unique()
    n_docked_v   = valid.filter(pl.col("zone") == "docked")["MMSI"].n_unique()
    n_waiting_v  = valid.filter(pl.col("zone") == "waiting")["MMSI"].n_unique()
    n_noise_ep   = int((cluster_df["cluster_label"] == -1).sum())

    legend_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
                padding:16px 20px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px;box-shadow:2px 2px 8px rgba(0,0,0,.18);">
      <b style="font-size:14px;">LA — {d.strftime('%d %b %Y')}</b>
      <hr style="margin:6px 0">
      <div style="margin:4px 0">
        <span style="color:{COLORS['docked']};font-size:18px;">■</span>
        <b>Docked (terminaux)</b>
        <span style="color:#555"> — {n_docked_cl} clusters, {n_docked_v} navires</span>
      </div>
      <div style="margin:4px 0">
        <span style="color:{COLORS['waiting']};font-size:18px;">■</span>
        <b>Waiting (anchorage)</b>
        <span style="color:#555"> — {n_waiting_cl} clusters, {n_waiting_v} navires</span>
      </div>
      <div style="margin:4px 0;color:#999">
        <span style="font-size:18px;">■</span> Hors zones — {n_cl_total - n_docked_cl - n_waiting_cl} clusters
      </div>
      <div style="margin:4px 0;color:#bbb">
        <span style="font-size:18px;">●</span> Bruit — {n_noise_ep} épisodes
      </div>
      <hr style="margin:6px 0">
      <div style="font-size:10px;color:#888">Zones délimitées via Google Earth KML<br>
        Classification géographique (pas heading_std)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(out_path))
    log.info("Carte : %s", out_path)


# ── Features quotidiennes ─────────────────────────────────────────────────────

def _f(val, default: float = 0.0) -> float:
    return float(val) if val is not None else default


def compute_features(
    prepared: pl.DataFrame,
    cluster_df: pl.DataFrame | None,
    d: date,
) -> dict:
    sog_col      = "SOG_corr" if "SOG_corr" in prepared.columns else "SOG"
    vessel_count = prepared["MMSI"].n_unique()
    static_mmsi  = prepared.filter(pl.col(sog_col) < SOG_STATIC_THRESHOLD)["MMSI"].n_unique()

    base = {
        "date":                 d.isoformat(),
        "vessel_count":         vessel_count,
        "SOG_mean":             round(_f(prepared[sog_col].mean()), 4),
        "SOG_std":              round(_f(prepared[sog_col].std()),  4),
        "utilization_rate_rho": round(static_mmsi / vessel_count, 4) if vessel_count else 0.0,
    }

    if cluster_df is None or len(cluster_df) == 0:
        return {**base,
                "n_clusters": 0, "n_docked_cl": 0, "n_waiting_cl": 0,
                "n_docked_v": 0, "n_waiting_v": 0, "n_noise_ep": 0,
                "noise_ratio": 1.0, "membership_mean": 0.0,
                "draft_mean": 0.0, "blocked_capacity": 0.0}

    labels   = cluster_df["cluster_label"].to_numpy()
    valid    = cluster_df.filter(pl.col("cluster_label") >= 0)
    docked   = valid.filter(pl.col("zone") == "docked")
    waiting  = valid.filter(pl.col("zone") == "waiting")
    non_noise = cluster_df.filter(pl.col("cluster_label") >= 0)

    draft_vals = cluster_df.filter(pl.col("Draft") > 0)["Draft"]
    cap_vals   = (
        cluster_df
        .filter((pl.col("Length") > 0) & (pl.col("Width") > 0))
        .select((pl.col("Length") * pl.col("Width")).alias("cap"))["cap"]
    )

    n_ep = len(cluster_df)
    return {**base,
            "n_clusters":       valid["cluster_label"].n_unique(),
            "n_docked_cl":      docked["cluster_label"].n_unique(),
            "n_waiting_cl":     waiting["cluster_label"].n_unique(),
            "n_docked_v":       docked["MMSI"].n_unique(),
            "n_waiting_v":      waiting["MMSI"].n_unique(),
            "n_noise_ep":       int((labels == -1).sum()),
            "noise_ratio":      round(int((labels == -1).sum()) / n_ep, 4),
            "membership_mean":  round(_f(non_noise["membership_score"].mean()), 4),
            "draft_mean":       round(_f(draft_vals.mean()), 2),
            "blocked_capacity": round(_f(cap_vals.sum()), 1)}


# ── Pipeline principal ────────────────────────────────────────────────────────

def process_file(
    parquet_path: Path,
    docked_poly: Polygon,
    waiting_poly: Polygon,
    out_dir: Path,
) -> dict | None:
    log.info("=== %s ===", parquet_path.name)
    raw      = pl.read_parquet(parquet_path)
    prepared = prepare_kinematics(raw)

    stem = parquet_path.stem.split("_")
    d    = date(int(stem[-3]), int(stem[-2]), int(stem[-1]))

    cluster_df = run_hdbscan(prepared, docked_poly, waiting_poly, label=parquet_path.name)

    map_path = out_dir / f"hdbscan_zones_{d.strftime('%Y_%m_%d')}.html"
    build_map(cluster_df, docked_poly, waiting_poly, d, map_path)

    return compute_features(prepared, cluster_df, d)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDBSCAN + classification par zones KML (docked/waiting)"
    )
    parser.add_argument("--parquet-dir", default="data/parquet/la_polygon")
    parser.add_argument("--out-dir",     default="outputs/hdbscan_test")
    parser.add_argument("--docked-kml",  default="docked.klm")
    parser.add_argument("--waiting-kml", default="waiting.klm")
    args = parser.parse_args()

    parquet_dir  = Path(args.parquet_dir)
    out_dir      = Path(args.out_dir)
    docked_path  = Path(args.docked_kml)
    waiting_path = Path(args.waiting_kml)
    out_dir.mkdir(parents=True, exist_ok=True)

    docked_poly  = load_zone_polygon(docked_path)
    waiting_poly = load_zone_polygon(waiting_path)

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        log.error("Aucun parquet dans %s", parquet_dir)
        return
    log.info("%d fichiers à traiter", len(files))

    all_features: list[dict] = []
    for f in files:
        feats = process_file(f, docked_poly, waiting_poly, out_dir)
        if feats:
            all_features.append(feats)

    if all_features:
        summary  = pl.DataFrame(all_features).sort("date")
        csv_path = out_dir / "hdbscan_zones_summary.csv"
        summary.write_csv(csv_path)

        w = 110
        print("\n" + "=" * w)
        print(f"{'date':<12} {'vessels':>7} {'clusters':>9} {'docked_cl':>10} {'waiting_cl':>11} "
              f"{'docked_v':>9} {'waiting_v':>10} {'bruit%':>7} {'draft':>6}")
        print("-" * w)
        for row in summary.iter_rows(named=True):
            print(
                f"{row['date']:<12} {row['vessel_count']:>7} {row['n_clusters']:>9} "
                f"{row['n_docked_cl']:>10} {row['n_waiting_cl']:>11} "
                f"{row['n_docked_v']:>9} {row['n_waiting_v']:>10} "
                f"{row['noise_ratio']*100:>6.1f}% {row['draft_mean']:>6.2f}"
            )
        print("=" * w)
        print(f"\nCartes : {out_dir.resolve()}")
        print(f"CSV    : {csv_path.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
