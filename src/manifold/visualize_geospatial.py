"""
Quatre visualisations géospatiales pour la section interprétabilité du rapport.

1. Carte Folium — zones constituantes sur fond satellite
2. Timeline gravity_score avec événements annotés
3. Scatter φ₁ vs φ₂ des zones, colorées par cluster_type
4. Double timeline : total_waiting_capacity (naïf) vs gravity_score (manifold)

Usage :
    python src/manifold/visualize_geospatial.py --location la
    python src/manifold/visualize_geospatial.py --location houston
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

OUT_DIR = Path("outputs/figures")

# Événements de référence par location
EVENTS = {
    "la": [
        {"date": "2020-03-19", "label": "COVID lockdown CA",  "color": "#E53935"},
        {"date": "2021-09-01", "label": "Port congestion peak", "color": "#FB8C00"},
        {"date": "2022-01-01", "label": "Décrue congestion",    "color": "#43A047"},
    ],
    "houston": [
        {"date": "2017-08-25", "label": "Harvey landfall",   "color": "#E53935"},
        {"date": "2017-09-15", "label": "Reprise trafic",    "color": "#43A047"},
    ],
}

MAP_CENTER = {
    "la":      {"center": [33.745, -118.22], "zoom": 10},
    "houston": {"center": [29.60,  -95.05],  "zoom": 11},
}


# ── 1. Carte Folium — zones constituantes ─────────────────────────────────────

def plot_constituent_map(zones: pl.DataFrame, location: str) -> Path:
    cfg = MAP_CENTER.get(location, MAP_CENTER["la"])
    m = folium.Map(location=cfg["center"], zoom_start=cfg["zoom"],
                   tiles="CartoDB positron")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=True, show=True,
    ).add_to(m)

    docked_group  = folium.FeatureGroup(name="Docked (à quai)",  show=True)
    waiting_group = folium.FeatureGroup(name="Waiting (attente)", show=True)

    for row in zones.iter_rows(named=True):
        ctype  = row.get("cluster_type", "waiting")
        color  = "#1565C0" if ctype == "docked" else "#E65100"
        radius = 6 if row.get("is_constituent", False) else 3
        group  = docked_group if ctype == "docked" else waiting_group

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(
                f"<b>{row['zone_key']}</b><br>"
                f"Type : {ctype}<br>"
                f"φ₁ : {row.get('phi_1', 0):.4f}<br>"
                f"φ₂ : {row.get('phi_2', 0):.4f}<br>"
                f"Navires (total) : {row.get('n_vessels_total', '?')}",
                max_width=200,
            ),
        ).add_to(group)

    docked_group.add_to(m)
    waiting_group.add_to(m)
    folium.LayerControl().add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;">
        <b>{location.upper()} — Zones constituantes</b><br><br>
        <span style="color:#1565C0">●</span> Docked ({zones.filter(pl.col('cluster_type')=='docked').shape[0]})<br>
        <span style="color:#E65100">●</span> Waiting ({zones.filter(pl.col('cluster_type')=='waiting').shape[0]})<br>
        <br><i>Taille = importance dans la variété</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = OUT_DIR / f"{location}_constituent_zones_map.html"
    m.save(str(out))
    print(f"[1/4] Carte zones constituantes → {out}")
    return out


# ── 2. Timeline gravity_score annotée ─────────────────────────────────────────

def plot_gravity_timeline(grav: pl.DataFrame, location: str) -> Path:
    dates   = grav["date"].to_list()
    gravity = grav["gravity_score"].to_numpy()

    # Normalisation pour lisibilité
    gravity_norm = gravity / gravity.max()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=gravity_norm,
        name="Gravity score (normalisé)",
        mode="lines",
        line=dict(color="#1565C0", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(21,101,192,0.1)",
    ))

    # Annotations événements
    for ev in EVENTS.get(location, []):
        if ev["date"] < dates[0] or ev["date"] > dates[-1]:
            continue
        fig.add_shape(
            type="line", x0=ev["date"], x1=ev["date"], y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dash", color=ev["color"], width=1.5),
        )
        fig.add_annotation(
            x=ev["date"], y=0.97, xref="x", yref="paper",
            text=ev["label"], showarrow=False,
            font=dict(size=10, color=ev["color"]),
            xanchor="left", bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_layout(
        title=dict(
            text=f"{location.upper()} — Gravity Score (manifold géospatial)",
            x=0.5, font_size=14,
        ),
        xaxis_title="Date",
        yaxis_title="Gravity score (normalisé)",
        template="plotly_white",
        height=380,
        margin=dict(t=60, b=50, l=60, r=20),
    )

    out = OUT_DIR / f"{location}_gravity_timeline.html"
    fig.write_html(str(out))
    print(f"[2/4] Timeline gravity score → {out}")
    return out


# ── 3. Scatter φ₁ vs φ₂ des zones ────────────────────────────────────────────

def plot_phi_scatter(zones: pl.DataFrame, location: str) -> Path:
    phi1  = zones["phi_1"].to_numpy()
    phi2  = zones["phi_2"].to_numpy()
    ctype = zones["cluster_type"].to_list()

    colors = ["#1565C0" if c == "docked" else "#E65100" for c in ctype]
    labels = ["Docked" if c == "docked" else "Waiting" for c in ctype]
    n_total = zones.shape[0]

    fig = go.Figure()

    for group_label, color in [("Docked", "#1565C0"), ("Waiting", "#E65100")]:
        mask = np.array([c == group_label for c in labels])
        fig.add_trace(go.Scatter(
            x=phi1[mask], y=phi2[mask],
            mode="markers",
            name=f"{group_label} ({mask.sum()})",
            marker=dict(color=color, size=5, opacity=0.6),
            text=[zones["zone_key"].to_list()[i] for i in range(n_total) if mask[i]],
            hovertemplate="<b>%{text}</b><br>φ₁: %{x:.4f}<br>φ₂: %{y:.4f}",
        ))

    fig.update_layout(
        title=dict(
            text=f"{location.upper()} — Zones dans l'espace propre (φ₁ × φ₂)",
            x=0.5, font_size=14,
        ),
        xaxis_title="φ₁ (1ᵉʳ vecteur propre LBO)",
        yaxis_title="φ₂ (2ᵉ vecteur propre LBO)",
        template="plotly_white",
        height=420,
        margin=dict(t=60, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )

    out = OUT_DIR / f"{location}_phi_scatter.html"
    fig.write_html(str(out))
    print(f"[3/4] Scatter φ₁ × φ₂ → {out}")
    return out


# ── 4. Double timeline : naïf vs manifold ────────────────────────────────────

def plot_double_timeline(grav: pl.DataFrame, location: str) -> Path:
    dates        = grav["date"].to_list()
    raw_cap      = grav["total_waiting_capacity"].to_numpy()
    gravity      = grav["gravity_score"].to_numpy()

    # Normalisation commune (même max pour comparer les formes)
    vmax = raw_cap.max()
    raw_norm  = raw_cap  / vmax
    grav_norm = gravity  / vmax

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Signal naïf : Σ capacité totale (tous navires stationnaires)",
            "Signal manifold : Gravity score (zones constituantes uniquement)",
        ],
        vertical_spacing=0.10,
    )

    fig.add_trace(go.Scatter(
        x=dates, y=raw_norm,
        name="Capacité brute",
        mode="lines",
        line=dict(color="#78909C", width=1.5),
        fill="tozeroy", fillcolor="rgba(120,144,156,0.15)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=grav_norm,
        name="Gravity score",
        mode="lines",
        line=dict(color="#1565C0", width=1.8),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.12)",
    ), row=2, col=1)

    # Annotations sur les deux sous-graphes
    for ev in EVENTS.get(location, []):
        if ev["date"] < dates[0] or ev["date"] > dates[-1]:
            continue
        for row_n in [1, 2]:
            fig.add_vline(
                x=ev["date"], line_dash="dash",
                line_color=ev["color"], line_width=1.2,
                row=row_n, col=1,
            )
        fig.add_annotation(
            x=ev["date"], y=1.02, yref="paper",
            text=ev["label"], showarrow=False,
            font=dict(size=10, color=ev["color"]),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(
            text=f"{location.upper()} — Signal naïf vs Gravity score (manifold géospatial)",
            x=0.5, font_size=14,
        ),
        template="plotly_white",
        height=520,
        margin=dict(t=80, b=50, l=60, r=20),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Capacité (normalisée)", row=1, col=1)
    fig.update_yaxes(title_text="Gravity (normalisé)",   row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    out = OUT_DIR / f"{location}_double_timeline.html"
    fig.write_html(str(out))
    print(f"[4/4] Double timeline naïf vs manifold → {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="4 visualisations géospatiales pour le rapport"
    )
    parser.add_argument("--location", default="la",
                        choices=["la", "houston"])
    args = parser.parse_args()
    loc = args.location

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zones_path = Path(f"data/features/{loc}_constituent_zones.parquet")
    grav_path  = Path(f"data/features/{loc}_gravity_daily.parquet")

    if not zones_path.exists():
        print(f"Fichier manquant : {zones_path}")
        print("Lancer d'abord : python src/manifold/manifold_pipeline.py --identify ...")
        return
    if not grav_path.exists():
        print(f"Fichier manquant : {grav_path}")
        print("Lancer d'abord : python src/manifold/manifold_pipeline.py --score ...")
        return

    zones = pl.read_parquet(zones_path)
    grav  = pl.read_parquet(grav_path).sort("date")

    print(f"\nLocation : {loc.upper()}")
    print(f"  Zones : {len(zones)} ({zones.filter(pl.col('is_constituent'))['is_constituent'].len()} constituantes)")
    print(f"  Jours : {len(grav)} ({grav['date'].min()} → {grav['date'].max()})")
    print()

    plot_constituent_map(zones, loc)
    plot_gravity_timeline(grav, loc)
    plot_phi_scatter(zones, loc)
    plot_double_timeline(grav, loc)

    print(f"\nTous les fichiers dans : {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
