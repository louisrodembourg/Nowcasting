"""
Génère la figure composite 2 panneaux pour le rapport :
  - Gauche : jours dans l'espace (φ₁, φ₂), colorés par mois, jours caractéristiques annotés
  - Droite : carte géospatiale des zones colorées par φ₁ (docked = carré, waiting = triangle)

Usage (depuis Nowcasting/) :
    python src/manifold/visualize_double_view.py
    python src/manifold/visualize_double_view.py --year 2021 --out outputs/figures/la_manifold_2021.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

DEFAULT_YEAR      = 2020
DEFAULT_LOCATION  = "la"
N_ANN_TOP         = 3   # jours à annoter côté φ₁ positif
N_ANN_BOT         = 2   # jours à annoter côté φ₁ négatif

# Offsets manuels par date pour éviter les chevauchements (clé = YYYY-MM-DD)
ANNOTATION_OFFSETS: dict[str, tuple[float, float]] = {
    "2020-02-24": ( 0.020,  0.015),
    "2020-08-09": ( 0.018,  0.018),
    "2020-05-23": (-0.040,  0.022),
    "2020-02-22": (-0.050, -0.020),
    "2020-07-09": ( 0.018, -0.020),
}


def _load_manifold(location: str, year: int) -> pl.DataFrame:
    path = Path(f"data/features/{location}_{year}_manifold.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pl.read_parquet(path)


def _load_zones(location: str) -> pl.DataFrame:
    path = Path(f"data/features/{location}_constituent_zones.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pl.read_parquet(path)


def plot_double_view(
    manifold: pl.DataFrame,
    zones: pl.DataFrame,
    year: int,
    location: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("white")

    # ── Panel gauche : jours dans (φ₁, φ₂) ──────────────────────────────────
    ax1 = axes[0]

    phi1   = manifold["phi_1"].to_numpy()
    phi2   = manifold["phi_2"].to_numpy()
    months = [d.month for d in manifold["date"].to_list()]

    sc1 = ax1.scatter(phi1, phi2, c=months, cmap="hsv", s=20, alpha=0.65,
                      vmin=1, vmax=12, zorder=2)

    # Sélection des jours à annoter
    top = (manifold.sort("phi_1", descending=True)
           .filter(pl.col("is_characteristic"))
           .head(N_ANN_TOP))
    bot = (manifold.sort("phi_1")
           .filter(pl.col("is_characteristic"))
           .head(N_ANN_BOT))
    to_ann = pl.concat([top, bot])

    for row in to_ann.iter_rows(named=True):
        key      = row["date"].isoformat()
        label    = row["date"].strftime("%d %b")
        dx, dy   = ANNOTATION_OFFSETS.get(key, (0.015, 0.015))
        ax1.annotate(
            label,
            xy=(row["phi_1"], row["phi_2"]),
            xytext=(row["phi_1"] + dx, row["phi_2"] + dy),
            fontsize=8, color="#1a1a1a", fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
            zorder=5,
        )
        ax1.scatter([row["phi_1"]], [row["phi_2"]], s=70,
                    facecolors="none", edgecolors="crimson", lw=1.4, zorder=4)

    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.72, pad=0.02)
    cbar1.set_label("Mois", fontsize=9)
    cbar1.set_ticks(range(1, 13))
    cbar1.set_ticklabels(
        ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
         "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"],
        fontsize=8,
    )

    ax1.axhline(0, color="#ccc", lw=0.6, ls="--")
    ax1.axvline(0, color="#ccc", lw=0.6, ls="--")
    ax1.grid(True, alpha=0.18)
    ax1.set_xlabel(r"$\varphi_1$", fontsize=12)
    ax1.set_ylabel(r"$\varphi_2$", fontsize=12)
    ax1.set_title(
        f"Approche temporelle — jours dans la variété\n"
        f"{location.upper()} {year}  (espace de référence 2019)",
        fontsize=10,
    )
    ax1.annotate("○  jours caractéristiques", xy=(0.03, 0.04),
                 xycoords="axes fraction", fontsize=8, color="crimson")

    # ── Panel droit : carte géospatiale des zones ─────────────────────────────
    ax2 = axes[1]

    z_phi1 = zones["phi_1"].to_numpy()
    z_lat  = zones["lat"].to_numpy()
    z_lon  = zones["lon"].to_numpy()
    z_type = zones["cluster_type"].to_list()

    vmax = np.percentile(np.abs(z_phi1), 97)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    mask_d = np.array([t == "docked"  for t in z_type])
    mask_w = np.array([t == "waiting" for t in z_type])

    ax2.scatter(z_lon[mask_d], z_lat[mask_d], c=z_phi1[mask_d],
                cmap="RdBu_r", norm=norm, s=12, marker="s",
                alpha=0.85, zorder=3)
    sc2 = ax2.scatter(z_lon[mask_w], z_lat[mask_w], c=z_phi1[mask_w],
                      cmap="RdBu_r", norm=norm, s=18, marker="^",
                      alpha=0.85, zorder=4)

    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.72, pad=0.02)
    cbar2.set_label(r"$\varphi_1$ (zone)", fontsize=9)

    ax2.legend(handles=[
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
                   markersize=8, label="waiting"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
                   markersize=8, label="docked"),
    ], fontsize=8, loc="lower right")

    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude",  fontsize=10)
    ax2.set_title(
        f"Approche géospatiale — zones dans la variété\n"
        f"{location.upper()}/Long Beach  (rouge = φ₁ élevé, bleu = φ₁ faible)",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.18)

    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure composite manifold double vue")
    parser.add_argument("--year",     type=int, default=DEFAULT_YEAR)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--out",      default=None,
                        help="Chemin de sortie (défaut : outputs/figures/<loc>_manifold_double_view.png)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else \
               Path(f"outputs/figures/{args.location}_manifold_double_view.png")

    manifold = _load_manifold(args.location, args.year)
    zones    = _load_zones(args.location)
    plot_double_view(manifold, zones, args.year, args.location, out_path)


if __name__ == "__main__":
    main()
