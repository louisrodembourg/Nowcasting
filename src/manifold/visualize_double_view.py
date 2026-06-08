"""
Figure composite 2 panneaux pour le rapport :
  - Gauche : jours dans l'espace (phi_1, phi_2), colores par mois,
             jours caracteristiques annotes
  - Droite : carte geospatiale des zones colorees par phi_1
             (docked = carre, waiting = triangle)

Usage (depuis Nowcasting/) :
    python src/manifold/visualize_double_view.py
    python src/manifold/visualize_double_view.py --year 2020 --out outputs/figures/la_manifold_2020.png
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

DEFAULT_YEAR     = 2020
DEFAULT_LOCATION = "la"
N_ANN_TOP        = 3   # days to annotate on the positive phi_1 side
N_ANN_BOT        = 2   # days to annotate on the negative phi_1 side

# Manual offsets per date to avoid label overlaps (key = YYYY-MM-DD)
ANNOTATION_OFFSETS: dict[str, tuple[float, float]] = {}


def _load_manifold(location: str, year: int) -> pl.DataFrame:
    """Load day-level manifold, filtered to the requested year."""
    path = Path(f"data/features/{location}_manifold.parquet")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pl.read_parquet(path)

    if df["date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("date").str.to_date())

    df = df.filter(pl.col("date").dt.year() == year)

    if len(df) == 0:
        available = (
            pl.read_parquet(path)
            .select(pl.col("date").dt.year().unique().sort())["date"]
            .to_list()
        )
        raise ValueError(
            f"No data for year {year} in {path}. Available years: {available}"
        )

    # Compute is_characteristic from phi_1 extremes if column is missing
    if "is_characteristic" not in df.columns:
        phi1 = df["phi_1"].to_numpy()
        threshold = np.percentile(np.abs(phi1), 95)
        df = df.with_columns(
            pl.Series("is_characteristic", (np.abs(phi1) >= threshold).tolist())
        )

    return df


def _load_zones(location: str) -> pl.DataFrame:
    path = Path(f"data/features/{location}_constituent_zones.parquet")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
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

    # ── Panel left: days in (phi_1, phi_2) ───────────────────────────────────
    ax1 = axes[0]

    phi1   = manifold["phi_1"].to_numpy()
    phi2   = manifold["phi_2"].to_numpy()
    months = [d.month for d in manifold["date"].to_list()]

    sc1 = ax1.scatter(phi1, phi2, c=months, cmap="hsv", s=20, alpha=0.65,
                      vmin=1, vmax=12, zorder=2)

    # Characteristic days to annotate (top phi_1 and bottom phi_1)
    char_df = manifold.filter(pl.col("is_characteristic"))
    top = char_df.sort("phi_1", descending=True).head(N_ANN_TOP)
    bot = char_df.sort("phi_1").head(N_ANN_BOT)
    to_ann = pl.concat([top, bot])

    for row in to_ann.iter_rows(named=True):
        key    = row["date"].isoformat()
        label  = row["date"].strftime("%d %b")
        dx, dy = ANNOTATION_OFFSETS.get(key, (0.015, 0.015))
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
    cbar1.set_label("Month", fontsize=9)
    cbar1.set_ticks(range(1, 13))
    cbar1.set_ticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=8,
    )

    ax1.axhline(0, color="#ccc", lw=0.6, ls="--")
    ax1.axvline(0, color="#ccc", lw=0.6, ls="--")
    ax1.grid(True, alpha=0.18)
    ax1.set_xlabel(r"$\varphi_1$", fontsize=12)
    ax1.set_ylabel(r"$\varphi_2$", fontsize=12)
    ax1.set_title(
        f"Temporal view — days in the manifold\n"
        f"{location.upper()} {year}",
        fontsize=10,
    )
    ax1.annotate(
        "o  characteristic days",
        xy=(0.03, 0.04), xycoords="axes fraction",
        fontsize=8, color="crimson",
    )

    # ── Panel right: geospatial view of zones ─────────────────────────────────
    ax2 = axes[1]

    z_phi1 = zones["phi_1"].to_numpy()
    z_lat  = zones["lat"].to_numpy()
    z_lon  = zones["lon"].to_numpy()
    z_type = zones["cluster_type"].to_list()

    vmax = np.percentile(np.abs(z_phi1), 97)
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    mask_d = np.array([t == "docked"  for t in z_type])
    mask_w = np.array([t == "waiting" for t in z_type])

    if mask_d.any():
        ax2.scatter(z_lon[mask_d], z_lat[mask_d], c=z_phi1[mask_d],
                    cmap="RdBu_r", norm=norm, s=12, marker="s",
                    alpha=0.85, zorder=3)

    sc2 = ax2.scatter(
        z_lon[mask_w] if mask_w.any() else z_lon,
        z_lat[mask_w] if mask_w.any() else z_lat,
        c=z_phi1[mask_w] if mask_w.any() else z_phi1,
        cmap="RdBu_r", norm=norm, s=18, marker="^",
        alpha=0.85, zorder=4,
    )

    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.72, pad=0.02)
    cbar2.set_label(r"$\varphi_1$ (zone)", fontsize=9)

    ax2.legend(
        handles=[
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
                       markersize=8, label="waiting"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
                       markersize=8, label="docked"),
        ],
        fontsize=8, loc="lower right",
    )

    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude",  fontsize=10)
    ax2.set_title(
        f"Geospatial view — zones in the manifold\n"
        f"{location.upper()}  (red = high phi_1, blue = low phi_1)",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.18)

    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifold double-view figure")
    parser.add_argument("--year",     type=int, default=DEFAULT_YEAR)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument(
        "--out", default=None,
        help="Output path (default: outputs/figures/<loc>_manifold_double_view_<year>.png)",
    )
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else Path(f"outputs/figures/{args.location}_manifold_double_view_{args.year}.png")
    )

    manifold = _load_manifold(args.location, args.year)
    zones    = _load_zones(args.location)
    plot_double_view(manifold, zones, args.year, args.location, out_path)


if __name__ == "__main__":
    main()
