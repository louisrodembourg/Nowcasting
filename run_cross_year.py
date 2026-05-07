"""
Cross-year manifold projection — 2019 reference → 2020-2024.

Projette les données 2020-2024 dans l'eigenspace du manifold 2019 via
l'extension de Nyström. Tous les scores sont comparables dans le même
espace de référence : "déviation par rapport au comportement normal 2019".

Usage:
    python run_cross_year.py
    python run_cross_year.py --years 2020 2021 2022
"""
import argparse
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

from src.manifold.lbo import normalize_features, run_lbo
from src.manifold.nystrom import load_reference, project_onto_manifold

log = logging.getLogger(__name__)

FIGS_DIR = Path("outputs/figures")
DATA_DIR = Path("data/features")

DEFAULT_LOCATION = "la"
DEFAULT_REF_YEAR = 2019
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

REF_EVENTS = {
    ("la", 2019): ("2019-06-01", "2019-09-30"),
    ("houston", 2017): ("2017-08-17", "2017-09-10"),
}

KEY_EVENTS_LA = [
    (date(2019,  5, 10), "Tarifs\nUS 25%",    "#C62828"),
    (date(2020,  3, 15), "COVID\nshutdown",    "#1565C0"),
    (date(2021,  9,  1), "Début\nbacklog",     "#E65100"),
    (date(2022,  1,  6), "Pic\nbacklog",       "#6A1B9A"),
    (date(2023, 11, 19), "Houthis\ndétournt.", "#2E7D32"),
]


def get_key_events(location: str) -> list[tuple[date, str, str]]:
    if location == "la":
        return KEY_EVENTS_LA
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def compute_deviation(phi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Déviation L1 normalisée dans l'eigenspace de référence."""
    z = np.abs((phi - mu) / sigma)
    return z.mean(axis=1)


def smooth_7d(arr: np.ndarray) -> np.ndarray:
    """Moyenne mobile 7 jours — edge-padding pour éviter les creux aux frontières."""
    kernel = np.ones(7) / 7
    padded = np.pad(arr, 3, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# Construction de la référence 2019
# ─────────────────────────────────────────────────────────────────────────────

def build_reference(
    ref_path: Path,
    location: str,
    ref_year: int,
    ref_event: tuple[str, str] | None,
) -> dict:
    """
    Charge la référence 2019 et calcule μ/σ de la baseline
    (tous les jours hors fenêtre trade war).
    """
    ref = load_reference(ref_path)

    df_ref = pl.read_parquet(DATA_DIR / f"{location}_{ref_year}_manifold.parquet").sort("date")
    dates_ref = df_ref["date"].to_list()

    if ref_event is None:
        baseline_mask = np.ones(len(dates_ref), dtype=bool)
    else:
        e_start = date.fromisoformat(ref_event[0])
        e_end = date.fromisoformat(ref_event[1])
        baseline_mask = np.array([not (e_start <= d <= e_end) for d in dates_ref])

    phi_cols = sorted([c for c in df_ref.columns if c.startswith("phi_")])
    phi_ref = df_ref.select(phi_cols).to_numpy()

    mu = phi_ref[baseline_mask].mean(axis=0)
    phi_sigma = phi_ref[baseline_mask].std(axis=0)
    phi_sigma[phi_sigma == 0] = 1.0

    cap_ref = df_ref["blocked_capacity"].to_numpy().astype(float)
    baseline_cap = float(np.median(cap_ref[baseline_mask & (cap_ref > 0)]))
    if baseline_cap == 0:
        baseline_cap = 1.0

    return {**ref, "mu": mu, "phi_sigma": phi_sigma, "baseline_cap": baseline_cap}


# ─────────────────────────────────────────────────────────────────────────────
# Projection d'une année
# ─────────────────────────────────────────────────────────────────────────────

def project_year(year: int, ref: dict, location: str) -> pl.DataFrame:
    """
    Projette les features d'une année dans l'eigenspace 2019 et calcule
    le gravity score cross-year.
    """
    feat_path = DATA_DIR / f"{location}_{year}_daily_features.parquet"
    if not feat_path.exists():
        log.warning("Features manquantes pour %d — ignoré", year)
        return pl.DataFrame()

    df = pl.read_parquet(feat_path).sort("date")
    X_new = normalize_features(df)

    phi_new = project_onto_manifold(
        X_new,
        X_train     = ref["X_train"],
        eigenvalues = ref["eigenvalues"],
        eigenvectors= ref["eigenvectors"],
        sigma       = ref["sigma"],
    )

    deviation = compute_deviation(phi_new, ref["mu"], ref["phi_sigma"])

    cap        = df["blocked_capacity"].to_numpy().astype(float)
    cap_weight = cap / ref["baseline_cap"]
    raw_score  = deviation * cap_weight

    dev_smooth  = smooth_7d(deviation)
    raw_smooth  = smooth_7d(raw_score)

    return df.select(["date", "blocked_capacity", "utilization_rate_rho"]).with_columns([
        pl.Series("phi_deviation",      deviation.tolist(),   dtype=pl.Float64),
        pl.Series("gravity_raw",        raw_score.tolist(),   dtype=pl.Float64),
        pl.Series("deviation_smooth",   dev_smooth.tolist(),  dtype=pl.Float64),
        pl.Series("gravity_smooth",     raw_smooth.tolist(),  dtype=pl.Float64),
        pl.lit(year).alias("year"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

YEAR_COLORS = {
    2019: "#1565C0",
    2020: "#E53935",
    2021: "#F57F17",
    2022: "#6A1B9A",
    2023: "#2E7D32",
    2024: "#00838F",
}


def build_years_tag(years: list[int]) -> str:
    years_sorted = sorted(set(years))
    if not years_sorted:
        return "none"
    if len(years_sorted) == (years_sorted[-1] - years_sorted[0] + 1):
        return f"{years_sorted[0]}-{years_sorted[-1]}"
    return "_".join(str(y) for y in years_sorted)


def plot_timeline(
    all_data: dict[int, pl.DataFrame],
    location: str,
    ref_year: int,
    years_tag: str,
    key_events: list[tuple[date, str, str]] | None,
    show_events: bool,
) -> Path:
    """
    Série temporelle complète 2019-2024 avec le gravity score cross-year lissé.
    Tous les scores sont dans le même espace de référence 2019.
    """
    fig, ax = plt.subplots(figsize=(20, 6))

    if not all_data:
        raise ValueError("Aucune donnée à tracer")

    plot_dates = [d for df in all_data.values() for d in df["date"].to_list()]
    min_date = min(plot_dates)
    max_date = max(plot_dates)

    for year, df in sorted(all_data.items()):
        if df.is_empty():
            continue
        dates = df["date"].to_list()
        score = df["gravity_smooth"].to_numpy()
        color = YEAR_COLORS.get(year, "gray")
        ax.fill_between(dates, score, alpha=0.15, color=color)
        ax.plot(dates, score, color=color, linewidth=1.2, label=str(year))

    if show_events and key_events:
        for ev_date, label, color in key_events:
            if not (min_date <= ev_date <= max_date):
                continue
            ax.axvline(ev_date, color=color, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(
                ev_date,
                ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.5,
                label,
                fontsize=6.5,
                color=color,
                ha="center",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Gravity score (espace manifold 2019)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_xlim(min_date, max_date)
    ax.set_title(
        f"{location.upper()} — Gravity Score Cross-Year (référence manifold)\n"
        "Scores comparables inter-années · lissage 7 jours · À comparer avec FBX01/WCI",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / f"{location}_ref{ref_year}_proj{years_tag}_timeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Timeline → %s", out)
    return out


def plot_seasonal(
    all_data: dict[int, pl.DataFrame],
    location: str,
    ref_year: int,
    years_tag: str,
) -> Path:
    """
    Overlay saisonnier : toutes les années sur le même axe Jan-Déc.
    Révèle si les pics de congestion sont récurrents aux mêmes saisons.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for year, df in sorted(all_data.items()):
        if df.is_empty():
            continue
        dates  = df["date"].to_list()
        score  = df["deviation_smooth"].to_numpy()
        color  = YEAR_COLORS.get(year, "gray")

        # Convertit les dates en "jour de l'année" pour l'axe commun
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        ax.plot(day_of_year, score, color=color, linewidth=1.4,
                alpha=0.85, label=str(year))

    # Axe mois
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                    "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=9)
    max_day = 366 if any(year % 4 == 0 for year in all_data.keys()) else 365
    ax.set_xlim(1, max_day)

    ax.set_ylabel("Déviation manifold (normalisée 2019)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"{location.upper()} — Overlay Saisonnier (toutes années dans l'espace manifold de référence)\n"
        "Chaque ligne = une année · Les pics qui se superposent = congestion saisonnière récurrente",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGS_DIR / f"{location}_ref{ref_year}_proj{years_tag}_seasonal.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saisonnier → %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-year manifold projection")
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--ref-year", type=int, default=DEFAULT_REF_YEAR)
    parser.add_argument("--years", type=int, nargs="*",
                        default=DEFAULT_YEARS)
    parser.add_argument("--ref-event-start", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--ref-event-end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    loc = args.location
    ref_year = args.ref_year
    years = list(dict.fromkeys(args.years))
    years_tag = build_years_tag(years)

    default_ref_event = REF_EVENTS.get((loc, ref_year))
    if args.ref_event_start and args.ref_event_end:
        ref_event = (args.ref_event_start, args.ref_event_end)
    elif args.ref_event_start or args.ref_event_end:
        raise ValueError("--ref-event-start et --ref-event-end doivent être donnés ensemble")
    else:
        ref_event = default_ref_event

    ref_path = DATA_DIR / f"{loc}_{ref_year}_manifold_ref.npz"

    # Génère la référence 2019 si nécessaire
    if not ref_path.exists():
        log.info("Génération du manifold de référence 2019...")
        run_lbo(
            features_path = DATA_DIR / f"{loc}_{ref_year}_daily_features.parquet",
            output_path   = DATA_DIR / f"{loc}_{ref_year}_manifold.parquet",
            save_reference= True,
        )

    log.info("Chargement référence 2019 depuis %s", ref_path)
    ref = build_reference(ref_path, location=loc, ref_year=ref_year, ref_event=ref_event)
    log.info("Référence : kernel_sigma=%.4f | baseline_cap=%.0f",
             ref["sigma"], ref["baseline_cap"])

    # Projection de chaque année
    all_data: dict[int, pl.DataFrame] = {}
    for year in years:
        log.info("--- Projection %d ---", year)
        df = project_year(year, ref, location=loc)
        if not df.is_empty():
            all_data[year] = df
            out = DATA_DIR / f"{loc}_{year}_cross_year.parquet"
            df.write_parquet(out)
            log.info("Sauvegardé → %s", out)

            top5 = df.sort("gravity_smooth", descending=True).head(5)
            print(f"\n  Top 5 [{year}]:")
            for row in top5.iter_rows(named=True):
                print(f"    {row['date']}  gravity={row['gravity_smooth']:.4f}"
                      f"  déviation={row['phi_deviation']:.3f}")

    # Figures
    log.info("--- Génération des figures ---")
    key_events = get_key_events(loc)
    plot_timeline(
        all_data,
        location=loc,
        ref_year=ref_year,
        years_tag=years_tag,
        key_events=key_events,
        show_events=not args.no_events,
    )
    plot_seasonal(
        all_data,
        location=loc,
        ref_year=ref_year,
        years_tag=years_tag,
    )

    print(f"\nFigures → {FIGS_DIR}/{loc}_ref{ref_year}_proj{years_tag}_timeline.png")
    print(f"         {FIGS_DIR}/{loc}_ref{ref_year}_proj{years_tag}_seasonal.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
