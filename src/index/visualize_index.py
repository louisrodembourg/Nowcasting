"""
Visualisation de la corrélation entre le gravity score (manifold pipeline)
et des indices financiers du fret (BDI, FBX, WCI).

Attendu dans data/parquet/market/ :
  - CSV  : colonnes 'Date' (DD/MM/YYYY), 'Dernier' (valeur, format FR)
  - Parquet : colonnes 'date', 'value' (ou 'close')
  Le nom du fichier doit commencer par le ticker : bdi_*, fbx_*, wci_*, etc.

  Parametres :
    --location : la | houston | suez  (correspond au gravity score à utiliser)
    --index    : ticker de l'indice à charger (ex: bdi, fbx, wci)
    --lag      : décalage en jours (gravity prédit l'indice à J+lag)
    
  
Usage :
    python src/index/visualize_index.py --location la
    python src/index/visualize_index.py --location houston --index bdi --lag 5
    python src/index/visualize_index.py --location la --index fbx --lag 0 7 14
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

log = logging.getLogger(__name__)

MARKET_DIR  = Path("data/parquet/market")
GRAVITY_DIR = Path("data/features")
OUT_DIR     = Path("outputs/figures")


# ============================================================================
# Data loading
# ============================================================================


def _parse_fr_float(s: str) -> float:
    """Convert French-locale number string '1.234,56' ->1234.56."""
    return float(s.replace(".", "").replace(",", "."))


def load_gravity(location: str) -> pl.DataFrame:
    path = GRAVITY_DIR / f"{location}_gravity_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Gravity score introuvable : {path}\n"
            f"Lance d'abord : python src/manifold/manifold_pipeline.py --both "
            f"--location {location} ..."
        )
    df = (
        pl.read_parquet(path)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
    )
    log.info("Gravity %s : %d jours (%s ->%s)", location, len(df),
             df["date"].min(), df["date"].max())
    return df


def load_index(ticker: str) -> pl.DataFrame:
    """
    Charge un fichier d'indice depuis MARKET_DIR.
    Cherche d'abord un CSV puis un Parquet dont le nom commence par ticker.
    Retourne un DataFrame avec colonnes ['date', 'value'].
    """
    candidates = sorted(MARKET_DIR.glob(f"{ticker.lower()}_*"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier pour l'indice '{ticker}' dans {MARKET_DIR}/\n"
            f"Fichiers présents : {[f.name for f in MARKET_DIR.iterdir()]}"
        )
    path = candidates[0]
    log.info("Chargement indice : %s", path)

    if path.suffix == ".csv":
        raw = pl.read_csv(path)
        # Detect column names (French investing.com export)
        date_col  = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
        value_col = next(
            (c for c in raw.columns if c.strip() in ("Dernier", "Close", "close", "Value")),
            raw.columns[1],
        )
        df = raw.select([
            pl.col(date_col).alias("date_raw"),
            pl.col(value_col).alias("value_raw"),
        ])
        # Parse French date DD/MM/YYYY
        dates = [
            pl.lit(d).str.strptime(pl.Date, "%d/%m/%Y").to_physical()
            for d in df["date_raw"].to_list()
        ]
        parsed_dates = []
        values = []
        from datetime import date as _date
        import datetime
        for d_str, v_str in zip(df["date_raw"].to_list(), df["value_raw"].to_list()):
            try:
                day, month, year = d_str.split("/")
                parsed_dates.append(_date(int(year), int(month), int(day)))
                values.append(_parse_fr_float(str(v_str)))
            except Exception:
                continue
        result = pl.DataFrame({
            "date":  pl.Series(parsed_dates, dtype=pl.Date),
            "value": pl.Series(values,       dtype=pl.Float64),
        })

    elif path.suffix == ".parquet":
        raw = pl.read_parquet(path)
        date_col  = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
        value_col = next(
            (c for c in raw.columns if c.lower() in ("value", "close", "price", "dernier")),
            raw.columns[1],
        )
        result = raw.select([
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(value_col).cast(pl.Float64).alias("value"),
        ])
    else:
        raise ValueError(f"Format non supporté : {path.suffix}")

    result = result.sort("date").drop_nulls()
    log.info("Indice %s : %d points (%s ->%s)", ticker.upper(), len(result),
             result["date"].min(), result["date"].max())
    return result


def align(gravity: pl.DataFrame, index: pl.DataFrame, lag_days: int = 0) -> pl.DataFrame:
    """
    Joint les deux séries sur la date.
    lag_days > 0 : décale l'indice vers l'avenir (gravity prédit l'indice à J+lag).
    """
    idx = index.with_columns(
        (pl.col("date") - pl.duration(days=lag_days)).alias("date")
    ) if lag_days != 0 else index

    merged = gravity.join(idx.rename({"value": "index_value"}), on="date", how="inner")
    return merged.sort("date")


# ============================================================================
# Plot
# ============================================================================


def plot_correlation(
    merged: pl.DataFrame,
    location: str,
    ticker: str,
    lag: int,
    out_path: Path,
) -> None:
    if len(merged) < 3:
        print(f"  [!] Seulement {len(merged)} points communs — graphique vide.")
        print(f"      Gravity couvre : voir la_gravity_daily.parquet")
        print(f"      Indice couvre  : voir fichier {ticker}")
        return

    dates   = merged["date"].to_list()
    gravity = merged["gravity_score"].to_numpy()
    index   = merged["index_value"].to_numpy()

    # Normalise pour affichage dual-axis
    g_norm = (gravity - gravity.mean()) / (gravity.std() or 1)
    i_norm = (index   - index.mean())   / (index.std()   or 1)

    # Corrélations
    r_pearson,  p_pearson  = stats.pearsonr(gravity, index)
    r_spearman, p_spearman = stats.spearmanr(gravity, index)

    lag_label = f"  (lag {lag}j)" if lag else ""
    title = f"{location.upper()} — Gravity Score vs {ticker.upper()}{lag_label}"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Séries temporelles normalisées",
            f"Scatter — r={r_pearson:.3f} (Pearson)",
            "Gravity Score brut",
            f"{ticker.upper()} brut",
        ),
        specs=[[{"colspan": 2}, None], [{}, {}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # --- Dual time series (normalisées) ---
    fig.add_trace(go.Scatter(
        x=dates, y=g_norm, name="Gravity (norm.)",
        mode="lines", line=dict(color="#1565C0", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=i_norm, name=f"{ticker.upper()} (norm.)",
        mode="lines", line=dict(color="#E65100", width=2, dash="dot"),
    ), row=1, col=1)

    # --- Scatter ---
    fig.add_trace(go.Scatter(
        x=gravity, y=index,
        mode="markers",
        marker=dict(color="#1565C0", size=5, opacity=0.6),
        name="Points",
        showlegend=False,
    ), row=2, col=1)
    # Droite de régression
    m, b = np.polyfit(gravity, index, 1)
    x_line = np.linspace(gravity.min(), gravity.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", line=dict(color="#E65100", width=2),
        name="Régression", showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text="Gravity Score", row=2, col=1)
    fig.update_yaxes(title_text=ticker.upper(), row=2, col=1)

    # --- Raw gravity ---
    fig.add_trace(go.Scatter(
        x=dates, y=gravity, name="Gravity brut",
        mode="lines+markers", marker=dict(size=3),
        line=dict(color="#1565C0", width=1.5),
        showlegend=False,
    ), row=2, col=2)

    # --- Raw index ---
    fig.add_trace(go.Scatter(
        x=dates, y=index, name=ticker.upper(),
        mode="lines", line=dict(color="#E65100", width=1.5),
        showlegend=False,
    ), row=2, col=2)

    # --- Annotation corrélation ---
    corr_text = (
        f"<b>Pearson  r = {r_pearson:.3f}</b>  (p={p_pearson:.3f})<br>"
        f"Spearman r = {r_spearman:.3f}  (p={p_spearman:.3f})<br>"
        f"N = {len(merged)} jours communs"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=1.04,
        text=corr_text, showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#ccc", borderwidth=1,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        template="plotly_white",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0.5, xanchor="center"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"Saved : {out_path.resolve()}")
    print(f"  Pearson  r = {r_pearson:.3f}  (p={p_pearson:.4f})")
    print(f"  Spearman r = {r_spearman:.3f}  (p={p_spearman:.4f})")
    print(f"  N = {len(merged)} jours communs")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrélation gravity score ↔ indice financier")
    parser.add_argument("--location", default="la", choices=["la", "houston", "suez"],
                        help="Localisation du gravity score (défaut: la)")
    parser.add_argument("--index", default="bdi",
                        help="Ticker de l'indice (ex: bdi, fbx, wci) — doit correspondre "
                             "au nom de fichier dans data/parquet/market/")
    parser.add_argument("--lag", type=int, nargs="+", default=[0],
                        metavar="N",
                        help="Décalage(s) en jours : gravity prédit l'indice à J+lag. "
                             "Plusieurs valeurs génèrent une figure par lag.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gravity = load_gravity(args.location)
    index   = load_index(args.index)

    # Résumé de couverture
    g_min, g_max = gravity["date"].min(), gravity["date"].max()
    i_min, i_max = index["date"].min(),   index["date"].max()
    print(f"\nGravity  : {g_min} ->{g_max}  ({len(gravity)} jours)")
    print(f"Indice   : {i_min} ->{i_max}  ({len(index)} points)")

    overlap_start = max(g_min, i_min)
    overlap_end   = min(g_max, i_max)
    if overlap_start > overlap_end:
        print(f"\n[ATTENTION] Aucun overlap temporel entre gravity ({g_min}->{g_max}) "
              f"et indice ({i_min}->{i_max}).")
        print("  ->Ajoute des données de marché couvrant la même période dans data/parquet/market/")
        return

    print(f"Overlap  : {overlap_start} ->{overlap_end}")

    for lag in args.lag:
        merged = align(gravity, index, lag_days=lag)
        lag_str = f"_lag{lag}" if lag else ""
        out_path = OUT_DIR / f"{args.location}_{args.index}_correlation{lag_str}.html"
        print(f"\n--- Lag {lag}j ---")
        plot_correlation(merged, args.location, args.index, lag, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
