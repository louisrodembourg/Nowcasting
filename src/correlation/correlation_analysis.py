"""
Analyse de corrélation : features AIS + rho_pred PINN → rendements MATX / BDRY / SBLK.

Pour chaque couple (feature, cible, horizon H, lag k) :
    Spearman( feature[t-k] , fwd_return[t+H] )

Produit :
  - outputs/financial/corr_lagscan.csv   : tableau complet
  - outputs/financial/corr_heatmap.png   : heatmap features × lags pour chaque cible/horizon
  - outputs/financial/corr_summary.csv   : top features par cible × horizon
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[2]
GRAVITY_PATH = ROOT / "data/features/la_gravity_2017_2022.parquet"
EQUITY_PATH  = ROOT / "data/financial/equity_proxies_wide.parquet"
TTC_DIR      = ROOT / "outputs/pinn_yearly"
OUT_DIR      = ROOT / "outputs/financial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS   = ["MATX", "BDRY", "SBLK"]
HORIZONS  = [5, 10, 21]
MAX_LAG   = 60
LAG_STEP  = 1          # granularité fine pour l'analyse


# ── Chargement ────────────────────────────────────────────────────────────────

def load_gravity() -> pd.DataFrame:
    df = pl.read_parquet(GRAVITY_PATH).with_columns(
        pl.col("date").str.to_date()
    ).sort("date").to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_prices(ticker: str) -> pd.Series:
    if ticker in ("MATX", "BDRY", "DAC", "CMRE", "GSL"):
        cached = pl.read_parquet(EQUITY_PATH).with_columns(
            pl.col("date").str.to_date()
        ).to_pandas()
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")
        if ticker in cached.columns:
            return cached[ticker].dropna()
    raw = yf.download(ticker, start="2017-01-01", end="2023-01-01",
                      progress=False, auto_adjust=True)["Close"].squeeze()
    raw.index = pd.to_datetime(raw.index)
    return raw.dropna()


def load_pinn_rho() -> pd.Series:
    frames = []
    for y in [2019, 2020, 2021, 2022]:
        p = TTC_DIR / f"la_{y}_time_to_clear.parquet"
        if p.exists():
            df = pl.read_parquet(p).to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df.set_index("date")["rho_pred"])
    return pd.concat(frames).sort_index() if frames else pd.Series(dtype=float)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(gravity: pd.DataFrame, pinn_rho: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=gravity.index)
    g = np.log1p(gravity["gravity_score"])

    f["log_gravity"]      = g
    f["gravity_raw"]      = gravity["gravity_score"]
    f["gravity_ma7"]      = g.rolling(7,  min_periods=3).mean()
    f["gravity_ma21"]     = g.rolling(21, min_periods=7).mean()

    roll60_mean = g.rolling(60, min_periods=20).mean()
    roll60_std  = g.rolling(60, min_periods=20).std().replace(0, np.nan)
    f["gravity_zscore60"] = (g - roll60_mean) / roll60_std

    f["gravity_delta5"]   = g.diff(5)
    f["gravity_delta21"]  = g.diff(21)
    f["log_waiting"]      = np.log1p(gravity["waiting_vessels"])
    f["log_capacity"]     = np.log1p(gravity["total_capacity"])
    f["congestion_flag"]  = (f["gravity_zscore60"] > 1.5).astype(float)
    f["pinn_rho"]         = pinn_rho.reindex(f.index)

    return f.dropna(subset=["gravity_zscore60"])


def forward_return(prices: pd.Series, h: int) -> pd.Series:
    return np.log(prices.shift(-h) / prices)


# ── Lag scan ─────────────────────────────────────────────────────────────────

def lag_scan(features: pd.DataFrame, prices: pd.Series,
             feat_cols: list, horizon: int, max_lag: int) -> pd.DataFrame:
    fwd = forward_return(prices, horizon).rename("fwd")
    rows = []
    for feat in feat_cols:
        series = features[feat]
        for lag in range(0, max_lag + 1, LAG_STEP):
            merged = pd.concat([series.shift(lag), fwd], axis=1).dropna()
            if len(merged) < 30:
                continue
            r, p = stats.spearmanr(merged.iloc[:, 0], merged.iloc[:, 1])
            rows.append({
                "feature": feat, "lag": lag,
                "spearman_r": round(float(r), 4),
                "p_value":    round(float(p), 4),
                "n":          len(merged),
            })
    return pd.DataFrame(rows)


# ── Heatmap ──────────────────────────────────────────────────────────────────

def plot_heatmap(scan: pd.DataFrame, ticker: str, horizon: int, out_path: Path):
    pivot = scan.pivot(index="feature", columns="lag", values="spearman_r")
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-0.25, vmax=0.25)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(f"Corrélation lag-scan — {ticker} fwd_return(H={horizon}j)\n"
                 f"Spearman(feature[t−lag], return[t+{horizon}])",
                 fontsize=11)
    ax.set_xlabel("Lag (jours)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Heatmap → {out_path.name}")


# ── Analyse épisode PINN ─────────────────────────────────────────────────────

def pinn_episode_corr(features: pd.DataFrame, prices: pd.Series,
                      ticker: str, horizons: list) -> pd.DataFrame:
    """Corrélation rho_pred vs forward return sur les jours d'épisode uniquement."""
    ep_mask = features["pinn_rho"].notna()
    rows = []
    for h in horizons:
        fwd = forward_return(prices, h)
        merged = pd.concat([
            features.loc[ep_mask, "pinn_rho"],
            fwd.rename("fwd")
        ], axis=1).dropna()
        if len(merged) < 10:
            continue
        r, p = stats.spearmanr(merged["pinn_rho"], merged["fwd"])
        rows.append({
            "ticker": ticker, "horizon": h,
            "n_episode": len(merged),
            "spearman_r": round(float(r), 4),
            "p_value":    round(float(p), 4),
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Chargement des données...")
    gravity  = load_gravity()
    pinn_rho = load_pinn_rho()
    features = build_features(gravity, pinn_rho)

    base_feats = [
        "log_gravity", "gravity_raw", "gravity_ma7", "gravity_ma21",
        "gravity_zscore60", "gravity_delta5", "gravity_delta21",
        "log_waiting", "log_capacity", "congestion_flag",
    ]
    pinn_feats = ["pinn_rho"]

    all_scans  = []
    pinn_corrs = []
    summaries  = []

    for ticker in TARGETS:
        print(f"\n{'═'*60} {ticker}")
        prices = load_prices(ticker)
        if prices.empty:
            print(f"  [!] Pas de données pour {ticker}"); continue

        for h in HORIZONS:
            print(f"  H={h}j — lag scan 0..{MAX_LAG}d...")

            # ── Scan features AIS ─────────────────────────────────────────
            scan = lag_scan(features, prices, base_feats, h, MAX_LAG)
            scan["ticker"]  = ticker
            scan["horizon"] = h
            all_scans.append(scan)

            # Heatmap
            plot_heatmap(scan, ticker, h,
                         OUT_DIR / f"corr_heatmap_{ticker}_h{h}.png")

            # Top 3 features au lag optimal
            top = (scan.assign(abs_r=scan["spearman_r"].abs())
                       .sort_values("abs_r", ascending=False)
                       .head(5))
            for _, row in top.iterrows():
                summaries.append({
                    "ticker": ticker, "horizon": h,
                    "feature": row["feature"], "lag": row["lag"],
                    "spearman_r": row["spearman_r"],
                    "p_value": row["p_value"],
                })

            # Top au lag 0 (signal instantané / nowcast)
            at0 = scan[scan["lag"] == 0].sort_values("spearman_r", key=abs, ascending=False)
            print(f"  Top features (lag=0) :")
            for _, r in at0.head(5).iterrows():
                sig = "***" if r["p_value"]<0.001 else ("**" if r["p_value"]<0.01
                      else ("*" if r["p_value"]<0.05 else ""))
                print(f"    {r['feature']:<22} ρ={r['spearman_r']:+.3f} p={r['p_value']:.3f}{sig}")

            # Lag optimal global
            best = scan.loc[scan["spearman_r"].abs().idxmax()]
            print(f"  Meilleur lag toutes features : {best['feature']} lag={best['lag']}j "
                  f"ρ={best['spearman_r']:+.3f} (p={best['p_value']:.3f})")

        # ── Corrélation rho_pred PINN en épisode ─────────────────────────
        pinn_df = pinn_episode_corr(features, prices, ticker, HORIZONS)
        if not pinn_df.empty:
            pinn_corrs.append(pinn_df)
            print(f"\n  Corrélation rho_pred PINN (jours d'épisode uniquement) :")
            print(pinn_df[["horizon","n_episode","spearman_r","p_value","significant"]].to_string(index=False))

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    df_scan    = pd.concat(all_scans, ignore_index=True)
    df_summary = pd.DataFrame(summaries)
    df_scan.to_csv(OUT_DIR / "corr_lagscan.csv", index=False)
    df_summary.to_csv(OUT_DIR / "corr_summary.csv", index=False)
    if pinn_corrs:
        pd.concat(pinn_corrs).to_csv(OUT_DIR / "corr_pinn_episode.csv", index=False)

    # ── Tableau récapitulatif final ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SYNTHÈSE — meilleure feature par (cible × horizon)")
    print(f"{'='*80}")
    print(f"{'Cible':<6} {'H':>4}  {'Feature':<22} {'Lag':>5} {'ρ':>8} {'p':>7}")
    print("─" * 60)
    for _, r in df_summary.drop_duplicates(["ticker","horizon"]).sort_values(
            ["ticker","horizon"]).iterrows():
        sig = "***" if r["p_value"]<0.001 else ("**" if r["p_value"]<0.01
              else ("*" if r["p_value"]<0.05 else "ns"))
        print(f"{r['ticker']:<6} {int(r['horizon']):>4}  {r['feature']:<22} "
              f"{int(r['lag']):>5} {r['spearman_r']:>+8.4f} {sig:>7}")
    print(f"{'='*80}")
    print(f"\nFichiers sauvegardés dans {OUT_DIR}")


if __name__ == "__main__":
    main()
