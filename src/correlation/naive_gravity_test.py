"""
Reproduit le test naïf du collègue :
  Spearman / Pearson ( gravity_score[t - lag] , fwd_return[t + horizon] )
  lags    : 1, 3, 5, 7, 14, 21 jours
  horizons: 5, 10, 15, 30 jours

Deux ports :
  - LA    (2017–2022) → cible MATX
  - Houston (2017–2022) → cible XOM (proxy port pétrolier)
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data/features"
OUT_DIR  = ROOT / "outputs/financial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAGS     = [1, 3, 5, 7, 14, 21]
HORIZONS = [5, 10, 15, 30]


# ── Chargement ────────────────────────────────────────────────────────────────

def load_la_gravity() -> pd.DataFrame:
    df = pl.read_parquet(FEAT_DIR / "la_gravity_2017_2022.parquet").to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["gravity_score"]]


def load_houston_gravity() -> pd.DataFrame:
    frames = [
        pl.read_parquet(f).to_pandas()
        for f in sorted(glob.glob(str(FEAT_DIR / "houston_20*_gravity_daily.parquet")))
        if "2017" <= Path(f).name[8:12] <= "2022"
    ]
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["gravity_score"]].sort_index()


def load_price(ticker: str, start="2017-01-01", end="2023-01-01") -> pd.Series:
    # Try cached equity parquet first
    equity_path = ROOT / "data/financial/equity_proxies_wide.parquet"
    if equity_path.exists():
        cached = pl.read_parquet(equity_path).to_pandas()
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")
        if ticker in cached.columns:
            return cached[ticker].dropna()
    raw = yf.download(ticker, start=start, end=end,
                      progress=False, auto_adjust=True)["Close"].squeeze()
    raw.index = pd.to_datetime(raw.index)
    return raw.dropna()


def fwd_return(prices: pd.Series, h: int) -> pd.Series:
    return np.log(prices.shift(-h) / prices)


# ── Test naïf ────────────────────────────────────────────────────────────────

def naive_lag_test(gravity: pd.DataFrame, prices: pd.Series,
                   port: str, ticker: str) -> pd.DataFrame:
    rows = []
    for lag in LAGS:
        g_lagged = gravity["gravity_score"].shift(lag)
        for h in HORIZONS:
            fwd = fwd_return(prices, h)
            merged = pd.concat([g_lagged.rename("g"), fwd.rename("r")],
                               axis=1).dropna()
            n = len(merged)
            if n < 30:
                continue
            pr, pp = stats.pearsonr(merged["g"], merged["r"])
            sr, sp = stats.spearmanr(merged["g"], merged["r"])
            rows.append({
                "port":          port,
                "ticker":        ticker,
                "gravity_lag_j": lag,
                "horizon_j":     h,
                "pearson_r":     round(float(pr), 4),
                "pearson_p":     round(float(pp), 4),
                "spearman_r":    round(float(sr), 4),
                "spearman_p":    round(float(sp), 4),
                "n_obs":         n,
                "sig_5pct":      sp < 0.05,
            })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Chargement des données...")
    la_grav  = load_la_gravity()
    hou_grav = load_houston_gravity()

    matx = load_price("MATX")
    xom  = load_price("XOM")

    print(f"LA gravity  : {la_grav.index.min().date()} → {la_grav.index.max().date()}")
    print(f"Houston gravity : {hou_grav.index.min().date()} → {hou_grav.index.max().date()}")
    print(f"MATX prices : {matx.index.min().date()} → {matx.index.max().date()}")
    print(f"XOM prices  : {xom.index.min().date()} → {xom.index.max().date()}")

    res_la  = naive_lag_test(la_grav,  matx, port="LA",      ticker="MATX")
    res_hou = naive_lag_test(hou_grav, xom,  port="Houston", ticker="XOM")

    results = pd.concat([res_la, res_hou], ignore_index=True)
    results.to_csv(OUT_DIR / "naive_gravity_test.csv", index=False)

    for port in ["LA", "Houston"]:
        sub = results[results["port"] == port]
        ticker = sub["ticker"].iloc[0]
        print(f"\n{'='*75}")
        print(f"PORT : {port}  |  Cible : {ticker}  |  Feature : gravity_score brut")
        print(f"{'='*75}")
        print(f"{'lag':>5} {'H':>5}  {'pearson_r':>10} {'pearson_p':>10} "
              f"{'spearman_r':>11} {'spearman_p':>11} {'n_obs':>6} {'sig':>5}")
        print("─" * 75)
        for _, r in sub.iterrows():
            sig = "  *" if r["sig_5pct"] else ""
            print(f"{int(r['gravity_lag_j']):>5} {int(r['horizon_j']):>5}  "
                  f"{r['pearson_r']:>10.4f} {r['pearson_p']:>10.4f} "
                  f"{r['spearman_r']:>11.4f} {r['spearman_p']:>11.4f} "
                  f"{int(r['n_obs']):>6}{sig}")

    n_sig = results["sig_5pct"].sum()
    n_tot = len(results)
    print(f"\n→ {n_sig}/{n_tot} tests significatifs à 5%  "
          f"(attendu par hasard : {n_tot*0.05:.1f})")
    print(f"Résultats sauvegardés : {OUT_DIR / 'naive_gravity_test.csv'}")


if __name__ == "__main__":
    main()
