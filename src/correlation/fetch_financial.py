"""
Fetch and persist financial proxy data for LA/Long Beach congestion correlation.

Sources:
- Equity proxies (yfinance): MATX (Matson, transpacific), DAC, CMRE, GSL (container lessors)
- Dry bulk ETF (yfinance): BDRY (Breakwave, BDI proxy, from 2018)
- PPI Water Transportation (FRED): PCU483111483111 — monthly, broadest macro signal
"""

import requests
import yfinance as yf
import polars as pl
import pandas as pd
from pathlib import Path
from io import StringIO

FINANCIAL_DIR = Path(__file__).parents[2] / "data" / "financial"
FINANCIAL_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_TICKERS = {
    "MATX": "Matson_transpacific",
    "DAC": "Danaos_container",
    "CMRE": "Costamare_container",
    "GSL": "GlobalShipLease",
    "BDRY": "Breakwave_DryBulk_ETF",
}

FRED_SERIES = {
    "PCU483111483111": "PPI_WaterTransport",
}

START = "2017-01-01"
END = "2024-12-31"


def fetch_equity(ticker: str, name: str) -> pl.DataFrame:
    df_pd = yf.download(ticker, start=START, end=END, progress=False)
    if df_pd.empty:
        print(f"  [WARN] {ticker}: no data")
        return pl.DataFrame()

    df_pd = df_pd[["Close"]].copy()
    df_pd.columns = ["close"]
    df_pd.index.name = "date"
    df_pd = df_pd.reset_index()
    df_pd["date"] = pd.to_datetime(df_pd["date"]).dt.strftime("%Y-%m-%d")

    df = pl.from_pandas(df_pd).with_columns(
        pl.lit(ticker).alias("ticker"),
        pl.lit(name).alias("series_name"),
    )
    print(f"  {ticker}: {len(df)} rows  {df['date'].min()} → {df['date'].max()}")
    return df


def fetch_fred(series_id: str, name: str) -> pl.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        print(f"  [WARN] FRED {series_id}: HTTP {r.status_code}")
        return pl.DataFrame()

    df_pd = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df_pd.columns = ["date", "close"]
    df_pd = df_pd[(df_pd["date"] >= START) & (df_pd["date"] <= END)].copy()
    df_pd["date"] = df_pd["date"].dt.strftime("%Y-%m-%d")
    df_pd["close"] = pd.to_numeric(df_pd["close"], errors="coerce")

    df = pl.from_pandas(df_pd).with_columns(
        pl.lit(series_id).alias("ticker"),
        pl.lit(name).alias("series_name"),
    )
    print(f"  {series_id}: {len(df)} rows  {df['date'].min()} → {df['date'].max()}")
    return df


def main():
    print("=== Fetching equity proxies ===")
    equity_frames = []
    for ticker, name in EQUITY_TICKERS.items():
        df = fetch_equity(ticker, name)
        if not df.is_empty():
            equity_frames.append(df)

    if equity_frames:
        equity_long = pl.concat(equity_frames)
        equity_long.write_parquet(FINANCIAL_DIR / "equity_proxies.parquet")
        print(f"Saved equity_proxies.parquet  shape={equity_long.shape}")

        # Wide format (one column per ticker, daily)
        equity_wide = equity_long.pivot(
            index="date", on="ticker", values="close"
        ).sort("date")
        equity_wide.write_parquet(FINANCIAL_DIR / "equity_proxies_wide.parquet")
        print(f"Saved equity_proxies_wide.parquet  shape={equity_wide.shape}")

    print("\n=== Fetching FRED macro series ===")
    fred_frames = []
    for series_id, name in FRED_SERIES.items():
        df = fetch_fred(series_id, name)
        if not df.is_empty():
            fred_frames.append(df)

    if fred_frames:
        fred_all = pl.concat(fred_frames)
        fred_all.write_parquet(FINANCIAL_DIR / "fred_macro.parquet")
        print(f"Saved fred_macro.parquet  shape={fred_all.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
