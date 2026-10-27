"""
Freight index scraper: BDI, FBX, WCI, SCFI.

Available strategies:
  1 (fred)      - FRED (Federal Reserve) through a free API key
                  BDI available under the code DBDI
                  Free key: https://fred.stlouisfed.org/docs/api/api_key.html
  2 (yfinance)  - Yahoo Finance via yfinance (pip install yfinance)
                  Tickers: BDI=F, ^BDI depending on Yahoo availability
  3 (barchart)  - HTML scraping, ~2 years free without an account
  4 (selenium)  - TradingEconomics via Selenium (Chrome + chromedriver required)

Output: data/parquet/market/<ticker>_<start>_<end>.csv
        columns: Date (YYYY-MM-DD), Close

Usage:
    python src/index/scrap_index.py --ticker bdi --start 2017-01-01 --end 2020-12-31
    python src/index/scrap_index.py --ticker bdi --start 2017-01-01 --strategy 1 --fred-key YOUR_KEY
    python src/index/scrap_index.py --ticker bdi --strategy 2
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

OUT_DIR = Path("data/parquet/market")

# Ticker -> FRED code mapping
FRED_CODES = {
    "bdi":  "DBDI",       # Baltic Dry Index (daily)
    "wci":  "DWCI",       # World Container Index (if available)
}

# Ticker -> Yahoo Finance symbol mapping
YAHOO_TICKERS = {
    "bdi":  "^BDI",
}

# Ticker -> Barchart URL mapping
BARCHART_URLS = {
    "bdi": "https://www.barchart.com/stocks/quotes/BDI.BI/price-history/historical",
    "fbx": "https://www.barchart.com/stocks/quotes/FBX/price-history/historical",
}

# Ticker -> TradingEconomics URL mapping
TE_URLS = {
    "bdi": "https://tradingeconomics.com/commodity/baltic",
    "fbx": "https://tradingeconomics.com/commodity/freightos-baltic-index",
    "wci": "https://tradingeconomics.com/commodity/drewry-world-container-index",
}


# ============================================================================
# Strategy 1 — FRED
# ============================================================================


def scrape_fred(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Download an index from FRED through the official API.
    Free key: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    fred_code = FRED_CODES.get(ticker.lower())
    if fred_code is None:
        raise ValueError(
            f"Ticker '{ticker}' not configured for FRED.\n"
            f"Available codes: {FRED_CODES}\n"
            f"Look up the code on https://fred.stlouisfed.org"
        )

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":        fred_code,
        "observation_start": start,
        "observation_end":   end,
        "api_key":           api_key,
        "file_type":         "json",
    }

    log.info("[FRED] Downloading %s (%s)...", ticker.upper(), fred_code)
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    data = r.json()
    if "observations" not in data:
        raise ValueError(f"Unexpected FRED response: {data.get('error_message', data)}")

    rows = [
        {"Date": pd.to_datetime(obs["date"]), "Close": float(obs["value"])}
        for obs in data["observations"]
        if obs["value"] != "."
    ]
    if not rows:
        raise ValueError(f"FRED returned no data for '{fred_code}'.")

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    log.info("[FRED] %d rows (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


# ============================================================================
# Strategy 2 — yfinance
# ============================================================================


def scrape_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Telecharge via Yahoo Finance (yfinance).
    pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    yahoo_sym = YAHOO_TICKERS.get(ticker.lower())
    if yahoo_sym is None:
        raise ValueError(
            f"Ticker '{ticker}' not configured for Yahoo Finance.\n"
            f"Available symbols: {YAHOO_TICKERS}"
        )

    log.info("[yfinance] Downloading %s (%s)...", ticker.upper(), yahoo_sym)
    raw = yf.download(yahoo_sym, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for '{yahoo_sym}'.")

    df = raw[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Close"] = df["Close"].astype(float)
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    log.info("[yfinance] %d rows (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


# ============================================================================
# Strategy 3 — Barchart HTML scraping
# ============================================================================


def scrape_barchart(ticker: str) -> pd.DataFrame:
    """
    Scrape the historical price table from Barchart (~2 years free).
    """
    url = BARCHART_URLS.get(ticker.lower())
    if url is None:
        raise ValueError(
            f"Barchart URL not configured for '{ticker}'.\n"
            f"Disponibles : {list(BARCHART_URLS)}"
        )

    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Referer": "https://www.barchart.com/",
    }

    log.info("[Barchart] Loading %s...", ticker.upper())
    r = requests.Session().get(url, headers=req_headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": lambda c: c and "historical" in c.lower()})
    if table is None:
        table = soup.find("table")
    if table is None:
        raise ValueError("Table not found on Barchart. The HTML structure may have changed.")

    thead = table.find("thead")
    cols = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError("tbody not found in the Barchart table.")

    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in tbody.find_all("tr")
        if tr.find_all("td")
    ]
    if not rows:
        raise ValueError("No data rows in the Barchart table.")

    df = pd.DataFrame(rows, columns=cols if cols else list(range(len(rows[0]))))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Last" in df.columns:
        df = df.rename(columns={"Last": "Close"})
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(
            df["Close"].astype(str).str.replace(",", ""), errors="coerce"
        )

    df = (
        df[["Date", "Close"]]
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    log.info("[Barchart] %d rows (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


# ============================================================================
# Strategy 4 — TradingEconomics via Selenium
# ============================================================================


def scrape_tradingeconomics(ticker: str) -> pd.DataFrame:
    """
    Capture the JSON chart data from TradingEconomics via Selenium.
    Requires: pip install selenium + Chrome + chromedriver
    """
    import json
    import re
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    url = TE_URLS.get(ticker.lower())
    if url is None:
        raise ValueError(
            f"TradingEconomics URL not configured for '{ticker}'.\n"
            f"Disponibles : {list(TE_URLS)}"
        )

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)
    try:
        log.info("[TradingEconomics] Loading %s...", ticker.upper())
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "canvas, .highcharts-root, svg"))
        )
        time.sleep(3)

        matches = re.findall(r'"data"\s*:\s*(\[\[.*?\]\])', driver.page_source, re.DOTALL)
        rows = []
        for match in matches[:1]:
            try:
                for point in json.loads(match):
                    if len(point) >= 2:
                        rows.append({
                            "Date":  pd.Timestamp(point[0], unit="ms"),
                            "Close": point[1],
                        })
            except Exception:
                continue

        if not rows:
            raise ValueError("Could not extract the JSON chart data.")

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        log.info("[TE] %d rows (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
        return df

    finally:
        driver.quit()


# ============================================================================
# Save
# ============================================================================


def save(df: pd.DataFrame, ticker: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_str = df["Date"].min().strftime("%Y-%m-%d")
    end_str   = df["Date"].max().strftime("%Y-%m-%d")
    out_path  = out_dir / f"{ticker.lower()}_{start_str}_{end_str}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Freight index scraper")
    parser.add_argument("--ticker",   default="bdi",
                        help="Index: bdi, fbx, wci, scfi (default: bdi)")
    parser.add_argument("--start",    default="2016-01-01",
                        help="Start date YYYY-MM-DD (default: 2016-01-01)")
    parser.add_argument("--end",      default=datetime.today().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--strategy", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1=FRED (default), 2=yfinance, 3=Barchart, 4=Selenium/TradingEconomics")
    parser.add_argument("--fred-key", default=None, metavar="KEY",
                        help="FRED API key (free on fred.stlouisfed.org). Required for --strategy 1.")
    parser.add_argument("--out-dir",  default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    try:
        if args.strategy == 1:
            if not args.fred_key:
                print(
                    "[ERROR] --strategy 1 (FRED) requires --fred-key.\n"
                    "  Free key in 30 seconds: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                    "  Then re-run: python src/index/scrap_index.py --ticker bdi "
                    "--start 2017-01-01 --fred-key YOUR_KEY\n\n"
                    "  Or try without a key: --strategy 2 (yfinance) or --strategy 3 (Barchart)"
                )
                sys.exit(1)
            df = scrape_fred(args.ticker, args.start, args.end, args.fred_key)

        elif args.strategy == 2:
            df = scrape_yfinance(args.ticker, args.start, args.end)

        elif args.strategy == 3:
            df = scrape_barchart(args.ticker)

        elif args.strategy == 4:
            df = scrape_tradingeconomics(args.ticker)

        print(f"\nPreview (last 5 rows):")
        print(df.tail(5).to_string(index=False))
        print(f"\nTotal : {len(df)} points  |  {df['Date'].min().date()} -> {df['Date'].max().date()}")

        out_path = save(df, args.ticker, out_dir)
        print(f"Saved: {out_path.resolve()}")

    except Exception as e:
        print(f"\nErreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
