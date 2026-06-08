"""
Scraper d'indices financiers fret : BDI, FBX, WCI, SCFI.

Strategies disponibles :
  1 (fred)      - FRED (Federal Reserve) via API key gratuite
                  BDI disponible sous le code DBDI
                  Cle gratuite : https://fred.stlouisfed.org/docs/api/api_key.html
  2 (yfinance)  - Yahoo Finance via yfinance (pip install yfinance)
                  Tickers : BDI=F, ^BDI selon disponibilite Yahoo
  3 (barchart)  - HTML scraping, ~2 ans gratuit sans compte
  4 (selenium)  - TradingEconomics via Selenium (Chrome + chromedriver requis)

Output : data/parquet/market/<ticker>_<start>_<end>.csv
         colonnes : Date (YYYY-MM-DD), Close

Usage :
    python src/index/scrap_index.py --ticker bdi --start 2017-01-01 --end 2020-12-31
    python src/index/scrap_index.py --ticker bdi --start 2017-01-01 --strategy 1 --fred-key VOTRE_CLE
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

# Correspondances ticker -> code FRED
FRED_CODES = {
    "bdi":  "DBDI",       # Baltic Dry Index (daily)
    "wci":  "DWCI",       # World Container Index (si disponible)
}

# Correspondances ticker -> symbole Yahoo Finance
YAHOO_TICKERS = {
    "bdi":  "^BDI",
}

# Correspondances ticker -> URL Barchart
BARCHART_URLS = {
    "bdi": "https://www.barchart.com/stocks/quotes/BDI.BI/price-history/historical",
    "fbx": "https://www.barchart.com/stocks/quotes/FBX/price-history/historical",
}

# Correspondances ticker -> URL TradingEconomics
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
    Telecharge un indice depuis FRED via l'API officielle.
    Cle gratuite : https://fred.stlouisfed.org/docs/api/api_key.html
    """
    fred_code = FRED_CODES.get(ticker.lower())
    if fred_code is None:
        raise ValueError(
            f"Ticker '{ticker}' non configure pour FRED.\n"
            f"Codes disponibles : {FRED_CODES}\n"
            f"Cherchez le code sur https://fred.stlouisfed.org"
        )

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":        fred_code,
        "observation_start": start,
        "observation_end":   end,
        "api_key":           api_key,
        "file_type":         "json",
    }

    log.info("[FRED] Telechargement %s (%s)...", ticker.upper(), fred_code)
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    data = r.json()
    if "observations" not in data:
        raise ValueError(f"Reponse FRED inattendue : {data.get('error_message', data)}")

    rows = [
        {"Date": pd.to_datetime(obs["date"]), "Close": float(obs["value"])}
        for obs in data["observations"]
        if obs["value"] != "."
    ]
    if not rows:
        raise ValueError(f"FRED n'a retourne aucune donnee pour '{fred_code}'.")

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    log.info("[FRED] %d lignes (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
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
        raise ImportError("yfinance non installe. Lancez : pip install yfinance")

    yahoo_sym = YAHOO_TICKERS.get(ticker.lower())
    if yahoo_sym is None:
        raise ValueError(
            f"Ticker '{ticker}' non configure pour Yahoo Finance.\n"
            f"Symboles disponibles : {YAHOO_TICKERS}"
        )

    log.info("[yfinance] Telechargement %s (%s)...", ticker.upper(), yahoo_sym)
    raw = yf.download(yahoo_sym, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(f"Yahoo Finance n'a retourne aucune donnee pour '{yahoo_sym}'.")

    df = raw[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Close"] = df["Close"].astype(float)
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    log.info("[yfinance] %d lignes (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


# ============================================================================
# Strategy 3 — Barchart HTML scraping
# ============================================================================


def scrape_barchart(ticker: str) -> pd.DataFrame:
    """
    Scrape la table de prix historiques sur Barchart (~2 ans gratuit).
    """
    url = BARCHART_URLS.get(ticker.lower())
    if url is None:
        raise ValueError(
            f"URL Barchart non configuree pour '{ticker}'.\n"
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

    log.info("[Barchart] Chargement %s...", ticker.upper())
    r = requests.Session().get(url, headers=req_headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": lambda c: c and "historical" in c.lower()})
    if table is None:
        table = soup.find("table")
    if table is None:
        raise ValueError("Table introuvable sur Barchart. La structure HTML a peut-etre change.")

    thead = table.find("thead")
    cols = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError("tbody introuvable dans la table Barchart.")

    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in tbody.find_all("tr")
        if tr.find_all("td")
    ]
    if not rows:
        raise ValueError("Aucune ligne de donnees dans la table Barchart.")

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
    log.info("[Barchart] %d lignes (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


# ============================================================================
# Strategy 4 — TradingEconomics via Selenium
# ============================================================================


def scrape_tradingeconomics(ticker: str) -> pd.DataFrame:
    """
    Capture les donnees JSON du graphique TradingEconomics via Selenium.
    Necessite : pip install selenium + Chrome + chromedriver
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
            f"URL TradingEconomics non configuree pour '{ticker}'.\n"
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
        log.info("[TradingEconomics] Chargement %s...", ticker.upper())
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
            raise ValueError("Impossible d'extraire les donnees JSON du graphique.")

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        log.info("[TE] %d lignes (%s -> %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
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
    parser = argparse.ArgumentParser(description="Scraper d'indices financiers fret")
    parser.add_argument("--ticker",   default="bdi",
                        help="Indice : bdi, fbx, wci, scfi (defaut: bdi)")
    parser.add_argument("--start",    default="2016-01-01",
                        help="Date debut YYYY-MM-DD (defaut: 2016-01-01)")
    parser.add_argument("--end",      default=datetime.today().strftime("%Y-%m-%d"),
                        help="Date fin YYYY-MM-DD (defaut: aujourd'hui)")
    parser.add_argument("--strategy", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1=FRED (defaut), 2=yfinance, 3=Barchart, 4=Selenium/TradingEconomics")
    parser.add_argument("--fred-key", default=None, metavar="KEY",
                        help="Cle API FRED (gratuite sur fred.stlouisfed.org). Requise pour --strategy 1.")
    parser.add_argument("--out-dir",  default=str(OUT_DIR),
                        help=f"Repertoire de sortie (defaut: {OUT_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    try:
        if args.strategy == 1:
            if not args.fred_key:
                print(
                    "[ERREUR] --strategy 1 (FRED) requiert --fred-key.\n"
                    "  Cle gratuite en 30 secondes : https://fred.stlouisfed.org/docs/api/api_key.html\n"
                    "  Puis relancez : python src/index/scrap_index.py --ticker bdi "
                    "--start 2017-01-01 --fred-key VOTRE_CLE\n\n"
                    "  Ou essayez sans cle : --strategy 2 (yfinance) ou --strategy 3 (Barchart)"
                )
                sys.exit(1)
            df = scrape_fred(args.ticker, args.start, args.end, args.fred_key)

        elif args.strategy == 2:
            df = scrape_yfinance(args.ticker, args.start, args.end)

        elif args.strategy == 3:
            df = scrape_barchart(args.ticker)

        elif args.strategy == 4:
            df = scrape_tradingeconomics(args.ticker)

        print(f"\nApercu (5 dernieres lignes) :")
        print(df.tail(5).to_string(index=False))
        print(f"\nTotal : {len(df)} points  |  {df['Date'].min().date()} -> {df['Date'].max().date()}")

        out_path = save(df, args.ticker, out_dir)
        print(f"Sauvegarde : {out_path.resolve()}")

    except Exception as e:
        print(f"\nErreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
