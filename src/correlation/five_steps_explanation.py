"""
Phase 4 — The 5 transformation steps of the AIS signal.

This script reproduces and explains each step of the progression:

  Step 1: Raw gravity score (lag=7d)
  Step 2: Log transform
  Step 3: MA21 smoothing
  Step 4: Direct log_waiting component (lag=4d)
  Step 5: Two-regime model + PINN

The Spearman IC of each step is written to
outputs/financial/five_steps_<target>_<horizon>d.csv.

Usage:
    python src/correlation/five_steps_explanation.py
    python src/correlation/five_steps_explanation.py --target SBLK
    python src/correlation/five_steps_explanation.py --target MATX --horizon 10
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
FEAT_DIR  = ROOT / "data/features"
PINN_DIR  = ROOT / "outputs/pinn_yearly"
OUT_DIR   = ROOT / "outputs/financial"

REGIME1_COLS = [
    "log_gravity", "gravity_ma7", "gravity_ma21",
    "gravity_zscore60", "gravity_delta5",
    "log_waiting", "log_capacity", "congestion_flag",
]
REGIME2_COLS = [
    "pinn_rho", "gravity_zscore60", "gravity_delta5",
    "log_waiting", "log_gravity",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_gravity() -> pd.DataFrame:
    df = (
        pl.read_parquet(FEAT_DIR / "la_gravity_2017_2022.parquet")
        .with_columns(pl.col("date").str.to_date())
        .sort("date")
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_pinn_rho() -> pd.Series:
    frames = []
    for y in [2019, 2020, 2021, 2022]:
        p = PINN_DIR / f"la_{y}_time_to_clear.parquet"
        if p.exists():
            df = pl.read_parquet(p).to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df.set_index("date")["rho_pred"])
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames).sort_index()


def load_prices(ticker: str) -> pd.Series:
    equity_path = ROOT / "data/financial/equity_proxies_wide.parquet"
    if equity_path.exists():
        cached = pl.read_parquet(equity_path).to_pandas()
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")
        if ticker in cached.columns:
            return cached[ticker].dropna()
    raw = yf.download(ticker, start="2017-01-01", end="2023-01-01",
                      progress=False, auto_adjust=True)["Close"].squeeze()
    raw.index = pd.to_datetime(raw.index)
    return raw.dropna()


def forward_return(prices: pd.Series, horizon: int) -> pd.Series:
    return np.log(prices.shift(-horizon) / prices)


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE CORRELATION (Spearman on aligned series)
# ══════════════════════════════════════════════════════════════════════════════

def spearman(signal: pd.Series, target: pd.Series) -> tuple[float, float, int]:
    merged = pd.concat([signal.rename("s"), target.rename("t")], axis=1).dropna()
    if len(merged) < 30:
        return float("nan"), float("nan"), 0
    r, p = stats.spearmanr(merged["s"], merged["t"])
    return round(float(r), 4), round(float(p), 4), len(merged)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD RIDGE (step 5 — normal regime)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_ridge(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    init_train: int = 252,
    test_block: int = 63,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    preds, actuals = [], []
    start = init_train
    while start < n:
        end = min(start + test_block, n)
        Xtr, ytr = X[:start], y[:start]
        Xte, yte = X[start:end], y[start:end]
        mtr = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
        mte = ~(np.isnan(Xte).any(1) | np.isnan(yte))
        if mtr.sum() < 50 or mte.sum() == 0:
            start += test_block
            continue
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr[mtr])
        Xte_s = scaler.transform(Xte[mte])
        m = Ridge(alpha=1.0)
        m.fit(Xtr_s, ytr[mtr])
        preds.extend(m.predict(Xte_s))
        actuals.extend(yte[mte])
        start += test_block
    return np.array(preds), np.array(actuals)


# ══════════════════════════════════════════════════════════════════════════════
# STEPS
# ══════════════════════════════════════════════════════════════════════════════

def run(ticker: str = "SBLK", horizon: int = 10) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gravity  = load_gravity()
    pinn_rho = load_pinn_rho()
    prices   = load_prices(ticker)
    fwd_ret  = forward_return(prices, horizon)

    results = []

    # ── Step 1: raw gravity score, lag 7d ────────────────────────────────────
    signal1 = gravity["gravity_score"].shift(7)
    r, p, n = spearman(signal1, fwd_ret)
    results.append({
        "step": "01",
        "label": "Gravity score brut (lag=7j)",
        "transformation": "gravity_score[t-7]  ←  aucune transformation",
        "rho_s": r, "p_value": p, "n_obs": n,
        "gain": None,
        "interpretation": (
            "Raw signal: gravity score = Σ(capacity × waiting_time × |φ₁|) "
            "shifted by 7 days. Very skewed distribution (0 most days, "
            "extreme spikes during crises) → unstable correlation."
        ),
    })

    # ── Step 2: log transform ────────────────────────────────────────────────
    log_gravity = np.log1p(gravity["gravity_score"])
    signal2 = log_gravity
    r, p, n = spearman(signal2, fwd_ret)
    results.append({
        "step": "02",
        "label": "Log-transformation",
        "transformation": "log1p(gravity_score)",
        "rho_s": r, "p_value": p, "n_obs": n,
        "gain": round(r - results[-1]["rho_s"], 4),
        "interpretation": (
            "log1p(x) = log(1+x) compresses extreme values and symmetrises the "
            "distribution. Markets react to relative changes in congestion "
            "(+10% vs +1000%), not to absolute levels."
        ),
    })

    # ── Step 3: MA21 smoothing ───────────────────────────────────────────────
    signal3 = log_gravity.rolling(21, min_periods=7).mean()
    r, p, n = spearman(signal3, fwd_ret)
    results.append({
        "step": "03",
        "label": "Lissage MA21",
        "transformation": "rolling(log_gravity, 21j).mean()",
        "rho_s": r, "p_value": p, "n_obs": n,
        "gain": round(r - results[-1]["rho_s"], 4),
        "interpretation": (
            "21-trading-day moving average (~1 month). Filters daily noise: "
            "markets react to congestion that persists over time, "
            "not to an isolated one-day spike."
        ),
    })

    # ── Step 4: direct log_waiting component, lag=4d ─────────────────────────
    signal4 = np.log1p(gravity["waiting_vessels"]).shift(4)
    r, p, n = spearman(signal4, fwd_ret)
    results.append({
        "step": "04",
        "label": "Direct log_waiting component (lag=4d)",
        "transformation": "log1p(waiting_vessels)[t-4]",
        "rho_s": r, "p_value": p, "n_obs": n,
        "gain": round(r - results[-1]["rho_s"], 4),
        "interpretation": (
            "Decompose the gravity score and isolate the raw number of waiting "
            "vessels. Removes the multiplicative noise of (waiting_time × |φ₁|). "
            "Lag=4d: empirical market reaction delay to the AIS signal."
        ),
    })

    # ── Step 5: two-regime model + PINN ──────────────────────────────────────
    roll60_mean = log_gravity.rolling(60, min_periods=20).mean()
    roll60_std  = log_gravity.rolling(60, min_periods=20).std().replace(0, np.nan)

    feat = pd.DataFrame(index=gravity.index)
    feat["log_gravity"]      = log_gravity
    feat["gravity_ma7"]      = log_gravity.rolling(7,  min_periods=3).mean()
    feat["gravity_ma21"]     = log_gravity.rolling(21, min_periods=7).mean()
    feat["gravity_zscore60"] = (log_gravity - roll60_mean) / roll60_std
    feat["gravity_delta5"]   = log_gravity.diff(5)
    feat["log_waiting"]      = np.log1p(gravity["waiting_vessels"])
    feat["log_capacity"]     = np.log1p(gravity["total_capacity"])
    feat["congestion_flag"]  = (feat["gravity_zscore60"] > 1.5).astype(float)
    feat["pinn_rho"]         = pinn_rho.reindex(feat.index)
    feat["in_episode"]       = feat["pinn_rho"].notna().astype(float)
    feat = feat.dropna(subset=["gravity_zscore60"])

    merged_all = feat.join(fwd_ret.rename("fwd_ret"), how="inner").dropna(subset=["fwd_ret"])
    feat_aligned = merged_all.drop(columns=["fwd_ret"])
    y_all  = merged_all["fwd_ret"].values
    dates_all = merged_all.index

    # Two-regime walk-forward (same logic as two_regime_model.py)
    init_train, test_block = 252, 63
    n = len(feat_aligned)
    oos_pred, oos_actual = [], []
    start = init_train

    while start < n:
        end = min(start + test_block, n)
        feat_tr = feat_aligned.iloc[:start]
        y_tr    = y_all[:start]
        feat_te = feat_aligned.iloc[start:end]
        y_te    = y_all[start:end]

        # Train regime 1 on all days
        X1_tr = feat_tr[REGIME1_COLS].values
        m1_mask = ~np.isnan(X1_tr).any(1) & ~np.isnan(y_tr)
        r1_model, s1 = Ridge(alpha=1.0), StandardScaler()
        r1_fitted = False
        if m1_mask.sum() >= 50:
            r1_model.fit(s1.fit_transform(X1_tr[m1_mask]), y_tr[m1_mask])
            r1_fitted = True

        # Train regime 2 on PINN episode days
        X2_tr = feat_tr[REGIME2_COLS].values
        in_ep_tr = feat_tr["in_episode"].values
        m2_mask = (in_ep_tr == 1) & ~np.isnan(X2_tr).any(1) & ~np.isnan(y_tr)
        r2_model, s2 = Ridge(alpha=0.5), StandardScaler()
        r2_fitted = False
        if m2_mask.sum() >= 15:
            r2_model.fit(s2.fit_transform(X2_tr[m2_mask]), y_tr[m2_mask])
            r2_fitted = True

        # Predict
        preds = np.full(end - start, np.nan)
        X1_te = feat_te[REGIME1_COLS].values
        X2_te = feat_te[REGIME2_COLS].values
        in_ep_te = feat_te["in_episode"].values

        if r1_fitted:
            m1_te = ~np.isnan(X1_te).any(1)
            if m1_te.sum() > 0:
                preds[m1_te] = r1_model.predict(s1.transform(X1_te[m1_te]))
        if r2_fitted:
            m2_te = (in_ep_te == 1) & ~np.isnan(X2_te).any(1)
            if m2_te.sum() > 0:
                preds[m2_te] = r2_model.predict(s2.transform(X2_te[m2_te]))

        valid_te = ~np.isnan(preds) & ~np.isnan(y_te)
        oos_pred.extend(preds[valid_te])
        oos_actual.extend(y_te[valid_te])
        start += test_block

    all_pred   = np.array(oos_pred)
    all_actual = np.array(oos_actual)
    valid = ~(np.isnan(all_pred) | np.isnan(all_actual))
    r5, p5 = stats.spearmanr(all_pred[valid], all_actual[valid])
    n5 = valid.sum()

    results.append({
        "step": "05",
        "label": "Two-regime model + PINN",
        "transformation": (
            "Regime 1 (normal days)  : Ridge on 8 AIS features\n"
            "                        Regime 2 (PINN episode): Ridge on pinn_rho + AIS"
        ),
        "rho_s": round(float(r5), 4), "p_value": round(float(p5), 4), "n_obs": int(n5),
        "gain": round(float(r5) - results[-1]["rho_s"], 4),
        "interpretation": (
            "Conditional architecture: two distinct models depending on the regime. "
            "During a crisis, pinn_rho (LWR density predicted by the PINN through the "
            "conservation law) replaces the empirical AIS features with a calibrated "
            "physical signal. This is the Phase 3 → Phase 4 junction."
        ),
    })

    # ── Affichage ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  THE 5 TRANSFORMATION STEPS — {ticker} | horizon {horizon}d")
    print(f"{'═'*80}")

    for i, res in enumerate(results):
        gain_str = f"  (+{res['gain']:.3f})" if res["gain"] is not None else ""
        sig = "★" if res["p_value"] < 0.05 else " "
        print(f"\n{'─'*80}")
        print(f"  STEP {res['step']} — {res['label']}")
        print(f"  Transformation : {res['transformation']}")
        print(f"  ρs = {res['rho_s']:+.4f}{gain_str}   p={res['p_value']:.4f}  n={res['n_obs']} {sig}")
        print(f"\n  → {res['interpretation']}")

    print(f"\n{'═'*80}")
    print(f"  PROGRESSION TOTALE : {results[0]['rho_s']:+.4f} → {results[-1]['rho_s']:+.4f}  "
          f"(+{results[-1]['rho_s'] - results[0]['rho_s']:.3f})")
    print(f"{'═'*80}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'Step':<6} {'Label':<42} {'ρs':>7}  {'gain':>7}  {'p':>8}  {'n':>6}")
    print(f"  {'─'*6} {'─'*42} {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}")
    prev_rho = None
    for res in results:
        gain = f"+{res['gain']:.3f}" if res["gain"] is not None else "—"
        sig  = " ★" if res["p_value"] < 0.05 else ""
        print(f"  {res['step']:<6} {res['label']:<42} {res['rho_s']:>+7.4f}  {gain:>7}  "
              f"{res['p_value']:>8.4f}  {res['n_obs']:>6}{sig}")

    print(f"\n  ★ = p < 0.05\n")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out = pd.DataFrame([{
        "step": r["step"], "label": r["label"],
        "rho_s": r["rho_s"], "gain": r["gain"],
        "p_value": r["p_value"], "n_obs": r["n_obs"],
    } for r in results])
    path = OUT_DIR / f"five_steps_{ticker}_{horizon}d.csv"
    out.to_csv(path, index=False)
    log.info("Results saved → %s", path)
    print(f"  CSV results → {path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="The 5 transformation steps of the AIS signal → freight returns"
    )
    parser.add_argument("--target",  default="SBLK",
                        help="Target ticker (default: SBLK)")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Return horizon in days (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    run(ticker=args.target, horizon=args.horizon)


if __name__ == "__main__":
    main()
