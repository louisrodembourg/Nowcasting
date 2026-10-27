"""
Phase 4 — First-Pass Prediction: LA port congestion → shipping equity returns.

Target hierarchy (smallest/most tradable first, then general):
  Tier 1 — Pure freight rate proxies:
    BDRY  : Breakwave Dry Bulk Shipping ETF (direct BDI proxy)
    SBLK  : Star Bulk Carriers (dry bulk, high BDI beta)
  Tier 2 — Container shippers (LA is a container port):
    MATX  : Matson Inc (US container shipping)
    DAC   : Danaos Corp (container ship owner)
    CMRE  : Costamare (container charter)
    GSL   : Global Ship Lease
    COMBO : Equal-weight basket {MATX, DAC, CMRE, GSL}

AIS features (from LA gravity score 2017–2022):
  - log1p(gravity_score)
  - rolling 7d / 21d mean of log gravity
  - z-score of log gravity vs. trailing 60-day baseline
  - 5-day delta of log gravity (momentum)
  - log1p(waiting_vessels)
  - log1p(total_capacity)
  - congestion flag: zscore > 1.5

Models:
  - Ridge regression (baseline, interpretable)
  - XGBoost (non-linear)

Walk-forward validation:
  - Initial train: 252 trading days (~1 year)
  - Test blocks: 63 days (1 quarter), rolling forward
  - Metrics: IC (rank corr), directional accuracy, RMSE, Sharpe of long-short

Horizons: 5, 10, 21 trading days

Usage:
    python src/correlation/predict_returns.py
    python src/correlation/predict_returns.py --horizon 10 --target BDRY
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
GRAVITY_PATH = ROOT / "data/features/la_gravity_2017_2022.parquet"
EQUITY_PATH  = ROOT / "data/financial/equity_proxies_wide.parquet"
OUT_DIR      = ROOT / "outputs/financial"

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [5, 10, 21]

TARGETS = {
    "BDRY":  ("BDRY",  "Breakwave Dry Bulk ETF"),
    "SBLK":  ("SBLK",  "Star Bulk Carriers"),
    "MATX":  ("MATX",  "Matson Inc"),
    "DAC":   ("DAC",   "Danaos Corp"),
    "CMRE":  ("CMRE",  "Costamare"),
    "GSL":   ("GSL",   "Global Ship Lease"),
    "COMBO": (None,    "Equal-weight basket MATX+DAC+CMRE+GSL"),
}

FEATURE_COLS = [
    "log_gravity",
    "gravity_ma7",
    "gravity_ma21",
    "gravity_zscore60",
    "gravity_delta5",
    "log_waiting",
    "log_capacity",
    "congestion_flag",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_gravity() -> pd.DataFrame:
    df = (
        pl.read_parquet(GRAVITY_PATH)
        .with_columns(pl.col("date").str.to_date().alias("date"))
        .sort("date")
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def load_equity_prices() -> pd.DataFrame:
    """Load equity prices: cached parquet + yfinance for SBLK."""
    cached = (
        pl.read_parquet(EQUITY_PATH)
        .with_columns(pl.col("date").str.to_date().alias("date"))
        .to_pandas()
    )
    cached["date"] = pd.to_datetime(cached["date"])
    cached = cached.set_index("date")

    # Fetch SBLK from yfinance (not in cached parquet)
    log.info("Fetching SBLK from yfinance...")
    try:
        sblk = yf.download("SBLK", start="2017-01-01", end="2023-01-01",
                            progress=False, auto_adjust=True)["Close"]
        sblk.name = "SBLK"
        sblk.index = pd.to_datetime(sblk.index)
        if hasattr(sblk, "squeeze"):
            sblk = sblk.squeeze()
        cached = cached.join(sblk, how="left")
        log.info("SBLK fetched: %d rows", sblk.notna().sum())
    except Exception as exc:
        log.warning("SBLK fetch failed: %s", exc)
        cached["SBLK"] = np.nan

    return cached


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(gravity: pd.DataFrame) -> pd.DataFrame:
    """Build AIS feature matrix from raw gravity data."""
    f = pd.DataFrame(index=gravity.index)

    g = np.log1p(gravity["gravity_score"])
    f["log_gravity"]      = g
    f["gravity_ma7"]      = g.rolling(7,  min_periods=3).mean()
    f["gravity_ma21"]     = g.rolling(21, min_periods=7).mean()

    roll60_mean = g.rolling(60, min_periods=20).mean()
    roll60_std  = g.rolling(60, min_periods=20).std().replace(0, np.nan)
    f["gravity_zscore60"] = (g - roll60_mean) / roll60_std

    f["gravity_delta5"]   = g.diff(5)
    f["log_waiting"]      = np.log1p(gravity["waiting_vessels"])
    f["log_capacity"]     = np.log1p(gravity["total_capacity"])
    f["congestion_flag"]  = (f["gravity_zscore60"] > 1.5).astype(float)

    # Drop the burn-in period (first 60 days with NaN rolling stats)
    f = f.dropna(subset=["gravity_zscore60"])
    return f


# ══════════════════════════════════════════════════════════════════════════════
# RETURN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    """Log return from t to t+horizon (in trading days)."""
    return np.log(prices.shift(-horizon) / prices)


def build_combo(equity: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return basket of MATX, DAC, CMRE, GSL."""
    basket_cols = [c for c in ["MATX", "DAC", "CMRE", "GSL"] if c in equity.columns]
    # Build from log returns then average → reconstruct price-like series
    log_rets = np.log(equity[basket_cols] / equity[basket_cols].shift(1))
    avg_ret  = log_rets.mean(axis=1)
    # Reconstruct a synthetic price starting at 100
    combo = np.exp(avg_ret.cumsum()) * 100
    combo.name = "COMBO"
    return combo


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_ridge(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    init_train: int = 252,
    test_block: int = 63,
) -> dict:
    """
    Expanding-window walk-forward Ridge regression.
    Returns dict with oos_pred, oos_actual, oos_dates.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    oos_pred, oos_actual, oos_dates_list = [], [], []

    start = init_train
    while start < n:
        end = min(start + test_block, n)

        X_train, y_train = X[:start], y[:start]
        X_test,  y_test  = X[start:end], y[start:end]

        mask_tr = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        mask_te = ~(np.isnan(X_test).any(axis=1)  | np.isnan(y_test))

        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            start += test_block
            continue

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[mask_tr])
        Xte = scaler.transform(X_test[mask_te])

        model = Ridge(alpha=1.0)
        model.fit(Xtr, y_train[mask_tr])
        preds = model.predict(Xte)

        oos_pred.extend(preds.tolist())
        oos_actual.extend(y_test[mask_te].tolist())
        oos_dates_list.extend(dates[start:end][mask_te].tolist())

        start += test_block

    return {
        "oos_pred":   np.array(oos_pred),
        "oos_actual": np.array(oos_actual),
        "oos_dates":  pd.DatetimeIndex(oos_dates_list),
    }


def walk_forward_xgb(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    init_train: int = 252,
    test_block: int = 63,
) -> dict:
    """Expanding-window walk-forward XGBoost."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {}

    n = len(X)
    oos_pred, oos_actual, oos_dates_list = [], [], []

    start = init_train
    while start < n:
        end = min(start + test_block, n)

        X_train, y_train = X[:start], y[:start]
        X_test,  y_test  = X[start:end], y[start:end]

        mask_tr = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        mask_te = ~(np.isnan(X_test).any(axis=1)  | np.isnan(y_test))

        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            start += test_block
            continue

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train[mask_tr], y_train[mask_tr])
        preds = model.predict(X_test[mask_te])

        oos_pred.extend(preds.tolist())
        oos_actual.extend(y_test[mask_te].tolist())
        oos_dates_list.extend(dates[start:end][mask_te].tolist())

        start += test_block

    return {
        "oos_pred":   np.array(oos_pred),
        "oos_actual": np.array(oos_actual),
        "oos_dates":  pd.DatetimeIndex(oos_dates_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, actual: np.ndarray, label: str) -> dict:
    """IC, directional accuracy, RMSE, Sharpe of long-short signal."""
    valid = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[valid], actual[valid]
    n = len(p)

    if n < 10:
        return {"label": label, "n": n, "note": "insufficient data"}

    ic_pearson,  p_pval  = stats.pearsonr(p, a)
    ic_spearman, sp_pval = stats.spearmanr(p, a)

    dir_acc = np.mean(np.sign(p) == np.sign(a))
    rmse    = float(np.sqrt(np.mean((p - a) ** 2)))

    # Long-short: long when pred > median, short otherwise
    threshold = np.median(p)
    ls_ret = np.where(p > threshold, a, -a)
    sharpe = float(np.mean(ls_ret) / (np.std(ls_ret) + 1e-10) * np.sqrt(252))

    return {
        "label":         label,
        "n_oos":         n,
        "IC_pearson":    round(float(ic_pearson),  4),
        "p_pearson":     round(float(p_pval),      4),
        "IC_spearman":   round(float(ic_spearman), 4),
        "p_spearman":    round(float(sp_pval),     4),
        "dir_accuracy":  round(float(dir_acc),     4),
        "rmse":          round(rmse,               6),
        "ls_sharpe_ann": round(float(sharpe),      4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAG SCAN (brute-force correlation over lags 0–60)
# ══════════════════════════════════════════════════════════════════════════════

def lag_scan(
    features: pd.DataFrame,
    prices: pd.Series,
    horizon: int,
    max_lag: int = 60,
) -> pd.DataFrame:
    """
    For each lag k (0..max_lag), compute Spearman(feature[t-k], fwd_return[t]).
    Identifies whether signal leads or is concurrent.
    """
    fwd_ret = compute_forward_returns(prices, horizon)
    results = []

    for feat in FEATURE_COLS:
        for lag in range(0, max_lag + 1, 5):
            merged = pd.DataFrame({
                "feat":    features[feat].shift(lag),
                "fwd_ret": fwd_ret,
            }).dropna()
            if len(merged) < 30:
                continue
            r, p = stats.spearmanr(merged["feat"], merged["fwd_ret"])
            results.append({
                "feature": feat,
                "lag":     lag,
                "spearman_r": round(float(r), 4),
                "p_value":    round(float(p), 4),
                "n":          len(merged),
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(target_key: str = "ALL", horizons: list[int] = None) -> pd.DataFrame:
    if horizons is None:
        horizons = HORIZONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading gravity data...")
    gravity  = load_gravity()
    features = build_features(gravity)

    log.info("Loading equity prices...")
    equity = load_equity_prices()

    # Build COMBO synthetic price
    equity["COMBO"] = build_combo(equity)

    targets_to_run = list(TARGETS.keys()) if target_key == "ALL" else [target_key]

    all_results = []

    for tgt in targets_to_run:
        ticker, description = TARGETS[tgt]
        col = tgt if tgt in equity.columns else ticker

        if col not in equity.columns:
            log.warning("Target %s not in equity data, skipping.", tgt)
            continue

        prices = equity[col].dropna()
        log.info("\n%s %s (%s) — %d price points", "─"*60, tgt, description, len(prices))

        for h in horizons:
            fwd_ret = compute_forward_returns(prices, h)

            # Align on business days that exist in both feature and return series
            merged = features[FEATURE_COLS].join(fwd_ret.rename("fwd_ret"), how="inner").dropna(subset=["fwd_ret"])

            X = merged[FEATURE_COLS].values
            y = merged["fwd_ret"].values
            dates = merged.index

            n_obs = (~np.isnan(y)).sum()
            log.info("  Horizon %dd | aligned obs = %d", h, n_obs)

            if n_obs < 300:
                log.warning("  Insufficient data for %s h=%d (%d obs)", tgt, h, n_obs)
                continue

            # ── Walk-forward Ridge ────────────────────────────────────────
            ridge_res = walk_forward_ridge(X, y, dates)
            if len(ridge_res.get("oos_pred", [])) > 10:
                ridge_metrics = compute_metrics(
                    ridge_res["oos_pred"], ridge_res["oos_actual"],
                    f"Ridge | {tgt} | h={h}d"
                )
                ridge_metrics.update({"target": tgt, "horizon": h, "model": "Ridge"})
                all_results.append(ridge_metrics)

                log.info(
                    "  Ridge OOS: IC_pearson=%.3f (p=%.3f) | IC_spearman=%.3f (p=%.3f) "
                    "| DirAcc=%.3f | Sharpe=%.3f | n=%d",
                    ridge_metrics["IC_pearson"],   ridge_metrics["p_pearson"],
                    ridge_metrics["IC_spearman"],  ridge_metrics["p_spearman"],
                    ridge_metrics["dir_accuracy"], ridge_metrics["ls_sharpe_ann"],
                    ridge_metrics["n_oos"],
                )

            # ── Walk-forward XGBoost ──────────────────────────────────────
            xgb_res = walk_forward_xgb(X, y, dates)
            if len(xgb_res.get("oos_pred", [])) > 10:
                xgb_metrics = compute_metrics(
                    xgb_res["oos_pred"], xgb_res["oos_actual"],
                    f"XGB | {tgt} | h={h}d"
                )
                xgb_metrics.update({"target": tgt, "horizon": h, "model": "XGBoost"})
                all_results.append(xgb_metrics)

                log.info(
                    "  XGB   OOS: IC_pearson=%.3f (p=%.3f) | IC_spearman=%.3f (p=%.3f) "
                    "| DirAcc=%.3f | Sharpe=%.3f | n=%d",
                    xgb_metrics["IC_pearson"],   xgb_metrics["p_pearson"],
                    xgb_metrics["IC_spearman"],  xgb_metrics["p_spearman"],
                    xgb_metrics["dir_accuracy"], xgb_metrics["ls_sharpe_ann"],
                    xgb_metrics["n_oos"],
                )

        # ── SHAP analysis (horizon=10d uniquement pour éviter la redondance) ──
        if 10 in horizons:
            h_shap = 10
            fwd_ret_shap = compute_forward_returns(prices, h_shap)
            merged_shap = features[FEATURE_COLS].join(fwd_ret_shap.rename("fwd_ret"), how="inner").dropna(subset=["fwd_ret"])
            if len(merged_shap) >= 50:
                compute_shap_analysis(
                    X=merged_shap[FEATURE_COLS].values,
                    y=merged_shap["fwd_ret"].values,
                    dates=merged_shap.index,
                    feature_names=FEATURE_COLS,
                    target_label=tgt,
                    horizon=h_shap,
                )

        # ── Lag scan on best horizon ──────────────────────────────────────
        log.info("  Lag scan (h=10d, lags 0-60)...")
        lag_df = lag_scan(features, prices, horizon=10, max_lag=60)
        best_by_feat = (
            lag_df.loc[lag_df["spearman_r"].abs().groupby(lag_df["feature"]).idxmax()]
            .sort_values("spearman_r", key=abs, ascending=False)
        )
        log.info("  Best lags:\n%s", best_by_feat[["feature", "lag", "spearman_r", "p_value"]].to_string(index=False))

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        log.warning("No results computed.")
        return results_df

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("PHASE 4 — FIRST PASS PREDICTION RESULTS (out-of-sample walk-forward)")
    print("="*90)
    print(f"{'Target':<8} {'H':>4} {'Model':<8} {'IC_pearson':>10} {'p':>6} {'IC_spearman':>12} {'p':>6} {'DirAcc':>8} {'Sharpe':>8} {'n_oos':>6}")
    print("─"*90)
    for _, row in results_df.sort_values(["target", "horizon", "model"]).iterrows():
        flag = " ★" if abs(row.get("IC_spearman", 0)) > 0.05 and row.get("p_spearman", 1) < 0.1 else ""
        print(
            f"{row['target']:<8} {int(row['horizon']):>4} {row['model']:<8} "
            f"{row['IC_pearson']:>10.4f} {row['p_pearson']:>6.3f} "
            f"{row['IC_spearman']:>12.4f} {row['p_spearman']:>6.3f} "
            f"{row['dir_accuracy']:>8.3f} {row['ls_sharpe_ann']:>8.3f} "
            f"{int(row['n_oos']):>6}{flag}"
        )
    print("="*90)
    print("★ = IC_spearman > 0.05 and p < 0.10")

    out_path = OUT_DIR / "firstpass_results.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_analysis(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    feature_names: list[str],
    target_label: str,
    horizon: int,
) -> None:
    """
    Entraîne un XGBoost final sur toutes les données disponibles et calcule
    les valeurs SHAP globales (summary bar) et locales (waterfall sur le jour
    de congestion maximale identifié par log_gravity).

    Sorties :
        outputs/financial/shap_{target}_{horizon}d_summary.png   — importance globale
        outputs/financial/shap_{target}_{horizon}d_waterfall.png — explication locale
    """
    try:
        import shap
        from xgboost import XGBRegressor
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        log.warning("SHAP non disponible (%s) — pip install shap", e)
        return

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean, y_clean, dates_clean = X[valid], y[valid], dates[valid]

    if len(X_clean) < 50:
        log.warning("SHAP: pas assez de données pour %s h=%dd", target_label, horizon)
        return

    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_clean, y_clean)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)  # (n_obs, n_features)

    slug = f"{target_label}_{horizon}d"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Summary bar (importance globale = mean |SHAP| par feature) ────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color="#1565C0",
    )
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"SHAP — Importance globale\n{target_label} | horizon {horizon}j")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    summary_path = OUT_DIR / f"shap_{slug}_summary.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    log.info("SHAP summary -> %s", summary_path)

    # ── Waterfall local : jour de congestion maximale (log_gravity max) ───────
    log_gravity_idx = feature_names.index("log_gravity") if "log_gravity" in feature_names else 0
    peak_idx = int(np.argmax(X_clean[:, log_gravity_idx]))
    peak_date = dates_clean[peak_idx]

    base_value = float(explainer.expected_value)
    sv = shap_values[peak_idx]           # (n_features,)
    fv = X_clean[peak_idx]               # valeurs des features ce jour-là
    pred = base_value + sv.sum()

    # Tri par contribution absolue décroissante
    order_wf = np.argsort(np.abs(sv))[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#D32F2F" if v > 0 else "#1565C0" for v in sv[order_wf]]
    labels = [f"{feature_names[i]}\n= {fv[i]:.3f}" for i in order_wf]
    ax.barh(labels, sv[order_wf], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution SHAP au rendement prédit")
    ax.set_title(
        f"SHAP — Explication locale\n{target_label} | h={horizon}j | "
        f"{peak_date.strftime('%Y-%m-%d')} (pic congestion)\n"
        f"Base={base_value:.4f}  Prédit={pred:.4f}"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    waterfall_path = OUT_DIR / f"shap_{slug}_waterfall.png"
    fig.savefig(waterfall_path, dpi=150)
    plt.close(fig)
    log.info("SHAP waterfall -> %s", waterfall_path)

    # ── Print résumé ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SHAP | {target_label} | h={horizon}j | pic congestion {peak_date.strftime('%Y-%m-%d')}")
    print(f"{'Feature':<22} {'SHAP':>10}  {'Valeur feature':>16}")
    print(f"{'─'*22} {'─'*10}  {'─'*16}")
    for i in order_wf:
        print(f"  {feature_names[i]:<20} {sv[i]:>+10.4f}  {fv[i]:>16.4f}")
    print(f"  {'Base value':<20} {base_value:>+10.4f}")
    print(f"  {'Prédiction':<20} {pred:>+10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — First-pass shipping index prediction")
    parser.add_argument("--target",   default="ALL",
                        choices=list(TARGETS.keys()) + ["ALL"],
                        help="Target equity (default: ALL)")
    parser.add_argument("--horizon",  type=int, nargs="+", default=HORIZONS,
                        help="Forward return horizon(s) in trading days (default: 5 10 21)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(target_key=args.target, horizons=args.horizon)


if __name__ == "__main__":
    main()
