"""
Phase 4 — Risk premium model.

Pipeline:
  1. Load gravity_score (daily, 2017-2024) + financial proxies
  2. Log-return transformation on financial series
  3. Lag-correlation analysis (Pearson/Spearman, 0-60 days)
  4. Granger causality test (gravity → returns)
  5. XGBoost + ElasticNet predictive models (walk-forward validation)
  6. SHAP feature importance
  7. Export results + plots

Usage:
    python -m src.correlation.risk_premium [--target MATX] [--max-lag 60]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
FINANCIAL_DIR = Path(__file__).parents[2] / "data" / "financial"
FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gravity(port: str = "la") -> pd.DataFrame:
    path = FEATURES_DIR / f"{port}_gravity_daily.parquet"
    df = pl.read_parquet(path).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Normalize gravity to [0,1] per column for interpretability
    df["gravity_norm"] = (df["gravity_score"] - df["gravity_score"].min()) / (
        df["gravity_score"].max() - df["gravity_score"].min() + 1e-9
    )
    df["waiting_ratio"] = df["waiting_vessels"] / (df["total_vessels"] + 1)
    return df


def load_financial(ticker: str) -> pd.Series:
    path = FINANCIAL_DIR / "equity_proxies_wide.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "equity_proxies_wide.parquet not found — run fetch_financial.py first"
        )
    df = pl.read_parquet(path).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if ticker not in df.columns:
        raise ValueError(f"Ticker '{ticker}' not in {list(df.columns)}")
    return df[ticker].dropna()


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).dropna()


def build_lag_features(gravity: pd.Series, max_lag: int) -> pd.DataFrame:
    """Build dataframe with gravity lagged 1..max_lag days."""
    frames = {"gravity_lag0": gravity}
    for lag in range(1, max_lag + 1):
        frames[f"gravity_lag{lag}"] = gravity.shift(lag)
    return pd.DataFrame(frames).dropna()


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def lag_correlation_analysis(
    gravity: pd.Series,
    returns: pd.Series,
    max_lag: int,
    label: str,
) -> pd.DataFrame:
    common = gravity.index.intersection(returns.index)
    g = gravity.loc[common]
    r = returns.loc[common]

    rows = []
    for lag in range(0, max_lag + 1):
        g_shifted = g.shift(lag).dropna()
        r_aligned = r.loc[g_shifted.index]
        mask = r_aligned.notna() & g_shifted.notna()
        if mask.sum() < 30:
            continue
        pearson_r, pearson_p = stats.pearsonr(g_shifted[mask], r_aligned[mask])
        spearman_r, spearman_p = stats.spearmanr(g_shifted[mask], r_aligned[mask])
        rows.append({
            "lag_days": lag,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_obs": int(mask.sum()),
        })

    df = pd.DataFrame(rows)
    out_path = FEATURES_DIR / f"lag_corr_{label}.parquet"
    pl.from_pandas(df).write_parquet(out_path)
    return df


def granger_test(gravity: pd.Series, returns: pd.Series, max_lag: int = 10) -> dict:
    common = gravity.index.intersection(returns.index)
    data = pd.DataFrame({"returns": returns.loc[common], "gravity": gravity.loc[common]}).dropna()
    results = grangercausalitytests(data[["returns", "gravity"]], maxlag=max_lag, verbose=False)
    pvalues = {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
    return pvalues


def walk_forward_xgb(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> dict:
    """Time-series walk-forward cross-validation for XGBoost."""
    n = len(X)
    min_train = n // (n_splits + 1)
    all_preds, all_true = [], []

    for fold in range(n_splits):
        train_end = min_train * (fold + 1)
        test_start = train_end
        test_end = min(train_end + min_train, n)
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0,
            device="cpu",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        preds = model.predict(X_te)
        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())

    r2 = r2_score(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    return {"r2": r2, "mae": mae, "preds": all_preds, "true": all_true}


def fit_models(
    gravity: pd.Series,
    returns: pd.Series,
    best_lag: int,
    label: str,
) -> dict:
    common = gravity.index.intersection(returns.index)
    g = gravity.shift(best_lag).loc[common]
    r = returns.loc[common]
    df = pd.DataFrame({"gravity": g, "returns": r}).dropna()

    X_raw = build_lag_features(gravity, max_lag=min(best_lag + 5, 20))
    X_raw = X_raw.loc[X_raw.index.intersection(r.index)]
    y = r.loc[X_raw.index].dropna()
    X_raw = X_raw.loc[y.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_raw.index, columns=X_raw.columns)

    # ElasticNet
    enet = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 1.0], max_iter=5000)
    enet.fit(X_scaled, y)
    enet_r2 = r2_score(y, enet.predict(X_scaled))

    # XGBoost walk-forward
    xgb_results = walk_forward_xgb(X_raw, y)

    # SHAP for XGBoost (fit on all data for explanation)
    xgb_final = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, verbosity=0,
        device="cpu",
    )
    xgb_final.fit(X_raw, y)
    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_raw)

    # Save SHAP summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_raw, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "enet_r2": enet_r2,
        "enet_alpha": enet.alpha_,
        "enet_l1_ratio": enet.l1_ratio_,
        "xgb_r2_walkforward": xgb_results["r2"],
        "xgb_mae_walkforward": xgb_results["mae"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lag_correlation(corr_df: pd.DataFrame, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, name in zip(
        axes, ["pearson_r", "spearman_r"], ["Pearson r", "Spearman ρ"]
    ):
        ax.bar(
            corr_df["lag_days"], corr_df[col],
            color=["steelblue" if p > 0.05 else "crimson"
                   for p in corr_df[col.replace("_r", "_p")]],
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(-0.1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Lag (jours)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} — gravity → {label} returns")
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color="crimson", label="p < 0.05"),
                plt.Rectangle((0, 0), 1, 1, color="steelblue", label="p ≥ 0.05"),
            ],
            fontsize=8,
        )
    plt.suptitle(f"Corrélation décalée : gravity score LA → {label}", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"lag_corr_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved lag_corr_{label}.png")


def plot_time_series_overlay(
    gravity: pd.Series, returns: pd.Series, label: str
) -> None:
    common = gravity.index.intersection(returns.index)
    g = gravity.loc[common]
    r = returns.loc[common]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.fill_between(g.index, g, alpha=0.4, color="steelblue", label="Gravity score (norm.)")
    ax2.plot(r.index, r.rolling(5).mean(), color="crimson", linewidth=1.2, label=f"{label} log-return (5d MA)")
    ax1.set_ylabel("Gravity score normalisé", color="steelblue")
    ax2.set_ylabel(f"{label} log-return", color="crimson")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    plt.title(f"Gravity score LA vs {label} — 2017–2024")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"overlay_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved overlay_{label}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target: str = "MATX", max_lag: int = 60, port: str = "la") -> None:
    print(f"\n=== Phase 4: Risk Premium Model | port={port} | target={target} ===\n")

    gravity_df = load_gravity(port)
    gravity_norm = gravity_df["gravity_norm"]
    waiting_ratio = gravity_df["waiting_ratio"]

    price = load_financial(target)
    returns = log_returns(price)

    label = f"{port}_{target}"

    # 1. Time-series overlay
    print("[1/5] Plotting time series overlay...")
    plot_time_series_overlay(gravity_norm, returns, label)

    # 2. Lag correlation
    print("[2/5] Lag correlation analysis...")
    corr_df = lag_correlation_analysis(gravity_norm, returns, max_lag, label)
    plot_lag_correlation(corr_df, label)

    best_lag_idx = corr_df["spearman_r"].abs().idxmax()
    best_lag = int(corr_df.loc[best_lag_idx, "lag_days"])
    best_r = corr_df.loc[best_lag_idx, "spearman_r"]
    best_p = corr_df.loc[best_lag_idx, "spearman_p"]
    print(f"  Best lag: {best_lag} days | Spearman ρ={best_r:.3f} | p={best_p:.3f}")

    # 3. Granger causality
    print("[3/5] Granger causality test (gravity → returns)...")
    granger_pvalues = granger_test(gravity_norm, returns, max_lag=min(10, max_lag))
    significant = [(lag, p) for lag, p in granger_pvalues.items() if p < 0.05]
    if significant:
        print(f"  Granger causal at lags: {significant}")
    else:
        print(f"  No significant Granger causality (min p={min(granger_pvalues.values()):.3f})")

    # 4. Predictive models
    print("[4/5] Fitting ElasticNet + XGBoost (walk-forward)...")
    model_results = fit_models(gravity_norm, returns, best_lag, label)
    print(f"  ElasticNet in-sample R²: {model_results['enet_r2']:.4f}")
    print(f"  XGBoost walk-forward R²: {model_results['xgb_r2_walkforward']:.4f}")
    print(f"  XGBoost walk-forward MAE: {model_results['xgb_mae_walkforward']:.6f}")

    # 5. Save summary
    summary = {
        "port": port,
        "target": target,
        "best_lag_days": best_lag,
        "spearman_r_at_best_lag": best_r,
        "spearman_p_at_best_lag": best_p,
        "granger_significant_lags": str(significant),
        **model_results,
    }
    summary_df = pl.DataFrame([summary])
    out_path = FEATURES_DIR / f"risk_premium_summary_{label}.parquet"
    summary_df.write_parquet(out_path)
    print(f"\n[5/5] Summary saved → {out_path.name}")

    print("\n=== Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="MATX", choices=list(["MATX", "DAC", "CMRE", "GSL", "BDRY"]))
    parser.add_argument("--port", default="la")
    parser.add_argument("--max-lag", type=int, default=60)
    args = parser.parse_args()
    main(target=args.target, max_lag=args.max_lag, port=args.port)
