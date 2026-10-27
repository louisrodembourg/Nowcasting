"""
correlation_pipeline.py — Continuous macro correlation analysis.

Approach: continuous time series (e.g. 2021-2024), without episodes or PINN.
The daily gravity score (from manifold_pipeline.py) is used as a leading
signal of future financial returns.

Step 1: Load and merge the continuous series (gravity + financial prices)
Step 2: Feature engineering (lags, rolling means, forward returns)
Step 3: Pearson/Spearman cross-correlogram (lag 0 -> max_lag)
Step 4: Continuous XGBoost with TimeSeriesSplit (out-of-fold predictions)

Commands:

  Houston 2021-2024 with the BOAT ETF:
    python src/correlation/correlation_pipeline.py --gravity data/features/houston_gravity_daily.parquet --financial data/financial/BOAT.csv --start 2021-08-01 --end 2024-11-17 --lags 1 3 5 7 14 21 --horizons 5 10 15 30 --target return_10d

  LA 2021-2024 with the S&P 500:
    python src/correlation/correlation_pipeline.py --gravity data/features/la_gravity_daily.parquet --financial data/financial/SP500.csv --start 2021-08-01 --end 2024-11-17 --lags 1 3 5 7 14 21 --horizons 5 10 15 30 --target return_10d
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

OUTPUT_DIR    = Path("outputs/financial")
DEFAULT_LAGS  = [1, 3, 5, 7, 14, 21]
DEFAULT_HORIZONS  = [5, 10, 15, 30]
DEFAULT_ROLLING   = [7, 14, 21]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Loading
# ════════════════════════════════════════════════════════════════════════════

def load_gravity(
    path: Path,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Load the daily gravity score (Parquet or CSV).
    Expected columns: date, gravity_score.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    # Detect the date column (case-insensitive)
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "ds")),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    if "gravity_score" not in df.columns:
        raise ValueError(
            f"Column 'gravity_score' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["date", "gravity_score"]].dropna().sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    log.info(
        "Gravity score: %d days | %s -> %s | non-zero: %d days",
        len(df),
        df["date"].min().date(), df["date"].max().date(),
        (df["gravity_score"] > 0).sum(),
    )
    return df.reset_index(drop=True)


def load_financial(
    path: Path,
    date_col:  str = "Date",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Load a CSV of financial prices. Returns date + price."""
    df = pd.read_csv(path, parse_dates=[date_col])
    if price_col not in df.columns:
        available = [c for c in df.columns if c != date_col]
        raise ValueError(
            f"Column '{price_col}' not found in {path}. "
            f"Available columns: {available}"
        )
    df = df[[date_col, price_col]].dropna()
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "Financial data: %d days | %s -> %s | col='%s'",
        len(df),
        df["date"].min().date(), df["date"].max().date(), price_col,
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature engineering
# ════════════════════════════════════════════════════════════════════════════

def build_continuous_dataset(
    gravity_df:  pd.DataFrame,
    financial_df: pd.DataFrame,
    lags:            list[int],
    horizons:        list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """
    Merge gravity + prices and build features / targets.

    Features (past only, no leakage):
      gravity_lag_Nd  : gravity score N days before t
      gravity_roll_Kd : K-day rolling mean of gravity (computed from t-1 backwards)

    Targets (future):
      return_Nd : forward return = (price[t+N] - price[t]) / price[t] * 100

    Missing prices (weekends, holidays) are forward-filled from the
    last available quote.
    """
    # Continuous calendar
    full_idx = pd.date_range(
        min(gravity_df["date"].min(), financial_df["date"].min()),
        max(gravity_df["date"].max(), financial_df["date"].max()),
        freq="D",
    )

    fin  = financial_df.set_index("date")["price"].reindex(full_idx).ffill()
    grav = gravity_df.set_index("date")["gravity_score"].reindex(full_idx)

    df = pd.DataFrame({
        "date":          full_idx,
        "gravity_score": grav.values,
        "price":         fin.values,
    })

    # Lagged gravity (features)
    for lag in lags:
        df[f"gravity_lag_{lag}d"] = df["gravity_score"].shift(lag)

    # Rolling means (shift(1) to avoid same-day leakage)
    for w in rolling_windows:
        df[f"gravity_roll_{w}d"] = df["gravity_score"].shift(1).rolling(w).mean()

    # Rendements forward (targets)
    for h in horizons:
        df[f"return_{h}d"] = (df["price"].shift(-h) / df["price"] - 1) * 100.0

    # Keep only days with gravity available
    df = df[df["gravity_score"].notna()].copy().reset_index(drop=True)

    feat_cols = (
        [f"gravity_lag_{l}d" for l in lags] +
        [f"gravity_roll_{w}d" for w in rolling_windows]
    )
    tgt_cols = [f"return_{h}d" for h in horizons]

    log.info(
        "Continuous dataset: %d days | %d features | %d target horizons",
        len(df), len(feat_cols), len(tgt_cols),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-correlogram
# ════════════════════════════════════════════════════════════════════════════

def run_cross_correlation(
    dataset:  pd.DataFrame,
    lags:     list[int],
    horizons: list[int],
) -> pd.DataFrame:
    """
    Compute corr(gravity[t - lag], return[t + horizon]) for every
    (lag, horizon) combination.

    Read as: "does the gravity score from LAG days ago predict the
    return in HORIZON days?"

    Returns a DataFrame with pearson_r, pearson_p, spearman_r, spearman_p,
    n_obs, and a 5% significance flag (*).
    """
    rows = []
    for lag in lags:
        feat = f"gravity_lag_{lag}d"
        if feat not in dataset.columns:
            continue
        for h in horizons:
            tgt = f"return_{h}d"
            if tgt not in dataset.columns:
                continue
            sub = dataset[[feat, tgt]].dropna()
            n = len(sub)
            if n < 5:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r,  pearson_p  = stats.pearsonr(sub[feat],  sub[tgt])
                spearman_r, spearman_p = stats.spearmanr(sub[feat], sub[tgt])

            rows.append({
                "gravity_lag_j":  lag,
                "horizon_j":      h,
                "pearson_r":      round(float(pearson_r),  4) if not np.isnan(pearson_r)  else np.nan,
                "pearson_p":      round(float(pearson_p),  4) if not np.isnan(pearson_p)  else np.nan,
                "spearman_r":     round(float(spearman_r), 4) if not np.isnan(spearman_r) else np.nan,
                "spearman_p":     round(float(spearman_p), 4) if not np.isnan(spearman_p) else np.nan,
                "n_obs":          n,
                "sig_5pct":       "*" if (not np.isnan(float(pearson_p or 1)) and float(pearson_p or 1) < 0.05) else "",
            })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — XGBoost on the continuous series
# ════════════════════════════════════════════════════════════════════════════

def train_xgboost_continuous(
    dataset:         pd.DataFrame,
    target_col:      str,
    lags:            list[int],
    rolling_windows: list[int],
    n_splits:        int = 5,
) -> tuple:
    """
    XGBoost with TimeSeriesSplit on the continuous time series.

    Features: gravity_lag_Nd + gravity_roll_Kd
    Target  : return_Nd (forward return)

    Returns (model_final, metrics_dict, feat_importance_df, oof_predictions_df).
    Out-of-fold (OOF) predictions allow a clean backtest without
    data leakage: each prediction is made on data never seen
    during training.
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        raise ImportError("Required: pip install xgboost scikit-learn")

    features = (
        [f"gravity_lag_{lag}d"  for lag in lags          if f"gravity_lag_{lag}d"  in dataset.columns] +
        [f"gravity_roll_{w}d"   for w   in rolling_windows if f"gravity_roll_{w}d" in dataset.columns]
    )

    cols_needed = ["date", "gravity_score"] + features + [target_col]
    sub = dataset[cols_needed].dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 20:
        raise ValueError(
            f"Not enough data ({len(sub)} rows after dropna). "
            f"Widen the period or reduce the lags."
        )

    X         = sub[features].values
    y         = sub[target_col].values
    dates     = sub["date"].values
    gravities = sub["gravity_score"].values

    n_splits = max(2, min(n_splits, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info(
        "=== Continuous XGBoost | target=%s | %d features | %d obs | %d folds",
        target_col, len(features), len(X), n_splits,
    )

    oof_preds  = np.full(len(y), np.nan)
    rmse_scores, mae_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        m = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[test_idx])
        oof_preds[test_idx] = preds

        rmse = float(np.sqrt(mean_squared_error(y[test_idx], preds)))
        mae  = float(mean_absolute_error(y[test_idx], preds))
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        log.info(
            "  Fold %d/%d : train=%d test=%d | RMSE=%.3f%% MAE=%.3f%%",
            fold + 1, n_splits, len(train_idx), len(test_idx), rmse, mae,
        )

    # Final model on the full dataset
    final_model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    final_model.fit(X, y)

    metrics = {
        "target":    target_col,
        "n_obs":     len(X),
        "n_splits":  n_splits,
        "features":  features,
        "rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "rmse_std":  round(float(np.std(rmse_scores)),  4),
        "mae_mean":  round(float(np.mean(mae_scores)),  4),
        "mae_std":   round(float(np.std(mae_scores)),   4),
    }

    feat_imp = pd.DataFrame({
        "feature":    features,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # OOF DataFrame: predicted vs realised (test folds only)
    mask = ~np.isnan(oof_preds)
    oof_df = pd.DataFrame({
        "date":       pd.to_datetime(dates[mask]),
        "gravity":    np.round(gravities[mask], 1),
        f"pred_{target_col}": np.round(oof_preds[mask], 3),
        f"actual_{target_col}": np.round(y[mask], 3),
        "error_%":   np.round(oof_preds[mask] - y[mask], 3),
    })

    log.info(
        "Final XGBoost: RMSE=%.3f+-%.3f%% | MAE=%.3f+-%.3f%% | OOF: %d/%d days",
        metrics["rmse_mean"], metrics["rmse_std"],
        metrics["mae_mean"],  metrics["mae_std"],
        mask.sum(), len(X),
    )
    return final_model, metrics, feat_imp, oof_df


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation continue gravity score -> rendements financiers"
    )

    # Inputs
    parser.add_argument("--gravity",   type=Path, required=True,
                        help="Parquet or CSV with columns date + gravity_score")
    parser.add_argument("--financial", type=Path, required=True,
                        help="CSV with daily historical prices")
    parser.add_argument("--price-col", default="Close",
                        help="Price column in the financial CSV (default: Close)")
    parser.add_argument("--date-col",  default="Date",
                        help="Date column in the financial CSV (default: Date)")
    parser.add_argument("--start",     help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",       help="End date (YYYY-MM-DD)")

    # Feature engineering
    parser.add_argument("--lags",     type=int, nargs="+", default=DEFAULT_LAGS,
                        help="Gravity score lags in days (default: 1 3 5 7 14 21)")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                        help="Forward return horizons in days (default: 5 10 15 30)")
    parser.add_argument("--rolling",  type=int, nargs="+", default=DEFAULT_ROLLING,
                        help="Gravity rolling-mean windows (default: 7 14 21)")

    # XGBoost
    parser.add_argument("--target",   default="return_10d",
                        help="XGBoost target column (default: return_10d)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="TimeSeriesSplit folds (default: 5)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save-model", action="store_true",
                        help="Save the XGBoost model (.json)")

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_label = args.financial.stem.upper()

    # ── STEP 1 ────────────────────────────────────────────────────────────
    log.info("=== STEP 1: Loading the series ===")
    gravity_df   = load_gravity(args.gravity, args.start, args.end)
    financial_df = load_financial(args.financial, args.date_col, args.price_col)

    # ── STEP 2 ────────────────────────────────────────────────────────────
    log.info("=== STEP 2: Feature engineering ===")
    dataset = build_continuous_dataset(
        gravity_df, financial_df,
        lags=args.lags, horizons=args.horizons, rolling_windows=args.rolling,
    )

    dataset_path = args.output_dir / f"continuous_dataset_{asset_label}.csv"
    dataset.to_csv(dataset_path, index=False)
    log.info("Dataset saved -> %s", dataset_path)

    # ── STEP 3: cross-correlogram ─────────────────────────────────────────
    log.info("=== STEP 3: Cross-correlogram ===")
    corr_df = run_cross_correlation(dataset, lags=args.lags, horizons=args.horizons)

    print(f"\n{'='*72}")
    print(f"  CROSS-CORRELOGRAMME  gravity_score(t-lag) -> return(t+horizon)")
    print(f"  Asset : {asset_label}  |  * = significant at 5%")
    print(f"{'='*72}")
    print(corr_df.to_string(index=False))

    corr_path = args.output_dir / f"crosscorr_{asset_label}.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info("Cross-correlogram saved -> %s", corr_path)

    # ── STEP 4: XGBoost ───────────────────────────────────────────────────
    if args.target not in dataset.columns:
        available = [c for c in dataset.columns if c.startswith("return_")]
        log.error("Target '%s' missing. Available: %s", args.target, available)
        sys.exit(1)

    try:
        xgb_model, metrics, feat_imp, oof_df = train_xgboost_continuous(
            dataset=dataset,
            target_col=args.target,
            lags=args.lags,
            rolling_windows=args.rolling,
            n_splits=args.n_splits,
        )

        print(f"\n{'='*72}")
        print(f"  CONTINUOUS XGBOOST | target: {args.target} | asset: {asset_label}")
        print(f"{'='*72}")
        print(f"  Observations  : {metrics['n_obs']}")
        print(f"  RMSE (OOF CV) : {metrics['rmse_mean']:.3f} +- {metrics['rmse_std']:.3f} %")
        print(f"  MAE  (OOF CV) : {metrics['mae_mean']:.3f} +- {metrics['mae_std']:.3f} %")
        print(f"\n  Feature importance :")
        for _, row in feat_imp.iterrows():
            bar = "#" * int(row["importance"] * 40)
            print(f"    {row['feature']:25s} {row['importance']:.4f}  {bar}")

        # Statistiques OOF
        errs = oof_df["error_%"].dropna()
        mae_oof  = float(errs.abs().mean())
        rmse_oof = float(np.sqrt((errs**2).mean()))
        n_ok     = int((errs.abs() < 1.5).sum())

        print(f"\n  OOF backtest ({len(oof_df)} days):")
        print(f"    MAE  = {mae_oof:.3f}%  |  RMSE = {rmse_oof:.3f}%")
        print(f"    |error| < 1.5% : {n_ok}/{len(oof_df)} days ({100*n_ok/len(oof_df):.0f}%)")

        # Save
        metrics_path = args.output_dir / f"xgb_metrics_{asset_label}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        oof_path = args.output_dir / f"oof_predictions_{asset_label}.csv"
        oof_df.to_csv(oof_path, index=False)
        log.info("OOF predictions saved -> %s", oof_path)

        if args.save_model:
            try:
                mp = args.output_dir / f"xgb_{asset_label}_{args.target}.json"
                xgb_model.save_model(str(mp))
                log.info("XGBoost model saved -> %s", mp)
            except Exception as exc:
                log.warning("Could not save the model: %s", exc)

        print(f"{'='*72}")

    except ValueError as exc:
        log.error("XGBoost impossible : %s", exc)

    print(f"\nResults -> {args.output_dir}/")


if __name__ == "__main__":
    main()
