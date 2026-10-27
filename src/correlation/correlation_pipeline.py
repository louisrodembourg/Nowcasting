"""
correlation_pipeline.py — Analyse de correlation macroeconomique continue.

Approche : serie temporelle continue (ex. 2021-2024), sans episodes ni PINN.
Le Gravity Score quotidien (issu de manifold_pipeline.py) est utilise comme
signal avance des rendements financiers futurs.

Etape 1 : Chargement et fusion des series continues (gravity + prix financiers)
Etape 2 : Feature engineering (lags, rolling means, rendements forward)
Etape 3 : Cross-correlogramme Pearson/Spearman (lag 0 -> max_lag)
Etape 4 : XGBoost continu avec TimeSeriesSplit (out-of-fold predictions)

Commandes de lancement (PowerShell, une seule ligne) :

  Houston 2021-2024 avec BOAT ETF :
    python src/correlation/correlation_pipeline.py --gravity data/features/houston_gravity_daily.parquet --financial data/financial/BOAT.csv --start 2021-08-01 --end 2024-11-17 --lags 1 3 5 7 14 21 --horizons 5 10 15 30 --target return_10d

  LA 2021-2024 avec S&P 500 :
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
# ETAPE 1 — Chargement
# ════════════════════════════════════════════════════════════════════════════

def load_gravity(
    path: Path,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Charge le gravity score quotidien (Parquet ou CSV).
    Colonnes attendues : date, gravity_score.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    # Detecte la colonne date (insensible a la casse)
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "ds")),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    if "gravity_score" not in df.columns:
        raise ValueError(
            f"Colonne 'gravity_score' introuvable dans {path}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df = df[["date", "gravity_score"]].dropna().sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    log.info(
        "Gravity score : %d jours | %s -> %s | non-zero : %d jours",
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
    """Charge un CSV de prix financiers. Retourne date + price."""
    df = pd.read_csv(path, parse_dates=[date_col])
    if price_col not in df.columns:
        available = [c for c in df.columns if c != date_col]
        raise ValueError(
            f"Colonne '{price_col}' introuvable dans {path}. "
            f"Colonnes disponibles : {available}"
        )
    df = df[[date_col, price_col]].dropna()
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "Donnees financieres : %d jours | %s -> %s | col='%s'",
        len(df),
        df["date"].min().date(), df["date"].max().date(), price_col,
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# ETAPE 2 — Feature engineering
# ════════════════════════════════════════════════════════════════════════════

def build_continuous_dataset(
    gravity_df:  pd.DataFrame,
    financial_df: pd.DataFrame,
    lags:            list[int],
    horizons:        list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """
    Fusionne gravity + prix et construit features / targets.

    Features (passees, pas de leakage) :
      gravity_lag_Nd  : gravity score N jours avant t
      gravity_roll_Kd : moyenne mobile du gravity sur K jours (calculee sur t-1 en arriere)

    Targets (futures) :
      return_Nd : rendement forward = (price[t+N] - price[t]) / price[t] * 100

    Les prix manquants (week-ends, feries) sont forward-filles depuis la
    derniere cotation disponible.
    """
    # Calendrier continu
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

    # Rolling means (utilise shift(1) pour eviter le leakage du jour J)
    for w in rolling_windows:
        df[f"gravity_roll_{w}d"] = df["gravity_score"].shift(1).rolling(w).mean()

    # Rendements forward (targets)
    for h in horizons:
        df[f"return_{h}d"] = (df["price"].shift(-h) / df["price"] - 1) * 100.0

    # Garder uniquement les jours avec gravity disponible
    df = df[df["gravity_score"].notna()].copy().reset_index(drop=True)

    feat_cols = (
        [f"gravity_lag_{l}d" for l in lags] +
        [f"gravity_roll_{w}d" for w in rolling_windows]
    )
    tgt_cols = [f"return_{h}d" for h in horizons]

    log.info(
        "Dataset continu : %d jours | %d features | %d horizons cibles",
        len(df), len(feat_cols), len(tgt_cols),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# ETAPE 3 — Cross-correlogramme
# ════════════════════════════════════════════════════════════════════════════

def run_cross_correlation(
    dataset:  pd.DataFrame,
    lags:     list[int],
    horizons: list[int],
) -> pd.DataFrame:
    """
    Calcule corr(gravity[t - lag], return[t + horizon]) pour toutes les
    combinaisons (lag, horizon).

    Interprete comme : "le gravity score d'il y a LAG jours predit-il le
    rendement dans HORIZON jours ?"

    Retourne un DataFrame avec pearson_r, pearson_p, spearman_r, spearman_p,
    n_obs, et un indicateur de significativite a 5% (*).
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
# ETAPE 4 — XGBoost sur serie continue
# ════════════════════════════════════════════════════════════════════════════

def train_xgboost_continuous(
    dataset:         pd.DataFrame,
    target_col:      str,
    lags:            list[int],
    rolling_windows: list[int],
    n_splits:        int = 5,
) -> tuple:
    """
    XGBoost avec TimeSeriesSplit sur la serie temporelle continue.

    Features : gravity_lag_Nd + gravity_roll_Kd
    Target   : return_Nd (rendement forward)

    Retourne (model_final, metrics_dict, feat_importance_df, oof_predictions_df).
    Les predictions out-of-fold (OOF) permettent un backtest propre sans
    data leakage : chaque prediction est faite sur des donnees jamais vues
    lors de l'entrainement.
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        raise ImportError("Requis : pip install xgboost scikit-learn")

    features = (
        [f"gravity_lag_{lag}d"  for lag in lags          if f"gravity_lag_{lag}d"  in dataset.columns] +
        [f"gravity_roll_{w}d"   for w   in rolling_windows if f"gravity_roll_{w}d" in dataset.columns]
    )

    cols_needed = ["date", "gravity_score"] + features + [target_col]
    sub = dataset[cols_needed].dropna().sort_values("date").reset_index(drop=True)

    if len(sub) < 20:
        raise ValueError(
            f"Pas assez de donnees ({len(sub)} lignes apres dropna). "
            f"Elargissez la periode ou reduisez les lags."
        )

    X         = sub[features].values
    y         = sub[target_col].values
    dates     = sub["date"].values
    gravities = sub["gravity_score"].values

    n_splits = max(2, min(n_splits, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info(
        "=== XGBoost continu | target=%s | %d features | %d obs | %d folds",
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

    # Modele final sur l'ensemble des donnees
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

    # DataFrame OOF : predictions vs realise (uniquement les folds de test)
    mask = ~np.isnan(oof_preds)
    oof_df = pd.DataFrame({
        "date":       pd.to_datetime(dates[mask]),
        "gravity":    np.round(gravities[mask], 1),
        f"pred_{target_col}": np.round(oof_preds[mask], 3),
        f"reel_{target_col}": np.round(y[mask], 3),
        "erreur_%":   np.round(oof_preds[mask] - y[mask], 3),
    })

    log.info(
        "XGBoost final : RMSE=%.3f+-%.3f%% | MAE=%.3f+-%.3f%% | OOF : %d/%d jours",
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
                        help="Parquet ou CSV avec colonnes date + gravity_score")
    parser.add_argument("--financial", type=Path, required=True,
                        help="CSV avec prix historiques quotidiens")
    parser.add_argument("--price-col", default="Close",
                        help="Colonne prix dans le CSV financier (defaut: Close)")
    parser.add_argument("--date-col",  default="Date",
                        help="Colonne date dans le CSV financier (defaut: Date)")
    parser.add_argument("--start",     help="Date de debut (YYYY-MM-DD)")
    parser.add_argument("--end",       help="Date de fin (YYYY-MM-DD)")

    # Feature engineering
    parser.add_argument("--lags",     type=int, nargs="+", default=DEFAULT_LAGS,
                        help="Lags du gravity score en jours (defaut: 1 3 5 7 14 21)")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                        help="Horizons de rendement forward en jours (defaut: 5 10 15 30)")
    parser.add_argument("--rolling",  type=int, nargs="+", default=DEFAULT_ROLLING,
                        help="Fenetres de rolling mean du gravity (defaut: 7 14 21)")

    # XGBoost
    parser.add_argument("--target",   default="return_10d",
                        help="Colonne cible XGBoost (defaut: return_10d)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Folds TimeSeriesSplit (defaut: 5)")

    # Sortie
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save-model", action="store_true",
                        help="Sauvegarder le modele XGBoost (.json)")

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

    # ── ETAPE 1 ───────────────────────────────────────────────────────────
    log.info("=== ETAPE 1 : Chargement des series ===")
    gravity_df   = load_gravity(args.gravity, args.start, args.end)
    financial_df = load_financial(args.financial, args.date_col, args.price_col)

    # ── ETAPE 2 ───────────────────────────────────────────────────────────
    log.info("=== ETAPE 2 : Feature engineering ===")
    dataset = build_continuous_dataset(
        gravity_df, financial_df,
        lags=args.lags, horizons=args.horizons, rolling_windows=args.rolling,
    )

    dataset_path = args.output_dir / f"continuous_dataset_{asset_label}.csv"
    dataset.to_csv(dataset_path, index=False)
    log.info("Dataset sauvegarde -> %s", dataset_path)

    # ── ETAPE 3 : Cross-correlogramme ─────────────────────────────────────
    log.info("=== ETAPE 3 : Cross-correlogramme ===")
    corr_df = run_cross_correlation(dataset, lags=args.lags, horizons=args.horizons)

    print(f"\n{'='*72}")
    print(f"  CROSS-CORRELOGRAMME  gravity_score(t-lag) -> return(t+horizon)")
    print(f"  Actif : {asset_label}  |  * = significatif a 5%")
    print(f"{'='*72}")
    print(corr_df.to_string(index=False))

    corr_path = args.output_dir / f"crosscorr_{asset_label}.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info("Cross-correlogramme sauvegarde -> %s", corr_path)

    # ── ETAPE 4 : XGBoost ─────────────────────────────────────────────────
    if args.target not in dataset.columns:
        available = [c for c in dataset.columns if c.startswith("return_")]
        log.error("Target '%s' absente. Disponibles : %s", args.target, available)
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
        print(f"  XGBOOST CONTINU | cible : {args.target} | actif : {asset_label}")
        print(f"{'='*72}")
        print(f"  Observations  : {metrics['n_obs']}")
        print(f"  RMSE (OOF CV) : {metrics['rmse_mean']:.3f} +- {metrics['rmse_std']:.3f} %")
        print(f"  MAE  (OOF CV) : {metrics['mae_mean']:.3f} +- {metrics['mae_std']:.3f} %")
        print(f"\n  Feature importance :")
        for _, row in feat_imp.iterrows():
            bar = "#" * int(row["importance"] * 40)
            print(f"    {row['feature']:25s} {row['importance']:.4f}  {bar}")

        # Statistiques OOF
        errs = oof_df["erreur_%"].dropna()
        mae_oof  = float(errs.abs().mean())
        rmse_oof = float(np.sqrt((errs**2).mean()))
        n_ok     = int((errs.abs() < 1.5).sum())

        print(f"\n  Backtest OOF ({len(oof_df)} jours) :")
        print(f"    MAE  = {mae_oof:.3f}%  |  RMSE = {rmse_oof:.3f}%")
        print(f"    |erreur| < 1.5% : {n_ok}/{len(oof_df)} jours ({100*n_ok/len(oof_df):.0f}%)")

        # Sauvegarde
        metrics_path = args.output_dir / f"xgb_metrics_{asset_label}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        oof_path = args.output_dir / f"oof_predictions_{asset_label}.csv"
        oof_df.to_csv(oof_path, index=False)
        log.info("Predictions OOF sauvegardees -> %s", oof_path)

        if args.save_model:
            try:
                mp = args.output_dir / f"xgb_{asset_label}_{args.target}.json"
                xgb_model.save_model(str(mp))
                log.info("Modele XGBoost sauvegarde -> %s", mp)
            except Exception as exc:
                log.warning("Sauvegarde modele impossible : %s", exc)

        print(f"{'='*72}")

    except ValueError as exc:
        log.error("XGBoost impossible : %s", exc)

    print(f"\nResultats -> {args.output_dir}/")


if __name__ == "__main__":
    main()
