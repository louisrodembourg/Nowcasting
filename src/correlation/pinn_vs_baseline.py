"""
Phase 4 — PINN features add-on: est-ce que rho_pred + TTC améliorent les prédictions ?

Compare deux jeux de features en walk-forward OOS pour les cibles BDRY et SBLK :
  A) Baseline  : gravity + AIS bruts uniquement
  B) PINN-aug  : baseline + rho_pred (PINN) + ttc_days + in_episode flag

Les TTC sont disponibles uniquement pendant les fenêtres d'épisodes PINN
(90j autour du pic annuel de congestion, années 2019–2022).

Usage:
    python src/correlation/pinn_vs_baseline.py
"""

import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
GRAVITY_PATH = ROOT / "data/features/la_gravity_2017_2022.parquet"
EQUITY_PATH  = ROOT / "data/financial/equity_proxies_wide.parquet"
TTC_DIR      = ROOT / "outputs/pinn_yearly"
OUT_DIR      = ROOT / "outputs/financial"


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_gravity() -> pd.DataFrame:
    df = (
        pl.read_parquet(GRAVITY_PATH)
        .with_columns(pl.col("date").str.to_date().alias("date"))
        .sort("date")
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_ttc_series(years=(2019, 2020, 2021, 2022)) -> pd.DataFrame:
    """
    Concat les fichiers TTC PINN par année.
    Colonnes utiles : rho_pred, v_pred, time_to_clear_days, in_episode (flag).
    En dehors des fenêtres d'épisodes, tout est NaN.
    """
    frames = []
    for pattern in [TTC_DIR / f"la_{y}_time_to_clear.parquet" for y in years]:
        if not pattern.exists():
            continue
        df = pl.read_parquet(pattern).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
        log.info("TTC %s: %d jours", pattern.stem, len(df))

    if not frames:
        return pd.DataFrame()

    ttc = pd.concat(frames).sort_values("date").drop_duplicates("date").set_index("date")
    ttc["in_episode"] = 1.0
    return ttc[["rho_pred", "v_pred", "time_to_clear_days", "in_episode"]]


def load_prices(ticker: str) -> pd.Series:
    """Charge depuis le parquet cache ou yfinance."""
    if ticker in ("BDRY", "MATX", "DAC", "CMRE", "GSL"):
        cached = (
            pl.read_parquet(EQUITY_PATH)
            .with_columns(pl.col("date").str.to_date().alias("date"))
            .to_pandas()
        )
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")
        if ticker in cached.columns:
            return cached[ticker].dropna()

    log.info("yfinance fetch: %s", ticker)
    raw = yf.download(ticker, start="2017-01-01", end="2023-01-01",
                      progress=False, auto_adjust=True)["Close"].squeeze()
    raw.index = pd.to_datetime(raw.index)
    return raw.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_baseline_features(gravity: pd.DataFrame) -> pd.DataFrame:
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
    return f.dropna(subset=["gravity_zscore60"])


def build_pinn_features(gravity_features: pd.DataFrame, ttc: pd.DataFrame) -> pd.DataFrame:
    """Fusionne baseline + PINN features (sparse)."""
    aug = gravity_features.copy()

    # Merge PINN data — inner reindex sur l'index gravity (quotidien)
    full_idx = gravity_features.index
    ttc_reindexed = ttc.reindex(full_idx)

    aug["pinn_rho"]        = ttc_reindexed["rho_pred"]
    aug["pinn_ttc"]        = ttc_reindexed["time_to_clear_days"].astype(float)
    aug["pinn_v"]          = ttc_reindexed["v_pred"]
    aug["in_episode"]      = ttc_reindexed["in_episode"].fillna(0.0)

    # TTC normalisé (0 quand hors épisode, valeur réelle pendant l'épisode)
    # On représente "hors épisode" par TTC = 0 pour donner un signal continu au modèle
    aug["pinn_ttc_filled"] = aug["pinn_ttc"].fillna(0.0)
    aug["pinn_rho_filled"] = aug["pinn_rho"].fillna(aug["pinn_rho"].mean())

    return aug


BASELINE_COLS = [
    "log_gravity", "gravity_ma7", "gravity_ma21",
    "gravity_zscore60", "gravity_delta5",
    "log_waiting", "log_capacity", "congestion_flag",
]

PINN_EXTRA_COLS = [
    "pinn_rho_filled", "pinn_ttc_filled", "pinn_v", "in_episode",
]

PINN_COLS = BASELINE_COLS + PINN_EXTRA_COLS


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def forward_return(prices: pd.Series, h: int) -> pd.Series:
    return np.log(prices.shift(-h) / prices)


def walk_forward(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
                 init_train=252, test_block=63, alpha=1.0) -> dict:
    n = len(X)
    oos_pred, oos_actual, oos_dates = [], [], []
    start = init_train
    while start < n:
        end = min(start + test_block, n)
        Xtr, ytr = X[:start], y[:start]
        Xte, yte = X[start:end], y[start:end]
        mask_tr = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
        mask_te = ~(np.isnan(Xte).any(1) | np.isnan(yte))
        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            start += test_block; continue
        sc = StandardScaler()
        m  = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(Xtr[mask_tr]), ytr[mask_tr])
        preds = m.predict(sc.transform(Xte[mask_te]))
        oos_pred.extend(preds)
        oos_actual.extend(yte[mask_te])
        oos_dates.extend(dates[start:end][mask_te])
        start += test_block
    return {
        "pred":   np.array(oos_pred),
        "actual": np.array(oos_actual),
        "dates":  pd.DatetimeIndex(oos_dates),
    }


def walk_forward_xgb(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
                     init_train=252, test_block=63) -> dict:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {}
    n = len(X)
    oos_pred, oos_actual, oos_dates = [], [], []
    start = init_train
    while start < n:
        end = min(start + test_block, n)
        Xtr, ytr = X[:start], y[:start]
        Xte, yte = X[start:end], y[start:end]
        mask_tr = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
        mask_te = ~(np.isnan(Xte).any(1) | np.isnan(yte))
        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            start += test_block; continue
        m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0)
        m.fit(Xtr[mask_tr], ytr[mask_tr])
        oos_pred.extend(m.predict(Xte[mask_te]))
        oos_actual.extend(yte[mask_te])
        oos_dates.extend(dates[start:end][mask_te])
        start += test_block
    return {"pred": np.array(oos_pred), "actual": np.array(oos_actual),
            "dates": pd.DatetimeIndex(oos_dates)}


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def metrics(pred, actual, label=""):
    valid = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[valid], actual[valid]
    if len(p) < 10:
        return {"label": label, "n": len(p)}
    ic_p, pv_p   = stats.pearsonr(p, a)
    ic_s, pv_s   = stats.spearmanr(p, a)
    dir_acc      = np.mean(np.sign(p) == np.sign(a))
    rmse         = float(np.sqrt(np.mean((p - a)**2)))
    ls           = np.where(p > np.median(p), a, -a)
    sharpe       = float(np.mean(ls) / (np.std(ls) + 1e-10) * np.sqrt(252))
    return {
        "label":       label,
        "n_oos":       len(p),
        "IC_pearson":  round(float(ic_p), 4),
        "p_pearson":   round(float(pv_p), 4),
        "IC_spearman": round(float(ic_s), 4),
        "p_spearman":  round(float(pv_s), 4),
        "dir_acc":     round(float(dir_acc), 4),
        "rmse":        round(rmse, 6),
        "ls_sharpe":   round(float(sharpe), 4),
    }


def delta_ic(m_aug, m_base, key="IC_spearman"):
    """Différence IC augmenté vs baseline, en absolu et en %."""
    base = m_base.get(key, np.nan)
    aug  = m_aug.get(key, np.nan)
    if np.isnan(base) or np.isnan(aug):
        return np.nan, np.nan
    diff = aug - base
    pct  = diff / (abs(base) + 1e-10) * 100
    return round(diff, 4), round(pct, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Chargement gravity...")
    gravity  = load_gravity()
    baseline = build_baseline_features(gravity)

    log.info("Chargement TTC PINN 2019–2022...")
    ttc      = load_ttc_series(years=(2019, 2020, 2021, 2022))
    aug_feat = build_pinn_features(baseline, ttc)

    log.info("Jours en épisode PINN: %d / %d total", int(aug_feat["in_episode"].sum()), len(aug_feat))
    log.info("TTC non-null: %d jours", aug_feat["pinn_ttc"].notna().sum())

    targets = {
        "BDRY": "Breakwave Dry Bulk ETF",
        "SBLK": "Star Bulk Carriers",
        "MATX": "Matson Inc",
    }
    horizons = [5, 10, 21]

    all_rows = []

    for ticker, desc in targets.items():
        prices = load_prices(ticker)
        log.info("\n%s %s (%s)", "═"*55, ticker, desc)

        for h in horizons:
            fwd = forward_return(prices, h)

            # ── Baseline ─────────────────────────────────────────────────
            merged_b = baseline[BASELINE_COLS].join(
                fwd.rename("fwd"), how="inner").dropna(subset=["fwd"])
            Xb = merged_b[BASELINE_COLS].values
            yb = merged_b["fwd"].values
            db = merged_b.index

            ridge_b = walk_forward(Xb, yb, db)
            xgb_b   = walk_forward_xgb(Xb, yb, db)

            mb_ridge = metrics(ridge_b["pred"], ridge_b["actual"], f"Ridge·base|{ticker}|h{h}")
            mb_xgb   = metrics(xgb_b["pred"],  xgb_b["actual"],   f"XGB·base|{ticker}|h{h}")

            # ── PINN-augmenté ─────────────────────────────────────────────
            merged_a = aug_feat[PINN_COLS].join(
                fwd.rename("fwd"), how="inner").dropna(subset=["fwd"])
            Xa = merged_a[PINN_COLS].values
            ya = merged_a["fwd"].values
            da = merged_a.index

            ridge_a = walk_forward(Xa, ya, da)
            xgb_a   = walk_forward_xgb(Xa, ya, da)

            ma_ridge = metrics(ridge_a["pred"], ridge_a["actual"], f"Ridge·PINN|{ticker}|h{h}")
            ma_xgb   = metrics(xgb_a["pred"],  xgb_a["actual"],   f"XGB·PINN|{ticker}|h{h}")

            # ── Calcul du delta ───────────────────────────────────────────
            diff_ridge_ic, diff_ridge_pct = delta_ic(ma_ridge, mb_ridge)
            diff_xgb_ic,   diff_xgb_pct  = delta_ic(ma_xgb,   mb_xgb)
            diff_ridge_sh, _              = delta_ic(ma_ridge, mb_ridge, "ls_sharpe")
            diff_xgb_sh,   _             = delta_ic(ma_xgb,   mb_xgb,   "ls_sharpe")

            log.info(
                "  h=%2dd Ridge | base IC_s=%.3f → PINN IC_s=%.3f | Δ=%+.3f (%+.1f%%) | "
                "Sharpe Δ=%+.3f",
                h,
                mb_ridge.get("IC_spearman", np.nan),
                ma_ridge.get("IC_spearman", np.nan),
                diff_ridge_ic if not np.isnan(diff_ridge_ic) else 0,
                diff_ridge_pct if not np.isnan(diff_ridge_pct) else 0,
                diff_ridge_sh if not np.isnan(diff_ridge_sh) else 0,
            )
            log.info(
                "  h=%2dd XGB   | base IC_s=%.3f → PINN IC_s=%.3f | Δ=%+.3f (%+.1f%%) | "
                "Sharpe Δ=%+.3f",
                h,
                mb_xgb.get("IC_spearman", np.nan),
                ma_xgb.get("IC_spearman", np.nan),
                diff_xgb_ic if not np.isnan(diff_xgb_ic) else 0,
                diff_xgb_pct if not np.isnan(diff_xgb_pct) else 0,
                diff_xgb_sh if not np.isnan(diff_xgb_sh) else 0,
            )

            for m_base, m_aug, model_name in [
                (mb_ridge, ma_ridge, "Ridge"),
                (mb_xgb,   ma_xgb,   "XGBoost"),
            ]:
                d_ic, d_pct = delta_ic(m_aug, m_base)
                d_sh, _     = delta_ic(m_aug, m_base, "ls_sharpe")
                all_rows.append({
                    "target":            ticker,
                    "horizon":           h,
                    "model":             model_name,
                    "IC_s_base":         m_base.get("IC_spearman", np.nan),
                    "IC_s_pinn":         m_aug.get("IC_spearman",  np.nan),
                    "delta_IC_s":        d_ic,
                    "delta_IC_s_pct":    d_pct,
                    "sharpe_base":       m_base.get("ls_sharpe", np.nan),
                    "sharpe_pinn":       m_aug.get("ls_sharpe",  np.nan),
                    "delta_sharpe":      d_sh,
                    "p_base":            m_base.get("p_spearman", np.nan),
                    "p_pinn":            m_aug.get("p_spearman",  np.nan),
                    "n_oos_base":        m_base.get("n_oos", 0),
                    "n_oos_pinn":        m_aug.get("n_oos", 0),
                })

    df = pd.DataFrame(all_rows)

    print("\n" + "="*100)
    print("COMPARAISON : Baseline (AIS brut) vs. PINN-augmenté (rho + TTC)")
    print("="*100)
    print(f"{'Target':<6} {'H':>4} {'Model':<8} "
          f"{'IC_s Base':>10} {'IC_s PINN':>10} {'Δ IC_s':>8} {'Δ%':>6} "
          f"{'Sharpe Base':>12} {'Sharpe PINN':>12} {'Δ Sharpe':>9} "
          f"{'p_base':>7} {'p_pinn':>7}")
    print("─"*100)

    for _, r in df.sort_values(["target", "horizon", "model"]).iterrows():
        better = " ▲" if r["delta_IC_s"] > 0.005 else (" ▼" if r["delta_IC_s"] < -0.005 else "  ")
        print(
            f"{r['target']:<6} {int(r['horizon']):>4} {r['model']:<8} "
            f"{r['IC_s_base']:>10.4f} {r['IC_s_pinn']:>10.4f} "
            f"{r['delta_IC_s']:>+8.4f} {r['delta_IC_s_pct']:>+5.1f}% "
            f"{r['sharpe_base']:>12.3f} {r['sharpe_pinn']:>12.3f} "
            f"{r['delta_sharpe']:>+9.3f} "
            f"{r['p_base']:>7.3f} {r['p_pinn']:>7.3f}{better}"
        )

    print("="*100)
    print("▲ = PINN améliore IC_s de >0.005  |  ▼ = PINN détériore IC_s de >0.005")

    out = OUT_DIR / "pinn_vs_baseline.csv"
    df.to_csv(out, index=False)
    log.info("Résultats sauvegardés -> %s", out)

    # ── Corrélation directe de rho_pred et TTC avec les rendements ───────────
    print("\n" + "═"*70)
    print("CORRÉLATION DIRECTE : rho_pred + TTC vs. rendements (lag scan)")
    print("═"*70)
    print("(sur les ~360 jours d'épisodes PINN disponibles 2019–2022)\n")

    for ticker in ["BDRY", "SBLK", "MATX"]:
        prices = load_prices(ticker)
        for h in [10, 21]:
            fwd = forward_return(prices, h)
            combined = aug_feat[["pinn_rho", "pinn_ttc", "in_episode"]].join(
                fwd.rename("fwd"), how="inner").dropna()
            episode_only = combined[combined["in_episode"] == 1].dropna()
            if len(episode_only) < 20:
                continue
            r_rho, p_rho = stats.spearmanr(episode_only["pinn_rho"], episode_only["fwd"])
            r_ttc, p_ttc = stats.spearmanr(episode_only["pinn_ttc"], episode_only["fwd"])
            print(
                f"{ticker} h={h:2d}d | episode days={len(episode_only):3d} | "
                f"rho_pred IC_s={r_rho:+.3f} (p={p_rho:.3f}) | "
                f"TTC IC_s={r_ttc:+.3f} (p={p_ttc:.3f})"
            )
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
