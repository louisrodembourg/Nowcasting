"""
Phase 4 — Modèle conditionnel à deux régimes.

Architecture
============
Régime 1 (jours normaux) : Ridge entraîné sur les features AIS brutes
Régime 2 (jours d'épisode) : Ridge entraîné sur rho_pred PINN + context AIS

Détection d'épisode : CAUSALE — gravity_zscore60 > seuil pendant min_days consécutifs.
Aucun look-ahead sur le timing PINN.

rho_pred du PINN est jointé quand disponible (épisodes 2019–2022). Hors fenêtre PINN,
on utilise les features AIS seules pour les deux régimes.

Walk-forward
============
- Fenêtre d'entraînement initiale : 252 jours
- Blocs OOS : 63 jours (un trimestre)
- Expansion de la fenêtre d'entraînement à chaque bloc

Métriques
=========
IC_spearman, directional accuracy, Sharpe L/S annualisé
Comparaison : baseline (Ridge unique) vs. deux régimes

Usage
=====
    python src/correlation/two_regime_model.py
    python src/correlation/two_regime_model.py --target BDRY --horizon 21
"""

import argparse
import glob
import logging
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

ROOT         = Path(__file__).resolve().parents[2]
GRAVITY_PATH = ROOT / "data/features/la_gravity_2017_2022.parquet"
EQUITY_PATH  = ROOT / "data/financial/equity_proxies_wide.parquet"
TTC_DIR      = ROOT / "outputs/pinn_yearly"
OUT_DIR      = ROOT / "outputs/financial"

# ── Colonnes features ─────────────────────────────────────────────────────────

REGIME1_COLS = [
    "log_gravity",
    "gravity_ma7",
    "gravity_ma21",
    "gravity_zscore60",
    "gravity_delta5",
    "log_waiting",
    "log_capacity",
    "congestion_flag",
]

REGIME2_COLS = [
    # Signal physique PINN — rho seul (delta et ma7 sont redondants sur 90j)
    "pinn_rho",
    # Context AIS
    "gravity_zscore60",
    "gravity_delta5",
    "log_waiting",
    "log_gravity",
]


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_gravity() -> pd.DataFrame:
    df = (
        pl.read_parquet(GRAVITY_PATH)
        .with_columns(pl.col("date").str.to_date())
        .sort("date")
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_pinn_rho(years=(2019, 2020, 2021, 2022)) -> pd.Series:
    """
    Série quotidienne de rho_pred PINN (dense sur les 90 jours de chaque épisode,
    NaN le reste du temps). rho_pred est toujours valide — contrairement à TTC
    qui est 0 ou null pour 2021 et 2022.
    """
    frames = []
    for y in years:
        path = TTC_DIR / f"la_{y}_time_to_clear.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df.set_index("date")["rho_pred"])
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames).sort_index()


def load_prices(ticker: str) -> pd.Series:
    if ticker in ("BDRY", "MATX", "DAC", "CMRE", "GSL"):
        cached = (
            pl.read_parquet(EQUITY_PATH)
            .with_columns(pl.col("date").str.to_date())
            .to_pandas()
        )
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")
        if ticker in cached.columns:
            return cached[ticker].dropna()
    raw = yf.download(ticker, start="2017-01-01", end="2023-01-01",
                      progress=False, auto_adjust=True)["Close"].squeeze()
    raw.index = pd.to_datetime(raw.index)
    return raw.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(gravity: pd.DataFrame, pinn_rho: pd.Series) -> pd.DataFrame:
    """
    Construit la matrice de features complète (régime 1 + régime 2 + flag d'épisode).

    Détection d'épisode : disponibilité de pinn_rho.
    NOTE : le PINN est entraîné sur la fenêtre de 90 jours complète (lookahead interne
    au modèle physique). rho_pred au début de la fenêtre "voit" donc des données
    futures. Ce biais est inhérent à l'utilisation d'un modèle PDE fitted offline.
    En production, le PINN serait ré-entraîné incrémentalement (hors scope ici).
    """
    f = pd.DataFrame(index=gravity.index)

    # ── Features de base (régime 1) ───────────────────────────────────────────
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

    # ── Feature PINN (régime 2) — rho seul, delta et ma7 redondants sur 90j ──
    rho_aligned   = pinn_rho.reindex(f.index)
    f["pinn_rho"] = rho_aligned

    # ── Indicateur de régime ──────────────────────────────────────────────────
    # Régime 2 = PINN rho disponible (le modèle physique est actif)
    # Régime 1 = jours hors fenêtre PINN (opération normale)
    f["in_episode"] = f["pinn_rho"].notna().astype(float)

    # Drop burn-in (60 premiers jours pour rolling stats)
    f = f.dropna(subset=["gravity_zscore60"])
    return f


def forward_return(prices: pd.Series, h: int) -> pd.Series:
    return np.log(prices.shift(-h) / prices)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE À DEUX RÉGIMES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeModels:
    r1: Ridge = field(default_factory=lambda: Ridge(alpha=1.0))
    r2: Ridge = field(default_factory=lambda: Ridge(alpha=0.5))
    s1: StandardScaler = field(default_factory=StandardScaler)
    s2: StandardScaler = field(default_factory=StandardScaler)
    r2_fitted: bool = False
    n_r1_train: int = 0
    n_r2_train: int = 0


def fit_two_regime(
    feat: pd.DataFrame,
    y: np.ndarray,
    regime1_cols: list[str],
    regime2_cols: list[str],
    models: RegimeModels | None = None,
) -> RegimeModels:
    if models is None:
        models = RegimeModels()

    in_ep = feat["in_episode"].values
    X1 = feat[regime1_cols].values
    X2 = feat[regime2_cols].values

    # ── Régime 1 : entraîné sur TOUS les jours (épisodes inclus)
    # Entraîner R1 uniquement sur les jours hors-épisode prive le modèle des
    # observations de pic de congestion — exactement les plus informatives.
    # R1 sert de fallback universel ; R2 override sur les jours d'épisode.
    mask1 = ~np.isnan(X1).any(1) & ~np.isnan(y)
    if mask1.sum() >= 50:
        models.n_r1_train = int(mask1.sum())
        models.r1.fit(models.s1.fit_transform(X1[mask1]), y[mask1])

    # ── Régime 2 : entraîné sur les jours d'épisode avec pinn_rho disponible
    mask2 = (in_ep == 1) & ~np.isnan(X2).any(1) & ~np.isnan(y)
    if mask2.sum() >= 15:
        models.n_r2_train = int(mask2.sum())
        models.r2.fit(models.s2.fit_transform(X2[mask2]), y[mask2])
        models.r2_fitted = True
    else:
        models.r2_fitted = False

    return models


def predict_two_regime(
    feat: pd.DataFrame,
    regime1_cols: list[str],
    regime2_cols: list[str],
    models: RegimeModels,
) -> np.ndarray:
    n = len(feat)
    pred = np.full(n, np.nan)
    in_ep = feat["in_episode"].values
    X1 = feat[regime1_cols].values
    X2 = feat[regime2_cols].values

    # Régime 1 : tous les jours normaux + fallback épisodes sans PINN
    mask1 = ~np.isnan(X1).any(1)
    if mask1.sum() > 0:
        pred[mask1] = models.r1.predict(models.s1.transform(X1[mask1]))

    # Régime 2 : épisodes avec rho_pred disponible
    if models.r2_fitted:
        mask2 = (in_ep == 1) & ~np.isnan(X2).any(1)
        if mask2.sum() > 0:
            pred[mask2] = models.r2.predict(models.s2.transform(X2[mask2]))

    return pred


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_two_regime(
    feat: pd.DataFrame,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    init_train: int = 252,
    test_block: int = 63,
) -> dict:
    n = len(feat)
    oos_pred, oos_actual, oos_dates = [], [], []
    regime_log = []

    start = init_train
    while start < n:
        end = min(start + test_block, n)

        feat_tr = feat.iloc[:start]
        y_tr    = y[:start]
        feat_te = feat.iloc[start:end]
        y_te    = y[start:end]
        dates_te = dates[start:end]

        m = fit_two_regime(feat_tr, y_tr, REGIME1_COLS, REGIME2_COLS)
        preds = predict_two_regime(feat_te, REGIME1_COLS, REGIME2_COLS, m)

        valid = ~np.isnan(preds) & ~np.isnan(y_te)
        in_ep_te = feat_te["in_episode"].values[valid]

        if valid.sum() > 0:
            oos_pred.extend(preds[valid])
            oos_actual.extend(y_te[valid])
            oos_dates.extend(dates_te[valid])
            regime_log.append({
                "block_start":    dates[start].date(),
                "n_days":         valid.sum(),
                "n_episode_days": int(in_ep_te.sum()),
                "r1_train":       m.n_r1_train,
                "r2_train":       m.n_r2_train,
                "r2_fitted":      m.r2_fitted,
            })

        start += test_block

    return {
        "pred":       np.array(oos_pred),
        "actual":     np.array(oos_actual),
        "dates":      pd.DatetimeIndex(oos_dates),
        "regime_log": pd.DataFrame(regime_log),
    }


def walk_forward_baseline(
    feat: pd.DataFrame,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    init_train: int = 252,
    test_block: int = 63,
) -> dict:
    """Modèle de référence : Ridge unique entraîné sur tous les jours."""
    n = len(feat)
    oos_pred, oos_actual, oos_dates = [], [], []

    start = init_train
    while start < n:
        end = min(start + test_block, n)

        X_tr = feat[REGIME1_COLS].values[:start]
        y_tr = y[:start]
        X_te = feat[REGIME1_COLS].values[start:end]
        y_te = y[start:end]

        mask_tr = ~np.isnan(X_tr).any(1) & ~np.isnan(y_tr)
        mask_te = ~np.isnan(X_te).any(1) & ~np.isnan(y_te)

        if mask_tr.sum() < 50 or mask_te.sum() == 0:
            start += test_block; continue

        sc = StandardScaler()
        m  = Ridge(alpha=1.0)
        m.fit(sc.fit_transform(X_tr[mask_tr]), y_tr[mask_tr])
        preds = m.predict(sc.transform(X_te[mask_te]))

        oos_pred.extend(preds)
        oos_actual.extend(y_te[mask_te])
        oos_dates.extend(dates[start:end][mask_te])
        start += test_block

    return {
        "pred":   np.array(oos_pred),
        "actual": np.array(oos_actual),
        "dates":  pd.DatetimeIndex(oos_dates),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES + RAPPORT
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, actual, label="") -> dict:
    valid = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[valid], actual[valid]
    if len(p) < 10:
        return {"label": label, "n_oos": len(p), "note": "insufficient"}

    ic_p,  pv_p  = stats.pearsonr(p, a)
    ic_s,  pv_s  = stats.spearmanr(p, a)
    dir_acc = float(np.mean(np.sign(p) == np.sign(a)))
    rmse    = float(np.sqrt(np.mean((p - a)**2)))
    ls      = np.where(p > np.median(p), a, -a)
    sharpe  = float(np.mean(ls) / (np.std(ls) + 1e-10) * np.sqrt(252))

    return {
        "label":       label,
        "n_oos":       int(len(p)),
        "IC_pearson":  round(float(ic_p), 4),
        "p_pearson":   round(float(pv_p), 4),
        "IC_spearman": round(float(ic_s), 4),
        "p_spearman":  round(float(pv_s), 4),
        "dir_acc":     round(dir_acc, 4),
        "rmse":        round(rmse, 6),
        "ls_sharpe":   round(float(sharpe), 4),
    }


def episode_metrics(pred, actual, feat_slice, label="") -> dict:
    """Métriques séparées pour jours d'épisode vs. jours normaux."""
    in_ep = feat_slice["in_episode"].values[:len(pred)]
    valid = ~(np.isnan(pred) | np.isnan(actual))

    results = {}
    for regime, mask_ep in [("normal", in_ep == 0), ("episode", in_ep == 1)]:
        mask = valid & mask_ep[:len(pred)]
        p, a = pred[mask], actual[mask]
        if len(p) < 5:
            results[regime] = {"n": len(p), "note": "insufficient"}
            continue
        ic_s, pv_s = stats.spearmanr(p, a)
        dir_acc    = float(np.mean(np.sign(p) == np.sign(a)))
        ls         = np.where(p > np.median(p), a, -a)
        sharpe     = float(np.mean(ls) / (np.std(ls) + 1e-10) * np.sqrt(252))
        results[regime] = {
            "n":           int(len(p)),
            "IC_spearman": round(float(ic_s), 4),
            "p_spearman":  round(float(pv_s), 4),
            "dir_acc":     round(dir_acc, 4),
            "ls_sharpe":   round(float(sharpe), 4),
        }
    return results


def print_summary(rows: list, title: str):
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    hdr = (f"{'Target':<6} {'H':>4} {'Model':<13} "
           f"{'IC_s':>8} {'p':>6} {'DirAcc':>7} {'Sharpe':>8} {'n_oos':>6}")
    print(hdr)
    print("─" * 100)
    for r in sorted(rows, key=lambda x: (x["target"], x["horizon"], x["model"])):
        star = " ★" if r.get("IC_spearman", 0) > 0.07 and r.get("p_spearman", 1) < 0.05 else ""
        print(
            f"{r['target']:<6} {int(r['horizon']):>4} {r['model']:<13} "
            f"{r.get('IC_spearman', np.nan):>8.4f} {r.get('p_spearman', np.nan):>6.3f} "
            f"{r.get('dir_acc', np.nan):>7.3f} {r.get('ls_sharpe', np.nan):>8.3f} "
            f"{r.get('n_oos', 0):>6}{star}"
        )
    print("=" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(targets=None, horizons=None):
    # Cibles retenues pour le modèle 2-régimes :
    #   MATX : lien causal direct (opérateur conteneurs LA/Pacifique → port LA)
    #   BDRY/SBLK : signal baseline AIS valide, mais la corrélation épisode PINN
    #               est probablement confoundue par le facteur COVID 2021 (BDI et
    #               LA congestion simultanément élevés par la même demande macro).
    #               On les conserve pour comparaison mais sans conclusion causale.
    if targets  is None: targets  = ["MATX", "BDRY", "SBLK"]
    if horizons is None: horizons = [5, 10, 21]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    log.info("Chargement gravity + PINN rho...")
    gravity  = load_gravity()
    pinn_rho = load_pinn_rho(years=(2019, 2020, 2021, 2022))
    feat     = build_feature_matrix(gravity, pinn_rho)

    n_episode = int(feat["in_episode"].sum())
    n_with_pinn = int(((feat["in_episode"] == 1) & feat["pinn_rho"].notna()).sum())
    log.info(
        "Features: %d jours | épisode causaux: %d (%d%%) | "
        "jours épisode + PINN rho: %d",
        len(feat), n_episode, int(n_episode / len(feat) * 100), n_with_pinn,
    )

    # Affichage des épisodes détectés
    ep_changes = feat["in_episode"].diff().fillna(0)
    entries = feat.index[ep_changes == 1]
    exits   = feat.index[ep_changes == -1]
    log.info("Épisodes détectés (%d) :", len(entries))
    for s, e in zip(entries, list(exits) + [feat.index[-1]]):
        pinn_cover = int(feat.loc[s:e, "pinn_rho"].notna().sum())
        log.info("  %s → %s (%d j) | PINN rho disponible: %d j",
                 s.date(), e.date(), (e - s).days, pinn_cover)

    all_rows = []
    ep_rows  = []

    for ticker in targets:
        log.info("\n%s %s", "═"*60, ticker)
        prices = load_prices(ticker)
        if prices.empty:
            log.warning("Pas de données pour %s", ticker); continue

        for h in horizons:
            fwd_ret = forward_return(prices, h)

            # Aligner features + rendements
            combined = feat.join(fwd_ret.rename("fwd_ret"), how="inner").dropna(subset=["fwd_ret"])
            y      = combined["fwd_ret"].values
            dates  = combined.index

            if len(combined) < 300:
                log.warning("Pas assez d'obs pour %s h=%d", ticker, h); continue

            log.info("  h=%2dd | %d obs | épisodes: %d j",
                     h, len(combined),
                     int(combined["in_episode"].sum()))

            # ── Baseline (Ridge unique) ───────────────────────────────────
            base_res = walk_forward_baseline(combined, y, dates)
            m_base   = compute_metrics(base_res["pred"], base_res["actual"],
                                       f"Baseline|{ticker}|h{h}")
            m_base.update({"target": ticker, "horizon": h, "model": "Baseline"})
            all_rows.append(m_base)

            log.info("  Baseline  : IC_s=%+.3f (p=%.3f) | DirAcc=%.3f | Sharpe=%.3f | n=%d",
                     m_base["IC_spearman"], m_base["p_spearman"],
                     m_base["dir_acc"], m_base["ls_sharpe"], m_base["n_oos"])

            # ── Deux régimes ──────────────────────────────────────────────
            tr_res = walk_forward_two_regime(combined, y, dates)
            m_tr   = compute_metrics(tr_res["pred"], tr_res["actual"],
                                     f"2Regime|{ticker}|h{h}")
            m_tr.update({"target": ticker, "horizon": h, "model": "2Regime"})
            all_rows.append(m_tr)

            delta_ic = m_tr["IC_spearman"] - m_base["IC_spearman"]
            delta_sh = m_tr["ls_sharpe"]   - m_base["ls_sharpe"]
            sign     = "▲" if delta_ic > 0.003 else ("▼" if delta_ic < -0.003 else "─")

            log.info("  2-Regime  : IC_s=%+.3f (p=%.3f) | DirAcc=%.3f | Sharpe=%.3f | "
                     "ΔIC_s=%+.3f | ΔSharpe=%+.3f  %s",
                     m_tr["IC_spearman"], m_tr["p_spearman"],
                     m_tr["dir_acc"], m_tr["ls_sharpe"],
                     delta_ic, delta_sh, sign)

            # ── Métriques par régime ──────────────────────────────────────
            ep_m = episode_metrics(tr_res["pred"], tr_res["actual"], combined,
                                   label=f"{ticker}|h{h}")
            for regime, rm in ep_m.items():
                if "IC_spearman" in rm:
                    ep_rows.append({
                        "target": ticker, "horizon": h, "regime": regime, **rm
                    })
                    log.info("    [%s] IC_s=%+.3f (p=%.3f) | DirAcc=%.3f | Sharpe=%.3f | n=%d",
                             regime, rm["IC_spearman"], rm["p_spearman"],
                             rm["dir_acc"], rm["ls_sharpe"], rm["n"])

            # ── Bloc de régime détaillé ───────────────────────────────────
            if not tr_res["regime_log"].empty:
                rl = tr_res["regime_log"]
                ep_blocks = rl[rl["n_episode_days"] > 0]
                if not ep_blocks.empty:
                    log.info("  Blocs avec épisodes (%d) : r2_fitted=%s | moy r2_train=%.0f j",
                             len(ep_blocks),
                             ep_blocks["r2_fitted"].any(),
                             ep_blocks["r2_train"].mean())

    # ── Tables de synthèse ────────────────────────────────────────────────────
    print_summary(all_rows, "COMPARAISON BASELINE vs. DEUX RÉGIMES (OOS walk-forward)")

    if ep_rows:
        print(f"\n{'='*80}")
        print("PERFORMANCE PAR RÉGIME (jours normaux vs. jours d'épisode) — modèle 2-Régimes")
        print(f"{'='*80}")
        print(f"{'Target':<6} {'H':>4} {'Régime':<10} {'IC_s':>8} {'p':>6} "
              f"{'DirAcc':>7} {'Sharpe':>8} {'n':>5}")
        print("─" * 80)
        for r in sorted(ep_rows, key=lambda x: (x["target"], x["horizon"], x["regime"])):
            print(f"{r['target']:<6} {int(r['horizon']):>4} {r['regime']:<10} "
                  f"{r['IC_spearman']:>8.4f} {r['p_spearman']:>6.3f} "
                  f"{r['dir_acc']:>7.3f} {r['ls_sharpe']:>8.3f} {r['n']:>5}")
        print("=" * 80)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    df_ep  = pd.DataFrame(ep_rows)
    df_all.to_csv(OUT_DIR / "two_regime_results.csv",     index=False)
    df_ep.to_csv(OUT_DIR  / "two_regime_by_regime.csv",   index=False)
    log.info("Résultats sauvegardés dans %s", OUT_DIR)

    return df_all, df_ep


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Modèle conditionnel à deux régimes")
    parser.add_argument("--target",   nargs="+", default=["BDRY", "SBLK", "MATX"],
                        help="Cibles equity (défaut: BDRY SBLK MATX)")
    parser.add_argument("--horizon",  type=int, nargs="+", default=[5, 10, 21],
                        help="Horizons en jours (défaut: 5 10 21)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    run(targets=args.target, horizons=args.horizon)


if __name__ == "__main__":
    main()
