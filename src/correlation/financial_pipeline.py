"""
Phase 4-5 — Pipeline Financier: nowcasting maritime → signal GSCSI.

Variable cible (Y) : variation du GSCSI (Global Supply Chain Stress Index,
Banque Mondiale) sur N jours. Le GSCSI mesure la tension mondiale sur les
chaînes d'approvisionnement — proxy direct de la valeur financière de la
capacité bloquée dans les ports/détroits.

Variables explicatives (X) : Gravity Score + Time to Clear (TTC) issus du PINN.

Étape 1 : build_events_dataset()      → Gravity Score + TTC via PINN pour chaque épisode
Étape 2 : align_with_financials()     → fusion avec le GSCSI, variation N-jours forward
Étape 3 : run_correlation_analysis()  → Pearson / Spearman + XGBRegressor (TimeSeriesSplit)
Étape 4 : generate_trading_signal()   → alerte structurée pour un épisode en temps réel

Usage:
    python src/correlation/financial_pipeline.py \\
        --model     outputs/models/pinn_houston_harvey.pt \\
        --financial data/financial/GSCSI.csv \\
        --location  houston \\
        --start 2017-01-01 --end 2017-12-31 \\
        --n-days 10 15 30 \\
        --target gscsi_delta_10d
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.pinns.lwr_pinn import LWRPINN, get_device
from src.pinns.pinn_pipeline import (
    Episode,
    LOCATION_DEFAULTS,
    extract_physical_profiles,
    structure_episodes,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/financial")
DEFAULT_N_DAYS = [10, 15, 30]
DEFAULT_RHO_THRESHOLD_FRAC = 0.3


# ════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ════════════════════════════════════════════════════════════════════════════

def _compute_ttc(
    profiles: pd.DataFrame,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
) -> Optional[float]:
    """
    Calcule le Time to Clear depuis un DataFrame de profils PINN.
    TTC = premier jour où max_x ρ(x,t) < rho_threshold_frac × pic initial.

    Réplique la logique de plot_congestion_clearance sans générer de figure,
    évitant ainsi l'overhead PNG pour chaque épisode du dataset.
    """
    t_series = (
        profiles.groupby("t_days")["rho_hat"]
        .max()
        .reset_index()
        .sort_values("t_days")
    )
    rho_max_t = t_series["rho_hat"].values
    t_vals    = t_series["t_days"].values

    if len(rho_max_t) == 0:
        return None

    threshold = rho_threshold_frac * float(rho_max_t[0])
    below = np.where(rho_max_t < threshold)[0]
    return float(t_vals[below[0]]) if len(below) > 0 else None


def _load_checkpoint(model_path: Path) -> tuple[LWRPINN, dict]:
    """Charge un checkpoint PINN et retourne (model, global_meta)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LWRPINN(hidden_layers=4, hidden_size=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(
        "Checkpoint charge: %s | best_loss=%.6f | %d episode(s) d'entrainement",
        model_path, ckpt["best_loss"], ckpt["n_episodes"],
    )
    return model, ckpt["global_meta"]


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Extraction des métriques physiques
# ════════════════════════════════════════════════════════════════════════════

def build_events_dataset(
    model: LWRPINN,
    global_meta: dict,
    location: str,
    gravity_threshold: Optional[float] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    gravity_path: Optional[Path] = None,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
    x_grid_points: int = 80,
    t_grid_points: int = 80,
) -> pd.DataFrame:
    """
    Étape 1 — Extrait Gravity Score + TTC pour chaque épisode de disruption.

    Pour chaque épisode détecté par structure_episodes():
      1. Reconstruit les coordonnées physiques (x_km, t_days) depuis Episode.
      2. Appelle extract_physical_profiles() → champ PINN dense (ρ̂, v̂).
      3. Calcule le TTC via _compute_ttc() (sans PNG).

    Parameters
    ----------
    model             : LWRPINN pré-entraîné (depuis checkpoint)
    global_meta       : métadonnées de normalisation du checkpoint
    location          : 'houston' ou 'la'
    gravity_threshold : seuil de déclenchement (défaut: LOCATION_DEFAULTS)
    start_date        : borne inférieure de la période de scan
    end_date          : borne supérieure de la période de scan
    gravity_path      : chemin custom vers gravity_daily.parquet
    rho_threshold_frac: fraction du pic définissant le seuil TTC
    x_grid_points     : résolution spatiale de la grille PINN
    t_grid_points     : résolution temporelle de la grille PINN

    Returns
    -------
    DataFrame pandas:
        episode_id    | start_date | end_date | duration_days
        gravity_score | ttc_days   | n_ais_points
    """
    if gravity_threshold is None:
        gravity_threshold = LOCATION_DEFAULTS.get(location, {}).get("gravity_threshold", 50_000)

    log.info(
        "=== ETAPE 1: Build events dataset | %s | threshold=%.0f | %s -> %s",
        location, gravity_threshold,
        start_date or "debut", end_date or "fin",
    )

    episodes, _ = structure_episodes(
        location=location,
        gravity_path=gravity_path,
        gravity_threshold=gravity_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    if not episodes:
        log.warning("Aucun episode detecte — DataFrame vide retourne.")
        return pd.DataFrame(columns=[
            "episode_id", "start_date", "end_date",
            "duration_days", "gravity_score", "ttc_days", "n_ais_points",
        ])

    log.info("%d episodes detectes. Extraction des profils PINN...", len(episodes))
    model.eval()
    records = []

    for ep in episodes:
        log.info(
            "  [%s] %s -> %s (%dj, %d pts AIS, trigger=%.0f)",
            ep.episode_id, ep.start_date, ep.end_date,
            ep.duration_days, ep.n_points, ep.trigger_score,
        )

        ttc = None
        if ep.n_points > 0:
            # Reconstruire les coordonnées physiques depuis les valeurs normalisées
            episode_data = pd.DataFrame({
                "x_km":   ep.x_norm * ep.channel_len_km,
                "t_days": ep.t_norm * ep.duration_days,
            })
            t_max = float(episode_data["t_days"].max()) * 1.2 + 1.0

            try:
                profiles = extract_physical_profiles(
                    model=model,
                    episode_data=episode_data,
                    global_meta=global_meta,
                    x_grid_points=x_grid_points,
                    t_grid_points=t_grid_points,
                    t_max_days=t_max,
                )
                ttc = _compute_ttc(profiles, rho_threshold_frac)
                log.info("    TTC = %s jours", f"{ttc:.1f}" if ttc is not None else "non atteint")
            except Exception as exc:
                log.warning("    Erreur profils PINN (%s) — TTC=None", exc)
        else:
            log.warning("    Aucun point AIS — TTC=None")

        records.append({
            "episode_id":    ep.episode_id,
            "start_date":    ep.start_date,
            "end_date":      ep.end_date,
            "duration_days": ep.duration_days,
            "gravity_score": float(ep.trigger_score),
            "ttc_days":      ttc,
            "n_ais_points":  ep.n_points,
        })

    df = pd.DataFrame(records)
    log.info(
        "Dataset: %d episodes | TTC disponible: %d/%d",
        len(df), df["ttc_days"].notna().sum(), len(df),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Alignement avec les données financières
# ════════════════════════════════════════════════════════════════════════════

def load_financial_data(path: Path, date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    """
    Charge un fichier CSV d'indice financier.
    Colonnes attendues: Date (YYYY-MM-DD), et une colonne prix (ex: Close, Price, BDI).
    Retourne un DataFrame avec index DatetimeIndex et colonne price_col.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    if date_col not in df.columns:
        raise ValueError(f"Colonne '{date_col}' introuvable dans {path}. Colonnes: {list(df.columns)}")
    if price_col not in df.columns:
        available = [c for c in df.columns if c != date_col]
        raise ValueError(
            f"Colonne '{price_col}' introuvable dans {path}. "
            f"Colonnes disponibles: {available}. Utilisez --price-col."
        )
    df = df[[date_col, price_col]].dropna().sort_values(date_col)
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)
    log.info(
        "Donnees financieres: %d jours de cotation | %s -> %s | col='%s'",
        len(df), df.index.min().date(), df.index.max().date(), price_col,
    )
    return df


def align_with_financials(
    events_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    price_col: str = "Close",
    n_days_list: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Étape 2 — Fusionne les épisodes avec l'indice financier et calcule
    les rendements forward sur N jours.

    Gestion des jours de fermeture de marché : forward-fill sur calendrier
    continu → la date d'un épisode tombe toujours sur un prix valide.

    Variation GSCSI N jours = (GSCSI[J+N] - GSCSI[J]) / |GSCSI[J]| × 100
    où J = start_date de l'épisode (ou prochaine date de publication valide).

    Note: le GSCSI peut être négatif (tension sous la normale), d'où la valeur
    absolue au dénominateur pour éviter une inversion de signe.

    Parameters
    ----------
    events_df    : sortie de build_events_dataset
    financial_df : sortie de load_financial_data (index DatetimeIndex)
    price_col    : nom de la colonne GSCSI dans financial_df
    n_days_list  : horizons forward en jours calendaires (défaut: [10, 15, 30])

    Returns
    -------
    DataFrame enrichi avec:
        gscsi_at_event    : valeur du GSCSI au J de l'épisode
        gscsi_delta_{N}d  : variation en % du GSCSI à J+N (une colonne par horizon)
    """
    if n_days_list is None:
        n_days_list = DEFAULT_N_DAYS

    # Calendrier continu + forward-fill pour combler les jours de fermeture
    full_idx = pd.date_range(financial_df.index.min(), financial_df.index.max(), freq="D")
    fin = financial_df[[price_col]].reindex(full_idx).ffill()

    result = events_df.copy()
    result["event_dt"] = pd.to_datetime(result["start_date"])

    gscsi_at_event = []
    fwd_deltas = {n: [] for n in n_days_list}

    for _, row in result.iterrows():
        t0 = row["event_dt"]
        g0 = float(fin.loc[t0, price_col]) if t0 in fin.index else np.nan
        gscsi_at_event.append(g0)

        for n in n_days_list:
            t_n = t0 + pd.Timedelta(days=n)
            denom = abs(g0) if not np.isnan(g0) else 0.0
            if t_n in fin.index and denom > 1e-10:
                g_n = float(fin.loc[t_n, price_col])
                fwd_deltas[n].append((g_n - g0) / denom * 100.0)
            else:
                fwd_deltas[n].append(np.nan)

    result["gscsi_at_event"] = gscsi_at_event
    for n in n_days_list:
        result[f"gscsi_delta_{n}d"] = fwd_deltas[n]

    result = result.drop(columns=["event_dt"])

    valid = result["gscsi_at_event"].notna().sum()
    log.info(
        "Alignement GSCSI: %d episodes | %d avec valeur disponible | horizons: %s jours",
        len(result), valid, n_days_list,
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Corrélations + XGBoost
# ════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(
    dataset: pd.DataFrame,
    features: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Étape 3a — Corrélations de Pearson et Spearman entre métriques physiques
    [gravity_score, ttc_days] et rendements financiers [return_Nd].

    Parameters
    ----------
    dataset  : sortie de align_with_financials
    features : variables explicatives (défaut: ['gravity_score', 'ttc_days'])
    targets  : variables cibles (défaut: toutes colonnes 'return_*')

    Returns
    -------
    DataFrame: feature × target avec pearson_r, pearson_p, spearman_r, spearman_p, n_obs
    """
    if features is None:
        features = ["gravity_score", "ttc_days"]
    if targets is None:
        targets = sorted(c for c in dataset.columns if c.startswith("gscsi_delta_"))

    rows = []
    for feat in features:
        for tgt in targets:
            sub = dataset[[feat, tgt]].dropna()
            n = len(sub)
            if n < 4:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r,  pearson_p  = stats.pearsonr(sub[feat], sub[tgt])
                spearman_r, spearman_p = stats.spearmanr(sub[feat], sub[tgt])
            rows.append({
                "feature":    feat,
                "target":     tgt,
                "pearson_r":  round(float(pearson_r),  4) if not np.isnan(pearson_r)  else np.nan,
                "pearson_p":  round(float(pearson_p),  4) if not np.isnan(pearson_p)  else np.nan,
                "spearman_r": round(float(spearman_r), 4) if not np.isnan(spearman_r) else np.nan,
                "spearman_p": round(float(spearman_p), 4) if not np.isnan(spearman_p) else np.nan,
                "n_obs":      n,
            })

    return pd.DataFrame(rows)


def train_xgboost(
    dataset: pd.DataFrame,
    target_col: str,
    features: Optional[list[str]] = None,
    n_splits: int = 3,
) -> tuple:
    """
    Étape 3b — XGBRegressor avec validation chronologique (TimeSeriesSplit).

    Séparation train/test strictement chronologique pour éviter tout data
    leakage : les épisodes futurs ne contaminent jamais l'entraînement.

    Parameters
    ----------
    dataset    : sortie de align_with_financials
    target_col : colonne cible (ex: 'return_10d')
    features   : features X (défaut: ['gravity_score', 'ttc_days'])
    n_splits   : nombre de folds chronologiques

    Returns
    -------
    (xgb_model_final, metrics_dict, feature_importance_df)
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        raise ImportError("Requis: pip install xgboost scikit-learn")

    if features is None:
        features = ["gravity_score", "ttc_days"]

    sort_col = "start_date" if "start_date" in dataset.columns else None
    sub = dataset[features + [target_col]].dropna()
    if sort_col:
        sub = dataset[features + [target_col, sort_col]].dropna().sort_values(sort_col)
        sub = sub.drop(columns=[sort_col])

    X = sub[features].values
    y = sub[target_col].values

    min_samples = 4
    if len(X) < min_samples:
        raise ValueError(
            f"Pas assez de données pour XGBoost ({len(X)} lignes après dropna — "
            f"minimum {min_samples}). Elargissez la fenêtre temporelle ou "
            f"abaissez le gravity_threshold."
        )

    n_splits = min(n_splits, len(X) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    log.info(
        "=== ETAPE 3b: XGBoost | target=%s | features=%s | %d obs | %d folds",
        target_col, features, len(X), n_splits,
    )

    rmse_scores, mae_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        m = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])

        rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        mae  = float(mean_absolute_error(y[test_idx], y_pred))
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        log.info(
            "  Fold %d/%d: train=%d test=%d | RMSE=%.3f%% MAE=%.3f%%",
            fold + 1, n_splits, len(train_idx), len(test_idx), rmse, mae,
        )

    # Modèle final entraîné sur l'ensemble des données disponibles
    final_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    metrics = {
        "target":    target_col,
        "features":  features,
        "n_obs":     len(X),
        "n_splits":  n_splits,
        "rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "rmse_std":  round(float(np.std(rmse_scores)),  4),
        "mae_mean":  round(float(np.mean(mae_scores)),  4),
        "mae_std":   round(float(np.std(mae_scores)),   4),
    }

    feat_importance = pd.DataFrame({
        "feature":    features,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    log.info(
        "XGBoost final: RMSE=%.3f±%.3f%% | MAE=%.3f±%.3f%%",
        metrics["rmse_mean"], metrics["rmse_std"],
        metrics["mae_mean"],  metrics["mae_std"],
    )
    return final_model, metrics, feat_importance


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Générateur de signal boursier
# ════════════════════════════════════════════════════════════════════════════

def generate_trading_signal(
    episode: Episode,
    pinn_model: LWRPINN,
    pinn_global_meta: dict,
    xgb_model,
    current_index_price: float,
    n_days: int = 10,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
) -> dict:
    """
    Étape 4 — Génère une alerte de marché structurée pour un épisode en direct.

    Chaîne complète:
        Episode → PINN → TTC → XGBoost → variation attendue → signal directionnel

    Parameters
    ----------
    episode             : Episode détecté en temps réel
    pinn_model          : LWRPINN pré-entraîné
    pinn_global_meta    : métadonnées du checkpoint PINN
    xgb_model           : XGBRegressor entraîné via train_xgboost()
    current_index_price : prix courant de l'indice (ex: BDI = 1250)
    n_days              : horizon de prédiction en jours
    rho_threshold_frac  : doit correspondre à celui utilisé à l'entraînement

    Returns
    -------
    dict: episode_id, gravity_score, ttc_days, predicted_return, signal, alert_text
    """
    # --- TTC via PINN ---
    ttc = None
    if episode.n_points > 0:
        episode_data = pd.DataFrame({
            "x_km":   episode.x_norm * episode.channel_len_km,
            "t_days": episode.t_norm * episode.duration_days,
        })
        t_max = float(episode_data["t_days"].max()) * 1.2 + 1.0
        try:
            pinn_model.eval()
            profiles = extract_physical_profiles(
                model=pinn_model,
                episode_data=episode_data,
                global_meta=pinn_global_meta,
                t_max_days=t_max,
            )
            ttc = _compute_ttc(profiles, rho_threshold_frac)
        except Exception as exc:
            log.warning("Signal: erreur TTC (%s)", exc)

    gravity = float(episode.trigger_score)

    # --- Prédiction XGBoost ---
    x_in = np.array([[gravity, ttc if ttc is not None else np.nan]])
    try:
        predicted_return = float(xgb_model.predict(x_in)[0])
    except Exception as exc:
        log.warning("Signal: erreur XGBoost (%s)", exc)
        predicted_return = np.nan

    # --- Classification du signal ---
    if np.isnan(predicted_return):
        signal, direction = "NEUTRE", "="
    elif predicted_return > 1.0:
        signal, direction = "HAUSSIER", "+"
    elif predicted_return < -1.0:
        signal, direction = "BAISSIER", "-"
    else:
        signal, direction = "NEUTRE", "="

    ttc_str = f"{ttc:.1f}" if ttc is not None else "non atteint dans la fenetre"
    ret_str = f"{predicted_return:+.2f}%" if not np.isnan(predicted_return) else "N/A"

    alert_text = (
        f"\n{'='*68}\n"
        f"  [{direction}] ALERTE NOWCASTING MARITIME — {episode.location.upper()}\n"
        f"{'='*68}\n"
        f"  Episode ID      : {episode.episode_id}\n"
        f"  Periode         : {episode.start_date} -> {episode.end_date} "
        f"({episode.duration_days} jours)\n"
        f"  Gravity Score   : {gravity:>12,.0f}\n"
        f"  TTC (PINN)      : {ttc_str} jours\n"
        f"  GSCSI actuel    : {current_index_price:>+12.4f}\n"
        f"  Signal GSCSI    : {signal} [{direction}]\n"
        f"  {'─'*60}\n"
        f"  ALERTE NOWCASTING : L'evenement detecte (ID: {episode.episode_id})\n"
        f"  a un Gravity Score de {gravity:,.0f}. Le modele physique PINN\n"
        f"  estime qu'il mettra {ttc_str} jours a se resorber.\n"
        f"  Consequence : Le modele XGBoost anticipe une variation de\n"
        f"  {ret_str} du GSCSI mondial, ce qui implique une pression\n"
        f"  imminente sur les prix du fret.\n"
        f"{'='*68}\n"
    )
    print(alert_text)

    return {
        "episode_id":       episode.episode_id,
        "location":         episode.location,
        "start_date":       episode.start_date,
        "gravity_score":    gravity,
        "ttc_days":         ttc,
        "predicted_return": predicted_return,
        "signal":           signal,
        "current_price":    current_index_price,
        "n_days_horizon":   n_days,
        "alert_text":       alert_text,
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4-5 — Pipeline financier: PINN + XGBoost → signal de marché"
    )

    # Données physiques
    parser.add_argument("--model",    type=Path, required=True,
                        help="Checkpoint PINN (.pt)")
    parser.add_argument("--location", default="houston", choices=["houston", "la"])
    parser.add_argument("--start",    help="Début de la fenêtre d'épisodes (YYYY-MM-DD)")
    parser.add_argument("--end",      help="Fin de la fenêtre d'épisodes (YYYY-MM-DD)")
    parser.add_argument("--gravity-threshold", type=float, default=None,
                        help="Seuil episodes (défaut: LOCATION_DEFAULTS)")
    parser.add_argument("--gravity-path", type=Path, default=None)
    parser.add_argument("--rho-threshold-frac", type=float, default=DEFAULT_RHO_THRESHOLD_FRAC,
                        help="Fraction du pic pour définir le TTC (défaut: 0.3)")

    # Données financières
    parser.add_argument("--financial",  type=Path, required=True,
                        help="CSV de l'indice financier (Date, Close)")
    parser.add_argument("--date-col",   default="Date",  help="Colonne date dans le CSV")
    parser.add_argument("--price-col",  default="GSCSI", help="Colonne GSCSI dans le CSV (défaut: GSCSI)")
    parser.add_argument("--n-days",     type=int, nargs="+", default=DEFAULT_N_DAYS,
                        help="Horizons forward en jours (défaut: 10 15 30)")

    # XGBoost
    parser.add_argument("--target", default="gscsi_delta_10d",
                        help="Colonne cible pour XGBoost (défaut: gscsi_delta_10d)")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Folds TimeSeriesSplit (défaut: 3)")

    # Sortie
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Dossier de sortie (CSV, modèle XGBoost)")
    parser.add_argument("--save-model", action="store_true",
                        help="Sauvegarder le modèle XGBoost (.json)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement du checkpoint PINN ──────────────────────────────────────
    if not args.model.exists():
        log.error("Checkpoint introuvable: %s", args.model)
        sys.exit(1)

    pinn_model, global_meta = _load_checkpoint(args.model)

    # ── ÉTAPE 1 ────────────────────────────────────────────────────────────
    start_d = date.fromisoformat(args.start) if args.start else None
    end_d   = date.fromisoformat(args.end)   if args.end   else None

    events_df = build_events_dataset(
        model=pinn_model,
        global_meta=global_meta,
        location=args.location,
        gravity_threshold=args.gravity_threshold,
        start_date=start_d,
        end_date=end_d,
        gravity_path=args.gravity_path,
        rho_threshold_frac=args.rho_threshold_frac,
    )

    if events_df.empty:
        log.error("Aucun episode — pipeline arrete.")
        sys.exit(1)

    events_path = args.output_dir / f"{args.location}_events.csv"
    events_df.to_csv(events_path, index=False)
    log.info("Events dataset sauvegarde -> %s", events_path)

    # ── ÉTAPE 2 ────────────────────────────────────────────────────────────
    if not args.financial.exists():
        log.error("Fichier financier introuvable: %s", args.financial)
        sys.exit(1)

    financial_df = load_financial_data(args.financial, args.date_col, args.price_col)

    log.info("=== ETAPE 2: Alignement avec donnees financieres ===")
    dataset = align_with_financials(
        events_df=events_df,
        financial_df=financial_df,
        price_col=args.price_col,
        n_days_list=args.n_days,
    )

    dataset_path = args.output_dir / f"{args.location}_dataset_aligned.csv"
    dataset.to_csv(dataset_path, index=False)
    log.info("Dataset aligne sauvegarde -> %s", dataset_path)

    # ── ÉTAPE 3a — Corrélations ────────────────────────────────────────────
    log.info("=== ETAPE 3a: Analyse de correlation ===")
    corr_df = run_correlation_analysis(dataset)

    print("\n── Corrélations Pearson / Spearman ──")
    print(corr_df.to_string(index=False))

    corr_path = args.output_dir / f"{args.location}_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info("Correlations sauvegardees -> %s", corr_path)

    # ── ÉTAPE 3b — XGBoost ────────────────────────────────────────────────
    if args.target not in dataset.columns:
        available_targets = [c for c in dataset.columns if c.startswith("return_")]
        log.error(
            "Target '%s' absente. Disponibles: %s. "
            "Vérifiez --target ou --n-days.",
            args.target, available_targets,
        )
        sys.exit(1)

    try:
        xgb_model, metrics, feat_imp = train_xgboost(
            dataset=dataset,
            target_col=args.target,
            n_splits=args.n_splits,
        )

        print(f"\n── XGBoost ({args.target}) ──")
        print(f"  Observations : {metrics['n_obs']}")
        print(f"  RMSE (CV)    : {metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} %")
        print(f"  MAE  (CV)    : {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} %")
        print(f"\n  Feature importance:")
        for _, row in feat_imp.iterrows():
            print(f"    {row['feature']:20s} {row['importance']:.4f}")

        metrics_path = args.output_dir / f"{args.location}_xgb_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        if args.save_model:
            try:
                xgb_path = args.output_dir / f"{args.location}_xgb_{args.target}.json"
                xgb_model.save_model(str(xgb_path))
                log.info("Modele XGBoost sauvegarde -> %s", xgb_path)
            except Exception as exc:
                log.warning("Impossible de sauvegarder le modèle XGBoost: %s", exc)

    except ValueError as exc:
        log.warning("XGBoost ignore: %s", exc)

    print(f"\nResultats -> {args.output_dir}/")


if __name__ == "__main__":
    main()
