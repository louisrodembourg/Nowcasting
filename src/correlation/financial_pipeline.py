"""
Phase 4-5 — Pipeline Financier: nowcasting maritime → signal de marché.

Variable cible (Y) : variation d'un indice financier sur N jours.

Actifs financiers cibles — routage automatique par port :
  - Houston → WTI Crude Oil (quotidien) : port pétrolier (~70% trafic energy)
              Source CSV : stooq.com/q/d/l/?s=cl.f&i=d  (col=Close)
  - LA      → S&P 500     (quotidien) : porte d'entrée biens de consommation US
              Source CSV : stooq.com/q/d/l/?s=%5Espx&i=d  (col=Close)
  - Override manuel possible via --financial pour tout autre indice (BDI, FBX, GSCSI…)

Variables explicatives (X) : Gravity Score + Time to Clear (TTC) issus du PINN.

Étape 1 : build_events_dataset()      → Gravity Score + TTC via PINN pour chaque épisode
Étape 2 : align_with_financials()     → fusion avec l'actif cible, rendement N-jours forward
Étape 3 : run_correlation_analysis()  → Pearson / Spearman + XGBRegressor (TimeSeriesSplit)
Étape 4 : generate_trading_signal()   → alerte structurée adaptée au port (WTI ou S&P 500)

Commandes de lancement (PowerShell — tout sur une ligne) :

  Houston → WTI :
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_houston_episodes.pt --location houston --wti data/financial/WTI.csv --start 2017-01-01 --end 2017-12-31 --n-days 5 10 15 30 --target return_10d

  LA → S&P 500 :
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_la_episodes.pt --location la --sp500 data/financial/SP500.csv --start 2017-01-01 --end 2017-12-31 --gravity-threshold 1000 --n-days 5 10 15 30 --target return_10d

  Override manuel (ex. BDI) :
    python src/correlation/financial_pipeline.py --model outputs/models/pinn_houston_episodes.pt --location houston --financial data/financial/BDI.csv --price-col Close --start 2017-01-01 --end 2017-12-31 --n-days 5 10 15 30 --target return_10d

Notes :
  - Données LA disponibles : 2017 uniquement (gravity_daily.parquet 2017-01-01→2017-12-31)
  - Données Houston disponibles : 2017 uniquement (idem)
  - Seuil gravity LA : 1000 (distribution bimodale basse, défaut=5000 trop restrictif)
  - Seuil gravity Houston : 50000 (défaut LOCATION_DEFAULTS, adapté à Harvey)
  - WTI CSV source : stooq.com  ticker cl.f  (col Close, quotidien)
  - SP500 CSV fourni : data/financial/SP500.csv  (converti depuis Investing.com)
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
DEFAULT_N_DAYS = [5, 10, 15, 30]
DEFAULT_RHO_THRESHOLD_FRAC = 0.3

# Mapping port → actif financier cible
# Justification économique :
#   Houston : ~70% trafic pétrolier/gazier  → WTI est l'actif le plus directement impacté
#   LA      : porte d'entrée biens conso US → S&P 500 capte l'impact sur la chaîne retail
LOCATION_ASSET_MAP: dict[str, dict] = {
    "houston": {
        "asset_name":  "PETROLE BRUT (WTI)",
        "asset_short": "WTI",
        "consequence": (
            "traduisant un choc d'offre energetique "
            "(Houston : ~70% de trafic petrolier et gazier)."
        ),
    },
    "la": {
        "asset_name":  "S&P 500",
        "asset_short": "SP500",
        "consequence": (
            "traduisant un impact sur la chaine d'approvisionnement "
            "des biens de consommation americains."
        ),
    },
}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ════════════════════════════════════════════════════════════════════════════

def _compute_ttc(
    profiles: pd.DataFrame,
    rho_threshold_frac: float = DEFAULT_RHO_THRESHOLD_FRAC,
    fallback_days: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """
    Calcule le Time to Clear depuis un DataFrame de profils PINN.
    TTC = premier jour où max_x ρ(x,t) < rho_threshold_frac × pic initial.

    Si le seuil n'est jamais atteint dans la fenêtre PINN, utilise fallback_days
    (= durée observée de l'épisode) comme borne basse conservative.

    Returns
    -------
    (ttc_value, source)  où source ∈ {"pinn", "duration_fallback", "unavailable"}
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
        if fallback_days is not None:
            return fallback_days, "duration_fallback"
        return None, "unavailable"

    threshold = rho_threshold_frac * float(rho_max_t[0])
    below = np.where(rho_max_t < threshold)[0]

    if len(below) > 0:
        return float(t_vals[below[0]]), "pinn"

    # PINN n'a pas prédit de résolution → fallback sur la durée observée
    if fallback_days is not None:
        return fallback_days, "duration_fallback"
    return None, "unavailable"


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
            "duration_days", "gravity_score", "ttc_days", "ttc_source", "n_ais_points",
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

        ttc, ttc_source = None, "unavailable"
        if ep.n_points > 0:
            # Reconstruire les coordonnées physiques depuis les valeurs normalisées
            episode_data = pd.DataFrame({
                "x_km":   ep.x_norm * ep.channel_len_km,
                "t_days": ep.t_norm * ep.duration_days,
            })
            # Fenêtre étendue à 3× la durée pour laisser le PINN extrapoler la résolution
            t_max = float(episode_data["t_days"].max()) * 3.0 + 2.0

            try:
                profiles = extract_physical_profiles(
                    model=model,
                    episode_data=episode_data,
                    global_meta=global_meta,
                    x_grid_points=x_grid_points,
                    t_grid_points=t_grid_points,
                    t_max_days=t_max,
                )
                ttc, ttc_source = _compute_ttc(
                    profiles, rho_threshold_frac,
                    fallback_days=float(ep.duration_days),
                )
                log.info(
                    "    TTC = %.1f jours [%s]",
                    ttc, ttc_source,
                )
            except Exception as exc:
                log.warning("    Erreur profils PINN (%s) — fallback duration", exc)
                ttc, ttc_source = float(ep.duration_days), "duration_fallback"
        else:
            log.warning("    Aucun point AIS — fallback duration")
            ttc, ttc_source = float(ep.duration_days), "duration_fallback"

        records.append({
            "episode_id":    ep.episode_id,
            "start_date":    ep.start_date,
            "end_date":      ep.end_date,
            "duration_days": ep.duration_days,
            "gravity_score": float(ep.trigger_score),
            "ttc_days":      ttc,
            "ttc_source":    ttc_source,
            "n_ais_points":  ep.n_points,
        })

    df = pd.DataFrame(records)
    n_pinn     = (df["ttc_source"] == "pinn").sum()
    n_fallback = (df["ttc_source"] == "duration_fallback").sum()
    log.info(
        "Dataset: %d episodes | TTC: %d via PINN, %d via fallback duree, %d indisponible",
        len(df), n_pinn, n_fallback, (df["ttc_source"] == "unavailable").sum(),
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1b — Mode rapide : épisodes depuis gravity parquet (sans PINN)
# ════════════════════════════════════════════════════════════════════════════

def build_events_from_gravity(
    location: str,
    gravity_threshold: Optional[float] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    gravity_path: Optional[Path] = None,
    min_episode_days: int = 3,
    gap_tolerance_days: int = 1,
) -> pd.DataFrame:
    """
    Mode --gravity-only : construit le dataset d'épisodes directement depuis
    le gravity parquet, sans charger les AIS ni faire tourner le PINN.

    TTC = durée observée de l'épisode (duration_fallback).
    Gravity score = max du score sur la durée de l'épisode.

    Avantage : traite des années entières en quelques secondes.
    Usage typique : grande période (2021-2024) quand les parquets AIS ont
    été supprimés ou quand on veut une corrélation rapide.

    Parameters
    ----------
    gap_tolerance_days : nombre de jours "sous le seuil" tolérés à l'intérieur
                         d'un épisode (évite de fragmenter sur les week-ends).
    """
    if gravity_threshold is None:
        gravity_threshold = LOCATION_DEFAULTS.get(location, {}).get("gravity_threshold", 50_000)

    # Chemin du fichier gravity
    if gravity_path is None:
        gravity_path = Path(f"data/features/{location}_gravity_daily.parquet")
    if not gravity_path.exists():
        raise FileNotFoundError(f"Gravity parquet introuvable : {gravity_path}")

    df = pd.read_parquet(gravity_path)
    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "=== ETAPE 1 (gravity-only) | %s | threshold=%.0f | %d jours | %s -> %s",
        location, gravity_threshold, len(df),
        df["date"].iloc[0].date() if len(df) else "?",
        df["date"].iloc[-1].date() if len(df) else "?",
    )

    # ── Groupement en épisodes ──────────────────────────────────────────────
    above = df["gravity_score"] >= gravity_threshold
    records = []
    i = 0
    while i < len(df):
        if not above.iloc[i]:
            i += 1
            continue

        # Début d'un épisode
        ep_start = i
        j = i + 1
        gap = 0
        while j < len(df):
            if above.iloc[j]:
                gap = 0
                j += 1
            elif gap < gap_tolerance_days:
                gap += 1
                j += 1
            else:
                break
        # Retrancher les jours de gap en fin d'épisode
        ep_end = j - gap - 1

        ep_df   = df.iloc[ep_start : ep_end + 1]
        dur     = len(ep_df)

        if dur >= min_episode_days:
            start_d = ep_df["date"].iloc[0].date()
            end_d   = ep_df["date"].iloc[-1].date()
            g_score = float(ep_df["gravity_score"].max())
            ep_id   = f"{location}_{str(start_d).replace('-', '')}_{str(end_d).replace('-', '')}"

            records.append({
                "episode_id":    ep_id,
                "start_date":    start_d,
                "end_date":      end_d,
                "duration_days": dur,
                "gravity_score": g_score,
                "ttc_days":      float(dur),
                "ttc_source":    "duration_fallback",
                "n_ais_points":  0,
            })
        i = ep_end + 1

    result = pd.DataFrame(records)
    log.info(
        "%d episodes detectes (threshold=%.0f, min_days=%d)",
        len(result), gravity_threshold, min_episode_days,
    )
    return result


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


def select_financial_data(
    location: str,
    wti_path: Optional[Path],
    sp500_path: Optional[Path],
    date_col: str = "Date",
    price_col: str = "Close",
) -> tuple[pd.DataFrame, dict]:
    """
    Sélectionne et charge l'actif financier cible selon le port d'étude.

    Houston → WTI Crude Oil  (port pétrolier, ~70% trafic energy)
    LA      → S&P 500        (porte d'entrée biens de consommation US)

    Parameters
    ----------
    location   : 'houston' ou 'la'
    wti_path   : chemin vers le CSV WTI  (requis si location=houston)
    sp500_path : chemin vers le CSV S&P 500 (requis si location=la)
    date_col   : colonne date dans les CSV
    price_col  : colonne prix dans les CSV (typiquement 'Close' pour Stooq)

    Returns
    -------
    (financial_df, asset_info_dict)
      asset_info_dict : entrée de LOCATION_ASSET_MAP (asset_name, asset_short, consequence)
    """
    asset_info = LOCATION_ASSET_MAP.get(location)
    if asset_info is None:
        raise ValueError(
            f"Location '{location}' inconnue dans LOCATION_ASSET_MAP. "
            f"Valeurs acceptees : {list(LOCATION_ASSET_MAP.keys())}"
        )

    if location == "houston":
        if wti_path is None:
            raise ValueError(
                "location=houston requiert --wti <chemin/WTI.csv>. "
                "Source gratuite : stooq.com/q/d/l/?s=cl.f&i=d"
            )
        if not wti_path.exists():
            raise FileNotFoundError(f"Fichier WTI introuvable : {wti_path}")
        log.info("Houston → actif financier cible : %s (%s)", asset_info["asset_name"], wti_path)
        df = load_financial_data(wti_path, date_col, price_col)

    else:  # la
        if sp500_path is None:
            raise ValueError(
                "location=la requiert --sp500 <chemin/SP500.csv>. "
                "Source gratuite : stooq.com/q/d/l/?s=%%5Espx&i=d"
            )
        if not sp500_path.exists():
            raise FileNotFoundError(f"Fichier S&P 500 introuvable : {sp500_path}")
        log.info("LA → actif financier cible : %s (%s)", asset_info["asset_name"], sp500_path)
        df = load_financial_data(sp500_path, date_col, price_col)

    return df, asset_info


def align_with_financials(
    events_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    price_col: str = "Close",
    n_days_list: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Étape 2 — Fusionne les épisodes avec l'indice financier et calcule
    les rendements forward sur N jours.

    Gestion des jours de fermeture de marché (week-ends, fériés) : forward-fill
    sur calendrier continu → la date d'un épisode tombe toujours sur un prix valide.
    Adapté à toute fréquence : quotidien (BDI), hebdo (FBX/WCI), mensuel (GSCSI).

    Rendement N jours = (P[J+N] - P[J]) / |P[J]| × 100
    où J = start_date de l'épisode.
    |P[J]| au dénominateur gère les indices pouvant être négatifs (ex. GSCSI).

    Parameters
    ----------
    events_df    : sortie de build_events_dataset
    financial_df : sortie de load_financial_data (index DatetimeIndex)
    price_col    : nom de la colonne prix dans financial_df
    n_days_list  : horizons forward en jours calendaires
                   (défaut: [5, 10, 15, 30] pour données quotidiennes ;
                    utiliser [30, 60, 90] pour données mensuelles)

    Returns
    -------
    DataFrame enrichi avec:
        index_at_event : valeur de l'indice au J de l'épisode
        return_{N}d    : variation en % de l'indice à J+N (une colonne par horizon)
    """
    if n_days_list is None:
        n_days_list = DEFAULT_N_DAYS

    # Calendrier continu + forward-fill pour combler les jours sans cotation
    full_idx = pd.date_range(financial_df.index.min(), financial_df.index.max(), freq="D")
    fin = financial_df[[price_col]].reindex(full_idx).ffill()

    result = events_df.copy()
    result["event_dt"] = pd.to_datetime(result["start_date"])

    index_at_event = []
    fwd_returns = {n: [] for n in n_days_list}

    for _, row in result.iterrows():
        t0 = row["event_dt"]
        p0 = float(fin.loc[t0, price_col]) if t0 in fin.index else np.nan
        index_at_event.append(p0)

        for n in n_days_list:
            t_n = t0 + pd.Timedelta(days=n)
            denom = abs(p0) if not np.isnan(p0) else 0.0
            if t_n in fin.index and denom > 1e-10:
                p_n = float(fin.loc[t_n, price_col])
                fwd_returns[n].append((p_n - p0) / denom * 100.0)
            else:
                fwd_returns[n].append(np.nan)

    result["index_at_event"] = index_at_event
    for n in n_days_list:
        result[f"return_{n}d"] = fwd_returns[n]

    result = result.drop(columns=["event_dt"])

    valid = result["index_at_event"].notna().sum()
    log.info(
        "Alignement financier (%s): %d episodes | %d avec valeur disponible | horizons: %s j",
        price_col, len(result), valid, n_days_list,
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
        targets = sorted(c for c in dataset.columns if c.startswith("return_"))

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

    # Supprimer automatiquement les features entièrement NaN (ex: ttc_days si jamais atteint)
    available = [f for f in features if f in dataset.columns and dataset[f].notna().any()]
    dropped = [f for f in features if f not in available]
    if dropped:
        log.warning("Features ignorees (toutes NaN) : %s", dropped)
    features = available
    if not features:
        raise ValueError("Aucune feature valide — toutes les colonnes sont NaN.")

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
            f"Pas assez de donnees pour XGBoost ({len(X)} lignes apres dropna — "
            f"minimum {min_samples}). Elargissez la fenetre temporelle ou "
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
    # --- TTC via PINN (fallback = durée observée de l'épisode) ---
    ttc, ttc_source = float(episode.duration_days), "duration_fallback"
    if episode.n_points > 0:
        episode_data = pd.DataFrame({
            "x_km":   episode.x_norm * episode.channel_len_km,
            "t_days": episode.t_norm * episode.duration_days,
        })
        t_max = float(episode_data["t_days"].max()) * 3.0 + 2.0
        try:
            pinn_model.eval()
            profiles = extract_physical_profiles(
                model=pinn_model,
                episode_data=episode_data,
                global_meta=pinn_global_meta,
                t_max_days=t_max,
            )
            ttc, ttc_source = _compute_ttc(
                profiles, rho_threshold_frac,
                fallback_days=float(episode.duration_days),
            )
        except Exception as exc:
            log.warning("Signal: erreur TTC (%s) — fallback duree", exc)

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

    ttc_str  = f"{ttc:.1f}" if ttc is not None else "N/A"
    ret_str  = f"{predicted_return:+.2f}%" if not np.isnan(predicted_return) else "N/A"
    src_label = {"pinn": "PINN", "duration_fallback": "duree observee", "unavailable": "N/A"}.get(
        ttc_source, ttc_source
    )

    asset_info  = LOCATION_ASSET_MAP.get(episode.location, {})
    asset_name  = asset_info.get("asset_name",  "l'indice de fret")
    asset_short = asset_info.get("asset_short", "IDX")
    consequence = asset_info.get("consequence", "traduisant une pression sur les prix du fret.")

    alert_text = (
        f"\n{'='*68}\n"
        f"  [{direction}] ALERTE NOWCASTING MARITIME — {episode.location.upper()}\n"
        f"{'='*68}\n"
        f"  Episode ID      : {episode.episode_id}\n"
        f"  Periode         : {episode.start_date} -> {episode.end_date} "
        f"({episode.duration_days} jours)\n"
        f"  Gravity Score   : {gravity:>12,.0f}\n"
        f"  TTC [{src_label:<13s}]: {ttc_str} jours\n"
        f"  {asset_short:<14s}: {current_index_price:>+12.4f}\n"
        f"  Signal          : {signal} [{direction}]\n"
        f"  {'─'*60}\n"
        f"  ALERTE NOWCASTING : L'evenement detecte (ID: {episode.episode_id})\n"
        f"  a un Gravity Score de {gravity:,.0f}. Le modele physique PINN\n"
        f"  estime qu'il mettra {ttc_str} jours a se resorber [{src_label}].\n"
        f"  Consequence : Le modele XGBoost anticipe une variation de\n"
        f"  {ret_str} sur les prix du {asset_name},\n"
        f"  {consequence}\n"
        f"{'='*68}\n"
    )
    print(alert_text)

    return {
        "episode_id":       episode.episode_id,
        "location":         episode.location,
        "start_date":       episode.start_date,
        "gravity_score":    gravity,
        "ttc_days":         ttc,
        "ttc_source":       ttc_source,
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
    parser.add_argument("--model",    type=Path, default=None,
                        help="Checkpoint PINN (.pt) — non requis en mode --gravity-only")
    parser.add_argument("--location", default="houston", choices=["houston", "la"])
    parser.add_argument("--start",    help="Début de la fenêtre d'épisodes (YYYY-MM-DD)")
    parser.add_argument("--end",      help="Fin de la fenêtre d'épisodes (YYYY-MM-DD)")
    parser.add_argument("--gravity-threshold", type=float, default=None,
                        help="Seuil episodes (défaut: LOCATION_DEFAULTS)")
    parser.add_argument("--gravity-path", type=Path, default=None)
    parser.add_argument("--rho-threshold-frac", type=float, default=DEFAULT_RHO_THRESHOLD_FRAC,
                        help="Fraction du pic pour définir le TTC (défaut: 0.3)")
    parser.add_argument("--gravity-only", action="store_true",
                        help="Mode rapide : episodes depuis gravity parquet uniquement, "
                             "sans charger les AIS ni faire tourner le PINN. "
                             "TTC = duree observee. Ideal pour grandes periodes (2021-2024).")
    parser.add_argument("--gap-tolerance", type=int, default=1,
                        help="Jours sous le seuil toleres a l'interieur d'un episode "
                             "(defaut: 1 — gere les week-ends)")

    # Données financières — routage automatique par port
    parser.add_argument("--wti",   type=Path, default=None,
                        help="CSV WTI Crude Oil (requis si location=houston). "
                             "Source: stooq.com/q/d/l/?s=cl.f&i=d")
    parser.add_argument("--sp500", type=Path, default=None,
                        help="CSV S&P 500 (requis si location=la). "
                             "Source: stooq.com/q/d/l/?s=%%5Espx&i=d")
    parser.add_argument("--financial", type=Path, default=None,
                        help="Override manuel: CSV d'un indice quelconque "
                             "(BDI, FBX, GSCSI…). Prioritaire sur --wti/--sp500.")
    parser.add_argument("--date-col",  default="Date",  help="Colonne date dans le CSV (défaut: Date)")
    parser.add_argument("--price-col", default="Close",
                        help="Colonne prix dans le CSV (défaut: Close)")
    parser.add_argument("--n-days",    type=int, nargs="+", default=DEFAULT_N_DAYS,
                        help="Horizons forward en jours (défaut: 5 10 15 30)")

    # XGBoost
    parser.add_argument("--target", default="return_10d",
                        help="Colonne cible pour XGBoost (défaut: return_10d)")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Folds TimeSeriesSplit (défaut: 3)")

    # Sortie
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Dossier de sortie (CSV, modèle XGBoost)")
    parser.add_argument("--save-model", action="store_true",
                        help="Sauvegarder le modèle XGBoost (.json)")

    args = parser.parse_args()

    # Forcer UTF-8 sur le terminal Windows (évite UnicodeEncodeError avec ─ ══ etc.)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_d = date.fromisoformat(args.start) if args.start else None
    end_d   = date.fromisoformat(args.end)   if args.end   else None

    # ── ÉTAPE 1 — Extraction des épisodes ─────────────────────────────────
    if args.gravity_only:
        # Mode rapide : gravity parquet uniquement, sans PINN ni AIS
        log.info("Mode --gravity-only active : pas de PINN, TTC = duree observee")
        try:
            events_df = build_events_from_gravity(
                location=args.location,
                gravity_threshold=args.gravity_threshold,
                start_date=start_d,
                end_date=end_d,
                gravity_path=args.gravity_path,
                gap_tolerance_days=args.gap_tolerance,
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
        pinn_model = None
    else:
        # Mode complet : PINN + AIS
        if args.model is None:
            log.error("--model requis en mode PINN (ou utilisez --gravity-only)")
            sys.exit(1)
        if not args.model.exists():
            log.error("Checkpoint introuvable: %s", args.model)
            sys.exit(1)
        pinn_model, global_meta = _load_checkpoint(args.model)
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
        log.error("Aucun episode detecte — pipeline arrete.")
        sys.exit(1)

    events_path = args.output_dir / f"{args.location}_events.csv"
    events_df.to_csv(events_path, index=False)
    log.info("Events dataset sauvegarde -> %s", events_path)

    # ── ÉTAPE 2 — Chargement de l'actif financier cible ───────────────────
    if args.financial is not None:
        # Override manuel — ignore la logique de routage par port
        if not args.financial.exists():
            log.error("Fichier financier introuvable: %s", args.financial)
            sys.exit(1)
        financial_df = load_financial_data(args.financial, args.date_col, args.price_col)
        asset_info = LOCATION_ASSET_MAP.get(args.location, {
            "asset_name": args.price_col, "asset_short": args.price_col, "consequence": "",
        })
        log.info(
            "Override manuel — indice: %s | col: %s",
            args.financial.name, args.price_col,
        )
    else:
        try:
            financial_df, asset_info = select_financial_data(
                args.location, args.wti, args.sp500,
                args.date_col, args.price_col,
            )
        except (ValueError, FileNotFoundError) as exc:
            log.error("%s", exc)
            sys.exit(1)

    log.info(
        "=== ETAPE 2: Alignement | port=%s | actif=%s ===",
        args.location, asset_info.get("asset_name", args.price_col),
    )
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
            "Verifiez --target ou --n-days.",
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

        # ── TABLEAU PAR ÉPISODE ────────────────────────────────────────────
        asset_info  = asset_info if "asset_info" in dir() else {}
        asset_short = asset_info.get("asset_short", args.price_col)
        features_used = metrics["features"]

        # Horizon cible (ex: return_10d → 10)
        n_target = int(args.target.replace("return_", "").replace("d", ""))

        ep_rows = []
        for _, row in dataset.sort_values("start_date").iterrows():
            x_ep = np.array([[row.get(f, np.nan) for f in features_used]])
            try:
                pred = float(xgb_model.predict(x_ep)[0])
            except Exception:
                pred = np.nan

            actual   = float(row[args.target]) if pd.notna(row.get(args.target)) else np.nan
            error    = pred - actual if (not np.isnan(pred) and not np.isnan(actual)) else np.nan
            ttc_val  = row.get("ttc_days")
            ttc_src  = row.get("ttc_source", "?")

            # Signal basé sur la prédiction
            if np.isnan(pred):
                sig = "="
            elif pred > 1.0:
                sig = "+"
            elif pred < -1.0:
                sig = "-"
            else:
                sig = "="

            # Qualité de la prédiction (si réalisé disponible)
            if not np.isnan(error):
                ok = "OK" if abs(error) < 1.5 else "KO"
            else:
                ok = "--"

            ep_rows.append({
                "episode_id":           row["episode_id"],
                "start_date":           str(row["start_date"]),
                "dur_j":                int(row["duration_days"]),
                "gravity":              int(row["gravity_score"]),
                "ttc_j":                f"{ttc_val:.0f}" if pd.notna(ttc_val) else "N/A",
                "ttc_src":              "PINN" if ttc_src == "pinn" else "obs",
                f"pred_{asset_short}%": f"{pred:+.2f}%" if not np.isnan(pred) else "N/A",
                f"reel_{asset_short}%": f"{actual:+.2f}%" if not np.isnan(actual) else "N/A",
                "erreur_%":             f"{error:+.2f}%" if not np.isnan(error) else "N/A",
                "sig":                  sig,
                "ok":                   ok,
            })

        ep_df = pd.DataFrame(ep_rows)

        # Métriques globales de backtest
        errs = [float(r["erreur_%"].replace("%","")) for r in ep_rows if r["erreur_%"] != "N/A"]
        mae_bt  = float(np.mean(np.abs(errs))) if errs else np.nan
        rmse_bt = float(np.sqrt(np.mean(np.array(errs)**2))) if errs else np.nan
        n_ok    = sum(1 for r in ep_rows if r["ok"] == "OK")

        print(f"\n{'='*80}")
        print(f"  NOWCASTING PAR EPISODE — {args.location.upper()} | cible: {asset_short} | horizon: {n_target}j")
        print(f"  ttc_j : duree estimee de resorption (obs=duree observee, PINN=calcul physique)")
        print(f"  pred  : variation anticipee par XGBoost | reel : variation effectivement realisee")
        print(f"  ok    : |erreur| < 1.5% (seuil indicatif)")
        print(f"{'='*80}")
        print(ep_df.to_string(index=False))
        print(f"{'─'*80}")
        if errs:
            print(f"  Backtest : MAE={mae_bt:.2f}%  RMSE={rmse_bt:.2f}%  "
                  f"Pred OK={n_ok}/{len(ep_rows)} episodes")
        print(f"{'='*80}")

        ep_path = args.output_dir / f"{args.location}_episodes_predictions.csv"
        ep_df.to_csv(ep_path, index=False)
        log.info("Predictions par episode -> %s", ep_path)

    except ValueError as exc:
        log.warning("XGBoost ignore: %s", exc)

    print(f"\nResultats -> {args.output_dir}/")


if __name__ == "__main__":
    main()
