"""
ttc_xgb_shap.py — Estimateur du Time-to-Clear par XGBoost + explicabilité SHAP.

Position dans la chaîne d'analyse :
    pinn_pipeline.py  →  build_events_dataset()
        ↓  (episode_id, start_date, gravity_score, ttc_days, …)
    [ttc_xgb_shap.py]            ← CE MODULE
        ↓  XGBRegressor entraîné + PNG SHAP (summary + waterfall)
    financial_pipeline.py → generate_trading_signal()

Features physiques d'entrée (une ligne = un épisode de disruption) :
    approach_speed  (float) : Vitesse moyenne d'approche dans rayon < 50 NM,
                              proxy du comportement de congestion amont
                              (Kontovas & Psaraftis, 2011).
    anchorage_queue (int)   : Navires immobiles dans zones d'attente au démarrage
                              de l'épisode (= waiting_vessels dans gravity_daily).
    complexity_p    (float) : Indice de complexité du trafic maritime (0–1 normalisé).
    gravity_score   (float) : Score topologique d'anomalie issu du Manifold Learning
                              (sortie de manifold_pipeline.py).

Cible supervisée :
    time_to_clear   (float) : Durée de résorption de l'épisode en jours.
                              Source : colonne ttc_days de build_events_dataset()
                              (calculée par PINN ou fallback durée observée).

Usage CLI :
    python src/correlation/ttc_xgb_shap.py \\
        --input outputs/financial/houston_events.csv \\
        --output-dir outputs/figures/ttc_shap \\
        --test-frac 0.20 \\
        --save-model

Usage programmatique :
    from src.correlation.ttc_xgb_shap import run_full_pipeline
    results = run_full_pipeline(df=events_df, output_dir=Path("outputs/figures/ttc_shap"))
    xgb_model = results["model"]  # injectables dans generate_trading_signal()
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend non-interactif — obligatoire en production (pas de GUI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Noms canoniques des 4 features physiques attendues dans le DataFrame d'entrée.
FEATURE_COLS: list[str] = [
    "approach_speed",   # nœuds — vitesse d'approche amont
    "anchorage_queue",  # navires — file d'attente au mouillage
    "complexity_p",     # [0,1]  — indice de complexité trafic
    "gravity_score",    # a.u.   — anomalie topologique Manifold
]

# Colonne cible : durée de résorption prédite par le PINN (ou fallback durée observée).
TARGET_COL: str = "time_to_clear"

# Colonne de date utilisée pour le tri chronologique (plusieurs alias tolérés).
DATE_COL_CANDIDATES: tuple[str, ...] = ("start_date", "date", "Date")

# Hyperparamètres XGBoost : conservatifs pour petits datasets (< 100 épisodes).
# max_depth=3 + min_child_weight=5 + reg_lambda=2 combattent l'overfitting
# plus efficacement que d'augmenter n_estimators.
XGB_PARAMS: dict = {
    "n_estimators":    300,
    "max_depth":       3,
    "learning_rate":   0.03,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,   # nombre minimal d'obs dans une feuille
    "gamma":           0.1,   # réduction minimale de perte pour un split
    "reg_alpha":       0.1,   # L1 — pénalise les poids proches de zéro
    "reg_lambda":      2.0,   # L2 — réduit la magnitude des poids
    "random_state":    42,
    "verbosity":       0,
}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_features(
    path: Path,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    alias_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Charge le CSV ou Parquet d'épisodes et vérifie la présence des colonnes requises.

    Compatible avec la sortie de build_events_dataset() (financial_pipeline.py).
    Dans ce cas, renommer "ttc_days" → "time_to_clear" est automatique.

    Parameters
    ----------
    path        : fichier CSV ou Parquet (détection automatique par extension)
    feature_cols: liste des colonnes features attendues
    target_col  : colonne cible (time_to_clear)
    alias_map   : mapping optionnel {colonne_source → colonne_cible}
                  ex: {"ttc_days": "time_to_clear", "waiting_vessels": "anchorage_queue"}

    Returns
    -------
    DataFrame trié chronologiquement, sans NaN sur features + target.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    log.info("Chargement : %d lignes depuis %s | colonnes : %s", len(df), path.name, list(df.columns))

    # Renommages automatiques (sortie build_events_dataset → format attendu)
    default_aliases: dict[str, str] = {
        "ttc_days":        "time_to_clear",
        "waiting_vessels": "anchorage_queue",
    }
    if alias_map:
        default_aliases.update(alias_map)

    for src, dst in default_aliases.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
            log.info("Alias appliqué : %s → %s", src, dst)

    # Tri chronologique (évite le data leakage dans le split)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    # Vérification des colonnes requises
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Colonnes features manquantes : {missing_features}. "
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"Conseil : utilisez alias_map pour mapper vos colonnes existantes."
        )
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. "
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"Conseil : renommez 'ttc_days' → 'time_to_clear' ou passez alias_map."
        )

    # Suppression des lignes incomplètes (NaN sur features ou cible)
    n_before = len(df)
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning("%d lignes supprimées (NaN sur features/target)", n_dropped)

    log.info(
        "Dataset final : %d épisodes | features=%s | target=%s | range TTC=[%.1f, %.1f] j",
        len(df), feature_cols, target_col,
        df[target_col].min(), df[target_col].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Séparation Train / Test
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test_chronological(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    test_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Séparation chronologique strict : les N derniers épisodes forment le test.

    Logique temporelle (pas de shuffle) : simuler un vrai nowcasting où le modèle
    est entraîné sur des crises passées et évalué sur des crises futures inédites.

    Returns
    -------
    X_train, X_test, y_train, y_test, df_train, df_test
    (les deux derniers permettent de récupérer les métadonnées, ex: episode_id)
    """
    if len(df) < 4:
        raise ValueError(
            f"Données insuffisantes : {len(df)} épisodes (minimum 4 requis). "
            "Élargissez la fenêtre temporelle ou abaissez le gravity_threshold."
        )

    n_test  = max(1, int(len(df) * test_frac))
    n_train = len(df) - n_test

    df_train = df.iloc[:n_train].copy()
    df_test  = df.iloc[n_train:].copy()

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train[target_col].values.astype(np.float32)
    X_test  = df_test[feature_cols].values.astype(np.float32)
    y_test  = df_test[target_col].values.astype(np.float32)

    log.info(
        "Split chronologique : train=%d épisodes | test=%d épisodes (%.0f%%)",
        n_train, n_test, 100 * n_test / len(df),
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Entraînement XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_ttc_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str] = FEATURE_COLS,
    xgb_params: dict = XGB_PARAMS,
):
    """
    Instancie et entraîne un XGBRegressor pour prédire le Time-to-Clear.

    Les hyperparamètres par défaut (XGB_PARAMS) sont conçus pour des petits
    datasets (< 100 épisodes) : arbres peu profonds, régularisation forte,
    taux d'apprentissage bas.

    Returns
    -------
    XGBRegressor entraîné (injectables dans generate_trading_signal de financial_pipeline.py)
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "XGBoost non installé. Installez avec : pip install xgboost"
        )

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Log de l'importance des features (gain normalisé)
    importances = dict(zip(feature_cols, model.feature_importances_))
    log.info("Feature importance (gain normalisé) :")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        log.info("  %-20s %.4f  %s", feat, imp, bar)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> dict:
    """
    Calcule RMSE, MAE et R² sur le jeu de test.

    Retourne aussi un DataFrame détaillé par épisode pour le rapport LaTeX.

    Returns
    -------
    dict avec clés : rmse, mae, r2, n_obs, predictions_df
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        raise ImportError("Requis : pip install scikit-learn")

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    log.info("=== Évaluation sur test set (%d épisodes) ===", len(y_test))
    log.info("  RMSE  = %.2f jours", rmse)
    log.info("  MAE   = %.2f jours", mae)
    log.info("  R²    = %.4f", r2)

    # Tableau par épisode (utile pour le rapport)
    pred_df = df_test[["episode_id", "start_date", target_col]].copy() if "episode_id" in df_test.columns \
        else df_test[[target_col]].copy()
    pred_df["ttc_predit_j"] = np.round(y_pred, 2)
    pred_df["ttc_reel_j"]   = np.round(y_test,  2)
    pred_df["erreur_j"]     = np.round(y_pred - y_test, 2)
    pred_df["abs_erreur_j"] = np.round(np.abs(y_pred - y_test), 2)

    return {
        "rmse":           rmse,
        "mae":            mae,
        "r2":             r2,
        "n_obs":          len(y_test),
        "predictions_df": pred_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — Moteur SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_explanation(
    model,
    X: np.ndarray,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple:
    """
    Calcule les SHAP values pour tout le dataset X.

    Utilise TreeExplainer (optimisé pour XGBoost) qui garantit des valeurs
    de Shapley exactes via l'algorithme TreeSHAP (Lundberg et al., 2020).

    Returns
    -------
    (shap_explanation, shap_values_np, X_df)
      - shap_explanation : objet Explanation (API moderne, pour waterfall_plot)
      - shap_values_np   : ndarray (N, n_features) (API legacy, pour summary_plot)
      - X_df             : DataFrame avec noms de colonnes (requis par summary_plot)
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP non installé. Installez avec : pip install shap"
        )

    explainer = shap.TreeExplainer(model)

    # API moderne → objet Explanation (nécessaire pour waterfall_plot)
    shap_explanation = explainer(X)

    # API legacy → numpy array (nécessaire pour summary_plot avec feature_names)
    shap_values_np = explainer.shap_values(X)

    X_df = pd.DataFrame(X, columns=feature_cols)

    log.info(
        "SHAP values calculées : %d épisodes × %d features | "
        "valeur absolue moyenne par feature : %s",
        X.shape[0], X.shape[1],
        {f: f"{float(np.abs(shap_values_np[:, i]).mean()):.3f}"
         for i, f in enumerate(feature_cols)},
    )
    return shap_explanation, shap_values_np, X_df


def plot_shap_summary(
    shap_values_np: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
    title: str = "SHAP — Impact global des features sur le Time-to-Clear",
    dpi: int = 150,
) -> None:
    """
    Génère le SHAP summary_plot (beeswarm) et le sauvegarde en PNG.

    Lecture du graphique :
      - Axe X  : contribution SHAP (positif = allonge le TTC, négatif = réduit le TTC)
      - Couleur : valeur de la feature (rouge = haute, bleu = basse)
      - Chaque point = un épisode de disruption
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    fig, ax = plt.subplots(figsize=(10, 5))
    # show=False : SHAP dessine sur la figure matplotlib courante sans ouvrir de fenêtre
    shap.summary_plot(shap_values_np, X_df, show=False, plot_size=None)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("SHAP summary_plot → %s", output_path)


def plot_worst_crisis_waterfall(
    shap_explanation,
    df: pd.DataFrame,
    output_path: Path,
    gravity_col: str = "gravity_score",
    target_col:  str = TARGET_COL,
    dpi: int = 150,
) -> int:
    """
    Génère un SHAP waterfall_plot pour l'épisode avec le gravity_score maximum.

    Le waterfall_plot décompose visuellement la prédiction XGBoost en contributions
    individuelles de chaque feature pour CE cas précis :
      - Baseline (E[f(X)]) : prédiction moyenne du modèle sur toutes les crises
      - Barre rouge : feature qui AUGMENTE le TTC prédit
      - Barre bleue : feature qui DIMINUE le TTC prédit
      - f(x)        : TTC final prédit pour cet épisode

    Parameters
    ----------
    shap_explanation : objet shap.Explanation calculé sur le même X que df
    df               : DataFrame source (doit contenir gravity_col)
    output_path      : fichier PNG de sortie
    gravity_col      : colonne utilisée pour localiser la pire crise

    Returns
    -------
    idx_worst (int) : indice de la pire crise dans df (pour log / rapport)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    # Localiser la pire crise (gravity_score maximum = disruption la plus intense)
    idx_worst = int(df[gravity_col].idxmax()) if gravity_col in df.columns else 0
    worst_episode_id = df.get("episode_id", pd.Series(["?"])).iloc[idx_worst] \
        if "episode_id" in df.columns else f"idx={idx_worst}"
    worst_gravity    = float(df[gravity_col].iloc[idx_worst]) if gravity_col in df.columns else np.nan
    worst_ttc        = float(df[target_col].iloc[idx_worst]) if target_col in df.columns else np.nan

    log.info(
        "Waterfall pour la pire crise : idx=%d | id=%s | gravity=%.0f | TTC réel=%.1f j",
        idx_worst, worst_episode_id, worst_gravity, worst_ttc,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_explanation[idx_worst], show=False)
    plt.title(
        f"SHAP Waterfall — Pire crise : {worst_episode_id}\n"
        f"gravity_score={worst_gravity:,.0f}  |  TTC réel={worst_ttc:.1f} j",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("SHAP waterfall_plot → %s", output_path)
    return idx_worst


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    output_path: Path,
    target_col: str = TARGET_COL,
    dpi: int = 150,
) -> None:
    """
    Scatter plot TTC prédit vs TTC réel (jeu de test) avec ligne de parfaite prédiction.

    Utile pour visualiser les biais systématiques du modèle (ex: sous-estimation
    des crises extrêmes).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    x = predictions_df["ttc_reel_j"]
    y = predictions_df["ttc_predit_j"]

    ax.scatter(x, y, alpha=0.7, edgecolors="k", linewidths=0.5, s=60, color="steelblue")

    # Ligne parfaite y = x
    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Prédiction parfaite")

    ax.set_xlabel("TTC réel (jours)", fontsize=11)
    ax.set_ylabel("TTC prédit par XGBoost (jours)", fontsize=11)
    ax.set_title("Time-to-Clear : prédit vs réel (test set)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Scatter prédit vs réel → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR — run_full_pipeline()
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    feature_cols: list[str] = FEATURE_COLS,
    target_col:   str = TARGET_COL,
    test_frac:    float = 0.20,
    xgb_params:   dict = XGB_PARAMS,
    save_model:   bool = False,
) -> dict:
    """
    Orchestrateur principal : entraîne XGBoost + génère les 3 PNG SHAP.

    Conçu pour être appelé depuis financial_pipeline.py après build_events_dataset() :
        events_df["time_to_clear"] = events_df["ttc_days"]
        # Enrichir avec approach_speed, anchorage_queue, complexity_p depuis les AIS...
        results = run_full_pipeline(df=events_df, output_dir=output_dir)
        xgb_model = results["model"]  # injecter dans generate_trading_signal()

    Parameters
    ----------
    df          : DataFrame enrichi avec features + target (voir load_episode_features)
    output_dir  : dossier de sortie pour les PNG et le modèle
    feature_cols: liste des colonnes features
    target_col  : colonne cible (time_to_clear)
    test_frac   : fraction du dataset pour le test (défaut: 20% des derniers épisodes)
    xgb_params  : hyperparamètres XGBoost (défaut: XGB_PARAMS)
    save_model  : sauvegarder le modèle XGBoost en JSON

    Returns
    -------
    dict avec clés :
        model           : XGBRegressor entraîné sur les données complètes
        metrics         : {rmse, mae, r2, n_obs} sur le test set
        predictions_df  : DataFrame par épisode (test set) avec TTC prédit vs réel
        shap_summary_path    : Path vers le PNG summary_plot
        shap_waterfall_path  : Path vers le PNG waterfall de la pire crise
        scatter_path         : Path vers le PNG scatter prédit vs réel
        idx_worst_crisis     : indice de la pire crise dans df
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  TTC XGBoost + SHAP — %d épisodes | %d features", len(df), len(feature_cols))
    log.info("=" * 60)

    # ── 1. Split chronologique ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, df_train, df_test = split_train_test_chronological(
        df, feature_cols, target_col, test_frac,
    )

    # ── 2. Entraînement ──────────────────────────────────────────────────────
    model = train_ttc_model(X_train, y_train, feature_cols, xgb_params)

    # ── 3. Évaluation ────────────────────────────────────────────────────────
    eval_results = evaluate_model(model, X_test, y_test, df_test, feature_cols, target_col)
    metrics       = {k: v for k, v in eval_results.items() if k != "predictions_df"}
    predictions_df = eval_results["predictions_df"]

    # ── 4. Ré-entraînement sur l'ensemble complet (modèle de production) ─────
    # Après évaluation out-of-sample, on entraîne sur tout pour maximiser la
    # robustesse du modèle déployé dans generate_trading_signal().
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[target_col].values.astype(np.float32)
    model_full = train_ttc_model(X_full, y_full, feature_cols, xgb_params)

    # ── 5. SHAP (calculé sur le dataset complet pour couvrir toutes les crises) ─
    shap_explanation, shap_values_np, X_df = compute_shap_explanation(
        model_full, X_full, feature_cols,
    )

    summary_path   = output_dir / "shap_summary.png"
    waterfall_path = output_dir / "shap_waterfall_worst_crisis.png"
    scatter_path   = output_dir / "ttc_predicted_vs_actual.png"

    plot_shap_summary(shap_values_np, X_df, summary_path)
    idx_worst = plot_worst_crisis_waterfall(shap_explanation, df, waterfall_path, target_col=target_col)
    plot_predictions_vs_actual(predictions_df, scatter_path, target_col)

    # ── 6. Sauvegarde optionnelle du modèle ───────────────────────────────────
    model_path = None
    if save_model:
        model_path = output_dir / "xgb_ttc_model.json"
        model_full.save_model(str(model_path))
        log.info("Modèle XGBoost sauvegardé → %s", model_path)

    # ── 7. Rapport synthétique ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  XGBoost TTC Estimator — Résultats (test set : {metrics['n_obs']} épisodes)")
    print(f"{'='*60}")
    print(f"  RMSE  = {metrics['rmse']:.2f} jours")
    print(f"  MAE   = {metrics['mae']:.2f} jours")
    print(f"  R²    = {metrics['r2']:.4f}")
    print(f"\n  Fichiers générés dans {output_dir} :")
    print(f"    shap_summary.png                  — importance globale des 4 features")
    print(f"    shap_waterfall_worst_crisis.png   — décomposition SHAP de la pire crise")
    print(f"    ttc_predicted_vs_actual.png       — prédit vs réel sur le test set")
    if model_path:
        print(f"    xgb_ttc_model.json                — modèle exportable")
    print(f"{'='*60}\n")

    pred_path = output_dir / "ttc_predictions_test.csv"
    predictions_df.to_csv(pred_path, index=False)
    log.info("Prédictions test set → %s", pred_path)

    return {
        "model":               model_full,
        "model_test_only":     model,
        "metrics":             metrics,
        "predictions_df":      predictions_df,
        "shap_summary_path":   summary_path,
        "shap_waterfall_path": waterfall_path,
        "scatter_path":        scatter_path,
        "idx_worst_crisis":    idx_worst,
        "model_path":          model_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES SYNTHÉTIQUES — pour tests et démo rapide
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(n_episodes: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Génère un DataFrame synthétique réaliste pour tester le pipeline sans données AIS.

    Relations simulées (inspirées de la littérature) :
      - anchorage_queue élevé → TTC long (files d'attente prolongent la résorption)
      - gravity_score élevé  → TTC long (amplitude de la disruption)
      - approach_speed faible → TTC long (précongestion, navires qui ralentissent)
      - complexity_p élevé   → TTC légèrement plus long (trafic dense = résorption lente)

    Ne remplace PAS les vraies données AIS — usage test uniquement.
    """
    rng = np.random.default_rng(seed)

    n = n_episodes
    approach_speed  = rng.uniform(3.0, 15.0, n)     # nœuds
    anchorage_queue = rng.integers(20, 200, n)       # navires
    complexity_p    = rng.uniform(0.1, 0.9,  n)     # [0,1]
    gravity_score   = rng.uniform(5_000, 400_000, n) # a.u.

    # TTC = signal réaliste (relation non-linéaire + bruit)
    time_to_clear = (
        5.0
        + 0.05  * anchorage_queue
        + 0.000015 * gravity_score
        + 3.0 * complexity_p
        - 0.2 * approach_speed
        + rng.normal(0, 1.5, n)
    ).clip(min=1.0)

    dates = pd.date_range("2017-01-01", periods=n, freq="5D")

    return pd.DataFrame({
        "episode_id":     [f"ep_{i:03d}" for i in range(n)],
        "start_date":     dates,
        "approach_speed":  np.round(approach_speed, 2),
        "anchorage_queue": anchorage_queue.astype(int),
        "complexity_p":    np.round(complexity_p, 3),
        "gravity_score":   np.round(gravity_score, 1),
        "time_to_clear":   np.round(time_to_clear, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimateur Time-to-Clear par XGBoost + explicabilité SHAP. "
            "Couche intermédiaire entre la sortie PINN et financial_pipeline.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :

  # Données réelles (sortie de build_events_dataset)
  python src/correlation/ttc_xgb_shap.py \\
      --input outputs/financial/houston_events.csv \\
      --alias-map ttc_days=time_to_clear waiting_vessels=anchorage_queue \\
      --output-dir outputs/figures/ttc_shap \\
      --save-model

  # Mode synthétique (test rapide sans données)
  python src/correlation/ttc_xgb_shap.py --synthetic --n-synthetic 80
        """,
    )

    # ── Données d'entrée ──────────────────────────────────────────────────────
    grp_in = parser.add_mutually_exclusive_group(required=True)
    grp_in.add_argument(
        "--input", type=Path,
        help="CSV ou Parquet d'épisodes (doit contenir les 4 features + time_to_clear). "
             "Sortie compatible avec build_events_dataset() de financial_pipeline.py.",
    )
    grp_in.add_argument(
        "--synthetic", action="store_true",
        help="Génère des données synthétiques réalistes pour tester le pipeline.",
    )

    parser.add_argument(
        "--n-synthetic", type=int, default=60,
        help="Nombre d'épisodes synthétiques (uniquement avec --synthetic, défaut=60)",
    )
    parser.add_argument(
        "--alias-map", nargs="+", metavar="SRC=DST",
        help=(
            "Renommages de colonnes au format SRC=DST (plusieurs acceptés). "
            "Ex: --alias-map ttc_days=time_to_clear waiting_vessels=anchorage_queue"
        ),
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--test-frac", type=float, default=0.20,
        help="Fraction du dataset réservée au test (défaut: 0.20 = 20%% des derniers épisodes)",
    )
    parser.add_argument(
        "--features", nargs="+", default=FEATURE_COLS,
        help=f"Colonnes features (défaut: {FEATURE_COLS})",
    )
    parser.add_argument(
        "--target", default=TARGET_COL,
        help=f"Colonne cible (défaut: {TARGET_COL})",
    )

    # ── Sortie ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/figures/ttc_shap"),
        help="Dossier de sortie pour les PNG SHAP et le CSV de prédictions",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Sauvegarder le modèle XGBoost final en JSON (réutilisable dans financial_pipeline.py)",
    )

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Chargement ───────────────────────────────────────────────────────────
    if args.synthetic:
        log.info("Mode --synthetic : génération de %d épisodes fictifs", args.n_synthetic)
        df = _make_synthetic_dataset(n_episodes=args.n_synthetic)
        log.info("Dataset synthétique créé : %d lignes | colonnes : %s", len(df), list(df.columns))
    else:
        alias_map: dict[str, str] = {}
        if args.alias_map:
            for pair in args.alias_map:
                if "=" not in pair:
                    log.warning("alias-map ignoré (format invalide, attenu SRC=DST) : %s", pair)
                    continue
                src, dst = pair.split("=", 1)
                alias_map[src.strip()] = dst.strip()

        df = load_episode_features(
            path=args.input,
            feature_cols=args.features,
            target_col=args.target,
            alias_map=alias_map,
        )

    if len(df) < 4:
        log.error("Trop peu d'épisodes (%d). Minimum 4 requis.", len(df))
        sys.exit(1)

    # ── Pipeline complet ─────────────────────────────────────────────────────
    results = run_full_pipeline(
        df=df,
        output_dir=args.output_dir,
        feature_cols=args.features,
        target_col=args.target,
        test_frac=args.test_frac,
        save_model=args.save_model,
    )

    # Affichage du tableau des prédictions sur le test set
    print("\n── Prédictions sur le test set ──")
    print(results["predictions_df"].to_string(index=False))
    print(f"\nResultats → {args.output_dir}/")


if __name__ == "__main__":
    main()
