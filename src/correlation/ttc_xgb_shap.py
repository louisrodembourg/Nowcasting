"""
ttc_xgb_shap.py — Time-to-clear estimator with XGBoost + SHAP explanations.

Position in the analysis chain:
    pinn_pipeline.py  →  extract_physical_profiles()  →  extract_pinn_summaries()
        ↓  (peak_rho, t_to_peak_days, area_under_rho, decay_rate, …)
    financial_pipeline.py  →  build_events_dataset()
        ↓  (episode_id, start_date, gravity_score, duration_days, pinn_summaries, …)
    [ttc_xgb_shap.py]            ← THIS MODULE
        ↓  trained XGBRegressor + SHAP PNGs (summary + waterfall)
    financial_pipeline.py → generate_trading_signal()

Physical input features (one row = one disruption episode):
    gravity_score   (float) : Topological anomaly score from manifold learning
                              (Phase 2 — manifold_pipeline.py).
    n_ais_points    (int)   : Number of AIS points observed during the episode.
    peak_rho        (float) : Maximum density [vessels/km] from the PINN field ρ̂(x,t).
    t_to_peak_days  (float) : Time [days] to reach the congestion peak.
    area_under_rho  (float) : Time integral of max_x ρ(x,t) — total load.
    decay_rate      (float) : Post-peak decay slope [v/km/d].
    flux_entry_mean (float) : Mean flux at the channel entrance [vessels·kn/km].

Supervised target:
    duration_days   (float) : Observed episode duration in days, derived from the
                              gravity score (length of the run above the threshold).
                              Empirical TTC proxy — more robust than an arbitrary
                              threshold on ρ̂(x,t).

CLI usage:
    python src/correlation/ttc_xgb_shap.py \\
        --input outputs/financial/houston_events.csv \\
        --output-dir outputs/figures/ttc_shap \\
        --test-frac 0.20 \\
        --save-model

Programmatic usage:
    from src.correlation.ttc_xgb_shap import run_full_pipeline
    results = run_full_pipeline(df=events_df, output_dir=Path("outputs/figures/ttc_shap"))
    xgb_model = results["model"]  # can be injected into generate_trading_signal()
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required in production (no GUI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Physical features: gravity score (Phase 2) + scalar summaries of the PINN field (Phase 3).
FEATURE_COLS: list[str] = [
    "gravity_score",    # topological anomaly from the manifold (Phase 2)
    "n_ais_points",     # AIS coverage of the episode
    "peak_rho",         # maximum density [vessels/km] — PINN field
    "t_to_peak_days",   # time to congestion peak [days] — PINN field
    "area_under_rho",   # total congestion load [v/km·d] — PINN field
    "decay_rate",       # post-peak clearing slope [v/km/d] — PINN field
    "flux_entry_mean",  # mean flux at the channel entrance [v·kn/km] — PINN field
]

# Target: observed episode duration (empirical TTC proxy, more robust than thresholding ρ̂).
TARGET_COL: str = "duration_days"

# Date column used for chronological sorting (several aliases accepted).
DATE_COL_CANDIDATES: tuple[str, ...] = ("start_date", "date", "Date")

# XGBoost hyper-parameters: conservative for small datasets (< 100 episodes).
# max_depth=3 + min_child_weight + reg_lambda=2 fight overfitting
# more effectively than increasing n_estimators.
XGB_PARAMS: dict = {
    "n_estimators":    300,
    "max_depth":       3,
    "learning_rate":   0.03,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,   # minimum number of observations in a leaf
    "gamma":           0.1,   # minimum loss reduction for a split
    "reg_alpha":       0.1,   # L1 — penalises weights close to zero
    "reg_lambda":      2.0,   # L2 — shrinks weight magnitude
    "random_state":    42,
    "verbosity":       0,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_features(
    path: Path,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    alias_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load the episode CSV or Parquet and check that the required columns exist.

    Compatible with the output of build_events_dataset() (financial_pipeline.py).
    In that case, renaming "ttc_days" → "time_to_clear" is automatic.

    Parameters
    ----------
    path        : CSV or Parquet file (detected from the extension)
    feature_cols: list of expected feature columns
    target_col  : target column
    alias_map   : optional mapping {source_column → target_column}
                  e.g. {"ttc_days": "time_to_clear", "waiting_vessels": "anchorage_queue"}

    Returns
    -------
    DataFrame sorted chronologically, without NaN in features + target.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    log.info("Loading: %d rows from %s | columns: %s", len(df), path.name, list(df.columns))

    # Automatic renames (build_events_dataset output → expected format)
    default_aliases: dict[str, str] = {
        "waiting_vessels": "anchorage_queue",
    }
    if alias_map:
        default_aliases.update(alias_map)

    for src, dst in default_aliases.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
            log.info("Alias applied: %s → %s", src, dst)

    # Chronological sort (avoids data leakage in the split)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    # Check the required columns
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}. "
            f"Available columns: {list(df.columns)}\n"
            f"Tip: use alias_map to map your existing columns."
        )
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}\n"
            f"Tip: rename 'ttc_days' → 'time_to_clear' or pass alias_map."
        )

    # Drop incomplete rows (NaN in features or target)
    n_before = len(df)
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning("%d rows dropped (NaN in features/target)", n_dropped)

    log.info(
        "Final dataset: %d episodes | features=%s | target=%s | TTC range=[%.1f, %.1f] d",
        len(df), feature_cols, target_col,
        df[target_col].min(), df[target_col].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test_chronological(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    test_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split: the last N episodes form the test set.

    Temporal logic (no shuffle): simulates real nowcasting, where the model
    is trained on past crises and evaluated on unseen future ones.

    Returns
    -------
    X_train, X_test, y_train, y_test, df_train, df_test
    (the last two give access to metadata, e.g. episode_id)
    """
    if len(df) < 4:
        raise ValueError(
            f"Not enough data: {len(df)} episodes (minimum 4 required). "
            "Widen the time window or lower gravity_threshold."
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
        "Chronological split: train=%d episodes | test=%d episodes (%.0f%%)",
        n_train, n_test, 100 * n_test / len(df),
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — XGBoost training
# ─────────────────────────────────────────────────────────────────────────────

def train_ttc_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str] = FEATURE_COLS,
    xgb_params: dict = XGB_PARAMS,
):
    """
    Instantiate and train an XGBRegressor to predict the time-to-clear.

    The default hyper-parameters (XGB_PARAMS) are designed for small
    datasets (< 100 episodes): shallow trees, strong regularisation,
    low learning rate.

    Returns
    -------
    Trained XGBRegressor (can be injected into generate_trading_signal of financial_pipeline.py)
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "XGBoost not installed. Install with: pip install xgboost"
        )

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Log feature importance (normalised gain)
    importances = dict(zip(feature_cols, model.feature_importances_))
    log.info("Feature importance (normalised gain):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        log.info("  %-20s %.4f  %s", feat, imp, bar)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluation
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
    Compute RMSE, MAE and R² on the test set.

    Also returns a per-episode DataFrame for the LaTeX report.

    Returns
    -------
    dict with keys: rmse, mae, r2, n_obs, predictions_df
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        raise ImportError("Required: pip install scikit-learn")

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    log.info("=== Evaluation on the test set (%d episodes) ===", len(y_test))
    log.info("  RMSE  = %.2f days", rmse)
    log.info("  MAE   = %.2f days", mae)
    log.info("  R²    = %.4f", r2)

    # Per-episode table (useful for the report)
    pred_df = df_test[["episode_id", "start_date", target_col]].copy() if "episode_id" in df_test.columns \
        else df_test[[target_col]].copy()
    pred_df["ttc_predicted_d"] = np.round(y_pred, 2)
    pred_df["ttc_actual_d"]   = np.round(y_test,  2)
    pred_df["error_d"]     = np.round(y_pred - y_test, 2)
    pred_df["abs_error_d"] = np.round(np.abs(y_pred - y_test), 2)

    return {
        "rmse":           rmse,
        "mae":            mae,
        "r2":             r2,
        "n_obs":          len(y_test),
        "predictions_df": pred_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SHAP engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_explanation(
    model,
    X: np.ndarray,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple:
    """
    Compute SHAP values for the whole dataset X.

    Uses TreeExplainer (optimised for XGBoost), which gives exact Shapley
    values through the TreeSHAP algorithm (Lundberg et al., 2020).

    Returns
    -------
    (shap_explanation, shap_values_np, X_df)
      - shap_explanation : Explanation object (modern API, for waterfall_plot)
      - shap_values_np   : ndarray (N, n_features) (legacy API, for summary_plot)
      - X_df             : DataFrame with column names (required by summary_plot)
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP not installed. Install with: pip install shap"
        )

    explainer = shap.TreeExplainer(model)

    # Modern API → Explanation object (needed for waterfall_plot)
    shap_explanation = explainer(X)

    # Legacy API → numpy array (needed for summary_plot with feature_names)
    shap_values_np = explainer.shap_values(X)

    X_df = pd.DataFrame(X, columns=feature_cols)

    log.info(
        "SHAP values computed: %d episodes × %d features | "
        "mean absolute value per feature: %s",
        X.shape[0], X.shape[1],
        {f: f"{float(np.abs(shap_values_np[:, i]).mean()):.3f}"
         for i, f in enumerate(feature_cols)},
    )
    return shap_explanation, shap_values_np, X_df


def plot_shap_summary(
    shap_values_np: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
    title: str = "SHAP — Global feature impact on the time-to-clear",
    dpi: int = 150,
) -> None:
    """
    Generate the SHAP summary_plot (beeswarm) and save it as PNG.

    How to read it:
      - X axis : SHAP contribution (positive = longer TTC, negative = shorter TTC)
      - Colour : feature value (red = high, blue = low)
      - Each point = one disruption episode
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    fig, ax = plt.subplots(figsize=(10, 5))
    # show=False: SHAP draws on the current matplotlib figure without opening a window
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
    Generate a SHAP waterfall_plot for the episode with the highest gravity_score.

    The waterfall_plot visually breaks the XGBoost prediction down into the
    individual contribution of each feature for THIS specific case:
      - Baseline (E[f(X)]) : mean model prediction over all crises
      - Red bar  : feature that INCREASES the predicted TTC
      - Blue bar : feature that DECREASES the predicted TTC
      - f(x)     : final predicted TTC for this episode

    Parameters
    ----------
    shap_explanation : shap.Explanation computed on the same X as df
    df               : source DataFrame (must contain gravity_col)
    output_path      : output PNG file
    gravity_col      : column used to locate the worst crisis

    Returns
    -------
    idx_worst (int) : index of the worst crisis in df (for log / report)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    # Locate the worst crisis (maximum gravity_score = most intense disruption)
    idx_worst = int(df[gravity_col].idxmax()) if gravity_col in df.columns else 0
    worst_episode_id = df.get("episode_id", pd.Series(["?"])).iloc[idx_worst] \
        if "episode_id" in df.columns else f"idx={idx_worst}"
    worst_gravity    = float(df[gravity_col].iloc[idx_worst]) if gravity_col in df.columns else np.nan
    worst_ttc        = float(df[target_col].iloc[idx_worst]) if target_col in df.columns else np.nan

    log.info(
        "Waterfall for the worst crisis: idx=%d | id=%s | gravity=%.0f | actual TTC=%.1f d",
        idx_worst, worst_episode_id, worst_gravity, worst_ttc,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_explanation[idx_worst], show=False)
    plt.title(
        f"SHAP Waterfall — Worst crisis: {worst_episode_id}\n"
        f"gravity_score={worst_gravity:,.0f}  |  actual TTC={worst_ttc:.1f} d",
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
    Scatter plot of predicted vs actual TTC (test set) with the perfect-prediction line.

    Useful to spot systematic model biases (e.g. underestimation
    of extreme crises).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    x = predictions_df["ttc_actual_d"]
    y = predictions_df["ttc_predicted_d"]

    ax.scatter(x, y, alpha=0.7, edgecolors="k", linewidths=0.5, s=60, color="steelblue")

    # Perfect line y = x
    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")

    ax.set_xlabel("Actual TTC (days)", fontsize=11)
    ax.set_ylabel("TTC predicted by XGBoost (days)", fontsize=11)
    ax.set_title("Time-to-clear: predicted vs actual (test set)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Predicted vs actual scatter → %s", output_path)


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
    Main orchestrator: train XGBoost + generate the 3 SHAP PNGs.

    Designed to be called from financial_pipeline.py after build_events_dataset():
        events_df["time_to_clear"] = events_df["ttc_days"]
        # Enrich with features computed from AIS...
        results = run_full_pipeline(df=events_df, output_dir=output_dir)
        xgb_model = results["model"]  # inject into generate_trading_signal()

    Parameters
    ----------
    df          : DataFrame with features + target (see load_episode_features)
    output_dir  : output folder for the PNGs and the model
    feature_cols: list of feature columns
    target_col  : target column
    test_frac   : fraction of the dataset kept for testing (default: last 20% of episodes)
    xgb_params  : XGBoost hyper-parameters (default: XGB_PARAMS)
    save_model  : save the XGBoost model as JSON

    Returns
    -------
    dict with keys:
        model           : XGBRegressor trained on the full data
        metrics         : {rmse, mae, r2, n_obs} on the test set
        predictions_df  : per-episode DataFrame (test set) with predicted vs actual TTC
        shap_summary_path    : Path to the summary_plot PNG
        shap_waterfall_path  : Path to the waterfall PNG of the worst crisis
        scatter_path         : Path to the predicted-vs-actual scatter PNG
        idx_worst_crisis     : index of the worst crisis in df
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  TTC XGBoost + SHAP — %d episodes | %d features", len(df), len(feature_cols))
    log.info("=" * 60)

    # ── 1. Split chronologique ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, df_train, df_test = split_train_test_chronological(
        df, feature_cols, target_col, test_frac,
    )

    # ── 2. Training ──────────────────────────────────────────────────────────
    model = train_ttc_model(X_train, y_train, feature_cols, xgb_params)

    # ── 3. Evaluation ────────────────────────────────────────────────────────
    eval_results = evaluate_model(model, X_test, y_test, df_test, feature_cols, target_col)
    metrics       = {k: v for k, v in eval_results.items() if k != "predictions_df"}
    predictions_df = eval_results["predictions_df"]

    # ── 4. Retrain on the full set (production model) ────────────────────────
    # After out-of-sample evaluation, train on everything to maximise the
    # robustness of the model used in generate_trading_signal().
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[target_col].values.astype(np.float32)
    model_full = train_ttc_model(X_full, y_full, feature_cols, xgb_params)

    # ── 5. SHAP (computed on the full dataset to cover every crisis) ──────────
    shap_explanation, shap_values_np, X_df = compute_shap_explanation(
        model_full, X_full, feature_cols,
    )

    summary_path   = output_dir / "shap_summary.png"
    waterfall_path = output_dir / "shap_waterfall_worst_crisis.png"
    scatter_path   = output_dir / "ttc_predicted_vs_actual.png"

    plot_shap_summary(shap_values_np, X_df, summary_path)
    idx_worst = plot_worst_crisis_waterfall(shap_explanation, df, waterfall_path, target_col=target_col)
    plot_predictions_vs_actual(predictions_df, scatter_path, target_col)

    # ── 6. Optional model saving ──────────────────────────────────────────────
    model_path = None
    if save_model:
        model_path = output_dir / "xgb_ttc_model.json"
        model_full.save_model(str(model_path))
        log.info("XGBoost model saved → %s", model_path)

    # ── 7. Summary report ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  XGBoost TTC Estimator — Results (test set: {metrics['n_obs']} episodes)")
    print(f"{'='*60}")
    print(f"  RMSE  = {metrics['rmse']:.2f} days")
    print(f"  MAE   = {metrics['mae']:.2f} days")
    print(f"  R²    = {metrics['r2']:.4f}")
    print(f"\n  Files generated in {output_dir}:")
    print(f"    shap_summary.png                  — global feature importance")
    print(f"    shap_waterfall_worst_crisis.png   — SHAP breakdown of the worst crisis")
    print(f"    ttc_predicted_vs_actual.png       — predicted vs actual on the test set")
    if model_path:
        print(f"    xgb_ttc_model.json                — exportable model")
    print(f"{'='*60}\n")

    pred_path = output_dir / "ttc_predictions_test.csv"
    predictions_df.to_csv(pred_path, index=False)
    log.info("Test-set predictions → %s", pred_path)

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
# SYNTHETIC DATA — for tests and quick demos
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(n_episodes: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic episode dataset to test the pipeline without AIS data.

    Columns match FEATURE_COLS + TARGET_COL (the output format of build_events_dataset).
    Simulated relationships:
      - higher gravity_score / peak_rho   → longer episodes (disruption magnitude)
      - larger area_under_rho             → longer episodes (total congestion load)
      - steeper (more negative) decay_rate → shorter episodes (fast clearing)
      - higher entry flux                  → slightly shorter episodes (channel still flowing)

    Does NOT replace real AIS data — test use only.
    """
    rng = np.random.default_rng(seed)

    n = n_episodes
    gravity_score   = rng.uniform(5_000, 400_000, n)   # a.u.
    n_ais_points    = rng.integers(200, 5_000, n)      # AIS coverage
    peak_rho        = rng.uniform(2.0, 30.0, n)        # vessels/km
    t_to_peak_days  = rng.uniform(1.0, 20.0, n)        # days
    decay_rate      = -rng.uniform(0.01, 0.5, n)       # vessels/km/day
    flux_entry_mean = rng.uniform(2.0, 20.0, n)        # vessels·kn/km
    area_under_rho  = peak_rho * rng.uniform(5.0, 40.0, n)

    # Episode duration = realistic signal (non-linear relationship + noise)
    duration_days = (
        3.0
        + 0.000015 * gravity_score
        + 0.02 * area_under_rho
        + 0.3 * t_to_peak_days
        + 8.0 * decay_rate
        - 0.1 * flux_entry_mean
        + rng.normal(0, 1.5, n)
    ).clip(min=1.0)

    dates = pd.date_range("2017-01-01", periods=n, freq="5D")

    return pd.DataFrame({
        "episode_id":      [f"ep_{i:03d}" for i in range(n)],
        "start_date":      dates,
        "gravity_score":   np.round(gravity_score, 1),
        "n_ais_points":    n_ais_points.astype(int),
        "peak_rho":        np.round(peak_rho, 3),
        "t_to_peak_days":  np.round(t_to_peak_days, 2),
        "area_under_rho":  np.round(area_under_rho, 2),
        "decay_rate":      np.round(decay_rate, 4),
        "flux_entry_mean": np.round(flux_entry_mean, 3),
        TARGET_COL:        np.round(duration_days, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Time-to-clear estimator with XGBoost + SHAP explanations. "
            "Intermediate layer between the PINN output and financial_pipeline.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Real data (output of build_events_dataset)
  python src/correlation/ttc_xgb_shap.py \\
      --input outputs/financial/houston_events.csv \\
      --alias-map ttc_days=time_to_clear waiting_vessels=anchorage_queue \\
      --output-dir outputs/figures/ttc_shap \\
      --save-model

  # Synthetic mode (quick test without data)
  python src/correlation/ttc_xgb_shap.py --synthetic --n-synthetic 80
        """,
    )

    # ── Input data ────────────────────────────────────────────────────────────
    grp_in = parser.add_mutually_exclusive_group(required=True)
    grp_in.add_argument(
        "--input", type=Path,
        help="Episode CSV or Parquet (must contain the features + target). "
             "Compatible with build_events_dataset() from financial_pipeline.py.",
    )
    grp_in.add_argument(
        "--synthetic", action="store_true",
        help="Generate realistic synthetic data to test the pipeline.",
    )

    parser.add_argument(
        "--n-synthetic", type=int, default=60,
        help="Number of synthetic episodes (only with --synthetic, default=60)",
    )
    parser.add_argument(
        "--alias-map", nargs="+", metavar="SRC=DST",
        help=(
            "Column renames in SRC=DST format (several allowed). "
            "Ex: --alias-map ttc_days=time_to_clear waiting_vessels=anchorage_queue"
        ),
    )

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--test-frac", type=float, default=0.20,
        help="Fraction of the dataset kept for testing (default: 0.20 = last 20%% of episodes)",
    )
    parser.add_argument(
        "--features", nargs="+", default=FEATURE_COLS,
        help=f"Feature columns (default: {FEATURE_COLS})",
    )
    parser.add_argument(
        "--target", default=TARGET_COL,
        help=f"Target column (default: {TARGET_COL})",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/figures/ttc_shap"),
        help="Output folder for the SHAP PNGs and the predictions CSV",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Save the final XGBoost model as JSON (reusable in financial_pipeline.py)",
    )

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Loading ──────────────────────────────────────────────────────────────
    if args.synthetic:
        log.info("--synthetic mode: generating %d fake episodes", args.n_synthetic)
        df = _make_synthetic_dataset(n_episodes=args.n_synthetic)
        log.info("Synthetic dataset created: %d rows | columns: %s", len(df), list(df.columns))
    else:
        alias_map: dict[str, str] = {}
        if args.alias_map:
            for pair in args.alias_map:
                if "=" not in pair:
                    log.warning("alias-map ignored (invalid format, expected SRC=DST): %s", pair)
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
        log.error("Too few episodes (%d). Minimum 4 required.", len(df))
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

    # Display the test-set prediction table
    print("\n── Test-set predictions ──")
    print(results["predictions_df"].to_string(index=False))
    print(f"\nResults → {args.output_dir}/")


if __name__ == "__main__":
    main()
