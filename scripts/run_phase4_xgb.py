"""
Script de lancement Phase 4 — XGBoost TTC
Résumés physiques extraits depuis les champs PINN interpolés (grille dense).
Modèles PINN annuels (1 sortie rho) chargés via LWRPINN_v1.
Dataset multi-années LA 2019-2024.
"""
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pinns.lwr_pinn import get_device
from src.pinns.pinn_pipeline import structure_episodes
from src.correlation.ttc_xgb_shap import TARGET_COL, XGB_PARAMS

FEATURE_COLS = [
    "gravity_score",
    "n_ais_points",
    "peak_rho",
    "t_to_peak_days",
    "area_under_rho",
    "decay_rate",
]

# ── Config ─────────────────────────────────────────────────────────────────────
LOCATION       = "la"
GRAVITY_PATH   = Path("data/features/la_gravity_all_years.parquet")
OUTPUT_CSV     = Path("outputs/financial/phase4_xgb_results.csv")
OUTPUT_DIR     = Path("outputs/figures/phase4_xgb")
GRAVITY_THRESH = 5_000
MIN_EP_DAYS    = 3
X_GRID         = 80
T_GRID         = 80
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modèles PINN disponibles par année (architecture 1-sortie)
YEAR_MODELS = {
    2019: Path("outputs/models/la_2019_lwr_pinn.pt"),
    2020: Path("outputs/models/la_2020_lwr_pinn.pt"),
    2021: Path("outputs/models/la_2021_lwr_pinn.pt"),
    2022: Path("outputs/models/la_2022_lwr_pinn.pt"),
    2023: Path("outputs/models/la_2023_lwr_pinn.pt"),
    2024: Path("outputs/models/la_2024_lwr_pinn.pt"),
}


# ── PINN legacy (1 sortie rho) ────────────────────────────────────────────────
class LWRPINN_v1(nn.Module):
    """Architecture originale des checkpoints annuels : (x,t) → rho uniquement."""
    def __init__(self, hidden_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        rho = torch.sigmoid(self.net(inp))
        return rho


def load_pinn_v1(path: Path) -> tuple[LWRPINN_v1, dict]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = LWRPINN_v1()
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    meta  = ckpt.get("global_meta", {
        "channel_len_km": 36.5, "global_rho_max": 30.0,
        "global_v_max": 12.0,
    })
    return model, meta


def pinn_summaries_v1(model: LWRPINN_v1, ep, meta: dict) -> dict:
    """
    Prédit rho sur grille dense via PINN, extrait les résumés scalaires.
    v n'est pas disponible (ancienne architecture 1 sortie).
    """
    device = get_device()
    model  = model.to(device)

    channel_len = meta.get("channel_len_km", 36.5)
    rho_max     = meta.get("global_rho_max", 30.0)
    t_max_days  = float(ep.duration_days) * 2.0 + 1.0

    x_grid = np.linspace(0.0, channel_len, X_GRID, dtype=np.float32)
    t_grid = np.linspace(0.0, t_max_days,  T_GRID, dtype=np.float32)
    xx, tt = np.meshgrid(x_grid, t_grid)

    x_norm = (xx.ravel() / channel_len).reshape(-1, 1)
    t_norm = (tt.ravel() / t_max_days).reshape(-1, 1)

    x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        rho_pred = model(x_t, t_t).cpu().numpy().ravel() * rho_max

    # Résumés sur max_x rho(x,t) par instant t
    rho_grid  = rho_pred.reshape(T_GRID, X_GRID)   # (t, x)
    rho_max_t = rho_grid.max(axis=1)                # max spatial par instant

    peak_idx      = int(np.argmax(rho_max_t))
    peak_rho      = float(rho_max_t[peak_idx])
    t_to_peak     = float(t_grid[peak_idx])
    area_under_rho = float(np.trapezoid(rho_max_t, t_grid))

    post_t   = t_grid[peak_idx:]
    post_rho = rho_max_t[peak_idx:]
    decay_rate = float(np.polyfit(post_t, post_rho, 1)[0]) if len(post_t) >= 2 else 0.0

    return {
        "peak_rho":        peak_rho,
        "t_to_peak_days":  t_to_peak,
        "area_under_rho":  area_under_rho,
        "decay_rate":      decay_rate,
    }


# ── Chargement PINN annuels ───────────────────────────────────────────────────
loaded_models: dict[int, tuple] = {}
for year, path in YEAR_MODELS.items():
    if path.exists():
        try:
            m, meta = load_pinn_v1(path)
            loaded_models[year] = (m, meta)
            log.info("PINN %d chargé (%s)", year, path.name)
        except Exception as e:
            log.warning("PINN %d échoué: %s", year, e)
    else:
        log.warning("PINN %d absent: %s", year, path)

# ── Détection des épisodes (toutes années) ────────────────────────────────────
log.info("Détection des épisodes 2017-2024 (threshold=%d)...", GRAVITY_THRESH)
episodes, _ = structure_episodes(
    location=LOCATION,
    gravity_path=GRAVITY_PATH,
    gravity_threshold=GRAVITY_THRESH,
    min_episode_days=MIN_EP_DAYS,
    start_date=date(2017, 1, 1),
    end_date=date(2024, 10, 31),
)
log.info("%d épisodes détectés", len(episodes))

# ── Construction du dataset avec résumés PINN ─────────────────────────────────
records = []
n_pinn_ok, n_pinn_fail = 0, 0
for ep in episodes:
    year = ep.start_date.year
    pinn_model, pinn_meta = loaded_models.get(year, (None, None))

    summaries = {"peak_rho": np.nan, "t_to_peak_days": np.nan,
                 "area_under_rho": np.nan, "decay_rate": np.nan}

    if pinn_model is not None and ep.n_points > 0:
        try:
            summaries = pinn_summaries_v1(pinn_model, ep, pinn_meta)
            n_pinn_ok += 1
        except Exception as e:
            log.warning("  PINN %s échoué: %s", ep.episode_id, e)
            n_pinn_fail += 1
    else:
        n_pinn_fail += 1

    log.info("  %s | %dj | %d pts | peak_rho=%.2f | decay=%.4f",
             ep.episode_id, ep.duration_days, ep.n_points,
             summaries["peak_rho"], summaries["decay_rate"])

    records.append({
        "episode_id":    ep.episode_id,
        "start_date":    ep.start_date,
        "end_date":      ep.end_date,
        "duration_days": ep.duration_days,
        "gravity_score": float(ep.trigger_score),
        "n_ais_points":  ep.n_points,
        **summaries,
    })

log.info("Résumés PINN: %d OK / %d fallback", n_pinn_ok, n_pinn_fail)

df_episodes = pd.DataFrame(records)
log.info("\nDataset épisodes:\n%s", df_episodes[["episode_id","duration_days","gravity_score","peak_rho","decay_rate"]].to_string())

# Supprimer les épisodes avec NaN sur les features PINN
n_before = len(df_episodes)
df_clean = df_episodes.dropna(subset=FEATURE_COLS).reset_index(drop=True)
log.info("%d/%d épisodes avec toutes les features PINN disponibles", len(df_clean), n_before)

if len(df_clean) < 4:
    log.error("Pas assez d'épisodes complets (%d) — on utilise gravity+n_ais seulement", len(df_clean))
    FEATURE_COLS_USED = ["gravity_score", "n_ais_points"]
    df_clean = df_episodes.dropna(subset=FEATURE_COLS_USED).reset_index(drop=True)
else:
    FEATURE_COLS_USED = FEATURE_COLS

# ── Split chronologique ───────────────────────────────────────────────────────
n_test  = max(1, int(len(df_clean) * 0.20))
n_train = len(df_clean) - n_test
df_train = df_clean.iloc[:n_train]
df_test  = df_clean.iloc[n_train:]

X_train = df_train[FEATURE_COLS_USED].values.astype(float)
y_train = df_train[TARGET_COL].values.astype(float)
X_test  = df_test[FEATURE_COLS_USED].values.astype(float)
y_test  = df_test[TARGET_COL].values.astype(float)

log.info("Split: train=%d  test=%d", n_train, n_test)

# ── Entraînement XGBoost ──────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
except ImportError:
    log.error("pip install xgboost"); sys.exit(1)

model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)

# ── Métriques ─────────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

def metrics(y_true, y_pred, label):
    mse  = float(np.mean((y_true - y_pred)**2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    # R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    # Baseline MAE (moyenne train)
    baseline_pred = np.full_like(y_pred, np.mean(y_train))
    mae_baseline  = float(np.mean(np.abs(y_true - baseline_pred)))
    log.info("%s | MSE=%.2f  MAE=%.2f j  R²=%.3f  MAE_baseline=%.2f j",
             label, mse, mae, r2, mae_baseline)
    return {"split": label, "mse": mse, "mae": mae, "r2": r2, "mae_baseline": mae_baseline}

results = [
    metrics(y_train, y_pred_train, "TRAIN"),
    metrics(y_test,  y_pred_test,  "TEST"),
]

# ── Importance des features ───────────────────────────────────────────────────
importances = dict(zip(FEATURE_COLS_USED, model.feature_importances_))
log.info("\nFeature importances (gain normalisé):")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    log.info("  %-20s %.4f  %s", feat, imp, bar)

# ── Détail épisodes test ──────────────────────────────────────────────────────
df_test = df_test.copy()
df_test["ttc_predicted"] = y_pred_test
df_test["error_days"]    = y_pred_test - y_test
log.info("\nDétail épisodes test:")
log.info(df_test[["episode_id","duration_days","ttc_predicted","error_days","gravity_score"]].to_string())

# ── SHAP ─────────────────────────────────────────────────────────────────────
try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_clean[FEATURE_COLS_USED].values)

    plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, df_clean[FEATURE_COLS_USED],
                      feature_names=FEATURE_COLS_USED, show=False)
    plt.tight_layout()
    shap_path = OUTPUT_DIR / "phase4_shap_summary.png"
    plt.savefig(shap_path, dpi=150)
    plt.close()
    log.info("SHAP summary → %s", shap_path)
except Exception as e:
    log.warning("SHAP échoué: %s", e)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
df_episodes.to_csv(OUTPUT_CSV, index=False)
log.info("Dataset épisodes → %s", OUTPUT_CSV)

log.info("\n=== RÉSUMÉ FINAL ===")
log.info("Épisodes total: %d | avec features PINN: %d", n_before, len(df_clean))
log.info("Features utilisées: %s", FEATURE_COLS_USED)
log.info("Target: %s", TARGET_COL)
for r in results:
    log.info("%s | MAE=%.1f j | R²=%.3f | vs baseline MAE=%.1f j",
             r["split"], r["mae"], r["r2"], r["mae_baseline"])
log.info("Feature importances: %s", {k: round(v,3) for k,v in sorted(importances.items(), key=lambda x:-x[1])})
