"""
Phase 3 — Inférence : Time to Clear.

Charge le PINN entraîné et calcule le "Time to Clear" (TTC) :
  Durée physique (en jours) pour que ρ(x, t) repasse sous le seuil baseline
  après le pic Harvey.

Méthode :
  1. On fixe x = 0.5 (centroid du chenal)
  2. On évalue ρ(0.5, t) sur une grille temporelle fine après le pic (t_peak)
  3. TTC = premier t où ρ_pred < ρ_baseline pendant N_consecutive jours consécutifs

Sauvegarde les résultats dans data/features/houston_time_to_clear.parquet.

Usage:
    python src/pinns/predict.py
    python src/pinns/predict.py --rho-threshold 0.85 --n-consecutive 3
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
import torch

from src.pinns.lwr_pinn import LWRPINN, get_device

log = logging.getLogger(__name__)

MODEL_PATH    = Path("outputs/models/lwr_pinn.pt")
FEATURES_PATH = Path("data/features/houston_daily_features.parquet")
OUTPUT_PATH   = Path("data/features/houston_time_to_clear.parquet")

HARVEY_PEAK   = date(2017, 8, 28)   # jour de densité minimale (port fermé)
RHO_THRESHOLD = 0.85                # fraction du ρ baseline → "retour à la normale"
N_CONSECUTIVE = 3                   # jours consécutifs sous le seuil pour valider


def load_model(device: torch.device, model_path: Path = MODEL_PATH) -> tuple[LWRPINN, dict]:
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {mp} — lancez train.py d'abord")

    checkpoint = torch.load(mp, map_location=device)
    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info("Modèle chargé (best_loss=%.6f, entraîné sur %s → %s)",
             checkpoint["best_loss"],
             checkpoint["train_start"],
             checkpoint["train_end"])
    return model, checkpoint


def compute_time_to_clear(
    rho_threshold:  float       = RHO_THRESHOLD,
    n_consecutive:  int         = N_CONSECUTIVE,
    n_eval_days:    int         = 60,
    features_path:  Path        = FEATURES_PATH,
    model_path:     Path        = MODEL_PATH,
    output_path:    Path        = OUTPUT_PATH,
    harvey_peak:    date | str  = HARVEY_PEAK,
) -> pl.DataFrame:
    """
    Évalue ρ(x=0.5, t) sur n_eval_days jours après le pic.
    Calcule le TTC et produit un DataFrame journalier avec :
        date, rho_pred, v_pred, is_cleared, time_to_clear_days
    """
    if isinstance(harvey_peak, str):
        harvey_peak = date.fromisoformat(harvey_peak)

    device = get_device()
    model, checkpoint = load_model(device, model_path=Path(model_path))

    train_start = date.fromisoformat(checkpoint["train_start"])
    train_end   = date.fromisoformat(checkpoint["train_end"])
    n_train_days = (train_end - train_start).days + 1

    # Baseline ρ : moyenne sur les jours normaux (hors fenêtre de crise)
    df_feat = pl.read_parquet(Path(features_path)).sort("date")
    # Use ±2 weeks around peak as "crisis" exclusion window
    crisis_start = harvey_peak - timedelta(days=14)
    crisis_end   = harvey_peak + timedelta(days=14)
    baseline_rho = float(
        df_feat
        .filter(~pl.col("date").is_between(crisis_start, crisis_end))
        .filter(pl.col("utilization_rate_rho") > 0)
        ["utilization_rate_rho"]
        .mean()
    )
    rho_threshold_abs = baseline_rho * rho_threshold
    log.info("Baseline ρ = %.4f | seuil TTC = %.4f", baseline_rho, rho_threshold_abs)

    # Grille temporelle d'évaluation : depuis le pic
    eval_dates = [harvey_peak + timedelta(days=i) for i in range(n_eval_days)]

    # Harvey fait chuter ρ → 0 (port fermé).
    # TTC = temps pour que ρ remonte AU-DESSUS du seuil (retour à la normale).
    rows = []
    consecutive_above = 0
    ttc_found         = False
    ttc_days          = None

    with torch.no_grad():
        for i, d in enumerate(eval_dates):
            t_norm = (d - train_start).days / max(n_train_days - 1, 1)
            t_norm = max(0.0, min(1.0, t_norm))

            x_t = torch.tensor([[0.5]], dtype=torch.float32, device=device)
            t_t = torch.tensor([[t_norm]], dtype=torch.float32, device=device)

            rho_pred, v_pred = model(x_t, t_t)
            rho_val = float(rho_pred.cpu().item())
            v_val   = float(v_pred.cpu().item())

            # Retour à la normale = ρ remonte au-dessus du seuil
            is_cleared = rho_val >= rho_threshold_abs
            if is_cleared:
                consecutive_above += 1
            else:
                consecutive_above = 0

            if consecutive_above >= n_consecutive and not ttc_found:
                ttc_days  = i - n_consecutive + 1
                ttc_found = True
                log.info("TTC détecté : %d jours après le pic (date=%s, rho=%.4f)",
                         ttc_days, eval_dates[ttc_days], rho_val)

            rows.append({
                "date":               d.isoformat(),
                "rho_pred":           round(rho_val, 6),
                "v_pred":             round(v_val,   6),
                "rho_threshold":      round(rho_threshold_abs, 6),
                "is_cleared":         is_cleared,
                "consecutive_above":  consecutive_above,
                "time_to_clear_days": ttc_days,
            })

    df_out = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())

    if ttc_found:
        log.info("Time to Clear = %d jours (depuis le pic %s)", ttc_days, harvey_peak)
    else:
        log.warning("TTC non atteint dans la fenêtre de %d jours — augmenter n_eval_days", n_eval_days)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out)
    log.info("Sauvegardé → %s", out)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — Time to Clear inference")
    parser.add_argument("--rho-threshold", type=float, default=RHO_THRESHOLD,
                        help="Fraction du ρ baseline (défaut 0.85)")
    parser.add_argument("--n-consecutive", type=int,   default=N_CONSECUTIVE,
                        help="Jours consécutifs sous le seuil (défaut 3)")
    parser.add_argument("--n-eval-days",   type=int,   default=60,
                        help="Jours à évaluer après le pic (défaut 60)")
    args = parser.parse_args()

    df = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
        n_eval_days=args.n_eval_days,
    )
    print(df.select(["date", "rho_pred", "v_pred", "is_cleared", "time_to_clear_days"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
