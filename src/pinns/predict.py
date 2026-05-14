"""
Phase 3 — Inférence : Time to Clear.

Nouveautés vs version Houston :
  - Seuil TTC adaptatif : modulé par le gravity_score du jour de pic (Phase 2).
    Un bouchon plus sévère (gravity_score élevé) requiert un retour à la normale
    plus complet avant d'être considéré comme "cleared".
  - lon_range et sog_max chargés depuis le checkpoint pour cohérence avec l'entraînement.

Méthode :
  1. x = 0.5 (centroïde du port)
  2. Évalue ρ(0.5, t) sur une grille temporelle fine après le pic
  3. TTC = premier t où ρ_pred ≥ ρ_threshold pendant N_consecutive jours consécutifs

Usage:
    python src/pinns/predict.py --location la
    python src/pinns/predict.py --location la --rho-threshold 0.85 --n-consecutive 3
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

RHO_THRESHOLD = 0.85
N_CONSECUTIVE = 3


def load_model(device: torch.device, model_path: Path) -> tuple[LWRPINN, dict]:
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {mp} — lancez train.py d'abord")
    checkpoint = torch.load(mp, map_location=device, weights_only=False)
    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info("Modèle chargé (best_loss=%.6f, %s → %s)",
             checkpoint["best_loss"],
             checkpoint["train_start"], checkpoint["train_end"])
    return model, checkpoint


def _get_peak_gravity(gravity_path: Path | None, peak_date: date) -> float:
    """Retourne le gravity_score du jour de pic (0.0 si non disponible)."""
    if gravity_path is None or not Path(gravity_path).exists():
        return 0.0
    try:
        gdf = pl.read_parquet(gravity_path).sort("date")
        row = gdf.filter(pl.col("date") == peak_date)
        if len(row) > 0:
            gs = float(row["gravity_score"][0])
            log.info("Gravity score au pic (%s) : %.4f", peak_date, gs)
            return gs
    except Exception as exc:
        log.warning("Impossible de lire gravity_path : %s", exc)
    return 0.0


def compute_time_to_clear(
    rho_threshold:  float          = RHO_THRESHOLD,
    n_consecutive:  int            = N_CONSECUTIVE,
    n_eval_days:    int            = 90,
    features_path:  Path | str     = Path("data/features/la_daily_features.parquet"),
    model_path:     Path | str     = Path("outputs/models/la_lwr_pinn.pt"),
    output_path:    Path | str     = Path("data/features/la_time_to_clear.parquet"),
    harvey_peak:    date | str     = date(2020, 8, 11),
    gravity_path:   Path | str | None = None,
    gravity_weight: float          = 0.3,
    time_mode:      str            = "train_window",
) -> pl.DataFrame:
    """
    Calcule le Time to Clear depuis le jour de pic.

        Seuil adaptatif :
        rho_threshold_abs = baseline_rho × rho_threshold × (1 + gravity_weight × gravity_peak)
    Un pic de gravity_score=1.0 avec gravity_weight=0.3 élève le seuil de 30 %.

        time_mode:
            - "train_window" : normalise et clippe t ∈ [0,1] (comportement historique)
            - "absolute"     : normalise par la fenêtre d'entraînement sans clip (extrapolation)
    """
    if isinstance(harvey_peak, str):
        harvey_peak = date.fromisoformat(harvey_peak)

    device = get_device()
    model, checkpoint = load_model(device, Path(model_path))

    train_start  = date.fromisoformat(checkpoint["train_start"])
    train_end    = date.fromisoformat(checkpoint["train_end"])
    n_train_days = (train_end - train_start).days + 1

    df_feat = pl.read_parquet(Path(features_path)).sort("date")
    crisis_start = harvey_peak - timedelta(days=14)
    crisis_end   = harvey_peak + timedelta(days=14)
    df_baseline  = df_feat.filter(~pl.col("date").is_between(crisis_start, crisis_end))

    # ρ normalisée via blocked_capacity (comme dans train.py) ou utilization_rate_rho
    if "blocked_capacity" in df_feat.columns and df_feat["blocked_capacity"].max() > 0:
        cap_arr  = df_feat["blocked_capacity"].to_numpy().astype(float)
        cap_95   = float(np.percentile(cap_arr[cap_arr > 0], 95))
        base_cap = df_baseline.filter(pl.col("blocked_capacity") > 0)["blocked_capacity"].to_numpy()
        baseline_rho = float(np.mean(base_cap) / cap_95)
        # TTC = quand la capacité bloquée REDESCEND sous le seuil (crisis clearing)
        ttc_direction = "below"
        log.info("ρ = blocked_capacity norm. (95p=%.0f) | baseline_ρ=%.4f", cap_95, baseline_rho)
    else:
        baseline_rho  = float(
            df_baseline.filter(pl.col("utilization_rate_rho") > 0)
            ["utilization_rate_rho"].mean()
        )
        ttc_direction = "above"

    # Seuil adaptatif : gravity_score élève le seuil (crise plus grave → besoin de descendre plus bas)
    gravity_peak = _get_peak_gravity(
        Path(gravity_path) if gravity_path else None, harvey_peak
    )
    if ttc_direction == "below":
        # Pour LA : seuil = baseline × (1 + gravity_weight × gravity_peak)
        # Plus grave → doit descendre PLUS BAS que baseline → seuil plus haut = plus difficile à franchir
        rho_threshold_abs = min(baseline_rho * (1.0 + gravity_weight * gravity_peak), 0.98)
    else:
        # Pour Houston : seuil = baseline × rho_threshold
        rho_threshold_abs = baseline_rho * rho_threshold

    log.info("Baseline ρ=%.4f | gravity_peak=%.4f | seuil TTC=%.4f | direction=%s",
             baseline_rho, gravity_peak, rho_threshold_abs, ttc_direction)

    eval_dates = [harvey_peak + timedelta(days=i) for i in range(n_eval_days)]

    rows = []
    consecutive_above = 0
    ttc_found         = False
    ttc_days          = None

    with torch.no_grad():
        for i, d in enumerate(eval_dates):
            t_norm = (d - train_start).days / max(n_train_days - 1, 1)
            if time_mode == "train_window":
                t_norm = float(np.clip(t_norm, 0.0, 1.0))
            elif time_mode == "absolute":
                t_norm = float(t_norm)
            else:
                raise ValueError(f"time_mode invalide: {time_mode}")

            x_t = torch.tensor([[0.5]], dtype=torch.float32, device=device)
            t_t = torch.tensor([[t_norm]], dtype=torch.float32, device=device)

            rho_pred = model(x_t, t_t)
            v_pred   = model.greenshields_v(rho_pred)
            rho_val = float(rho_pred.cpu().item())
            v_val   = float(v_pred.cpu().item())

            if ttc_direction == "below":
                is_cleared = rho_val <= rho_threshold_abs
            else:
                is_cleared = rho_val >= rho_threshold_abs

            if is_cleared:
                consecutive_above += 1
            else:
                consecutive_above = 0

            if consecutive_above >= n_consecutive and not ttc_found:
                ttc_days  = i - n_consecutive + 1
                ttc_found = True
                log.info("TTC détecté : %d jours après le pic (date=%s, ρ=%.4f)",
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
        log.warning("TTC non atteint dans la fenêtre de %d jours", n_eval_days)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out)
    log.info("Sauvegardé → %s", out)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — Time to Clear inference")
    parser.add_argument("--location",       default="la", choices=["houston", "la"])
    parser.add_argument("--rho-threshold",  type=float, default=RHO_THRESHOLD)
    parser.add_argument("--n-consecutive",  type=int,   default=N_CONSECUTIVE)
    parser.add_argument("--n-eval-days",    type=int,   default=90)
    parser.add_argument("--peak-date",      default="2020-08-11")
    parser.add_argument("--gravity-weight", type=float, default=0.3)
    parser.add_argument("--time-mode",      default="train_window",
                        choices=["train_window", "absolute"])
    parser.add_argument("--features-path",  default=None)
    parser.add_argument("--gravity-path",   default=None)
    parser.add_argument("--model-path",     default=None)
    parser.add_argument("--output-path",    default=None)
    args = parser.parse_args()

    loc = args.location
    features_path = Path(args.features_path) if args.features_path else Path(
        f"data/features/{loc}_daily_features.parquet"
    )
    gravity_path = Path(args.gravity_path) if args.gravity_path else Path(
        f"data/features/{loc}_gravity_daily.parquet"
    )
    model_path = Path(args.model_path) if args.model_path else Path(
        f"outputs/models/{loc}_lwr_pinn.pt"
    )
    output_path = Path(args.output_path) if args.output_path else Path(
        f"data/features/{loc}_time_to_clear.parquet"
    )

    df = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
        n_eval_days=args.n_eval_days,
        features_path=features_path,
        model_path=model_path,
        output_path=output_path,
        harvey_peak=args.peak_date,
        gravity_path=gravity_path,
        gravity_weight=args.gravity_weight,
        time_mode=args.time_mode,
    )
    print(df.select(["date", "rho_pred", "v_pred", "is_cleared", "time_to_clear_days"]).head(20))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
