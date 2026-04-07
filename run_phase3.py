"""
Phase 3 orchestrator — PINNs / Time to Clear.

Usage (run from Nowcasting/ root):
    python run_phase3.py                        # entraîne + prédit
    python run_phase3.py --skip-train           # prédit seulement (modèle existant)
    python run_phase3.py --epochs 5000 --lr 1e-3
"""
import argparse
import logging
from pathlib import Path

from src.pinns.train import train, TRAIN_START, TRAIN_END, DEFAULT_EPOCHS, DEFAULT_LR
from src.pinns.predict import compute_time_to_clear


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINNs + Time to Clear")
    parser.add_argument("--skip-train",   action="store_true",
                        help="Sauter l'entraînement (utiliser le modèle existant)")
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde",   type=float, default=0.1)
    parser.add_argument("--lambda-bc",    type=float, default=0.1)
    parser.add_argument("--rho-threshold",type=float, default=0.85)
    parser.add_argument("--n-consecutive",type=int,   default=3)
    args = parser.parse_args()

    log.info("=== Phase 3 — PINNs / Time to Clear ===")

    if not args.skip_train:
        log.info("--- Entraînement PINN (%d epochs) ---", args.epochs)
        train(
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
        )
    else:
        model_path = Path("outputs/models/lwr_pinn.pt")
        if not model_path.exists():
            log.error("Modèle non trouvé — relancez sans --skip-train")
            return
        log.info("--- Skip entraînement (modèle existant) ---")

    log.info("--- Inférence Time to Clear ---")
    df = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
    )

    ttc = df["time_to_clear_days"].drop_nulls()
    if len(ttc) > 0:
        ttc_val = int(ttc[0])
        log.info("=" * 50)
        log.info("TIME TO CLEAR : %d jours après le pic Harvey (2017-08-28)", ttc_val)
        log.info("Date de retour à la normale estimée : %s",
                 (TRAIN_START.__class__(2017, 8, 28)
                  .__class__.fromordinal(
                      __import__("datetime").date(2017, 8, 28).toordinal() + ttc_val
                  )))
        log.info("=" * 50)
    else:
        log.warning("Time to Clear non atteint dans la fenêtre d'évaluation")

    import polars as pl
    print("\n--- Évolution ρ prédit (premiers 20 jours post-Harvey) ---")
    print(df.select(["date", "rho_pred", "v_pred", "is_cleared", "time_to_clear_days"]).head(20))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
