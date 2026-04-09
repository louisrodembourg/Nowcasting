"""
Phase 3 orchestrator — PINNs / Time to Clear.

Usage (run from Nowcasting/ root):
    python run_phase3.py                              # houston (default)
    python run_phase3.py --location la
    python run_phase3.py --location la --skip-train   # inférence seule
    python run_phase3.py --location la --epochs 5000 --lr 1e-3
"""
import argparse
import logging
from pathlib import Path

from src.ingestion.download import LOCATIONS
from src.pinns.train import train, DEFAULT_EPOCHS, DEFAULT_LR
from src.pinns.predict import compute_time_to_clear

log = logging.getLogger(__name__)

# Training windows + peak dates per location
TRAIN_CONFIG = {
    "houston": {
        "train_start": "2017-08-15",
        "train_end":   "2017-09-10",
        "peak_date":   "2017-08-28",
    },
    "la": {
        "train_start": "2019-05-01",
        "train_end":   "2019-10-31",
        "peak_date":   "2019-07-01",  # placeholder — update after Phase 2 analysis
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINNs + Time to Clear")
    parser.add_argument("--location",      default="houston", choices=list(LOCATIONS.keys()),
                        help="Target port (default: houston)")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde",    type=float, default=0.1)
    parser.add_argument("--lambda-bc",     type=float, default=0.1)
    parser.add_argument("--rho-threshold", type=float, default=0.85)
    parser.add_argument("--n-consecutive", type=int,   default=3)
    parser.add_argument("--train-start",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--train-end",     default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--peak-date",     default=None, metavar="YYYY-MM-DD")
    args = parser.parse_args()

    loc = args.location
    cfg = TRAIN_CONFIG.get(loc, TRAIN_CONFIG["houston"])

    features_path = Path(f"data/features/{loc}_daily_features.parquet")
    model_path    = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    output_path   = Path(f"data/features/{loc}_time_to_clear.parquet")

    train_start = args.train_start or cfg["train_start"]
    train_end   = args.train_end   or cfg["train_end"]
    peak_date   = args.peak_date   or cfg["peak_date"]

    log.info("=== Phase 3 — %s ===", loc.upper())
    log.info("Features    : %s", features_path)
    log.info("Model       : %s", model_path)
    log.info("Train window: %s → %s", train_start, train_end)
    log.info("Peak date   : %s", peak_date)

    if not features_path.exists():
        log.error("Features file not found: %s — run Phase 1 first", features_path)
        return

    if not args.skip_train:
        log.info("--- Entraînement PINN (%d epochs) ---", args.epochs)
        train(
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            train_start=train_start,
            train_end=train_end,
            features_path=features_path,
            model_path=model_path,
        )
    else:
        if not model_path.exists():
            log.error("Modèle non trouvé : %s — relancez sans --skip-train", model_path)
            return
        log.info("--- Skip entraînement (modèle existant : %s) ---", model_path)

    log.info("--- Inférence Time to Clear ---")
    df = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
        features_path=features_path,
        model_path=model_path,
        output_path=output_path,
        harvey_peak=peak_date,
    )

    import datetime
    ttc = df["time_to_clear_days"].drop_nulls()
    if len(ttc) > 0:
        ttc_val = int(ttc[0])
        peak_dt = datetime.date.fromisoformat(peak_date)
        clear_dt = peak_dt + datetime.timedelta(days=ttc_val)
        log.info("=" * 50)
        log.info("TIME TO CLEAR : %d jours après le pic (%s)", ttc_val, peak_date)
        log.info("Date de retour à la normale estimée : %s", clear_dt)
        log.info("=" * 50)
    else:
        log.warning("Time to Clear non atteint dans la fenêtre d'évaluation")

    import polars as pl
    print(f"\n--- Évolution ρ prédit {loc.upper()} (premiers 20 jours post-pic) ---")
    print(df.select(["date", "rho_pred", "v_pred", "is_cleared", "time_to_clear_days"]).head(20))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
