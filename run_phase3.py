"""
Phase 3 orchestrator — PINNs / Time to Clear.

Usage (run from Nowcasting/ root):
    python run_phase3.py --location la
    python run_phase3.py --location la --skip-train
    python run_phase3.py --location la --epochs 3000 --lr 1e-3
    python run_phase3.py --location la --skip-train --skip-viz
"""
import argparse
import datetime
import logging
from pathlib import Path

from src.ingestion.download import LOCATIONS

import numpy as np
import polars as pl
from src.pinns.train import train, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_LAMBDA_KIN
from src.pinns.predict import compute_time_to_clear

log = logging.getLogger(__name__)

TRAIN_CONFIG = {
    "houston": {
        "train_start": "2017-06-01",
        "train_end":   "2018-06-30",   # Harvey (sept 2017) + récupération complète (13 mois)
        "peak_date":   "2017-09-07",   # pic réel waiting_capacity post-Harvey
    },
    "la": {
        "train_start": "2021-01-01",
        "train_end":   "2022-12-31",   # backlog (pic jan 2022) + clearing (T3-T4 2022)
        "peak_date":   "2022-01-06",   # pic réel waiting_capacity COVID backlog
    },
}


def _select_peak_from_gravity(
    gravity_path: Path,
    percentile: float,
) -> tuple[str | None, float | None, float | None]:
    if not gravity_path.exists():
        return None, None, None
    gdf = pl.read_parquet(gravity_path).sort("date")
    scores = gdf["gravity_score"].to_numpy().astype(float)
    if scores.size == 0:
        return None, None, None
    threshold = float(np.quantile(scores, percentile))
    candidates = (
        gdf.filter(pl.col("gravity_score") >= threshold)
        .sort("gravity_score", descending=True)
    )
    if len(candidates) == 0:
        return None, None, threshold
    peak_date = candidates["date"][0].isoformat()
    peak_score = float(candidates["gravity_score"][0])
    return peak_date, peak_score, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — PINNs + Time to Clear")
    parser.add_argument("--location",      default="la", choices=list(LOCATIONS.keys()))
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--skip-viz",      action="store_true")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda-pde",         type=float, default=0.1)
    parser.add_argument("--lambda-bc",          type=float, default=0.1)
    parser.add_argument("--lambda-kin",         type=float, default=DEFAULT_LAMBDA_KIN)
    parser.add_argument("--curriculum-warmup",  type=int,   default=500)
    parser.add_argument("--obs-per-day",        type=int,   default=5)
    parser.add_argument("--lra",                action="store_true", default=True)
    parser.add_argument("--no-lra",             dest="lra", action="store_false")
    parser.add_argument("--rho-threshold",      type=float, default=0.85)
    parser.add_argument("--n-consecutive",      type=int,   default=3)
    parser.add_argument("--gravity-weight",     type=float, default=0.3)
    parser.add_argument("--ttc-mode",           default="threshold",
                        choices=["threshold", "threshold_relaxed", "ma_threshold", "inflection"])
    parser.add_argument("--relaxation-margin",  type=float, default=0.05)
    parser.add_argument("--ma-window",          type=int,   default=7)
    parser.add_argument("--train-start",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--train-end",     default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--peak-date",     default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--auto-peak",     action="store_true")
    parser.add_argument("--gravity-percentile", type=float, default=0.9)
    args = parser.parse_args()

    loc = args.location
    cfg = TRAIN_CONFIG.get(loc, TRAIN_CONFIG["la"])

    train_start  = args.train_start or cfg["train_start"]
    train_end    = args.train_end   or cfg["train_end"]
    peak_date    = args.peak_date   or cfg["peak_date"]

    features_path = Path(f"data/features/{loc}_daily_features.parquet")
    zones_path    = Path(f"data/features/{loc}_constituent_zones.parquet")
    gravity_path  = Path(f"data/features/{loc}_gravity_daily.parquet")
    model_path    = Path(f"outputs/models/{loc}_lwr_pinn.pt")
    output_path   = Path(f"data/features/{loc}_time_to_clear.parquet")

    log.info("=== Phase 3 — %s ===", loc.upper())
    log.info("Features     : %s", features_path)
    log.info("Zones        : %s (%s)", zones_path, "✓" if zones_path.exists() else "absent")
    log.info("Gravity      : %s (%s)", gravity_path, "✓" if gravity_path.exists() else "absent")
    log.info("Model        : %s", model_path)
    log.info("Train window : %s → %s", train_start, train_end)
    if args.auto_peak:
        auto_peak, auto_score, auto_threshold = _select_peak_from_gravity(
            gravity_path, args.gravity_percentile,
        )
        if auto_peak is None:
            log.error("Gravity peak introuvable — impossible de declencher le PINN")
            return
        peak_date = auto_peak
        log.info("Peak date    : %s (auto, score=%.4f, %.0fth pct=%.4f)",
                 peak_date, auto_score, args.gravity_percentile * 100, auto_threshold)
    else:
        log.info("Peak date    : %s", peak_date)

    if not features_path.exists():
        log.error("Features file not found: %s — run Phase 1+2 first", features_path)
        return

    # ── Entraînement ──────────────────────────────────────────────────────────
    if not args.skip_train:
        if not zones_path.exists():
            log.error(
                "Manifold géospatial introuvable : %s\n"
                "Lancez d'abord : python run_phase2.py --location %s ...",
                zones_path, loc,
            )
            return
        log.info("--- Entraînement PINN (%d epochs, curriculum=%d, LRA=%s) ---",
                 args.epochs, args.curriculum_warmup, args.lra)
        train(
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_kin=args.lambda_kin,
            train_start=train_start,
            train_end=train_end,
            features_path=features_path,
            zones_path=zones_path,
            gravity_path=gravity_path if gravity_path.exists() else None,
            curriculum_warmup=args.curriculum_warmup,
            obs_per_day=args.obs_per_day,
            use_lra=args.lra,
            model_path=model_path,
        )
    else:
        if not model_path.exists():
            log.error("Modèle non trouvé : %s — relancez sans --skip-train", model_path)
            return
        log.info("--- Skip entraînement (modèle existant) ---")

    # ── Inférence Time to Clear ────────────────────────────────────────────────
    log.info("--- Inférence Time to Clear (mode=%s) ---", args.ttc_mode)
    df = compute_time_to_clear(
        rho_threshold=args.rho_threshold,
        n_consecutive=args.n_consecutive,
        features_path=features_path,
        model_path=model_path,
        output_path=output_path,
        harvey_peak=peak_date,
        gravity_path=gravity_path if gravity_path.exists() else None,
        gravity_weight=args.gravity_weight,
        ttc_mode=args.ttc_mode,
        relaxation_margin=args.relaxation_margin,
        ma_window=args.ma_window,
    )

    # Résumé TTC
    ttc = df["time_to_clear_days"].drop_nulls()
    if len(ttc) > 0:
        ttc_val  = int(ttc[0])
        peak_dt  = datetime.date.fromisoformat(peak_date)
        clear_dt = peak_dt + datetime.timedelta(days=ttc_val)
        log.info("=" * 55)
        log.info("TIME TO CLEAR : %d jours après le pic (%s)", ttc_val, peak_date)
        log.info("Date de retour à la normale estimée : %s", clear_dt)
        log.info("=" * 55)
    else:
        log.warning("Time to Clear non atteint dans la fenêtre d'évaluation")

    import polars as pl
    print(f"\n--- Évolution ρ prédit {loc.upper()} (premiers 20 jours post-pic) ---")
    print(df.select(["date", "rho_pred", "v_pred", "is_cleared", "time_to_clear_days"]).head(20))

    # ── Visualisation ──────────────────────────────────────────────────────────
    if not args.skip_viz:
        log.info("--- Génération des visualisations ---")
        try:
            from src.pinns.visualize_pinn import visualize_all
            paths = visualize_all(
                location=loc,
                peak_date=peak_date,
                train_start=train_start,
                train_end=train_end,
            )
            print("\n--- Figures générées ---")
            for name, p in paths.items():
                print(f"  {name:<14} → {p}")
        except Exception as exc:
            log.warning("Visualisation échouée : %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger(__name__)
    main()
