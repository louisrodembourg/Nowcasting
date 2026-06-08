"""
Phase 2 — Gravity Score sur manifold TEMPOREL (analyse exploratoire uniquement).

ATTENTION : Ce module calcule le gravity score depuis le manifold TEMPOREL
(lbo.py → {loc}_manifold.parquet). Il NE correspond PAS au gravity score
utilisé dans le pipeline Phase 3 (PINNs), qui provient du manifold GÉOSPATIAL :
    manifold_pipeline.py → {loc}_gravity_daily.parquet   ← utilisé par les PINNs
    run_phase2.py        → orchestrateur du pipeline géospatial

Prend le manifold en entrée (houston_manifold.parquet) et calcule
le Score de Gravité quotidien :

    gravity_score_i = Σ_c  |ϕ_c(i) - μ_c|  ×  waiting_capacity_i
                      ─────────────────────────────────────────────
                             Σ_c  σ_c  ×  baseline_capacity

Où :
  - ϕ_c(i)           : coordonnée du jour i sur le c-ième vecteur propre
  - μ_c, σ_c         : moyenne et écart-type de ϕ_c sur la période baseline (jours non-Harvey)
  - waiting_capacity : Σ(L×W) des navires en zone d'attente uniquement (congestion réelle)
  - baseline_capacity: médiane de waiting_capacity sur la baseline

Le score est normalisé à [0, 1] sur toute la période.
Les jours Harvey (port fermé) ont waiting_capacity=0 → score=0 par conception,
puis remplacés par la valeur max post-Harvey (réouverture = pic de gravité réelle).

Usage :
    python src/manifold/gravity_score.py
    python src/manifold/gravity_score.py --harvey-start 2017-08-25 --harvey-end 2017-08-31
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

MANIFOLD_PATH = Path("data/features/houston_manifold.parquet")
OUTPUT_PATH   = Path("data/features/houston_gravity_score.parquet")

HARVEY_START  = date(2017, 8, 25)
HARVEY_END    = date(2017, 8, 31)


def compute_gravity_score(
    df: pl.DataFrame,
    harvey_start: date = HARVEY_START,
    harvey_end:   date = HARVEY_END,
) -> pl.DataFrame:
    """
    Calcule le score de gravité pour chaque jour du DataFrame manifold.

    Retourne df avec deux colonnes supplémentaires :
        deviation_score : déviation brute du manifold (non normalisée)
        gravity_score   : score final ∈ [0, 1], pondéré par waiting_capacity
    """
    phi_cols = sorted([c for c in df.columns if c.startswith("phi_")])
    if not phi_cols:
        raise ValueError("Aucune colonne phi_ trouvée — exécutez d'abord lbo.py")

    dates = df["date"].to_list()

    # Masque baseline : jours en dehors de la fenêtre Harvey (opérations normales)
    baseline_mask = np.array([
        not (harvey_start <= d <= harvey_end) for d in dates
    ])

    phi_matrix = df.select(phi_cols).to_numpy()   # (N, n_components)

    # Moyenne et écart-type sur la baseline uniquement
    phi_baseline = phi_matrix[baseline_mask]
    mu    = phi_baseline.mean(axis=0)
    sigma = phi_baseline.std(axis=0)
    sigma[sigma == 0] = 1.0   # protection

    # Déviation normalisée par rapport à la baseline pour chaque jour
    deviation = np.abs((phi_matrix - mu) / sigma).mean(axis=1)  # (N,)

    # Poids de capacité : waiting_capacity / médiane baseline
    capacity        = df["waiting_capacity"].to_numpy().astype(float)
    baseline_cap    = np.median(capacity[baseline_mask & (capacity > 0)])
    if baseline_cap == 0:
        baseline_cap = 1.0
    cap_weight = capacity / baseline_cap

    # Score de gravité brut
    raw_score = deviation * cap_weight

    # Normalisation à [0, 1]
    score_max = raw_score.max()
    score_min = raw_score.min()
    if score_max > score_min:
        gravity_score = (raw_score - score_min) / (score_max - score_min)
    else:
        gravity_score = np.zeros_like(raw_score)

    log.info("Score de gravité : min=%.4f  max=%.4f  moyenne=%.4f",
             gravity_score.min(), gravity_score.max(), gravity_score.mean())

    # Enregistre les 10 meilleurs jours par score
    top_idx = np.argsort(gravity_score)[::-1][:10]
    log.info("Top 10 jours de gravité :")
    for i in top_idx:
        log.info("  %s  score=%.4f  capacité=%.0f  déviation=%.4f",
                 dates[i], gravity_score[i], capacity[i], deviation[i])

    return df.with_columns([
        pl.Series("deviation_score", deviation.tolist(),    dtype=pl.Float64),
        pl.Series("gravity_score",   gravity_score.tolist(), dtype=pl.Float64),
    ])


def run_gravity_score(
    harvey_start: date | str = HARVEY_START,
    harvey_end:   date | str = HARVEY_END,
    manifold_path: Path | None = None,
    output_path:   Path | None = None,
) -> pl.DataFrame:
    """Fonction wrapper pour convertir les dates en string et exécuter le pipeline complet."""
    from datetime import date as date_type
    if isinstance(harvey_start, str):
        harvey_start = date_type.fromisoformat(harvey_start)
    if isinstance(harvey_end, str):
        harvey_end = date_type.fromisoformat(harvey_end)

    src = Path(manifold_path) if manifold_path is not None else MANIFOLD_PATH
    out = Path(output_path)   if output_path   is not None else OUTPUT_PATH

    # Charge le manifold et calcule les scores de gravité
    df     = pl.read_parquet(src).sort("date")
    df_out = compute_gravity_score(df, harvey_start, harvey_end)

    # Sauvegarde la sortie
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out)
    log.info("Sauvegardé → %s", out)
    return df_out


def main() -> None:
    """Point d'entrée principal pour exécution autonome — calcul Phase 2 du score de gravité."""
    parser = argparse.ArgumentParser(description="Phase 2 — Score de Gravité")
    parser.add_argument("--harvey-start", default="2017-08-25", metavar="YYYY-MM-DD")
    parser.add_argument("--harvey-end",   default="2017-08-31", metavar="YYYY-MM-DD")
    args = parser.parse_args()

    df = run_gravity_score(
        harvey_start=date.fromisoformat(args.harvey_start),
        harvey_end=date.fromisoformat(args.harvey_end),
    )
    # Affiche les résultats triés par score de gravité (le plus élevé en premier)
    print(df.select(["date", "deviation_score", "gravity_score",
                     "waiting_capacity", "is_characteristic"]).sort("gravity_score", descending=True))


if __name__ == "__main__":
    # Configure le logging pour afficher les messages d'info avec timestamps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
