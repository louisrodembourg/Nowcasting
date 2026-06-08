"""
Phase 3 — PINN Physical Engine: pipeline en 4 étapes pour la dynamique
du trafic maritime lors d'événements de disruption.

Implémente un réseau de neurones informé par la physique (PINN) qui modélise
l'équation macroscopique de Lighthill-Whitham-Richards (LWR):
    ∂ρ/∂t + ∂(ρ·v)/∂x = 0

Structure en 4 étapes:
  Étape 1 — structure_episodes()        : découper les données en épisodes
                                           depuis le gravity score
  Étape 2 — LWRPINN (lwr_pinn.py)      : MLP (x, t) → (ρ̂, v̂)
  Étape 3 — hybrid loss (lwr_pinn.py)  : L_data + λ·L_LWR + λ_kin·L_cinématique
  Étape 4 — train_on_episodes()         : boucle d'entraînement épisodique
             extract_physical_profiles(): inférence dense avec comblement des
                                          trous par la physique

Usage:
    python src/pinns/pinn_pipeline.py --train  --location la --start 2017-01-01 --end 2017-12-31
    python src/pinns/pinn_pipeline.py --train  --location houston --start 2017-01-01 --end 2017-12-31
    python src/pinns/pinn_pipeline.py --infer  --model outputs/models/pinn_la_episodes.pt --start 2019-06-01 --end 2024-08-01
    python src/pinns/pinn_pipeline.py --both   --location houston
    python src/pinns/pinn_pipeline.py --data-only --location houston --start 2017-01-01 --end 2017-12-31

    python src/pinns/pinn_pipeline.py --infer `
  --model outputs/models/pinn_houston_episodes.pt `
  --start 2020-08-01 --end 2020-12-30 `
  --fine-tune `
  --fine-tune-epochs 300 `
  --fine-tune-lr 1e-4
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import polars as pl
import torch

from src.pinns.lwr_pinn import LWRPINN, get_device, total_loss
from src.pinns.data_prep import (
    CHANNEL_AXES,
    EPOCH_DATE,
    T_DAYS_MAX,
    build_rho_v_tensors,
    compute_channel_length,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/figures")
MODEL_DIR = Path("outputs/models")
FEATURES_DIR = Path("data/features")

# Points de collocation pour la contrainte physique
N_COLLOC = 2000       # nombre de points par tirage
COLLOC_REFRESH = 500  # reéchantillonnage tous les N époques
LOG_EVERY = 200       # fréquence de log

# ── Paramètres par localisation ──────────────────────────────────────────────
# gravity_threshold : seuil de déclenchement des épisodes.
#   La distribution du gravity score est bimodale (0 ou très grand).
#   Choisir un seuil au-dessus du bruit de fond mais en dessous des vrais pics.
#   → Ajuste ces valeurs si tu obtiens trop ou trop peu d'épisodes.
LOCATION_DEFAULTS: dict[str, dict] = {
    # Houston: non-zero scores 4k–370k | p50=37k | p75=83k | p90=134k
    # Harvey (août 2017) génère des pics >300k → seuil 50k capture Harvey + gros événements
    "houston": {
        "gravity_threshold": 5_000,
        "min_episode_days": 2,
    },
    # LA: distribution bimodale — 250 jours à 580–7500 (normal) + pics 100k–214k (crises)
    # p90=5700, p95=7500, p99=109k → seuil 5000 isole les vraies perturbations
    "la": {
        "gravity_threshold": 5_000,
        "min_episode_days": 3,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — STRUCTURATION DES DONNÉES EN ÉPISODES
#
# Un épisode commence (t=0) quand le gravity score dépasse un seuil de déclenchement.
# Pour chaque épisode, on:
#   1. Extrait les données AIS sur la fenêtre temporelle
#   2. Projette les coordonnées géographiques sur l'axe 1D du chenal (x en km)
#   3. Calcule la densité de navires (ρ, vessels/km) et la vitesse moyenne (v, noeuds)
#   4. Normalise pour le PINN:
#        x_norm = x_km / channel_len_km          ∈ [0, 1]
#        t_norm = (jour − début_épisode) / durée  ∈ [0, 1]  (relatif à l'épisode)
#        ρ_norm = ρ / ρ_max_global               ∈ [0, 1]
#        v_norm = v / v_max_global               ∈ [0, 1]
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """Un événement de disruption extrait de la série temporelle du gravity score."""
    episode_id: str
    location: str
    start_date: date
    end_date: date
    trigger_score: float    # gravity score au déclenchement
    duration_days: int
    # Coordonnées normalisées pour le PINN (N points d'observation)
    x_norm: np.ndarray      # position le long du chenal ∈ [0, 1]
    t_norm: np.ndarray      # temps relatif à l'épisode ∈ [0, 1]
    # Valeurs physiques brutes (avant normalisation globale)
    rho_physical: np.ndarray  # densité en vessels/km
    v_physical: np.ndarray    # vitesse en noeuds
    n_points: int
    channel_len_km: float


def _detect_episode_ranges(
    gravity_df: pl.DataFrame,
    threshold: float,
    min_days: int,
) -> list[tuple[date, date, float]]:
    """
    Parcourt la série de gravity score et retourne les plages
    (start_date, end_date, trigger_score) de chaque épisode.

    Un épisode est une suite continue de jours où gravity_score >= threshold,
    d'une durée d'au moins min_days jours.
    """
    sorted_df = gravity_df.sort("date")
    episodes: list[tuple[date, date, float]] = []
    in_run = False
    run_start: Optional[date] = None
    trigger_score = 0.0
    last_date: Optional[date] = None

    for row in sorted_df.iter_rows(named=True):
        d = date.fromisoformat(row["date"]) if isinstance(row["date"], str) else row["date"]
        score = float(row["gravity_score"])
        last_date = d

        if not in_run and score >= threshold:
            in_run = True
            run_start = d
            trigger_score = score

        elif in_run and score < threshold:
            run_end = d - timedelta(days=1)
            if (run_end - run_start).days + 1 >= min_days:
                episodes.append((run_start, run_end, trigger_score))
            in_run = False

    # Fermer un épisode qui s'étend jusqu'à la fin de la série
    if in_run and last_date is not None:
        if (last_date - run_start).days + 1 >= min_days:
            episodes.append((run_start, last_date, trigger_score))

    return episodes


def structure_episodes(
    location: str = "houston",
    gravity_path: Optional[Path] = None,
    gravity_threshold: float = 0.7,
    min_episode_days: int = 3,
    dx_km: float = 2.0,
    constituent_path: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> tuple[list[Episode], dict]:
    """
    Étape 1 — Découpe les données en épisodes de disruption et extrait
    les observations physiques (x, ρ, v) pour chaque épisode.

    Parameters
    ----------
    location          : 'houston' ou 'la'
    gravity_path      : chemin vers le gravity_daily.parquet
                        (défaut: data/features/<location>_gravity_daily.parquet)
    gravity_threshold : seuil de déclenchement d'un épisode
    min_episode_days  : durée minimale pour qu'un run soit considéré comme épisode
    dx_km             : largeur des bins spatiaux le long du chenal
    constituent_path  : filtrage optionnel sur les zones constituantes
    start_date        : borne inférieure de la période de scan (incluse)
    end_date          : borne supérieure de la période de scan (incluse)

    Returns
    -------
    episodes    : liste d'objets Episode (coordonnées normalisées pour le PINN)
    global_meta : {'channel_len_km', 'global_rho_max', 'global_v_max', 'dx_km', 'location'}
    """
    if gravity_path is None:
        gravity_path = FEATURES_DIR / f"{location}_gravity_daily.parquet"
    if not gravity_path.exists():
        raise FileNotFoundError(f"Fichier gravity score introuvable: {gravity_path}")

    gravity_df = pl.read_parquet(gravity_path)

    # Filtrer par période si fournie
    if start_date is not None:
        gravity_df = gravity_df.filter(pl.col("date") >= start_date.isoformat())
    if end_date is not None:
        gravity_df = gravity_df.filter(pl.col("date") <= end_date.isoformat())

    period_str = (
        f"{start_date or 'debut'} -> {end_date or 'fin'}"
        if (start_date or end_date)
        else "periode complete"
    )
    log.info(
        "Gravity score: %d lignes | periode: %s | fichier: %s",
        len(gravity_df), period_str, gravity_path,
    )

    date_ranges = _detect_episode_ranges(gravity_df, gravity_threshold, min_episode_days)
    if not date_ranges:
        log.warning(
            "Aucun épisode trouvé (threshold=%.2f, min_days=%d). Essayez un seuil plus bas.",
            gravity_threshold, min_episode_days,
        )
        return [], {}

    log.info("%d épisode(s) candidat(s) détecté(s) (threshold=%.2f)", len(date_ranges), gravity_threshold)

    waypoints = CHANNEL_AXES[location]
    channel_len_km = compute_channel_length(waypoints)

    raw_episodes: list[Episode] = []

    for start_d, end_d, trigger_score in date_ranges:
        ep_id = f"{location}_{start_d.strftime('%Y%m%d')}_{end_d.strftime('%Y%m%d')}"
        duration = (end_d - start_d).days + 1
        log.info("Traitement de l'épisode %s (%d jours)", ep_id, duration)

        try:
            X_raw, y_raw, meta_ep = build_rho_v_tensors(
                start_d,
                end_d,
                location,
                dx_km=dx_km,
                constituent_path=str(constituent_path) if constituent_path else None,
                use_raw_velocity=True,
            )
        except Exception as exc:
            log.warning("Épisode %s ignoré: %s", ep_id, exc)
            continue

        if len(X_raw) == 0:
            log.warning("Épisode %s ignoré: aucune donnée AIS.", ep_id)
            continue

        # x_norm = x_km / channel_len est déjà dans X_raw[:, 0] ∈ [0, 1]
        x_norm = X_raw[:, 0].astype(np.float32)

        # Convertir le temps global normalisé → temps relatif à l'épisode ∈ [0, 1]
        # t_global_norm = (jour - EPOCH_DATE).days / T_DAYS_MAX
        t_start_global = (start_d - EPOCH_DATE).days / T_DAYS_MAX
        t_end_global = (end_d - EPOCH_DATE).days / T_DAYS_MAX
        duration_norm = max(t_end_global - t_start_global, 1e-8)
        t_norm = np.clip(
            (X_raw[:, 1] - t_start_global) / duration_norm, 0.0, 1.0
        ).astype(np.float32)

        # Récupérer les valeurs physiques brutes (dénormaliser depuis les stats par épisode)
        # y_raw[:, 0] = (ρ - ρ_min) / (ρ_max - ρ_min)  →  ρ_physical = y_raw[:,0]*(ρ_max-ρ_min)+ρ_min
        rho_range = max(meta_ep["rho_max"] - meta_ep["rho_min"], 1e-8)
        rho_physical = (y_raw[:, 0] * rho_range + meta_ep["rho_min"]).astype(np.float32)

        # y_raw[:, 1] = v / v_max  →  v_physical = y_raw[:,1] * v_max
        v_physical = (y_raw[:, 1] * max(meta_ep["v_max"], 1e-8)).astype(np.float32)

        raw_episodes.append(Episode(
            episode_id=ep_id,
            location=location,
            start_date=start_d,
            end_date=end_d,
            trigger_score=trigger_score,
            duration_days=duration,
            x_norm=x_norm,
            t_norm=t_norm,
            rho_physical=rho_physical,
            v_physical=v_physical,
            n_points=len(x_norm),
            channel_len_km=channel_len_km,
        ))

    if not raw_episodes:
        log.error("Aucun épisode avec données AIS trouvé.")
        return [], {}

    # Calculer les constantes de normalisation globales (99e percentile pour ignorer les outliers)
    all_rho = np.concatenate([ep.rho_physical for ep in raw_episodes])
    all_v = np.concatenate([ep.v_physical for ep in raw_episodes])
    global_rho_max = float(max(np.percentile(all_rho, 99), 1e-4))
    global_v_max = float(max(np.percentile(all_v, 99), 1e-4))

    log.info(
        "Normalisation globale: ρ_max=%.2f vessels/km  v_max=%.2f kn  chenal=%.1f km",
        global_rho_max, global_v_max, channel_len_km,
    )
    log.info(
        "Total: %d points d'observation sur %d épisode(s)",
        sum(ep.n_points for ep in raw_episodes), len(raw_episodes),
    )

    global_meta = {
        "location": location,
        "channel_len_km": channel_len_km,
        "global_rho_max": global_rho_max,
        "global_v_max": global_v_max,
        "dx_km": dx_km,
    }

    return raw_episodes, global_meta


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — ARCHITECTURE DU PINN  (implémentée dans lwr_pinn.py)
#
# class LWRPINN(nn.Module):
#   Entrées : [x_norm, t_norm] ∈ [0, 1]²  (coordonnées normalisées épisode-relatives)
#   Sorties : [ρ̂_norm, v̂_norm] ∈ (0, 1)  (bornées par sigmoid)
#   Corps   : Linear(2, H) → Tanh → [Linear(H, H) → Tanh] × (L−1) → Linear(H, 2)
#   Défaut  : L=4 couches, H=64 unités
#
# Interprétation physique des sorties:
#   ρ_physique [vessels/km] = ρ̂_norm × global_rho_max
#   v_physique [noeuds]     = v̂_norm × global_v_max
# ════════════════════════════════════════════════════════════════════════════

# (LWRPINN importé depuis src.pinns.lwr_pinn)


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — FONCTION DE PERTE HYBRIDE  (implémentée dans lwr_pinn.py)
#
#   L_total = L_data  +  λ_pde · L_LWR  +  λ_kin · L_cinématique
#
#   L_data  = MSE(ρ̂, ρ_obs) + MSE(v̂, v_obs)
#             calculée sur les points d'observation AIS
#
#   L_LWR   = mean[ (∂ρ̂/∂t + ∂(ρ̂·v̂)/∂x)² ]
#             résidu de l'équation LWR, calculé via torch.autograd.grad
#             sur des points de collocation aléatoires dans [0,1]²
#
#   L_kin   = mean[ relu(|∂v̂/∂t| − a_max)  ]  (limite d'accélération navire)
#           + mean[ relu(|∂v̂/∂x| − g_max)  ]  (lissage spatial)
#           + mean[ relu(ρ̂ − 1) + relu(−ρ̂) ]  (bornes de densité)
# ════════════════════════════════════════════════════════════════════════════

# (total_loss importé depuis src.pinns.lwr_pinn)


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — BOUCLE D'ENTRAÎNEMENT + INFÉRENCE
# ════════════════════════════════════════════════════════════════════════════

def train_on_episodes(
    episodes: list[Episode],
    global_meta: dict,
    epochs: int = 2000,
    lr: float = 5e-4,
    lambda_pde: float = 0.1,
    lambda_kin: float = 0.05,
    n_colloc: int = N_COLLOC,
    colloc_refresh: int = COLLOC_REFRESH,
    model_name: Optional[str] = None,
) -> tuple[LWRPINN, list[float]]:
    """
    Étape 4a — Entraîne le PINN conjointement sur tous les épisodes historiques.

    Un seul modèle est entraîné sur l'ensemble poolé des observations (x_norm, t_norm),
    normalisées de façon cohérente entre épisodes grâce aux constantes globales.

    La contrainte physique LWR (Étape 3) est évaluée sur N_COLLOC points de
    collocation aléatoires dans [0,1]², rééchantillonnés tous les COLLOC_REFRESH
    époques pour couvrir le domaine complet.

    Le checkpoint sauvegardé inclut global_meta pour permettre la dénormalisation
    dans extract_physical_profiles.

    Returns
    -------
    model   : LWRPINN entraîné (meilleur état selon la loss totale)
    history : liste de la loss totale par époque
    """
    device = get_device()
    log.info("Dispositif: %s", device)

    global_rho_max = global_meta["global_rho_max"]
    global_v_max = global_meta["global_v_max"]

    # Pooler tous les épisodes en un seul jeu (X, y) normalisé globalement
    x_list, t_list, rho_list, v_list = [], [], [], []
    for ep in episodes:
        rho_norm = np.clip(ep.rho_physical / global_rho_max, 0.0, 1.0)
        v_norm = np.clip(ep.v_physical / global_v_max, 0.0, 1.0)
        x_list.append(ep.x_norm)
        t_list.append(ep.t_norm)
        rho_list.append(rho_norm.astype(np.float32))
        v_list.append(v_norm.astype(np.float32))

    x_data = torch.tensor(np.concatenate(x_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_data = torch.tensor(np.concatenate(t_list), dtype=torch.float32, device=device).unsqueeze(1)
    rho_obs = torch.tensor(np.concatenate(rho_list), dtype=torch.float32, device=device).unsqueeze(1)
    v_obs = torch.tensor(np.concatenate(v_list), dtype=torch.float32, device=device).unsqueeze(1)

    log.info(
        "Dataset poolé: %d points issus de %d épisode(s)",
        len(x_data), len(episodes),
    )

    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)

    best_loss = float("inf")
    best_state = None
    history: list[float] = []

    # Tirage initial des points de collocation (feuilles requises pour autograd)
    x_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)
    t_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)

    for epoch in range(1, epochs + 1):

        # Rééchantillonner les points de collocation périodiquement
        if epoch % colloc_refresh == 1 and epoch > 1:
            x_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)
            t_col = torch.rand(n_colloc, 1, device=device, requires_grad=True)

        optimizer.zero_grad()

        loss, comps = total_loss(
            model,
            x_data, t_data, rho_obs, v_obs,
            x_col, t_col,
            lambda_pde=lambda_pde,
            lambda_kin=lambda_kin,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        history.append(comps["total"])

        if comps["total"] < best_loss:
            best_loss = comps["total"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Époque %4d/%d  total=%.5f  data=%.5f  pde=%.5f  kin=%.5f",
                epoch, epochs,
                comps["total"], comps["data"], comps["pde"], comps["kin"],
            )

    if best_state:
        model.load_state_dict(best_state)
        log.info("Meilleur état restauré (loss=%.6f)", best_loss)

    # Sauvegarde du checkpoint avec les métadonnées de normalisation
    location = global_meta.get("location", "unknown")
    name = model_name or f"pinn_{location}_episodes"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{name}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "global_meta": global_meta,
            "best_loss": best_loss,
            "epochs": epochs,
            "n_episodes": len(episodes),
            "episode_ids": [ep.episode_id for ep in episodes],
        },
        save_path,
    )
    log.info("Checkpoint sauvegardé → %s", save_path)

    return model, history


# ════════════════════════════════════════════════════════════════════════════
# FINE-TUNING — Adaptation du PINN aux données d'un nouvel événement
# ════════════════════════════════════════════════════════════════════════════

def fine_tune_on_episode(
    model: LWRPINN,
    x_km: np.ndarray,
    t_days: np.ndarray,
    rho_physical: np.ndarray,
    v_physical: np.ndarray,
    global_meta: dict,
    t_max_days: float,
    epochs: int = 300,
    lr: float = 1e-4,
    lambda_pde: float = 0.1,
    lambda_kin: float = 0.05,
    n_colloc: int = 500,
) -> LWRPINN:
    """
    Fine-tune rapide du PINN sur les observations AIS d'un nouvel événement.

    Part des poids pré-entraînés et adapte le champ (ρ̂, v̂) aux données
    réelles de l'événement courant tout en conservant la contrainte physique
    LWR (∂ρ/∂t + ∂(ρv)/∂x = 0).

    Principe : le pré-entraînement a appris la physique générale du chenal
    (forme des ondes de choc, relation fondamentale). Le fine-tuning ajuste
    l'amplitude et la localisation spécifiques à l'événement observé.
    LR petit (1e-4) pour ne pas écraser la physique apprise.

    Parameters
    ----------
    model         : LWRPINN pré-entraîné (modifié en place + retourné)
    x_km          : positions x [km] des observations AIS
    t_days        : temps [jours] des observations (0 = onset de l'épisode)
    rho_physical  : densité [vessels/km] des observations
    v_physical    : vitesse [knots] des observations
    global_meta   : métadonnées de normalisation issues du checkpoint
    t_max_days    : durée de l'épisode en jours (pour normaliser t_norm)
    epochs        : nombre d'itérations (défaut: 300)
    lr            : learning rate (défaut: 1e-4, petit pour préserver la physique)
    lambda_pde    : poids de la contrainte LWR
    lambda_kin    : poids de la contrainte cinématique
    n_colloc      : points de collocation pour la contrainte physique

    Returns
    -------
    model fine-tuné (même objet, modifié en place)
    """
    if len(x_km) == 0:
        log.warning("Fine-tuning: aucune donnee AIS disponible — etape ignoree.")
        return model

    device = get_device()
    model = model.to(device)
    model.train()

    channel_len_km = global_meta["channel_len_km"]
    global_rho_max = global_meta["global_rho_max"]
    global_v_max   = global_meta["global_v_max"]

    # Normaliser les observations vers l'espace [0, 1]² du PINN
    x_norm   = np.clip(x_km / max(channel_len_km, 1e-8), 0, 1).astype(np.float32)
    t_norm   = np.clip(t_days / max(t_max_days, 1e-8), 0, 1).astype(np.float32)
    rho_norm = np.clip(rho_physical / max(global_rho_max, 1e-8), 0, 1).astype(np.float32)
    v_norm   = np.clip(v_physical / max(global_v_max, 1e-8), 0, 1).astype(np.float32)

    x_t   = torch.tensor(x_norm,   device=device).unsqueeze(1)
    t_t   = torch.tensor(t_norm,   device=device).unsqueeze(1)
    rho_t = torch.tensor(rho_norm, device=device).unsqueeze(1)
    v_t   = torch.tensor(v_norm,   device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log.info(
        "Fine-tuning: %d pts AIS | %d epochs | lr=%.0e | device=%s",
        len(x_km), epochs, lr, device,
    )

    x_col = t_col = None
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Reéchantillonner les collocation points tous les 100 epochs
        if (epoch - 1) % 100 == 0:
            x_col_np = np.random.rand(n_colloc).astype(np.float32)
            t_col_np = np.random.rand(n_colloc).astype(np.float32)
            x_col = torch.tensor(x_col_np, device=device, requires_grad=True).unsqueeze(1)
            t_col = torch.tensor(t_col_np, device=device, requires_grad=True).unsqueeze(1)

        optimizer.zero_grad()
        loss, _ = total_loss(
            model, x_t, t_t, rho_t, v_t,
            x_col, t_col, lambda_pde, lambda_kin,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv

        if epoch % 100 == 0 or epoch == 1:
            log.info("  FT [%d/%d]  loss=%.6f  best=%.6f", epoch, epochs, lv, best_loss)

    model.eval()
    log.info("Fine-tuning termine. best_loss=%.6f sur %d pts AIS", best_loss, len(x_km))
    return model


def extract_physical_profiles(
    model: LWRPINN,
    episode_data: pd.DataFrame,
    global_meta: dict,
    x_grid_points: int = 100,
    t_grid_points: int = 100,
    t_max_days: Optional[float] = None,
) -> pd.DataFrame:
    """
    Étape 4b — Inférence dense sur un nouvel épisode en cours.

    Prend les observations des premières heures/jours d'un épisode, passe les
    coordonnées (x, t) dans le PINN entraîné, et retourne un DataFrame propre
    contenant les profils physiques lissés et continus de densité et de vitesse.

    Les trous de données sont comblés intelligemment par la physique LWR intégrée
    dans le réseau: le PINN prédit sur une grille dense même là où il n'y a pas
    d'observation AIS.

    Parameters
    ----------
    model        : LWRPINN entraîné
    episode_data : DataFrame avec colonnes [x_km, t_days]
                     x_km   = position le long du chenal en km depuis l'entrée
                     t_days = jours écoulés depuis le début de l'épisode (t=0 à l'onset)
    global_meta  : métadonnées de normalisation issues de l'entraînement
                   (channel_len_km, global_rho_max, global_v_max)
    x_grid_points: résolution spatiale de la grille de sortie
    t_grid_points: résolution temporelle de la grille de sortie
    t_max_days   : horizon de prédiction en jours
                   (défaut: fenêtre observée × 1.2 + 1 jour)

    Returns
    -------
    DataFrame avec colonnes:
        x_km      : position le long du chenal [km]
        t_days    : temps depuis l'onset de l'épisode [jours]
        rho_hat   : densité prédite [vessels/km]   — lissée par physique LWR
        v_hat     : vitesse prédite [noeuds]       — lissée par physique LWR
        flux_hat  : débit prédit [vessels·kn/km]   = rho_hat × v_hat
    """
    model.eval()
    device = get_device()
    model = model.to(device)

    channel_len_km = global_meta["channel_len_km"]
    global_rho_max = global_meta["global_rho_max"]
    global_v_max = global_meta["global_v_max"]

    # Horizon de prédiction
    if t_max_days is None:
        t_max_days = float(episode_data["t_days"].max()) * 1.2 + 1.0

    # Grille d'évaluation dense en coordonnées physiques
    x_grid = np.linspace(0.0, channel_len_km, x_grid_points, dtype=np.float32)
    t_grid = np.linspace(0.0, t_max_days, t_grid_points, dtype=np.float32)
    xx, tt = np.meshgrid(x_grid, t_grid)  # shape: (t_grid_points, x_grid_points)

    x_flat = xx.ravel()
    t_flat = tt.ravel()

    # Normalisation vers l'espace d'entrée du PINN [0, 1]²
    x_norm = x_flat / max(channel_len_km, 1e-8)
    t_norm = t_flat / max(t_max_days, 1e-8)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(1)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        rho_pred_norm, v_pred_norm = model(x_tensor, t_tensor)

    # Dénormalisation vers les unités physiques
    rho_hat = rho_pred_norm.cpu().numpy().ravel() * global_rho_max
    v_hat = v_pred_norm.cpu().numpy().ravel() * global_v_max

    return pd.DataFrame({
        "x_km": x_flat,
        "t_days": t_flat,
        "rho_hat": rho_hat,
        "v_hat": v_hat,
        "flux_hat": rho_hat * v_hat,
    })


def plot_physical_profiles(
    profiles: pd.DataFrame,
    episode_data: Optional[pd.DataFrame] = None,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Visualise les profils physiques du PINN sous forme de 3 heatmaps interactives:
        - Densité   ρ(x, t)  [vessels/km]
        - Vitesse   v(x, t)  [knots]
        - Débit     q(x, t)  = ρ·v

    Axe x (horizontal) = temps depuis l'onset de l'épisode [jours]
    Axe y (vertical)   = position le long du chenal [km]

    Si episode_data est fourni (colonnes [x_km, t_days]), les observations AIS
    réelles sont superposées en scatter sur chaque panneau.

    Parameters
    ----------
    profiles     : DataFrame issu de extract_physical_profiles
    episode_data : observations AIS brutes à superposer (optionnel)
    title_suffix : texte ajouté au titre (ex: nom de l'épisode)
    output_path  : chemin du fichier HTML de sortie (défaut: outputs/figures/)

    Returns
    -------
    Path du fichier HTML sauvegardé
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly requis pour la visualisation: pip install plotly")

    # Pivoter le DataFrame long → grille 2D pour les heatmaps
    t_vals = np.sort(profiles["t_days"].unique())
    x_vals = np.sort(profiles["x_km"].unique())

    def _to_grid(col: str) -> np.ndarray:
        pivot = profiles.pivot_table(index="x_km", columns="t_days", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=x_vals, columns=t_vals)
        return pivot.values  # shape (n_x, n_t)

    rho_grid = _to_grid("rho_hat")
    v_grid = _to_grid("v_hat")
    flux_grid = _to_grid("flux_hat")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Densite rho(x,t)  [vessels/km]",
            "Vitesse v(x,t)    [knots]",
            "Debit  q(x,t)     [vessels*kn/km]",
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colorscales = ["Blues", "RdYlGn_r", "Oranges"]
    grids = [rho_grid, v_grid, flux_grid]
    rows = [1, 2, 3]

    for row, grid, cscale in zip(rows, grids, colorscales):
        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=t_vals,
                y=x_vals,
                colorscale=cscale,
                zsmooth="best",
                colorbar=dict(len=0.28, y=1.0 - (row - 1) * 0.36, thickness=12),
                hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km  val=%{z:.2f}<extra></extra>",
            ),
            row=row, col=1,
        )

        # Superposer les observations AIS réelles si disponibles
        if episode_data is not None and "x_km" in episode_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=episode_data["t_days"],
                    y=episode_data["x_km"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="white",
                        line=dict(color="black", width=1),
                        symbol="circle",
                    ),
                    name="Observations AIS",
                    showlegend=(row == 1),
                    hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km<extra>AIS</extra>",
                ),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="x [km]", row=1, col=1)
    fig.update_yaxes(title_text="x [km]", row=2, col=1)
    fig.update_yaxes(title_text="x [km]", row=3, col=1)
    fig.update_xaxes(title_text="Temps depuis l'onset [jours]", row=3, col=1)

    title = "PINN LWR — Profils physiques"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=900,
        template="plotly_white",
        legend=dict(x=1.02, y=0.5),
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn_profiles"
        output_path = OUTPUT_DIR / f"{stem}_physical_profiles.png"

    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Visualisation sauvegardee -> %s", output_path)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS LWR AVANCÉES
# ════════════════════════════════════════════════════════════════════════════

def plot_shockwave_heatmap(
    profiles: pd.DataFrame,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Diagramme spatio-temporel LWR — onde de choc.

    Convention LWR classique:
      - Axe X : espace x [km] (x=0 = entrée du chenal côté large, x=max = port)
      - Axe Y : temps t [heures] depuis l'onset (t=0 en haut, croissant vers le bas)
      - Couleur : densité ρ̂  (bleu=fluide → rouge foncé=saturé)

    L'onde de choc se manifeste comme une bande diagonale rouge partant du port
    (x élevé) et se propageant vers le large (x faible) au fil du temps.
    C'est la signature visuelle que le PINN a bien capturé la dynamique LWR.

    Returns path_html.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly requis: pip install plotly")

    profiles_h = profiles.copy()
    profiles_h["t_hours"] = profiles_h["t_days"] * 24.0

    t_u = np.sort(profiles_h["t_hours"].unique())
    x_u = np.sort(profiles_h["x_km"].unique())

    # z shape: (n_t, n_x) — t on rows (Y), x on columns (X)
    rho_grid = (
        profiles_h.pivot_table(index="t_hours", columns="x_km", values="rho_hat", aggfunc="mean")
        .reindex(index=t_u, columns=x_u)
        .values
    )

    colorscale = [
        [0.00, "#0a1628"],
        [0.25, "#1565C0"],
        [0.50, "#FFD600"],
        [0.75, "#E65100"],
        [1.00, "#7B0000"],
    ]

    rho_max = float(np.nanmax(rho_grid)) if not np.all(np.isnan(rho_grid)) else 1.0
    rho_mean = float(np.nanmean(rho_grid)) if not np.all(np.isnan(rho_grid)) else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=x_u,
            y=t_u,
            colorscale=colorscale,
            zsmooth="best",
            colorbar=dict(title=dict(text="rho<br>[v/km]", side="right"), thickness=16),
            hovertemplate=(
                "x = %{x:.1f} km<br>t = %{y:.1f} h<br>rho = %{z:.2f} v/km<extra></extra>"
            ),
        )
    )

    # Contour au front de choc (50% du pic) pour matérialiser l'onde
    shock_level = 0.5 * rho_max
    if shock_level > rho_mean * 0.1 and not np.isnan(shock_level):
        fig.add_trace(
            go.Contour(
                z=rho_grid,
                x=x_u,
                y=t_u,
                contours=dict(
                    start=shock_level, end=shock_level, size=1,
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(color="white", size=10),
                ),
                line=dict(color="white", width=2, dash="dot"),
                showscale=False,
                name=f"Front ({shock_level:.1f} v/km)",
            )
        )

    title = "PINN LWR — Onde de Choc  |  rho(x, t)"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(title="Distance le long du chenal [km]  (entree -> port)", showgrid=False),
        yaxis=dict(
            title="Temps depuis l'onset [heures]",
            autorange="reversed",
            showgrid=False,
        ),
        height=620,
        template="plotly_dark",
        annotations=[
            dict(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text="<b>-- Front de choc</b><br><i>se propage vers x=0 (large)</i>",
                showarrow=False,
                font=dict(color="white", size=11),
                align="left",
                bgcolor="rgba(0,0,0,0.45)",
                bordercolor="white", borderwidth=1,
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_shockwave.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Diagramme onde de choc sauvegarde -> %s", output_path)
    return output_path


def plot_ais_vs_pinn(
    profiles: pd.DataFrame,
    raw_x_km: np.ndarray,
    raw_t_days: np.ndarray,
    raw_rho: np.ndarray,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Comparaison cote-a-cote: donnees AIS brutes (trouees) vs sortie PINN (lisse).

    Demontre la capacite du PINN a combler les lacunes de couverture satellitaire
    via la contrainte physique LWR.

    Panneau gauche: observations AIS binees sur la grille PINN — zones blanches
                    = absence de signal AIS (trous de couverture).
    Panneau droit : sortie dense du PINN — continue et physiquement coherente
                    meme dans les zones sans observation.

    Parameters
    ----------
    profiles    : DataFrame issu de extract_physical_profiles (grille dense)
    raw_x_km    : positions x [km] des observations AIS
    raw_t_days  : temps [jours] des observations AIS (episode-relatif)
    raw_rho     : densite [vessels/km] des observations AIS
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly requis: pip install plotly")

    t_u = np.sort(profiles["t_days"].unique())
    x_u = np.sort(profiles["x_km"].unique())

    pinn_grid = (
        profiles.pivot_table(index="t_days", columns="x_km", values="rho_hat", aggfunc="mean")
        .reindex(index=t_u, columns=x_u)
        .values
    )

    # Binner les observations AIS brutes sur la grille PINN
    raw_grid = np.full((len(t_u), len(x_u)), np.nan, dtype=np.float32)
    if len(raw_x_km) > 0:
        dx = (x_u[1] - x_u[0]) / 2 if len(x_u) > 1 else 0.5
        dt = (t_u[1] - t_u[0]) / 2 if len(t_u) > 1 else 0.5
        x_edges = np.concatenate([[x_u[0] - dx], (x_u[:-1] + x_u[1:]) / 2, [x_u[-1] + dx]])
        t_edges = np.concatenate([[t_u[0] - dt], (t_u[:-1] + t_u[1:]) / 2, [t_u[-1] + dt]])
        raw_acc = np.zeros((len(t_u), len(x_u)), dtype=np.float64)
        raw_cnt = np.zeros((len(t_u), len(x_u)), dtype=np.int32)
        for xi, ti, ri in zip(raw_x_km, raw_t_days, raw_rho):
            xi_i = int(np.clip(np.searchsorted(x_edges, xi, side="right") - 1, 0, len(x_u) - 1))
            ti_i = int(np.clip(np.searchsorted(t_edges, ti, side="right") - 1, 0, len(t_u) - 1))
            raw_acc[ti_i, xi_i] += ri
            raw_cnt[ti_i, xi_i] += 1
        mask = raw_cnt > 0
        raw_grid[mask] = (raw_acc[mask] / raw_cnt[mask]).astype(np.float32)

    zmax = float(np.nanmax(pinn_grid)) if not np.all(np.isnan(pinn_grid)) else 1.0
    colorscale = [
        [0.00, "#0a1628"],
        [0.25, "#1565C0"],
        [0.50, "#FFD600"],
        [0.75, "#E65100"],
        [1.00, "#7B0000"],
    ]
    coverage_pct = 100.0 * float(np.sum(~np.isnan(raw_grid))) / raw_grid.size

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Ground Truth AIS  ({coverage_pct:.0f}% de couverture)",
            "Sortie PINN  (continu, sans lacunes)",
        ),
        horizontal_spacing=0.10,
    )

    for col, (grid, name) in enumerate([(raw_grid, "AIS brut"), (pinn_grid, "PINN")], start=1):
        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=x_u,
                y=t_u,
                colorscale=colorscale,
                zmin=0, zmax=zmax,
                zsmooth="best",
                showscale=(col == 2),
                colorbar=dict(title="rho<br>[v/km]", len=0.8, y=0.5, thickness=14)
                if col == 2 else None,
                hovertemplate=(
                    f"x=%{{x:.1f}}km  t=%{{y:.1f}}j<br>"
                    f"rho=%{{z:.2f}}<br><i>{name}</i><extra></extra>"
                ),
            ),
            row=1, col=col,
        )

    fig.update_yaxes(title_text="Temps [jours]", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="x [km]", row=1, col=1)
    fig.update_xaxes(title_text="x [km]", row=1, col=2)

    title = "PINN LWR — Lissage physique: AIS brut vs. Sortie PINN"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=550,
        template="plotly_dark",
        annotations=[
            dict(
                x=0.50, y=-0.14, xref="paper", yref="paper",
                text=(
                    "<i>Les zones blanches (gauche) = absence de signal AIS. "
                    "Le PINN (droite) comble ces lacunes par la contrainte physique LWR.</i>"
                ),
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_ais_vs_pinn.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Comparaison AIS vs PINN sauvegardee -> %s", output_path)
    return output_path


def plot_density_profiles_1d(
    profiles: pd.DataFrame,
    n_slices: int = 6,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Profils 1D de densite rho(x) a differents instants d'un episode.

    Montre l'evolution de la bosse de masse bloquee: plate au debut,
    elle gonfle devant l'entree du port (x eleve) et se propage vers le large.

    Chaque courbe est une coupe transversale rho(x) a t fixe.
    Gradient de couleur: bleu (debut) -> rouge (fin de l'episode).

    Parameters
    ----------
    profiles   : DataFrame issu de extract_physical_profiles
    n_slices   : nombre de coupes temporelles a tracer (defaut: 6)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly requis: pip install plotly")

    t_all = np.sort(profiles["t_days"].unique())

    indices = np.round(np.linspace(0, len(t_all) - 1, n_slices)).astype(int)
    t_slices = t_all[indices]

    # Gradient: bleu (hue=240) -> rouge (hue=0)
    colors = [
        f"hsl({int(240 - 240 * i / max(n_slices - 1, 1))}, 80%, 50%)"
        for i in range(n_slices)
    ]

    fig = go.Figure()
    for i, (t_val, color) in enumerate(zip(t_slices, colors)):
        idx = int(np.argmin(np.abs(t_all - t_val)))
        t_actual = t_all[idx]
        slice_df = profiles[np.isclose(profiles["t_days"], t_actual)].sort_values("x_km")
        if len(slice_df) == 0:
            continue

        label = f"t = {t_actual*24:.0f}h" if t_actual < 2 else f"t = {t_actual:.1f}j"
        fill_color = color.replace("hsl", "hsla").replace(")", ", 0.08)")

        fig.add_trace(
            go.Scatter(
                x=slice_df["x_km"].values,
                y=slice_df["rho_hat"].values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                fill="tozeroy",
                fillcolor=fill_color,
                hovertemplate=(
                    f"x=%{{x:.1f}} km<br>rho=%{{y:.2f}} v/km<br>{label}<extra></extra>"
                ),
            )
        )

    title = "PINN LWR — Profils de densite rho(x) a differents instants"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="Distance le long du chenal [km]  (entree -> port)",
            showgrid=True, gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title="Densite rho  [vessels/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        height=520,
        template="plotly_white",
        legend=dict(title="Instant (bleu=debut, rouge=fin)", x=1.02, y=0.5),
        annotations=[
            dict(
                x=0.50, y=-0.14, xref="paper", yref="paper",
                text=(
                    "<i>La bosse se forme a x eleve (entree du port) "
                    "et se propage vers x=0 (large) : onde cinematique LWR.</i>"
                ),
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_profiles_1d.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Profils 1D sauvegardes -> %s", output_path)
    return output_path


def plot_fundamental_diagram(
    profiles: pd.DataFrame,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Diagramme fondamental du trafic: flux q = rho * v en fonction de rho.

    Selon l'equation LWR (modele de Greenshields), la relation (rho, q) forme
    une parabole : q augmente avec rho jusqu'a la capacite maximale du chenal,
    puis s'effondre (embouteillage — vitesse -> 0).

    Si le PINN a bien appris la physique, le nuage de points reproduit
    naturellement cette parabole sans qu'elle soit imposee explicitement.
    C'est la preuve que le PINN agit comme un simulateur de physique,
    pas comme une boite noire.

    Le nuage est colore par le temps t [jours] pour montrer l'evolution
    chronologique de l'etat du trafic.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly requis: pip install plotly")

    rho = profiles["rho_hat"].values.astype(np.float64)
    q   = profiles["flux_hat"].values.astype(np.float64)
    t   = profiles["t_days"].values

    rho_p99 = float(np.percentile(rho, 99))

    # Fit Greenshields: q = a*rho + b*rho^2 (no intercept, b < 0)
    valid = (rho > 0.01 * rho_p99) & np.isfinite(q) & np.isfinite(rho)
    has_fit = False
    rho_opt_fit = rho_jam_fit = q_max_fit = v_free_fit = 0.0
    rho_theory = q_theory = np.array([])

    if valid.sum() >= 10:
        A = np.stack([rho[valid], rho[valid] ** 2], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, q[valid], rcond=None)
        a_c, b_c = float(coeffs[0]), float(coeffs[1])
        if b_c < 0:
            has_fit    = True
            rho_jam_fit = -a_c / b_c
            v_free_fit  = a_c
            q_max_fit   = a_c ** 2 / (-4.0 * b_c)
            rho_opt_fit = rho_jam_fit / 2.0
            rho_theory  = np.linspace(0, min(rho_jam_fit * 1.05, rho_p99 * 2.5), 300)
            q_theory    = a_c * rho_theory + b_c * rho_theory ** 2

    fig = go.Figure()

    # Nuage de points colore par le temps
    fig.add_trace(
        go.Scatter(
            x=rho, y=q,
            mode="markers",
            marker=dict(
                size=5,
                color=t,
                colorscale="RdYlBu_r",
                colorbar=dict(title="t [jours]", thickness=14),
                opacity=0.55,
                line=dict(width=0),
            ),
            name="Predictions PINN",
            hovertemplate=(
                "rho=%{x:.2f} v/km<br>q=%{y:.2f} v*kn/km<br>"
                "t=%{marker.color:.1f}j<extra></extra>"
            ),
        )
    )

    if has_fit:
        fig.add_trace(
            go.Scatter(
                x=rho_theory, y=q_theory,
                mode="lines",
                line=dict(color="red", width=2.5, dash="dash"),
                name=(
                    f"Greenshields ajuste<br>"
                    f"v_free={v_free_fit:.1f} kn, rho_jam={rho_jam_fit:.1f} v/km"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[rho_opt_fit], y=[q_max_fit],
                mode="markers+text",
                marker=dict(size=14, color="red", symbol="star"),
                text=[f"  q* = {q_max_fit:.2f}<br>  rho* = {rho_opt_fit:.2f}"],
                textposition="top right",
                name="Capacite maximale q*",
                textfont=dict(color="red", size=12),
            )
        )

    title = "PINN LWR — Diagramme Fondamental du Trafic  q = rho * v"
    if title_suffix:
        title += f"  |  {title_suffix}"

    caption = (
        "<i>La parabole (Greenshields) est une consequence de l'equation LWR. "
        "Si le PINN la reproduit naturellement, il a bien appris la physique "
        "du trafic maritime — pas une boite noire.</i>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(
            title="Densite rho  [vessels/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        yaxis=dict(
            title="Flux q = rho * v  [vessels*kn/km]",
            showgrid=True, gridcolor="#e0e0e0", rangemode="tozero",
        ),
        height=560,
        template="plotly_white",
        legend=dict(x=1.02, y=0.7),
        annotations=[
            dict(
                x=0.50, y=-0.13, xref="paper", yref="paper",
                text=caption,
                showarrow=False,
                font=dict(color="gray", size=11),
                align="center",
            )
        ],
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_fundamental_diagram.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format="png", scale=2)
    log.info("Diagramme fondamental sauvegarde -> %s", output_path)
    return output_path


def plot_congestion_clearance(
    profiles: pd.DataFrame,
    rho_threshold_frac: float = 0.3,
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> tuple[Path, Optional[float]]:
    """
    Visualise la dynamique de résorption de la congestion et calcule le TTC.

    La congestion est mesurée par max_x ρ(x,t) — la densité maximale le long
    du chenal à chaque instant. Le TTC est le premier jour où cette valeur
    passe sous le seuil = rho_threshold_frac × pic initial.

    Panel 1 — Courbe de résorption:
        max_x ρ(x,t) en fonction du temps, avec:
        - ligne de seuil pointillée rouge
        - zone grisée "congestion active"
        - marqueur vertical vert au TTC
        - annotation "TTC = X jours"

    Panel 2 — Heatmap spatiale ρ(x,t):
        Le même champ que plot_physical_profiles mais avec l'isoline rouge
        au niveau du seuil — montre visuellement quel bout du chenal se
        désengorge en premier et à quelle vitesse.

    Parameters
    ----------
    profiles            : DataFrame issu de extract_physical_profiles
    rho_threshold_frac  : fraction du pic initial définissant le seuil de
                          retour à la normale (défaut 0.3 = 30%)
    title_suffix        : texte ajouté au titre
    output_path         : chemin HTML de sortie

    Returns
    -------
    (path_html, ttc_days)  — ttc_days est None si non atteint dans la fenêtre
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly requis: pip install plotly")

    # ── Courbe de résorption: max_x ρ(x,t) ──────────────────────────────────
    t_series = (
        profiles.groupby("t_days")["rho_hat"]
        .max()
        .reset_index()
        .sort_values("t_days")
    )
    t_vals_c = t_series["t_days"].values
    rho_max_t = t_series["rho_hat"].values

    rho_peak = float(rho_max_t[0]) if len(rho_max_t) > 0 else 1.0
    threshold = rho_threshold_frac * rho_peak

    # TTC = premier t où max ρ < seuil
    ttc_days: Optional[float] = None
    below = np.where(rho_max_t < threshold)[0]
    if len(below) > 0:
        ttc_days = float(t_vals_c[below[0]])

    # ── Grille 2D pour la heatmap ────────────────────────────────────────────
    t_u = np.sort(profiles["t_days"].unique())
    x_u = np.sort(profiles["x_km"].unique())
    rho_grid = (
        profiles.pivot_table(index="x_km", columns="t_days", values="rho_hat", aggfunc="mean")
        .reindex(index=x_u, columns=t_u)
        .values
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.42, 0.58],
        subplot_titles=(
            "Courbe de resorption: max(rho(x,t)) le long du chenal",
            "Champ de densite rho(x,t)  [vessels/km]  avec isoline de seuil",
        ),
        vertical_spacing=0.12,
    )

    # — Panel 1 : courbe de résorption ————————————————————————————————
    # Zone grisée "congestion active" (rho > seuil)
    rho_clipped = np.where(rho_max_t > threshold, rho_max_t, threshold)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_vals_c, t_vals_c[::-1]]),
            y=np.concatenate([rho_clipped, np.full(len(t_vals_c), threshold)]),
            fill="toself",
            fillcolor="rgba(200, 50, 50, 0.15)",
            line=dict(width=0),
            name="Zone congestionnee",
            showlegend=True,
        ),
        row=1, col=1,
    )
    # Courbe principale
    fig.add_trace(
        go.Scatter(
            x=t_vals_c,
            y=rho_max_t,
            mode="lines",
            line=dict(color="#1565C0", width=2.5),
            name="max rho(x,t)",
        ),
        row=1, col=1,
    )
    # Ligne de seuil
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        annotation_text=f"Seuil ({rho_threshold_frac*100:.0f}% du pic = {threshold:.1f} v/km)",
        annotation_position="top right",
        row=1, col=1,
    )
    # Marqueur TTC
    if ttc_days is not None:
        fig.add_vline(
            x=ttc_days,
            line_dash="dot",
            line_color="green",
            line_width=2,
            row=1, col=1,
        )
        fig.add_annotation(
            x=ttc_days,
            y=rho_peak * 0.85,
            text=f"<b>TTC = {ttc_days:.1f} jours</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(color="green", size=13),
            ax=40, ay=-30,
            row=1, col=1,
        )
    else:
        fig.add_annotation(
            x=t_vals_c[-1] * 0.6,
            y=rho_peak * 0.5,
            text="TTC non atteint dans la fenetre",
            font=dict(color="orange", size=12),
            showarrow=False,
            row=1, col=1,
        )

    fig.update_yaxes(title_text="max rho [vessels/km]", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)

    # — Panel 2 : heatmap + isoline ──────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=rho_grid,
            x=t_u,
            y=x_u,
            colorscale="Blues",
            zsmooth="best",
            colorbar=dict(title="rho<br>[v/km]", len=0.45, y=0.22, thickness=14),
            hovertemplate="t=%{x:.1f}j  x=%{y:.1f}km  rho=%{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Isoline au seuil
    fig.add_trace(
        go.Contour(
            z=rho_grid,
            x=t_u,
            y=x_u,
            contours=dict(
                start=threshold, end=threshold, size=1,
                coloring="none",
                showlabels=True,
                labelfont=dict(color="red", size=11),
            ),
            line=dict(color="red", width=2, dash="dash"),
            showscale=False,
            name=f"Seuil {threshold:.1f} v/km",
        ),
        row=2, col=1,
    )
    # Ligne verticale TTC sur la heatmap
    if ttc_days is not None:
        fig.add_vline(
            x=ttc_days,
            line_dash="dot",
            line_color="green",
            line_width=2,
            row=2, col=1,
        )

    fig.update_yaxes(title_text="x [km]", row=2, col=1)
    fig.update_xaxes(title_text="Temps depuis l'onset [jours]", row=2, col=1)

    # ── Mise en page ─────────────────────────────────────────────────────────
    ttc_label = f"{ttc_days:.1f}j" if ttc_days is not None else "non atteint"
    title = f"PINN LWR — Resorption de la congestion  |  TTC = {ttc_label}"
    if title_suffix:
        title += f"  |  {title_suffix}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        height=750,
        template="plotly_white",
        legend=dict(x=1.02, y=0.85),
    )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = title_suffix.replace(" ", "_").replace("/", "-") or "pinn"
        output_path = OUTPUT_DIR / f"{stem}_clearance.png"

    fig.write_image(str(output_path), format="png", scale=2)
    log.info(
        "Clearance plot sauvegardes -> %s  (TTC=%s)",
        output_path, f"{ttc_days:.1f}j" if ttc_days is not None else "n/a",
    )
    return output_path, ttc_days


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — PINN physical engine (Étapes 1–4)"
    )
    parser.add_argument("--location", default="houston", choices=["houston", "la"])

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--train", action="store_true",
                      help="Étapes 1+2+3+4a: détecter les épisodes et entraîner le PINN")
    mode.add_argument("--infer", action="store_true",
                      help="Étape 4b: extraire les profils physiques depuis un checkpoint")
    mode.add_argument("--both", action="store_true",
                      help="Entraîner puis inférer (--train puis --infer)")
    mode.add_argument("--data-only", action="store_true",
                      help="Étape 1 uniquement: afficher les épisodes détectés")

    # Données
    parser.add_argument("--start", help="Début de la fenêtre d'inférence (YYYY-MM-DD)")
    parser.add_argument("--end",   help="Fin de la fenêtre d'inférence   (YYYY-MM-DD)")
    parser.add_argument("--gravity-path", type=Path, default=None,
                        help="Chemin vers gravity_daily.parquet")
    parser.add_argument("--gravity-threshold", type=float, default=None,
                        help="Seuil de déclenchement des épisodes (défaut: valeur dans LOCATION_DEFAULTS)")
    parser.add_argument("--min-episode-days", type=int, default=3,
                        help="Durée minimale d'un épisode en jours (défaut: 3)")
    parser.add_argument("--constituent-path", type=Path, default=None)
    parser.add_argument("--dx-km", type=float, default=2.0,
                        help="Largeur des bins spatiaux le long du chenal (km)")

    # Modèle
    parser.add_argument("--model", type=Path, default=None,
                        help="Chemin vers le checkpoint .pt (pour --infer)")
    parser.add_argument("--model-name", default=None,
                        help="Nom du fichier de sortie (sans extension)")

    # Hyperparamètres d'entraînement
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-kin", type=float, default=0.05)

    # Résolution de la grille d'inférence
    parser.add_argument("--x-grid", type=int, default=100)
    parser.add_argument("--t-grid", type=int, default=100)
    parser.add_argument("--t-max-days", type=float, default=None,
                        help="Horizon de prédiction en jours")
    parser.add_argument("--rho-threshold-frac", type=float, default=0.3,
                        help="Fraction du pic initial définissant le retour à la normale (défaut: 0.3)")

    # Fine-tuning
    parser.add_argument("--fine-tune", action="store_true",
                        help="Adapter le PINN aux données AIS de la fenêtre --start/--end avant inférence")
    parser.add_argument("--fine-tune-epochs", type=int, default=300,
                        help="Nombre d'epochs de fine-tuning (défaut: 300)")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4,
                        help="Learning rate du fine-tuning (défaut: 1e-4)")

    args = parser.parse_args()

    if not any([args.train, args.infer, args.both, args.data_only]):
        parser.print_help()
        return

    # ── Étape 1: détection des épisodes (modes train / both / data-only) ────
    episodes: list[Episode] = []
    global_meta: dict = {}

    if args.train or args.both or args.data_only:
        log.info("=== ÉTAPE 1: Structuration des données en épisodes ===")
        # Résoudre le seuil : --gravity-threshold > défaut de la localisation
        loc_defaults = LOCATION_DEFAULTS.get(args.location, {})
        effective_threshold = (
            args.gravity_threshold
            if args.gravity_threshold is not None
            else loc_defaults.get("gravity_threshold", 50_000)
        )
        effective_min_days = (
            args.min_episode_days
            if args.min_episode_days != 3
            else loc_defaults.get("min_episode_days", args.min_episode_days)
        )
        log.info(
            "Localisation=%s | threshold=%.0f | min_episode_days=%d",
            args.location, effective_threshold, effective_min_days,
        )
        train_start = date.fromisoformat(args.start) if args.start else None
        train_end = date.fromisoformat(args.end) if args.end else None
        episodes, global_meta = structure_episodes(
            location=args.location,
            gravity_path=args.gravity_path,
            gravity_threshold=effective_threshold,
            min_episode_days=effective_min_days,
            dx_km=args.dx_km,
            constituent_path=args.constituent_path,
            start_date=train_start,
            end_date=train_end,
        )

        if args.data_only:
            print(f"\n=== Épisodes détectés ({args.location}, threshold={effective_threshold:.0f}) ===")
            for ep in episodes:
                print(
                    f"  {ep.episode_id}  |  {ep.start_date} -> {ep.end_date}"
                    f"  ({ep.duration_days}j)  |  {ep.n_points} pts"
                    f"  |  trigger={ep.trigger_score:.3f}"
                )
            if global_meta:
                print(f"\n  rho_max global: {global_meta['global_rho_max']:.2f} vessels/km")
                print(f"  v_max global:   {global_meta['global_v_max']:.2f} knots")
            return

    # ── Étapes 2-3-4a: entraînement ──────────────────────────────────────────
    trained_model: Optional[LWRPINN] = None

    if args.train or args.both:
        if not episodes:
            log.error("Aucun épisode disponible pour l'entraînement.")
            return

        log.info("=== ÉTAPE 2: Architecture LWRPINN — MLP(x, t) → (ρ̂, v̂) ===")
        log.info("=== ÉTAPE 3: Loss hybride LWR + contraintes cinématiques ===")
        log.info("=== ÉTAPE 4a: Boucle d'entraînement épisodique ===")

        trained_model, history = train_on_episodes(
            episodes=episodes,
            global_meta=global_meta,
            epochs=args.epochs,
            lr=args.lr,
            lambda_pde=args.lambda_pde,
            lambda_kin=args.lambda_kin,
            model_name=args.model_name or f"pinn_{args.location}_episodes",
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        loss_path = OUTPUT_DIR / f"{args.location}_pinn_episode_loss.npy"
        np.save(loss_path, np.array(history))
        log.info("Courbe de loss sauvegardée → %s", loss_path)

        if args.both:
            args.model = MODEL_DIR / f"{args.model_name or f'pinn_{args.location}_episodes'}.pt"

    # ── Étape 4b: extraction des profils physiques ────────────────────────────
    if args.infer or args.both:
        log.info("=== ÉTAPE 4b: extract_physical_profiles ===")

        if args.model is None:
            log.error("--model requis pour --infer")
            return
        if not Path(args.model).exists():
            log.error("Checkpoint introuvable: %s", args.model)
            return

        # Charger modèle et métadonnées (ou réutiliser depuis --both)
        if args.both and trained_model is not None:
            infer_model = trained_model
            infer_meta = global_meta
        else:
            ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
            infer_meta = ckpt["global_meta"]
            infer_model = LWRPINN(hidden_layers=4, hidden_size=64)
            infer_model.load_state_dict(ckpt["model_state"])
            log.info(
                "Checkpoint chargé: %s (best_loss=%.6f, %d épisode(s))",
                args.model, ckpt["best_loss"], ckpt["n_episodes"],
            )

        # Tableaux AIS bruts pour fine-tuning + plot_ais_vs_pinn
        raw_x_km_arr: np.ndarray = np.array([], dtype=np.float32)
        raw_t_arr:    np.ndarray = np.array([], dtype=np.float32)
        raw_rho_arr:  np.ndarray = np.array([], dtype=np.float32)
        raw_v_arr:    np.ndarray = np.array([], dtype=np.float32)

        # Construire episode_data depuis la fenêtre --start / --end si fournie
        if args.start and args.end:
            start_d = date.fromisoformat(args.start)
            end_d = date.fromisoformat(args.end)
            try:
                X_raw, y_raw, meta_ep = build_rho_v_tensors(
                    start_d, end_d, infer_meta["location"],
                    dx_km=infer_meta["dx_km"],
                    use_raw_velocity=True,
                )
                if len(X_raw) > 0:
                    x_km = X_raw[:, 0] * infer_meta["channel_len_km"]
                    t_start_g = (start_d - EPOCH_DATE).days / T_DAYS_MAX
                    t_days = (X_raw[:, 1] - t_start_g) * T_DAYS_MAX
                    episode_data = pd.DataFrame({"x_km": x_km, "t_days": t_days})
                    # Denormaliser rho et v vers les unités physiques
                    rho_range    = meta_ep["rho_max"] - meta_ep.get("rho_min", 0.0)
                    raw_rho_arr  = (y_raw[:, 0] * rho_range + meta_ep.get("rho_min", 0.0)).astype(np.float32)
                    raw_v_arr    = (y_raw[:, 1] * meta_ep["v_max"]).astype(np.float32)
                    raw_x_km_arr = x_km.astype(np.float32)
                    raw_t_arr    = t_days.astype(np.float32)
                else:
                    log.warning("Aucune donnée AIS dans la fenêtre d'inférence. Grille par défaut.")
                    episode_data = pd.DataFrame({
                        "x_km": [0.0, infer_meta["channel_len_km"]],
                        "t_days": [0.0, float((end_d - start_d).days)],
                    })
            except Exception as exc:
                log.warning("Chargement des données d'inférence impossible (%s). Grille par défaut.", exc)
                episode_data = pd.DataFrame({
                    "x_km": [0.0, infer_meta["channel_len_km"]],
                    "t_days": [0.0, 30.0],
                })
        else:
            # Grille par défaut: chenal complet sur 30 jours
            episode_data = pd.DataFrame({
                "x_km": [0.0, infer_meta["channel_len_km"]],
                "t_days": [0.0, 30.0],
            })

        # ── Fine-tuning optionnel ────────────────────────────────────────────
        # Calcule t_max_days ici pour le passer aussi au fine-tuning
        t_max_days_infer = args.t_max_days
        if t_max_days_infer is None:
            ep_t_max = float(episode_data["t_days"].max()) if len(episode_data) > 0 else 0.0
            t_max_days_infer = ep_t_max * 1.2 + 1.0

        if args.fine_tune:
            if len(raw_x_km_arr) == 0:
                log.warning(
                    "Fine-tuning demande (--fine-tune) mais aucune donnee AIS disponible "
                    "dans la fenetre --start/--end. Donnez --start et --end pour charger les donnees."
                )
            else:
                log.info("=== FINE-TUNING sur l'evenement courant ===")
                infer_model = fine_tune_on_episode(
                    model=infer_model,
                    x_km=raw_x_km_arr,
                    t_days=raw_t_arr,
                    rho_physical=raw_rho_arr,
                    v_physical=raw_v_arr,
                    global_meta=infer_meta,
                    t_max_days=t_max_days_infer,
                    epochs=args.fine_tune_epochs,
                    lr=args.fine_tune_lr,
                )

        profiles = extract_physical_profiles(
            model=infer_model,
            episode_data=episode_data,
            global_meta=infer_meta,
            x_grid_points=args.x_grid,
            t_grid_points=args.t_grid,
            t_max_days=t_max_days_infer,
        )

        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        model_stem = Path(args.model).stem
        out_path = FEATURES_DIR / f"{model_stem}_profiles.parquet"
        profiles.to_parquet(out_path, index=False)
        log.info("Profils physiques sauvegardes -> %s", out_path)

        # Visualisation
        infer_period = (
            f"{args.start}_to_{args.end}" if args.start and args.end else model_stem
        )

        # Sous-dossier résultats : <location>_<start>_pinn_results/
        location_label = infer_meta["location"]
        date_label = (args.start or model_stem).replace("-", "")[:8]  # YYYYMMDD
        png_dir = OUTPUT_DIR / f"{location_label}_{date_label}_pinn_results"
        png_dir.mkdir(parents=True, exist_ok=True)
        log.info("Resultats PNG -> %s", png_dir)

        title_tag = f"{location_label.upper()} | {infer_period}"

        plot_path = plot_physical_profiles(
            profiles=profiles,
            episode_data=episode_data if len(episode_data) > 2 else None,
            title_suffix=title_tag,
            output_path=png_dir / "01_profiles_rho_v_q.png",
        )

        clearance_path, ttc_days = plot_congestion_clearance(
            profiles=profiles,
            rho_threshold_frac=args.rho_threshold_frac,
            title_suffix=title_tag,
            output_path=png_dir / "02_clearance_ttc.png",
        )

        shock_path = plot_shockwave_heatmap(
            profiles=profiles,
            title_suffix=title_tag,
            output_path=png_dir / "03_shockwave_lwr.png",
        )

        ais_path = plot_ais_vs_pinn(
            profiles=profiles,
            raw_x_km=raw_x_km_arr,
            raw_t_days=raw_t_arr,
            raw_rho=raw_rho_arr,
            title_suffix=title_tag,
            output_path=png_dir / "04_ais_vs_pinn.png",
        )

        slices_path = plot_density_profiles_1d(
            profiles=profiles,
            n_slices=6,
            title_suffix=title_tag,
            output_path=png_dir / "05_profiles_1d_slices.png",
        )

        fd_path = plot_fundamental_diagram(
            profiles=profiles,
            title_suffix=title_tag,
            output_path=png_dir / "06_fundamental_diagram.png",
        )

        ttc_str = f"{ttc_days:.1f} jours" if ttc_days is not None else "non atteint dans la fenetre"
        print(f"\n=== Profils physiques (Etape 4b) ===")
        print(f"  Grille:  {args.x_grid} x {args.t_grid} = {len(profiles)} points")
        print(f"  Chenal:  {infer_meta['channel_len_km']:.1f} km")
        print(f"  rho_hat: [{profiles['rho_hat'].min():.2f}, {profiles['rho_hat'].max():.2f}] vessels/km")
        print(f"  v_hat:   [{profiles['v_hat'].min():.2f}, {profiles['v_hat'].max():.2f}] knots")
        print(f"  TTC:     {ttc_str}")
        print(f"  Parquet: {out_path}")
        print(f"\n  Figures -> {png_dir}/")
        print(f"    01_profiles_rho_v_q.png      (3 heatmaps rho / v / q)")
        print(f"    02_clearance_ttc.png          (courbe resorption + TTC)")
        print(f"    03_shockwave_lwr.png          (onde de choc LWR)")
        print(f"    04_ais_vs_pinn.png            (AIS brut vs PINN lisse)")
        print(f"    05_profiles_1d_slices.png     (coupes rho(x) a 6 instants)")
        print(f"    06_fundamental_diagram.png    (diagramme fondamental Greenshields)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
