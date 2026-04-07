# Avancement du projet — Nowcasting maritime

Projet de recherche académique — Mars–Juin 2026
Focus actuel : **Houston Ship Channel / Hurricane Harvey (août 2017)**

---

## Vue d'ensemble

| Phase | Nom | Statut |
|-------|-----|--------|
| 1 | Collecte, nettoyage, HDBSCAN | ✅ Complète |
| 2 | Manifold Learning + Gravity Score | ✅ Complète |
| 3 | PINNs / Time to Clear | ✅ Complète |
| 4 | Prime de risque (XGBoost / Elastic Net) | 🔜 À coder |

---

## Phase 1 — Collecte & HDBSCAN ✅

### Scripts

| Fichier | Rôle | Statut |
|---------|------|--------|
| `src/ingestion/download.py` | Télécharge les ZIP NOAA → Parquet filtré (bbox, MMSI, SOG) | ✅ |
| `src/ingestion/kinematic_filter.py` | Correction SOG via haversine + détection de gaps → `traj_id` | ✅ |
| `src/clustering/hdbscan_daily.py` | HDBSCAN par jour sur épisodes stationnaires, classification docked/waiting | ✅ |
| `src/clustering/features_daily.py` | Calcul des 13 features quotidiennes | ✅ |
| `src/utils/visualize.py` | Cartes Folium (jour unique + heatmap animée période) | ✅ |
| `run_phase1.py` | Orchestrateur : download → kinematic → HDBSCAN → features | ✅ |

### Ce que fait le pipeline

```
Marine Cadastre ZIP
        ↓  download.py          (bbox + MMSI + SOG filter → Parquet ZSTD)
data/parquet/houston/houston_YYYY_MM_DD.parquet
        ↓  kinematic_filter.py  (SOG_corr haversine + traj_id)
        ↓  hdbscan_daily.py     (1 épisode/MMSI+traj_id → HDBSCAN haversine)
        ↓  features_daily.py    (13 features journalières)
data/features/houston_daily_features.parquet
```

### 13 features produites

`vessel_count` · `SOG_mean` · `SOG_std` · `SOG_median` · `utilization_rate_rho`
`hdbscan_cluster_count` · `hdbscan_noise_ratio` · `membership_score_mean` · `membership_score_std`
`draft_mean` · `draft_std` · `blocked_capacity` · `tanker_ratio`

### Données disponibles

| Période | Fichiers parquet | Features calculées |
|---------|------------------|--------------------|
| 2017-07-01 → 2017-09-30 | ✅ 92 fichiers | ✅ dans `houston_daily_features.parquet` |

### Ce qui N'est PAS dans Phase 1 (déféré à Phase 3)
- Interpolation de trajectoires (linéaire / Cubic Hermite) → codé dans `src/ingestion/trajectory.py`
- Compression Douglas-Peucker → codé dans `src/ingestion/trajectory.py`

---

## Phase 2 — Manifold Learning & Gravity Score ✅

### Scripts

| Fichier | Rôle |
|---------|------|
| `src/manifold/lbo.py` | Normalisation L2, KNN (k=7), matrice W gaussienne, LBO + décomposition propre, points caractéristiques |
| `src/manifold/gravity_score.py` | Score de gravité pondéré par `blocked_capacity`, normalisé [0,1] |
| `run_phase2.py` | Orchestrateur |

### Résultats Houston

- **18 points caractéristiques** sur 92 jours
- **Pic gravity score** : 2017-08-04 (score=1.0, accumulation pré-Harvey) + 2017-09-07 (score=0.86, embouteillage post-Harvey)
- Jours Harvey 28–29 août : score≈0 (port fermé, capacité=0)

### Outputs

| Fichier | Contenu |
|---------|---------|
| `data/features/houston_manifold.parquet` | phi_1…phi_8, eigenvalue_1…8, is_characteristic |
| `data/features/houston_gravity_score.parquet` | + deviation_score, gravity_score |
| `outputs/figures/houston_gravity_timeseries.png` | Time series 3 panels |
| `outputs/figures/houston_manifold_2d.png` | Scatter ϕ₁ vs ϕ₂ coloré par gravity score |

---

## Phase 3 — PINNs / Time to Clear ✅

### Scripts

| Fichier | Rôle |
|---------|------|
| `src/ingestion/trajectory.py` | Interpolation linéaire / Cubic Hermite + Douglas-Peucker (ε=0.0001°) |
| `src/pinns/lwr_pinn.py` | MLP PyTorch (x,t)→(ρ,v), loss = données + PDE LWR + BC |
| `src/pinns/train.py` | Entraînement sur 2017-08-15 → 2017-09-10, backend MPS |
| `src/pinns/predict.py` | Inférence TTC : ρ remonte au-dessus de 85% du baseline |
| `run_phase3.py` | Orchestrateur |

### Résultats Houston / Harvey

- **Time to Clear = 2 jours** après le pic (2017-08-28 → retour à la normale estimé 2017-08-30)
- Baseline ρ = 0.954, seuil TTC = 0.811
- Best loss = 0.009650 (2000 epochs, MPS)

> Note : le TTC de 2 jours reflète la réouverture rapide du HSC après Harvey — historiquement, le canal a rouvert partiellement le 31 août. Le signal est physiquement cohérent.

### Outputs

| Fichier | Contenu |
|---------|---------|
| `outputs/models/lwr_pinn.pt` | Poids du PINN + métadonnées |
| `data/features/houston_time_to_clear.parquet` | ρ_pred, v_pred, is_cleared, time_to_clear_days (60 jours post-Harvey) |

---

## Phase 4 — Prime de risque 🔜

**Prérequis :** Gravity Score (Phase 2) + Time to Clear (Phase 3) + indices BDI/FBX/WCI

### Étapes à coder

| Étape | Description | Fichier cible |
|-------|-------------|---------------|
| 4.1 | Collecte indices financiers (yfinance / CSV) | `src/correlation/fetch_indices.py` |
| 4.2 | Alignement temporel AIS ↔ indices | `src/correlation/align.py` |
| 4.3 | Modèle XGBoost / Elastic Net | `src/correlation/risk_model.py` |
| 4.4 | Explicabilité SHAP | `src/correlation/explain.py` |
| 4.5 | Générateur d'alertes (livrable final) | `src/correlation/alert.py` |

**Output attendu :** alerte du type `"Houston Ship Channel bloqué : +X jours, -Y% BDI"`

---

## Commandes utiles

```bash
# Phase 3 complète (entraînement + prédiction)
python run_phase3.py --epochs 2000

# Phase 3 prédiction seulement (modèle existant)
python run_phase3.py --skip-train

# Phase 2
python run_phase2.py

# Phase 1 (features seulement, données déjà téléchargées)
python run_phase1.py --start 2017-07-01 --end 2017-09-30 --no-download
```

---

## Visualisations disponibles

```bash
# Phase 2 — gravity score + manifold 2D
python src/utils/visualize.py --gravity

# Phase 1 — carte des clusters HDBSCAN pour un jour
python src/utils/visualize.py --date 2017-08-25

# Phase 1 — heatmap animée sur une période
python src/utils/visualize.py --start 2017-07-01 --end 2017-09-30

# Phase 1 — heatmap + carte par jour
python src/utils/visualize.py --start 2017-07-01 --end 2017-09-30 --each-day
```

Outputs → `outputs/figures/`
