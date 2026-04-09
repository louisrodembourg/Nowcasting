# Architecture technique — Maritime Nowcasting

Modélisation de l'impact des disruptions maritimes sur les indices de fret (BDI, FBX, WCI)
à partir de données AIS (Automatic Identification System).

---

## Schéma de fonctionnement global

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         DONNÉES BRUTES (Marine Cadastre / NOAA)                 ║
║              ZIP journaliers — 17 colonnes — ~100k messages/jour/port            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                        │
                               download.py
                          (DuckDB bbox filter → Parquet ZSTD)
                                        │
                                        ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 1 — FEATURE ENGINEERING  (run_phase1.py)                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  kinematic_filter.py          hdbscan_daily.py          features_daily.py        ║
║  ─────────────────            ───────────────           ───────────────          ║
║  • SOG haversine corr         • Filtre SOG < 1 kt       • 13 features/jour       ║
║  • traj_id (gap > 30min)      • HDBSCAN haversine       • vessel_count           ║
║  • speed_computed_kt          • docked / waiting        • SOG_mean/std/median    ║
║                               • membership_score        • utilization_rate_rho   ║
║                                                         • cluster_count/noise    ║
║                                                         • draft_mean/std         ║
║                                                         • blocked_capacity       ║
║                                                         • tanker_ratio           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                        │
                            <location>_daily_features.parquet
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
╔═══════════════════════════════╗      ╔═══════════════════════════════════════════╗
║  PHASE 2 — MANIFOLD           ║      ║  PHASE 3 — PINNs / TIME TO CLEAR          ║
║  (run_phase2.py)              ║      ║  (run_phase3.py)                           ║
╠═══════════════════════════════╣      ╠═══════════════════════════════════════════╣
║                               ║      ║                                           ║
║  lbo.py                       ║      ║  lwr_pinn.py                              ║
║  ────────                     ║      ║  ──────────                               ║
║  • L2 normalize (13D→13D)     ║      ║  • MLP (x,t) → (ρ, v)                    ║
║  • KNN k=7 + Gaussian W       ║      ║  • tanh activations                       ║
║  • LBO = A⁻¹W                 ║      ║  • Loss = L_data + L_PDE + L_BC           ║
║  • Eigenvectors ϕ₁…ϕ₈        ║      ║  • PDE : ∂ρ/∂t + ∂(ρv)/∂x = 0           ║
║  • Characteristic points      ║      ║                                           ║
║                               ║      ║  train.py                                 ║
║  gravity_score.py             ║      ║  ─────────                                ║
║  ────────────────             ║      ║  • ρ = utilization_rate_rho               ║
║  • Baseline vs event window   ║      ║  • v = SOG_mean normalisé                 ║
║  • deviation = |ϕ - μ| / σ   ║      ║  • x = 0.5 (proxy chenal)                ║
║  • gravity = dev × capacity   ║      ║  • Adam + ReduceLROnPlateau               ║
║  • Normalisé [0, 1]           ║      ║  • Gradient clipping max_norm=1.0         ║
║                               ║      ║                                           ║
╚═══════════════════════════════╝      ║  predict.py                               ║
                │                      ║  ──────────                               ║
                │                      ║  • ρ(x=0.5, t) post-peak                 ║
                │                      ║  • TTC = ρ ≥ 85% baseline × 3 jours      ║
                │                      ╚═══════════════════════════════════════════╝
                │                                       │
                ▼                                       ▼
   <location>_gravity_score.parquet      <location>_time_to_clear.parquet
   <location>_manifold.parquet           outputs/models/<location>_lwr_pinn.pt
                │
                ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VISUALISATION  (src/utils/visualize.py)                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  --date        → Folium cluster map (1 jour)                                     ║
║  --start/--end → Folium HeatMapWithTime (période animée)                         ║
║  --gravity     → Matplotlib time series + scatter manifold 2D                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Phase 1 — Feature Engineering

### `src/ingestion/download.py`

**Rôle :** Télécharge les ZIP journaliers Marine Cadastre (NOAA) et les filtre vers Parquet.

**Entrée :** URL NOAA `AIS_{year}_{month}_{day}.zip`

**Sortie :** `data/parquet/<location>/<location>_YYYY_MM_DD.parquet` (ZSTD, ~30–100× compression)

**Fonctionnement :**
1. Téléchargement HTTP avec 3 tentatives + backoff 10s
2. Extraction CSV depuis le ZIP en mémoire
3. Filtrage DuckDB par bbox géographique + MMSI valide [200M–999M] + SOG [0–50 kt]
4. `ignore_errors=true` pour les lignes avec guillemets dans les noms de navires

**Locations disponibles :**
| Location | Bbox LAT | Bbox LON | Événement de référence |
|----------|---------|---------|----------------------|
| `houston` | [29.3, 29.85] | [-95.4, -94.7] | Hurricane Harvey (août 2017) |
| `la` | [33.65, 33.85] | [-118.40, -118.05] | Tensions US-Chine (2019) |

**Colonnes brutes conservées (17) :**
`MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass`

---

### `src/ingestion/kinematic_filter.py`

**Rôle :** Corriger les vitesses aberrantes et segmenter les trajectoires.

**Entrée :** DataFrame Polars brut (17 colonnes)

**Sortie :** DataFrame enrichi avec `SOG_corr`, `speed_computed_kt`, `traj_id`

**Fonctionnement :**

```
Pour chaque navire (MMSI) :
  ├── Calculer la distance haversine entre positions consécutives
  ├── speed_computed_kt = distance / Δt
  ├── SOG_corr = speed_computed_kt si SOG déclaré > 50 kt (erreur capteur)
  │             sinon SOG déclaré
  └── traj_id = nouveau segment si Δt > 30 min (gap = trajet distinct)
              = np.cumsum(gap_mask) par MMSI
```

**Pourquoi haversine et non distance euclidienne ?**
À lat ≈ 29–33°N, 1° longitude ≠ 1° latitude en km. La formule haversine corrige la sphéricité terrestre.

---

### `src/clustering/hdbscan_daily.py`

**Rôle :** Identifier les zones de stationnement et classifier docked / waiting.

**Entrée :** Parquet journalier ou DataFrame préparé (via `cluster_day_from_df`)

**Sortie :** `(cluster_df, prepared_df)` — un tuple

**Fonctionnement :**

```
1. Filtre SOG_corr < 1 kt  →  navires stationnaires uniquement
2. Déduplication : 1 position médiane par (MMSI, traj_id)
   → chaque épisode stationnaire = 1 point distinct
3. HDBSCAN(min_cluster_size=3, min_samples=2, metric="haversine")
   → labels (cluster ou bruit = -1) + probabilities (membership score)
4. Classification par cluster :
   - Heading_std < 25° → "docked"  (caps alignés sur le quai)
   - Heading_std ≥ 25° → "waiting" (caps dispersés par vent/courant)
```

**Deux fonctions d'entrée :**
- `cluster_day(parquet_path)` — lit depuis fichier (pipeline production)
- `cluster_day_from_df(prepared_df)` — reçoit un DataFrame (notebooks, tests)

---

### `src/clustering/features_daily.py`

**Rôle :** Calculer les 13 features agrégées par jour.

**Entrée :** `(prepared_df, cluster_df, date)`

**Sortie :** `dict` avec 13 features + date → une ligne dans la matrice features

**Les 13 features :**

| Feature | Calcul | Signal capturé |
|---------|--------|----------------|
| `vessel_count` | n_unique(MMSI) | Volume de trafic brut |
| `SOG_mean` | mean(SOG_corr) | Fluidité globale |
| `SOG_std` | std(SOG_corr) | Hétérogénéité des vitesses |
| `SOG_median` | median(SOG_corr) | Vitesse typique (robuste aux outliers) |
| `utilization_rate_rho` | n_stationary / n_total | Taux de congestion (ρ dans LWR) |
| `hdbscan_cluster_count` | n clusters (label ≥ 0) | Zones d'accumulation actives |
| `hdbscan_noise_ratio` | n_bruit / n_stationary | Dispersion spatiale des stationnaires |
| `membership_score_mean` | mean(probabilities) | Qualité/netteté des clusters |
| `membership_score_std` | std(probabilities) | Stabilité des zones |
| `draft_mean` | mean(Draft) | Profondeur chargement moyen |
| `draft_std` | std(Draft) | Mix de types de navires |
| `blocked_capacity` | Σ(Draft × Length) stationnaires | Tonnage immobilisé |
| `tanker_ratio` | n_tankers / n_total | Part du segment pétrolier |

---

## Phase 2 — Manifold Learning

### `src/manifold/lbo.py`

**Rôle :** Projeter la matrice de features dans un espace de faible dimension capturant la géométrie des régimes de trafic.

**Entrée :** `<location>_daily_features.parquet` (N jours × 13 features)

**Sortie :** `<location>_manifold.parquet` (+ colonnes `phi_1…phi_8`, `eigenvalue_1…8`, `is_characteristic`)

**Fonctionnement — 4 étapes :**

```
1. NORMALISATION L2
   Xi ← Xi / ||Xi||₂   (chaque jour = vecteur unitaire dans R¹³)
   → Supprime les effets d'échelle entre features

2. MATRICE DE POIDS KNN
   Pour chaque point i, trouver les k=7 voisins les plus proches
   W_ij = exp(-||Xi - Xj||² / σ²)   si j ∈ KNN(i)
   W_ij = 0                          sinon
   W = symmetrize(W)   →  matrice sparse

3. OPÉRATEUR DE LAPLACE-BELTRAMI
   A = diag(sommes de lignes de W)   (matrice degrés)
   LBO = A⁻¹ · W
   → Diffusion sur la variété des régimes de trafic

4. DÉCOMPOSITION PROPRE
   W · ϕ = λ · A · ϕ
   → ϕ₀ trivial (vecteur constant), on garde ϕ₁…ϕ₈
   → Points caractéristiques = extrema locaux dans le graphe KNN
```

**Hyperparamètres :** `k=7` voisins, `n_eigenvectors=8`

---

### `src/manifold/gravity_score.py`

**Rôle :** Quantifier l'anomalie de chaque jour par rapport au régime normal.

**Entrée :** `<location>_manifold.parquet` + fenêtre d'événement (harvey_start / harvey_end)

**Sortie :** `<location>_gravity_score.parquet` (+ `gravity_score`, `deviation_score`)

**Fonctionnement :**

```
baseline = jours HORS fenêtre événement
μ_c = mean(ϕ_c sur baseline)    pour chaque composante c
σ_c = std(ϕ_c sur baseline)

deviation_i = mean_c(|ϕ_c(i) - μ_c| / σ_c)   (déviation normalisée z-score)

cap_weight_i = blocked_capacity_i / max(blocked_capacity)

raw_score_i = deviation_i × cap_weight_i
gravity_score_i = (raw_score_i - min) / (max - min)   →  [0, 1]
```

**Interprétation :** Un score élevé = jour très anormal (loin du baseline) avec beaucoup de tonnage immobilisé.

---

## Phase 3 — PINNs

### `src/pinns/lwr_pinn.py`

**Rôle :** Réseau de neurones qui respecte l'équation de conservation du trafic maritime.

**Architecture :**
```
Entrée : (x, t) ∈ [0,1]²    x = position chenal, t = temps normalisé
  ↓
MLP : Linear(2→64) → tanh → [4 couches cachées 64→64 + tanh] → Linear(64→2)
  ↓
Sortie : (ρ, v) avec sigmoid → ∈ (0,1)
```

**Équation LWR (Lighthill-Whitham-Richards) intégrée dans la loss :**
```
∂ρ/∂t + ∂(ρ·v)/∂x = 0    (conservation de la "densité" de navires)

Modèle de Greenshields : v = v_max · (1 - ρ/ρ_max)
```

---

### `src/pinns/train.py`

**Rôle :** Entraîner le PINN sur la fenêtre temporelle de la disruption.

**Données d'entrée (depuis `_daily_features.parquet`) :**
- `ρ_obs` = `utilization_rate_rho` ∈ [0,1]
- `v_obs` = `SOG_mean` normalisé par max observé
- `x` = 0.5 (proxy — centroid du chenal)
- `t` = index jour normalisé ∈ [0,1]

**Loss composite :**
```
L_total = L_data + 0.1 · L_PDE + 0.1 · L_BC

L_data = MSE(ρ_pred, ρ_obs) + MSE(v_pred, v_obs)
L_PDE  = résidu ∂ρ/∂t + ∂(ρv)/∂x sur points de collocation aléatoires
L_BC   = condition limite : ρ(x=0, t) = ρ(x=1, t) (chenal fermé aux bords)
```

**Entraînement :** 2000 epochs, Adam + ReduceLROnPlateau, gradient clipping max_norm=1.0

---

### `src/pinns/predict.py`

**Rôle :** Calculer le Time to Clear (TTC) — nombre de jours pour retour à la normale.

**Fonctionnement :**
```
1. Calculer ρ_baseline = mean(ρ_obs hors fenêtre de crise)
2. Seuil TTC = 0.85 × ρ_baseline
3. Évaluer ρ_pred(x=0.5, t) pour chaque jour post-pic
4. TTC = premier jour où ρ_pred ≥ seuil × 3 jours consécutifs
```

---

## Fichier non intégré : `trajectory.py`

### `src/ingestion/trajectory.py`

**Statut : codé mais non appelé par le pipeline principal**

**Ce qu'il fait :**
- **Douglas-Peucker** (ε=0.0001° ≈ 11m) : compression des trajectoires haute fréquence
- **Interpolation à 120s** : uniformisation de la résolution entre navires
- **Cubic Hermite Spline** si changement de cap > 30° : courbes réalistes dans les virages

**Pourquoi pas utilisé actuellement :**
Les phases 1–3 travaillent sur des features agrégées par jour — la géométrie fine des trajectoires n'est pas nécessaire. Sera utile pour la **Phase 4** si on veut un x(t) spatial réel dans le PINN plutôt que le proxy x=0.5.

---

## Visualisation

### `src/utils/visualize.py`

**Trois modes :**

| Commande | Output | Technologie |
|----------|--------|-------------|
| `--date YYYY-MM-DD` | Carte interactive clusters HDBSCAN | Folium + CircleMarker |
| `--start … --end …` | Heatmap animée navires stationnaires | Folium HeatMapWithTime |
| `--gravity` | Time series gravity score + scatter manifold 2D | Matplotlib |

---

## Commandes de lancement

```bash
# Phase 1 — features (données déjà téléchargées)
python run_phase1.py --location la --start 2019-01-01 --end 2019-12-31 --no-download
python run_phase1.py --location houston --start 2017-07-01 --end 2017-09-30 --no-download

# Phase 2 — manifold + gravity score
python run_phase2.py --location la
python run_phase2.py --location houston

# Phase 3 — PINN + Time to Clear
python run_phase3.py --location la
python run_phase3.py --location houston

# Visualisation
python src/utils/visualize.py --gravity --location la
python src/utils/visualize.py --location la --date 2019-06-05
python src/utils/visualize.py --location la --start 2019-05-01 --end 2019-09-30
```

---

## Flux de données complet

```
Marine Cadastre NOAA
        │  ZIP (~1 GB/jour)
        ▼
download.py  ──────────────────────────────────────────────────────
        │  Parquet ZSTD (~8 MB/jour, ×125 compression)
        ▼
kinematic_filter.py  ──────────────────────────────────────────────
        │  + SOG_corr, traj_id
        ▼
hdbscan_daily.py  ─────────────────────────────────────────────────
        │  cluster_df (episodes stationnaires + labels)
        │  prepared_df (DataFrame complet enrichi)
        ▼
features_daily.py  ────────────────────────────────────────────────
        │  13 scalaires par jour
        ▼
<location>_daily_features.parquet  (N jours × 13 features)
        │
   ┌────┴────┐
   ▼         ▼
lbo.py    train.py
   │         │
   ▼         ▼
manifold  lwr_pinn.pt
   │         │
   ▼         ▼
gravity   predict.py
_score        │
   │          ▼
   │    time_to_clear.parquet
   │
   ▼
visualize.py
   │
   ▼
outputs/figures/*.png  /  *.html
```

---

## Données produites

| Fichier | Contenu | Taille typique |
|---------|---------|---------------|
| `data/parquet/<loc>/<loc>_YYYY_MM_DD.parquet` | Positions AIS filtrées | ~1–5 MB/jour |
| `data/features/<loc>_daily_features.parquet` | 13 features × N jours | < 100 KB |
| `data/features/<loc>_manifold.parquet` | Features + ϕ₁…ϕ₈ + is_characteristic | < 200 KB |
| `data/features/<loc>_gravity_score.parquet` | + gravity_score, deviation_score | < 200 KB |
| `data/features/<loc>_time_to_clear.parquet` | ρ_pred, v_pred, TTC par jour | < 50 KB |
| `outputs/models/<loc>_lwr_pinn.pt` | Poids PyTorch du PINN | ~1 MB |
