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
║    si >50kts correction       • HDBSCAN haversine       • vessel_count           ║
║  • traj_id (gap > 30min)      • docked / waiting        • SOG_mean/std/median    ║
║  • speed_computed_kt          • membership_score        • utilization_rate_rho   ║
║                                                         • cluster_count/noise    ║
║                                                         • draft_mean/std         ║
║                                                         • blocked_capacity       ║
║                                                         • tanker_ratio           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                        │
                            <location>_daily_features.parquet
                                        │
                 ┌──────────────────────────┴──────────────────────────┐
                 ▼                                               ▼
╔═══════════════════════════════╗         ╔═══════════════════════════════════════════╗
║  PHASE 2 — MANIFOLD          ║         ║  PHASE 3 — PINNs / TIME TO CLEAR          ║
║  (corrigé géospatial)        ║         ║  (run_phase3.py)                           ║
╠═══════════════════════════════╣         ╠═══════════════════════════════════════════╣
║                               ║         ║                                           ║
║  manifold_pipeline.py         ║         ║  lwr_pinn.py                              ║
║  ─────────────────           ║         ║  ──────────                               ║
║  • Matrice zones × jours      ║         ║  • MLP (x,t) → (ρ, v)                    ║
║  • L2 normalize par zone    ║         ║  • tanh activations                       ║
║  • KNN + Gaussian W         ║         ║  • Loss = L_data + L_PDE + L_BC           ║
║  • LBO spectral             ║         ║  • PDE : ∂ρ/∂t + ∂(ρv)/∂x = 0           ║
║  • Constituent zones ⭐     ║         ║                                           ║
║                               ║         ║  train.py                                 ║
║  Scoring quotidien          ║         ║  ─────────                                ║
║  ─────────────────           ║         ║  • ρ = utilization_rate_rho               ║
║  • Compter navires           ║         ║  • v = SOG_mean normalisé                 ║
║    dans zones constituantes║         ║  • x = 0.5 (proxy chenal)                ║
║  • Sum(capacité × is_const)  ║         ║  • Adam + ReduceLROnPlateau               ║
║                               ║         ║  • Gradient clipping max_norm=1.0         ║
╚═══════════════════════════════╝         ║  predict.py                               ║
                 │                            ║  ──────────                               ║
                 │                            ║  • ρ(x=0.5, t) post-peak                 ║
                 │                            ║  • TTC = ρ ≥ 85% baseline × 3 jours      ║
                 │                            ║  ────────────────────────────────           ║
                 ▼                            ║
    <location>_constituent_zones.parquet       ║  <location>_time_to_clear.parquet
    <location>_gravity_daily.parquet           ║  outputs/models/<location>_lwr_pinn.pt
                 │
                 ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         VISUALISATION                                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  visualize_clusters.py → Polygones sémantiques (MBR/hull) sur carte satellite      ║
║  manifold_pipeline.py --plot → Courbe gravity score temporelle                  ║
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
| `la` | [33.65, 33.85] | [-118.40, [-118.05]] | Tensions US-Chine (2019) |

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

## Phase 2 — Manifold Learning (corrigé)

### `src/manifold/manifold_pipeline.py`

**Rôle :** Identifier les zones géographiques "constituantes" (les plus structurantes) 
puis calculer le gravity score quotidien sur ces zones.

**Entrée :** `<location>_daily_features.parquet` + données Parquet journalières

**Sortie :** 
- `<location>_constituent_zones.parquet` (zones + is_constituent + phi)
- `<location>_gravity_daily.parquet` (date, gravity_score, etc.)
- `outputs/figures/<location>_gravity_score.png`

**Fonctionnement — 2 phases :**

#### Phase 2A : Identification des zones constituantes

```
1. CONSTRUIRE LA MATRICE ZONES × JOURS
   Pour chaque jour :
     - Extraire les clusters HDBSCAN
     - Créer une clé de zone (lat_rondé_3, lon_rondé_3)
     - Compter les navire par zone
   
   Matrice X : (n_zones, n_jours)
   - Ligne = zone géographique unique
   - Colonne = jour
   - Valeur = occupation normalisée [0,1]

2. NORMALISATION L2 PAR LIGNE
   Xi ← Xi / ||Xi||₂   (chaque zone = vecteur unitaire dans R^jours)

3. MATRICE DE POIDS KNN
   KNN(k=5) + noyau gaussien → W symétrique sparse

4. OPÉRATEUR DE LAPLACE-BELTRAMI
   LBO = A⁻¹ · W   (A = matrice degrés)

5. DÉCOMPOSITION PROPRE
   W·φ = λ·A·φ
   → φ₁…φ₅ vecteurs propres

6. ZONES CONSTITUANTES ⭐
   = extrema locaux dans le graphe KNN sur φ₁…φ₃
   → Ce sont les zones géographiques qui capturent la structure du trafic
```

#### Phase 2B : Scoring quotidien

```
Pour chaque nouveau jour :
  1. Charger les clusters HDBSCAN du jour
  2. Identifier quais sont dans les zones constituantes
  3. gravity_score = Σ(capacité × is_constituent)
  4. Sauvegarder dans _gravity_daily.parquet
```

**Commandes :**

```bash
# Identifier les zones constituantes (une seule fois)
python src/manifold/manifold_pipeline.py --identify --start 2020-01-01 --end 2020-01-31 --location la

# Scoring quotidien
python src/manifold/manifold_pipeline.py --score --date 2020-02-01 --location la

# Scoring sur période + graphique
python src/manifold/manifold_pipeline.py --score --start 2020-02-01 --end 2020-02-07 --location la
```

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

**Données d'entrée (depuis `_gravity_daily.parquet`) :**
- `ρ_obs` = `gravity_score` ∈ [0,1]
- `t` = index jour normalisé ∈ [0,1]

**Loss composite :**
```
L_total = L_data + 0.1 · L_PDE + 0.1 · L_BC

L_data = MSE(ρ_pred, ρ_obs)
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

### `src/clustering/visualize_clusters.py`

| Commande | Output |
|----------|--------|
| `--date YYYY-MM-DD` | Carte interactive avec polygones sémantiques |
| `--start … --end …` | Carte avec clusters par jour |

**Fonctionnalités :**
- Rectangles orientés (MBR) pour clusters "docked" (bleu)
- Convex Hulls pour clusters "waiting" (orange)
- Couche satellite ESRI en overlay
- Popups avec vignette satellite

### `src/manifold/manifold_pipeline.py --plot`

**Sortie :** `outputs/figures/<location>_gravity_score.png`

Courbe temporelle du gravity score avec :
- Axe X : dates
- Axe Y : gravity score (capacité bloquée dans zones constituantes)
- Marqueur sur le maximum

---

## Commandes de lancement

```bash
# Phase 1 — features (données déjà téléchargées)
python run_phase1.py --location la --start 2019-01-01 --end 2019-12-31 --no-download

# Phase 2 — Identificaton zones constituantes (une seule fois)
python src/manifold/manifold_pipeline.py --identify --start 2019-01-01 --end 2019-12-31 --location la

# Phase 2 — Scoring quotidien
python src/manifold/manifold_pipeline.py --score --date 2019-06-05 --location la

# Phase 2 — Scoring période + graphique
python src/manifold/manifold_pipeline.py --score --start 2019-06-01 --end 2019-06-30 --location la

# Phase 3 — PINN + Time to Clear
python run_phase3.py --location la

# Visualisation clusters géometriques
python src/clustering/visualize_clusters.py --date 2019-06-05 --location la
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
manifold_pipeline.py    train.py
    │         │
    ▼         ▼
constituent_zones  lwr_pinn.pt
    │         │
    ▼         ▼
gravity_daily   predict.py
    │          │
    │    time_to_clear.parquet
    │
    ▼
outputs/figures/*.png
```

---

## Données produites

| Fichier | Contenu | Taille typique |
|---------|--------|---------------|
| `data/parquet/<loc>/<loc>_YYYY_MM_DD.parquet` | Positions AIS filtrées | ~1–5 MB/jour |
| `data/features/<loc>_daily_features.parquet` | 13 features × N jours | < 100 KB |
| `data/features/<loc>_constituent_zones.parquet` | Zones + is_constituent + phi | < 200 KB |
| `data/features/<loc>_gravity_daily.parquet` | date + gravity_score | < 100 KB |
| `data/features/<loc>_time_to_clear.parquet` | ρ_pred, v_pred, TTC par jour | < 50 KB |
| `outputs/models/<loc>_lwr_pinn.pt` | Poids PyTorch du PINN | ~1 MB |
| `outputs/figures/<loc>_gravity_score.png` | Courbe temporelle | ~50 KB |

---

## Différenciation Houston / LA

| Port | Source AIS | Période disponible | Événement de référence |
|------|----------|-----------------|-------------------|
| Houston | Marine Cadastre (NOAA) | 2017–2017 | Hurricane Harvey |
| LA | AISStream | 2020–2020 | COVID-19 / trade tensions |