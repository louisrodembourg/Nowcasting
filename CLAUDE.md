# CLAUDE.md — Contexte Projet

## Objectif

Modéliser l'impact d'événements de disruption maritime sur les marchés financiers du fret (BDI, FBX, WCI) en utilisant des données alternatives AIS (suivi de navires). L'output cible est un système d'alerte interprétable :

> "La congestion au détroit de X devrait prendre N jours à se résorber et provoquer une variation de Y% sur l'indice Z."

Projet de recherche académique encadré, semestre Mars–Juin 2026.

---

## Architecture en 4 phases

| Phase | Nom | Description |
|-------|-----|-------------|
| 1 | **Collecte** | Ingestion AIS, nettoyage, feature engineering (13 features quotidiennes) |
| 2 | **Quantification** | Score de gravité maritime via Manifold Learning (UMAP + HDBSCAN) |
| 3 | **Estimation** | Prédiction de durée (Time to Clear) via PINNs + équations LWR de trafic |
| 4 | **Corrélation / Exploitation** | Modèle de prime de risque reliant disruptions maritimes aux indices fret |

## Périmètres géographiques

- **Paire 1 — Houston Ship Channel + Port of Houston** : données Marine Cadastre NOAA (gratuites, accès immédiat). Événement de référence : **Hurricane Harvey (août 2017)**.
- **Paire 2 — Canal de Suez + Port de Rotterdam** : Global Fishing Watch (historique long terme) + MarineTraffic/UNGP (haute fréquence). Événement de référence : **crise Houthi/Mer Rouge (2023–2024)**.

### Distinction architecturale clé

Les **détroits** sont les points de transit où se produisent les disruptions. Les **ports** sont des nœuds de propagation en aval. Cette distinction structure le pipeline et le cadrage de la théorie des files d'attente.

---

## Stack technique

| Catégorie | Outils |
|-----------|--------|
| Langages | Python 3.x |
| Data | DuckDB, Polars (lazy eval sur Parquet), Pandas si nécessaire |
| ML / DL | PyTorch (backend MPS pour Apple Silicon), UMAP, HDBSCAN, XGBoost, BiGRU |
| PINNs | PyTorch (équations LWR intégrées dans la loss) |
| Recherche de similarité | FAISS |
| Explicabilité | SHAP |
| Visualisation | Folium (cartes interactives), Matplotlib, Seaborn |
| Rapport | LaTeX (rapport formel), TikZ (diagrammes architecture) |
| Notes | Notion (Markdown) |

## Contrainte hardware

MacBook Pro M3 Pro, 18 GB RAM unifiée, ~150 GB stockage libre. Toutes les décisions data/compute doivent respecter cette contrainte. Utiliser le backend MPS pour PyTorch.

---

## Pipeline de données AIS

### Sources

| Source | Couverture | Usage |
|--------|-----------|-------|
| Marine Cadastre / NOAA | Eaux US, schéma riche (tirant d'eau, dimensions, cargo, cap) | Houston — Phase 1-2 (source unique) |
| Global Fishing Watch | Historique long terme, global | Suez — Phase 2 (manifold baseline) |
| MarineTraffic / UNGP | Haute fréquence | Suez — Phase 3 (PINNs) |

> **Note Houston :** Marine Cadastre remplace à la fois GFW (baseline manifold) et MarineTraffic (haute fréquence crise). C'est la source unique pour Houston — gratuitement disponible, historique 2009–présent, résolution ~2–15 min/navire.

---

## Phase 1 — Pipeline Houston (Marine Cadastre)

Focus actuel du projet. Événement de référence : **Hurricane Harvey (août 2017)**.

### Étape 1 — Collecte ciblée

**Bounding box Houston Ship Channel :** LAT [29.5, 29.9], LON [-95.4, -94.7]

- Télécharger les CSV mensuels depuis marinecadastre.gov pour la période **2015–2017** (baseline pré-Harvey + crise Harvey)
- Charger via DuckDB, filtrer par bbox stricte
- Convertir en Parquet, supprimer les CSV bruts immédiatement (discipline stockage)
- Colonnes clés Marine Cadastre : `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, VesselType, Status, Length, Width, Draft, Cargo`

> Contrairement au pipeline Suez (deux flux GFW + MarineTraffic), Marine Cadastre est la source unique pour Houston : même résolution sur le baseline et sur la période de crise Harvey.

### Étape 2 — Filtrage cinématique

- **MMSI invalides :** exclure si hors plage [200 000 000 – 999 999 999]
- **Coordonnées hors bbox :** filtrage strict sur LAT/LON
- **VesselType :** exclure les classes hors pertinence cargo (ex. type 0 = inconnu si masse de données trop bruyante)
- **SOG aberrant :** recalculer la vitesse réelle via la formule haversine entre deux positions consécutives du même MMSI et comparer à la valeur déclarée

### Étape 3 — Restructuration et interpolation des trajectoires

- **Gap threshold :** 30 minutes (Marine Cadastre est moins dense qu'un flux AIS temps réel ; un trou > 30 min scinde la trajectoire en deux segments distincts)
- **Houston Ship Channel = canal quasi-linéaire** (comme Suez) → interpolation linéaire pour la majorité des navires
- **Zone turning basin** (lat ≈ 29.75) → si changement de cap > 30° entre deux points, utiliser une interpolation **Cubic Hermite**
- **Compression :** appliquer Douglas-Peucker (ε ≈ 0.0001°, soit ~11 m) pour réduire la redondance des trajectoires haute fréquence

### Étape 4 — Identification spatiale (HDBSCAN)

- **Filtre préalable :** conserver uniquement les navires avec SOG < 1 nœud (stationnaires sûrs)
- **Features de clustering :** `[LAT, LON, Heading]` — le cap (Heading) permet de distinguer les navires à quai des navires au mouillage
- **Paramètres :** `min_cluster_size=5`, `min_samples=3`
- **Distinction quai / mouillage :**
  - À quai : caps alignés sur le terminal (faible écart-type du Heading au sein du cluster)
  - Au mouillage / en attente : caps dispersés (vent + courant imposent des orientations variées)
- **Enrichissement optionnel :** pondérer par `Draft` pour mesurer la capacité bloquée

**Résultat Phase 1 :** trajectoires continues physiquement fiables + clusters étiquetés (docked / waiting) → input direct pour Phase 2 (Manifold + Score de gravité).

---

### 13 features quotidiennes (input manifold)

`vessel_count, SOG_mean, SOG_std, SOG_median, utilization_rate_rho, hdbscan_cluster_count, hdbscan_noise_ratio, membership_score_mean, membership_score_std, draft_mean, draft_std, blocked_capacity, tanker_ratio`

---

## État d'avancement

### Fait ✅

- Rapport LaTeX complet rédigé (intro, état de l'art, roadmap, Gantt 16 semaines, diagramme TikZ)
- État de l'art en français académique couvrant : AIS engineering, théorie des files d'attente portuaire, Manifold Learning, PINNs/LWR
- Bibliographie nettoyée (papiers faibles/redondants retirés)
- Implémentation démarrée sur Marine Cadastre août 2017 (Harvey) : chargement DuckDB, filtrage bbox, conversion Parquet, déduplication, HDBSCAN (~86 clusters significatifs, ~620 navires stationnaires)
- Pipeline de features quotidiennes défini

### En cours / À faire 🔜

**Phase 1 — Houston (priorité immédiate)**
- Étapes 2–4 sur données août 2017 (filtrage cinématique, interpolation, HDBSCAN journalier)
- Étendre le pipeline sur 2015–2017 (baseline pré-Harvey + crise) → construire la matrice de features quotidiennes (13 features)
- Vérifier/supprimer le CSV brut `data/raw/AIS_2017_08_01.csv` (1 GB — parquet existant dans `data/parquet/houston/`)

**Phase 2 — Manifold / Score de gravité**
- UMAP sur la matrice de features Houston → extraction du gravity score
- Ensuite : pipeline équivalent Suez/Rotterdam (GFW + MarineTraffic)

**Phases 3–4**
- PINNs / LWR pour Time to Clear
- Modèle de prime de risque (XGBoost/Elastic Net), validation sur crise Mer Rouge 2023–2024

---

## Principes de design

- **HDBSCAN > DBSCAN** : robuste à la densité hétérogène, score de membership continu utilisable dans le gravity score
- **Manifold Learning (UMAP) sert deux rôles** : compression des features AIS en coordonnées de régime de marché + pont entre couche physique et financière
- **Résoudre la redondance temporelle AVANT le clustering** : une position par navire par jour
- **Start simple, then scale** : Houston/Marine Cadastre d'abord, Suez/Rotterdam ensuite
- **Discipline stockage** : un an de données à la fois, Parquet immédiat, suppression des CSV bruts, jamais >70 GB total

---

## Conventions de code

- Langue du code et commentaires : **anglais**
- Langue du rapport et de la documentation : **français académique**
- Privilégier Polars (lazy) sur Pandas quand possible
- Fichiers Parquet comme format de stockage intermédiaire systématique
- Noms de variables descriptifs, pas d'abréviations cryptiques
- Scripts autonomes et reproductibles (pas de notebooks Jupyter en production)
- Toujours vérifier la compatibilité MPS avant d'utiliser .to("mps") sur PyTorch

---

## Structure de fichiers

```
Nowcasting/
├── CLAUDE.md                       # Ce fichier
├── data/
│   ├── raw/                        # CSV temporaires (supprimer après conversion Parquet)
│   ├── parquet/
│   │   ├── houston/                # Données Marine Cadastre nettoyées
│   │   └── suez/
│   │       ├── ais/                # Données AIS Suez (GFW)
│   │       └── presence/           # Données présence/détection
│   ├── features/                   # Matrices de features quotidiennes
│   └── financial/                  # Indices BDI, FBX, WCI
├── src/
│   ├── ingestion/                  # Scripts collecte + nettoyage AIS
│   ├── clustering/                 # HDBSCAN, déduplication
│   ├── manifold/                   # UMAP, gravity score
│   ├── pinns/                      # PINNs + LWR
│   ├── correlation/                # Modèle prime de risque
│   └── utils/                      # Helpers communs
├── notebooks/                      # Exploration seulement (pas de production)
├── outputs/
│   ├── figures/                    # Cartes Folium, plots, PNG
│   └── models/                     # Checkpoints PyTorch
├── report/                         # Sources LaTeX du rapport
└── references/                     # PDFs articles, BibTeX
```

---

## Indices financiers cibles

| Indice | Segment | Fréquence |
|--------|---------|-----------|
| BDI (Baltic Dry Index) | Vrac sec | Quotidien |
| FBX (Freightos Baltic Index) | Conteneurs | Hebdomadaire |
| WCI (World Container Index) | Conteneurs | Hebdomadaire |

---

## Événements de référence pour validation

| Événement | Période | Zone | Usage |
|-----------|---------|------|-------|
| Hurricane Harvey | Août 2017 | Houston Ship Channel | Test initial pipeline (Phase 1-2) |
| Crise Houthi / Mer Rouge | 2023–2024 | Canal de Suez | Expérience naturelle Phase 4 |
