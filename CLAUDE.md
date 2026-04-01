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
| Marine Cadastre / NOAA | Eaux US, colonnes riches (tirant d'eau, dimensions, cargo) | Houston — Phase 1-2 |
| Global Fishing Watch | Historique long terme, global | Suez — Phase 2 (manifold) |
| MarineTraffic / UNGP | Haute fréquence | Suez — Phase 3 (PINNs) |
| AISStream.io | WebSocket temps réel | Optionnel, temps réel |
| AISHub | API REST | Optionnel |

### Workflow type (Marine Cadastre)

1. Télécharger les CSV mensuels depuis Marine Cadastre
2. Charger via DuckDB, filtrer par bounding box géographique
3. Convertir en Parquet, supprimer les CSV bruts (discipline stockage)
4. Dédupliquer : agréger à **une position par navire** avant clustering (résolution de la redondance temporelle — critique pour éviter l'inflation de clusters)
5. Appliquer HDBSCAN pour détecter les clusters de navires stationnaires

### 13 features quotidiennes (input manifold)

vessel_count, SOG_mean, SOG_std, SOG_median, utilization_rate_rho, hdbscan_cluster_count, hdbscan_noise_ratio, membership_score_mean, membership_score_std, draft_mean, draft_std, blocked_capacity, tanker_ratio

---

## État d'avancement

### Fait ✅

- Rapport LaTeX complet rédigé (intro, état de l'art, roadmap, Gantt 16 semaines, diagramme TikZ)
- État de l'art en français académique couvrant : AIS engineering, théorie des files d'attente portuaire, Manifold Learning, PINNs/LWR
- Bibliographie nettoyée (papiers faibles/redondants retirés)
- Implémentation démarrée sur Marine Cadastre août 2017 (Harvey) : chargement DuckDB, filtrage bbox, conversion Parquet, déduplication, HDBSCAN (~86 clusters significatifs, ~620 navires stationnaires)
- Pipeline de features quotidiennes défini

### En cours / À faire 🔜

- Appliquer le pipeline de features sur plusieurs jours → construire la matrice d'entrée du manifold
- Pipeline d'accès données Suez/Rotterdam (GFW + MarineTraffic)
- UMAP + construction du gravity score (Phase 2)
- PINNs / LWR pour Time to Clear (Phase 3)
- Modèle de prime de risque (Phase 4), validation sur crise Mer Rouge

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

## Structure de fichiers attendue

```
project/
├── CLAUDE.md                  # Ce fichier
├── data/
│   ├── raw/                   # CSV temporaires (supprimer après conversion)
│   ├── parquet/               # Données nettoyées
│   └── features/              # Matrices de features quotidiennes
├── src/
│   ├── ingestion/             # Scripts collecte + nettoyage AIS
│   ├── clustering/            # HDBSCAN, déduplication
│   ├── manifold/              # UMAP, gravity score
│   ├── pinns/                 # PINNs + LWR
│   ├── correlation/           # Modèle prime de risque
│   └── utils/                 # Helpers communs
├── notebooks/                 # Exploration seulement (pas de production)
├── outputs/
│   ├── figures/               # Cartes Folium, plots
│   └── models/                # Checkpoints PyTorch
├── report/                    # Sources LaTeX du rapport
└── references/                # PDFs articles, BibTeX
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
