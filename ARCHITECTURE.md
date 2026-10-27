# Architecture

Module-level overview of the pipeline. See [README.md](README.md) for context and results.

```
NOAA Marine Cadastre (daily ZIP)
   │  src/ingestion/download.py        bbox + MMSI + SOG filter → Parquet (ZSTD)
   ▼
data/parquet/<location>/<location>_YYYY_MM_DD.parquet
   │  src/ingestion/kinematic_filter.py  haversine speed check, trajectory split (gap > 30 min)
   │  src/clustering/hdbscan_daily.py    HDBSCAN on stationary vessels → docked / waiting
   │  src/clustering/features_daily.py   13 daily features
   ▼
data/features/<location>_daily_features.parquet
   │  src/manifold/manifold_pipeline.py  zone × day matrix → KNN graph → LBO spectrum
   ▼
data/features/<location>_gravity_daily.parquet         (daily gravity score)
   │  src/pinns/pinn_pipeline.py         episodes → LWR PINN → ρ(x,t), v(x,t), time-to-clear
   ▼
outputs/models/pinn_<location>_episodes.pt + physical profiles (Parquet)
   │  src/correlation/*                  alignment with prices, walk-forward models, SHAP
   ▼
outputs/financial/*.csv, figures
```

## Phase 1 — Ingestion & clustering

| Module | Role |
|---|---|
| `ingestion/download.py` | Downloads the NOAA daily archives and filters them with DuckDB (bbox, valid MMSI, SOG ≤ 50 kt). Defines `LOCATIONS` (Houston, LA). |
| `ingestion/kinematic_filter.py` | Recomputes speed from consecutive positions (haversine), corrects aberrant SOG, assigns a `traj_id` at each gap longer than 30 min. |
| `ingestion/trajectory.py` | Linear / cubic Hermite interpolation (Hermite when the heading changes by more than 30°) and Douglas–Peucker compression. |
| `clustering/hdbscan_daily.py` | Daily HDBSCAN (haversine metric) on stationary episodes (SOG < 1 kt). Labels clusters as *docked* or *waiting* using heading dispersion and geographic zones. |
| `clustering/features_daily.py` | 13 daily features: vessel count, SOG statistics, utilisation rate, cluster/noise statistics, membership scores, draft, waiting capacity, tanker ratio. |
| `clustering/visualize_clusters.py`, `utils/visualize.py` | Folium / Plotly maps of the clusters (single day or animated period). |
| `scripts/run_phase1.py` | Orchestrator: download → clustering → features over a date range. |

## Phase 2 — Gravity score (manifold learning)

| Module | Role |
|---|---|
| `manifold/manifold_pipeline.py` | Builds a zone × day occupancy matrix, then a Gaussian-weighted KNN graph, then the Laplace–Beltrami eigenvectors. It identifies the *constituent zones* on a calm baseline (`--identify`) and scores each day by the weighted capacity in those zones (`--score`, `--both`). A rolling-window mode is also available (`--rolling`). |
| `manifold/manifold_zones.py` | Zone-level LBO variant used for the constituent-zone analysis. |
| `manifold/pca_pipeline.py`, `manifold/nmf_pipeline.py` | Linear baselines (PCA, NMF) used for comparison. |
| `manifold/visualize_double_view.py` | Report figure: days in eigen-space alongside the geographic map of the zones. |

## Phase 3 — PINN / time-to-clear

| Module | Role |
|---|---|
| `pinns/lwr_pinn.py` | `LWRPINN`: MLP (x, t) → (ρ, v) with sigmoid outputs and an optional `FourierEmbedding`. The loss is L_data + λ_pde·L_LWR + λ_kin·L_kinematic. The LWR residual is scaled by the Courant number α = v_max·T/L so that both PDE terms have the same order of magnitude. |
| `pinns/data_prep.py` | Projects vessels onto the channel axis, bins them in space and builds the (x, t, ρ, v) tensors. |
| `pinns/pinn_pipeline.py` | Main entry point with four steps: `structure_episodes` (episodes defined by gravity score > threshold), `train_on_episodes` (pooled training with curriculum warm-up), `extract_physical_profiles` (dense inference on a grid), and time-to-clear computation with `extract_pinn_summaries`. Also covers strict nowcast validation (`--nowcast-end`), per-episode fine-tuning and diagnostic plots (shock waves, fundamental diagram, LWR residual field, characteristic lines, episode gallery). |
| `scripts/run_train_pinn_per_episode.py` | Trains one PINN per episode and exports the summaries of the physical field. |

## Phase 4 — Market link

| Module | Role |
|---|---|
| `correlation/financial_pipeline.py` | Builds the per-episode dataset (gravity score, PINN TTC, PINN summaries) and aligns it with an asset's forward returns. Ends with the trading-signal output. |
| `correlation/correlation_pipeline.py` | Continuous approach: cross-correlogram and XGBoost (`TimeSeriesSplit`) on the daily gravity score. |
| `correlation/ttc_xgb_shap.py` + `scripts/run_phase4_xgb.py` | XGBoost regression of episode duration from PINN features, with SHAP summary and waterfall plots. |
| `correlation/correlation_analysis.py` | Spearman lag-scan: features × targets × horizons × lags. |
| `correlation/predict_returns.py` | Walk-forward Ridge / XGBoost baselines on MATX, BDRY, SBLK and container stocks. |
| `correlation/two_regime_model.py` | Causal episode detection that switches between a normal-day model and an episode model fed with PINN ρ. |
| `correlation/pinn_vs_baseline.py`, `five_steps_explanation.py`, `naive_gravity_test.py` | Ablations: what the PINN adds, the step-by-step IC progression (0.07 → 0.25), and the naive gravity-score test. |
| `index/scrap_index.py`, `index/visualize_index.py`, `ingestion/download_index.py` | Download freight indices and equity prices, and plot them against the gravity score. |

## Conventions

- Intermediate storage is Parquet. Polars (lazy) is preferred over pandas.
- All scripts are run from the repository root. `data/` and `outputs/` are git-ignored.
- PyTorch picks CUDA, then MPS (Apple Silicon), then CPU, through `lwr_pinn.get_device()`.
