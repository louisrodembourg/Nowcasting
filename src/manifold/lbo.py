"""
Phase 2 — Manifold Learning : Laplace-Beltrami Operator (LBO).

Étapes 1→4 du pipeline manifold :
  1. Normalisation des 13 features (L2 par ligne → variations, pas valeurs absolues)
  2. Graphe KNN + matrice de poids W (décroissance gaussienne sur distance euclidienne)
  3. Construction et décomposition de l'opérateur LBO → vecteurs propres (ϕ, λ)
  4. Détection des points caractéristiques (extremums locaux dans le graphe KNN)

Usage (standalone):
    python src/manifold/lbo.py
    python src/manifold/lbo.py --k 7 --n-eigenvectors 10
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/features/houston_daily_features.parquet")
OUTPUT_PATH   = Path("data/features/houston_manifold.parquet")

FEATURE_COLS = [
    "vessel_count", "SOG_mean", "SOG_std", "SOG_median",
    "utilization_rate_rho", "hdbscan_cluster_count", "hdbscan_noise_ratio",
    "membership_score_mean", "membership_score_std",
    "draft_mean", "draft_std", "blocked_capacity", "tanker_ratio",
]

# Default hyperparameters
DEFAULT_K              = 7   # K-nearest neighbours
DEFAULT_N_EIGENVECTORS = 8   # number of eigenvectors to compute (excl. trivial ϕ0)


# ---------------------------------------------------------------------------
# Step 1 — Normalisation
# ---------------------------------------------------------------------------

def normalize_features(df: pl.DataFrame) -> np.ndarray:
    """
    Extract and L2-normalize the feature matrix.
    Each row (day) is divided by its L2 norm so the manifold captures
    *relative* congestion patterns rather than absolute magnitudes.
    Rows with zero norm (e.g. Harvey blackout days) are left as zeros.
    Returns a (N, 13) float64 numpy array.
    """
    X = df.select(FEATURE_COLS).cast(pl.Float64).to_numpy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # avoid div-by-zero for Harvey days
    return X / norms


# ---------------------------------------------------------------------------
# Step 2 — KNN graph + weight matrix W
# ---------------------------------------------------------------------------

def build_weight_matrix(X_norm: np.ndarray, k: int) -> sparse.csr_matrix:
    """
    Build the symmetric weight matrix W using KNN + Gaussian kernel.

    W_ij = exp(-||xi - xj||² / sigma²)  if j ∈ KNN(i) or i ∈ KNN(j)
    W_ij = 0                              otherwise

    Sigma is set to the mean of all KNN distances (data-driven bandwidth).
    Symmetry enforced via W = (W + W^T) / 2.
    """
    n = len(X_norm)
    k_safe = min(k, n - 1)

    nbrs = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(X_norm)
    distances, indices = nbrs.kneighbors(X_norm)

    # Exclude self (index 0 is always the point itself)
    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    sigma = distances.mean()
    if sigma == 0:
        sigma = 1.0
    log.debug("KNN sigma (bandwidth) = %.6f", sigma)

    # Build sparse W
    rows = np.repeat(np.arange(n), k_safe)
    cols = indices.ravel()
    vals = np.exp(-(distances.ravel() ** 2) / (sigma ** 2))

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2   # enforce symmetry
    return W


# ---------------------------------------------------------------------------
# Step 3 — Laplace-Beltrami operator + eigendecomposition
# ---------------------------------------------------------------------------

def compute_eigenvectors(
    W: sparse.csr_matrix, n_eigenvectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the normalized LBO L = A^{-1} W and solve the generalised
    eigenproblem  W ϕ = λ A ϕ.

    Returns (eigenvalues, eigenvectors) sorted by ascending eigenvalue.
    eigenvalues  : (n_eigenvectors+1,) — includes trivial λ0 ≈ 1
    eigenvectors : (N, n_eigenvectors+1) — ϕ[:,0] is the trivial constant vector
    """
    # Diagonal degree matrix A
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-10   # guard against isolated nodes
    A = sparse.diags(degree)

    n_eigs = min(n_eigenvectors + 1, W.shape[0] - 1)

    # Solve W ϕ = λ A ϕ  →  largest eigenvalues of A^{-1} W
    A_inv = sparse.diags(1.0 / degree)
    L     = A_inv @ W   # normalized LBO

    eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which="LM")

    # Sort descending (largest eigenvalue = smoothest mode = structural info)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    log.info("Eigenvalues: %s", np.round(eigenvalues, 4))
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Step 4 — Characteristic points (local extrema in the KNN graph)
# ---------------------------------------------------------------------------

def find_characteristic_points(
    eigenvectors: np.ndarray,
    W: sparse.csr_matrix,
    n_components: int = 3,
) -> np.ndarray:
    """
    Detect local extrema of the first n_components non-trivial eigenvectors.

    A point i is a local maximum (resp. minimum) of eigenvector ϕ if
    ϕ[i] > ϕ[j] (resp. <) for all neighbours j in the KNN graph.

    Returns a boolean mask of shape (N,) — True = characteristic point.
    """
    # Skip trivial eigenvector (ϕ[:,0] ≈ constant)
    phi = eigenvectors[:, 1: n_components + 1]
    n   = phi.shape[0]

    # Neighbour index list from sparse W
    W_coo    = W.tocoo()
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(W_coo.row, W_coo.col):
        if i != j:
            neighbours[i].append(j)

    is_characteristic = np.zeros(n, dtype=bool)
    for i in range(n):
        nbr = neighbours[i]
        if not nbr:
            continue
        for c in range(phi.shape[1]):
            vals_nbr = phi[nbr, c]
            if phi[i, c] > vals_nbr.max() or phi[i, c] < vals_nbr.min():
                is_characteristic[i] = True
                break

    n_char = int(is_characteristic.sum())
    log.info("Characteristic points: %d / %d days", n_char, n)
    return is_characteristic


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lbo(
    k: int = DEFAULT_K,
    n_eigenvectors: int = DEFAULT_N_EIGENVECTORS,
) -> pl.DataFrame:
    """
    Full LBO pipeline on houston_daily_features.parquet.
    Saves results to houston_manifold.parquet and returns the DataFrame.

    Output columns (appended to original features):
        phi_1 … phi_N  : eigenvector coordinates (manifold embedding)
        eigenvalue_1…N : corresponding eigenvalues
        is_characteristic: True if local extremum in any of the first 3 eigenvectors
    """
    df = pl.read_parquet(FEATURES_PATH).sort("date")
    log.info("Loaded %d days × %d features", *df.select(FEATURE_COLS).shape)

    # Step 1 — normalise
    X_norm = normalize_features(df)

    # Step 2 — KNN + weight matrix
    W = build_weight_matrix(X_norm, k=k)

    # Step 3 — LBO eigendecomposition
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)

    # Step 4 — characteristic points
    is_characteristic = find_characteristic_points(eigenvectors, W)

    # Assemble output DataFrame
    extra_cols = {"is_characteristic": is_characteristic.tolist()}
    # Skip trivial eigenvector (index 0)
    for i in range(1, eigenvectors.shape[1]):
        extra_cols[f"phi_{i}"]        = eigenvectors[:, i].tolist()
        extra_cols[f"eigenvalue_{i}"] = float(eigenvalues[i])

    df_out = df.with_columns([
        pl.Series(name, vals)
        for name, vals in extra_cols.items()
        if isinstance(extra_cols[name], list)
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(OUTPUT_PATH)
    log.info("Saved manifold output → %s", OUTPUT_PATH)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — LBO Manifold Learning")
    parser.add_argument("--k",               type=int, default=DEFAULT_K,
                        help=f"KNN neighbours (default {DEFAULT_K})")
    parser.add_argument("--n-eigenvectors",  type=int, default=DEFAULT_N_EIGENVECTORS,
                        help=f"Number of eigenvectors (default {DEFAULT_N_EIGENVECTORS})")
    args = parser.parse_args()

    df = run_lbo(k=args.k, n_eigenvectors=args.n_eigenvectors)
    print(df.select(["date", "is_characteristic"] +
                    [c for c in df.columns if c.startswith("phi_")]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
