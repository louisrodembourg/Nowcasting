"""
Phase 2 — Manifold Learning : Laplace-Beltrami Operator (LBO).

Étapes 1→4 du pipeline manifold :
  1. Normalisation des 13 features (L2 par ligne → variations, pas valeurs absolues)
  2. Graphe KNN + matrice de poids W (décroissance gaussienne sur distance euclidienne)
  3. Construction et décomposition de l'opérateur LBO → vecteurs propres (ϕ, λ)
  4. Détection des points caractéristiques (extremums locaux dans le graphe KNN)

Usage (standalone):
    python src/manifold/lbo.py
    python src/manifold/lbo.py --location houston --k 7 --n-eigenvectors 10
    python src/manifold/lbo.py --location la
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

DEFAULT_LOCATION = "houston"

FEATURE_COLS = [
    "vessel_count", "SOG_mean", "SOG_std", "SOG_median",
    "utilization_rate_rho", "waiting_cluster_count", "hdbscan_noise_ratio",
    "membership_score_mean", "membership_score_std",
    "draft_mean", "draft_std", "waiting_capacity", "tanker_ratio",
]

# Paramètres par défaut pour le clustering KNN et la décomposition spectrale
DEFAULT_K              = 7   # K-plus proches voisins
DEFAULT_N_EIGENVECTORS = 8   # nombre de vecteurs propres à calculer (hors ϕ0 trivial)


# ---------------------------------------------------------------------------
# Étape 1 — Normalisation
# ---------------------------------------------------------------------------

def normalize_features(df: pl.DataFrame) -> np.ndarray:
    """
    Extrait et normalise en L2 la matrice de features.
    Chaque ligne (jour) est divisée par sa norme L2 pour que la variété capture
    les *patterns relatifs* de congestion plutôt que les magnitudes absolues.
    Les lignes avec une norme zéro (ex. jours de blackout Harvey) restent zéro.
    Retourne une matrice (N, 13) float64 numpy.
    """
    X = df.select(FEATURE_COLS).cast(pl.Float64).to_numpy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # évite division par zéro pour les jours de blackout
    return X / norms


# ---------------------------------------------------------------------------
# Étape 2 — Graphe KNN + matrice de poids W
# ---------------------------------------------------------------------------

def build_weight_matrix(X_norm: np.ndarray, k: int) -> sparse.csr_matrix:
    """
    Construit la matrice symétrique de poids W en utilisant KNN + noyau gaussien.

    W_ij = exp(-||xi - xj||² / sigma²)  si j ∈ KNN(i) ou i ∈ KNN(j)
    W_ij = 0                              sinon

    Sigma est fixé à la moyenne de toutes les distances KNN (largeur de bande orientée par les données).
    La symétrie est appliquée via W = (W + W^T) / 2.
    """
    n = len(X_norm)
    k_safe = min(k, n - 1)

    # Construit le graphe KNN
    nbrs = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(X_norm)
    distances, indices = nbrs.kneighbors(X_norm)

    # Exclut l'auto-voisinage (index 0 est toujours le point lui-même)
    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    # Détermine sigma (largeur de bande) comme la moyenne des distances KNN
    sigma = distances.mean()
    if sigma == 0:
        sigma = 1.0
    log.debug("KNN sigma (largeur de bande) = %.6f", sigma)

    # Construit W creuse (sparse)
    rows = np.repeat(np.arange(n), k_safe)
    cols = indices.ravel()
    vals = np.exp(-(distances.ravel() ** 2) / (sigma ** 2))

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2   # force la symétrie
    return W


# ---------------------------------------------------------------------------
# Étape 3 — Opérateur de Laplace-Beltrami + décomposition en vecteurs propres
# ---------------------------------------------------------------------------

def compute_eigenvectors(
    W: sparse.csr_matrix, n_eigenvectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit l'opérateur LBO normalisé L = A^{-1} W et résout le problème
    généralisé W ϕ = λ A ϕ.

    Retourne (valeurs_propres, vecteurs_propres) triés par valeur propre croissante.
    eigenvalues  : (n_eigenvectors+1,) — inclut λ0 ≈ 1 triviale
    eigenvectors : (N, n_eigenvectors+1) — ϕ[:,0] est le vecteur constant trivial
    """
    # Matrice diagonale du degré A
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-10   # protection contre les nœuds isolés
    A = sparse.diags(degree)

    n_eigs = min(n_eigenvectors + 1, W.shape[0] - 1)

    # Résout W ϕ = λ A ϕ  →  plus grandes valeurs propres de A^{-1} W
    A_inv = sparse.diags(1.0 / degree)
    L     = A_inv @ W   # opérateur LBO normalisé

    eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which="LM")

    # Trie en ordre décroissant (plus grande valeur propre = mode le plus lisse = info structurelle)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    log.info("Valeurs propres : %s", np.round(eigenvalues, 4))
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Étape 4 — Points caractéristiques (extremums locaux dans le graphe KNN)
# ---------------------------------------------------------------------------

def find_characteristic_points(
    eigenvectors: np.ndarray,
    W: sparse.csr_matrix,
    n_components: int = 3,
) -> np.ndarray:
    """
    Détecte les extremums locaux des premiers n_components vecteurs propres non triviaux.

    Un point i est un maximum local (resp. minimum) du vecteur propre ϕ si
    ϕ[i] > ϕ[j] (resp. <) pour tous les voisins j dans le graphe KNN.

    Retourne un masque booléen de forme (N,) — True = point caractéristique.
    """
    # Ignore le vecteur propre trivial (ϕ[:,0] ≈ constante)
    phi = eigenvectors[:, 1: n_components + 1]
    n   = phi.shape[0]

    # Liste d'index des voisins depuis la matrice W creuse
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
        # Vérifie si i est un extremum local dans au moins une composante
        for c in range(phi.shape[1]):
            vals_nbr = phi[nbr, c]
            if phi[i, c] > vals_nbr.max() or phi[i, c] < vals_nbr.min():
                is_characteristic[i] = True
                break

    n_char = int(is_characteristic.sum())
    log.info("Points caractéristiques : %d / %d jours", n_char, n)
    return is_characteristic


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def run_lbo(
    k: int = DEFAULT_K,
    n_eigenvectors: int = DEFAULT_N_EIGENVECTORS,
    location: str = DEFAULT_LOCATION,
    features_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """
    Pipeline LBO complet sur un parquet de features.
    Sauvegarde les résultats dans data/features/{location}_manifold.parquet et retourne le DataFrame.

    Colonnes de sortie (ajoutées aux features originales):
        phi_1 … phi_N  : coordonnées des vecteurs propres (plongement de variété)
        eigenvalue_1…N : valeurs propres correspondantes
        is_characteristic: True si extremum local dans l'un des 3 premiers vecteurs propres
    """
    path = (Path(features_path) if features_path is not None
            else Path(f"data/features/{location}_daily_features.parquet"))
    out  = (Path(output_path)   if output_path   is not None
            else Path(f"data/features/{location}_manifold.parquet"))
    df = pl.read_parquet(path).sort("date")
    log.info("Chargé %d jours × %d features", *df.select(FEATURE_COLS).shape)

    # Étape 1 — normalise
    X_norm = normalize_features(df)

    # Étape 2 — KNN + matrice de poids
    W = build_weight_matrix(X_norm, k=k)

    # Étape 3 — décomposition en vecteurs propres LBO
    eigenvalues, eigenvectors = compute_eigenvectors(W, n_eigenvectors)

    # Étape 4 — points caractéristiques
    is_characteristic = find_characteristic_points(eigenvectors, W)

    # Assemble le DataFrame de sortie
    extra_cols = {"is_characteristic": is_characteristic.tolist()}
    # Ignore le vecteur propre trivial (index 0)
    for i in range(1, eigenvectors.shape[1]):
        extra_cols[f"phi_{i}"]        = eigenvectors[:, i].tolist()
        extra_cols[f"eigenvalue_{i}"] = float(eigenvalues[i])

    df_out = df.with_columns([
        pl.Series(name, vals)
        for name, vals in extra_cols.items()
        if isinstance(extra_cols[name], list)
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out)
    log.info("Sortie de variété sauvegardée → %s", out)
    return df_out


def main() -> None:
    """Point d'entrée pour l'exécution autonome — Phase 2 du pipeline manifold."""
    parser = argparse.ArgumentParser(description="Phase 2 — LBO Manifold Learning")
    parser.add_argument("--location",        default=DEFAULT_LOCATION,
                        help=f"Port cible (défaut {DEFAULT_LOCATION})")
    parser.add_argument("--k",               type=int, default=DEFAULT_K,
                        help=f"Voisins KNN (défaut {DEFAULT_K})")
    parser.add_argument("--n-eigenvectors",  type=int, default=DEFAULT_N_EIGENVECTORS,
                        help=f"Nombre de vecteurs propres (défaut {DEFAULT_N_EIGENVECTORS})")
    args = parser.parse_args()

    df = run_lbo(k=args.k, n_eigenvectors=args.n_eigenvectors, location=args.location)
    print(df.select(["date", "is_characteristic"] +
                    [c for c in df.columns if c.startswith("phi_")]))


if __name__ == "__main__":
    # Configure le logging pour afficher les messages d'info avec timestamps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
