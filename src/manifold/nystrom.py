"""
Extension de Nyström — projection out-of-sample dans un eigenspace de référence.

Permet de projeter des données 2020-2024 dans l'espace du manifold 2019 :
  φ_c(x_new) ≈ (1/λ_c) × Σ_i [ k(x_new, x_i) / Σ_j k(x_new, x_j) ] × φ_c(x_i)

où k est le même noyau gaussien utilisé pour construire W en 2019.

Usage:
    ref  = load_reference("data/features/la_2019_manifold_ref.npz")
    phi  = project_onto_manifold(X_new_norm, **ref)
"""
from pathlib import Path

import numpy as np


def load_reference(ref_path: Path) -> dict:
    """
    Charge le fichier .npz de référence généré par run_lbo(save_reference=True).
    Retourne un dict avec clés : X_norm, sigma, eigenvalues, eigenvectors.
    """
    data = np.load(ref_path)
    return {
        "X_train":     data["X_norm"],
        "sigma":       float(data["sigma"][0]),
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": data["eigenvectors"],
    }


def project_onto_manifold(
    X_new: np.ndarray,
    X_train: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Projette X_new (M, 13) dans l'eigenspace de référence (N, k) via Nyström.

    Formule (random walk D^{-1}W) :
        K[i, j]       = exp(-||x_new[i] - x_train[j]||² / σ²)
        K_norm[i, :]  = K[i, :] / sum(K[i, :])
        φ_c(x_new[i]) = (1/λ_c) × K_norm[i, :] @ φ_c_train

    Retourne phi_new (M, k) — coordonnées dans l'eigenspace 2019.
    """
    # Distances au carré entre X_new et X_train — vectorisé sans boucle
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    sq_a = (X_new   ** 2).sum(axis=1, keepdims=True)   # (M, 1)
    sq_b = (X_train ** 2).sum(axis=1, keepdims=True).T  # (1, N)
    cross = X_new @ X_train.T                            # (M, N)
    sq_dists = sq_a + sq_b - 2 * cross                  # (M, N)
    sq_dists = np.maximum(sq_dists, 0.0)                # évite les -ε numériques

    # Noyau gaussien
    K = np.exp(-sq_dists / (sigma ** 2))  # (M, N)

    # Normalisation ligne (random walk)
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    K_norm = K / row_sums  # (M, N)

    # Projection Nyström : φ(x) ≈ (1/λ) × K_norm @ Φ_train
    phi_new = (K_norm @ eigenvectors) / eigenvalues  # (M, k)

    return phi_new
