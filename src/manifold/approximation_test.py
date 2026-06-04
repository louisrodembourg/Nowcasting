"""
Test d'approximation des données — Xu et al.

Compare la qualité d'approximation de 4 méthodes de sélection de zones constituantes :
  - LBO    (Laplace-Beltrami Operator — variété riemannienne)
  - NMF    (Non-negative Matrix Factorization)
  - PCA    (Analyse en Composantes Principales — linéaire, baseline)
  - Random (sélection aléatoire — baseline inférieur)

Protocole (Xu et al.) :
  Matrice X (N_zones × N_jours) = taux d'occupation quotidien par zone.

  Test 1 — MSE vs k zones (comparaison équitable) :
    Pour k ∈ {1, 2, ..., N_zones} zones sélectionnées par chaque méthode :
      signal_réduit = moyenne des k zones sélectionnées (courbe temporelle)
      signal_global = moyenne de toutes les zones (courbe temporelle)
      MSE(k) = ||signal_réduit - signal_global||² / N_jours
    → La méthode avec le MSE le plus bas pour un k donné est la plus efficace.

  Test 2 — MSE vs n_components (version Xu et al. stricte) :
    Pour n_comp ∈ {2, 3, ..., 10} composantes :
      Extraire les zones constituantes (extremums locaux dans l'espace réduit)
      Calculer MSE entre leur moyenne et la moyenne globale
    → LBO doit atteindre un MSE faible avec peu de composantes.

Résultat attendu (Xu et al.) :
  LBO < NMF < PCA ≈ Random à faible k / faible n_comp.
  LBO capture la structure non-linéaire de la variété avec moins de zones.

Usage :
    python src/manifold/approximation_test.py --start 2019-01-01 --end 2019-06-30 --location la
    python src/manifold/approximation_test.py --start 2017-07-01 --end 2017-09-15 --location houston
    python src/manifold/approximation_test.py --start 2019-01-01 --end 2019-06-30 --location la --max-zones 30
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import NearestNeighbors

from src.manifold.nmf_pipeline import _make_config, build_zone_matrix
from src.ingestion.download import LOCATIONS

log = logging.getLogger(__name__)

# ── Styles visuels ────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "LBO":    {"color": "#1565C0", "dash": "solid",  "width": 2.5},
    "NMF":    {"color": "#2E7D32", "dash": "solid",  "width": 2.0},
    "PCA":    {"color": "#F57F17", "dash": "dash",   "width": 2.0},
    "Random": {"color": "#BDBDBD", "dash": "dot",    "width": 1.5},
}


# ── Normalisation et graphe KNN ───────────────────────────────────────────────

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _build_knn_kernel(X_norm: np.ndarray, k: int) -> sparse.csr_matrix:
    n      = len(X_norm)
    k_safe = min(k, n - 1)
    nbrs   = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(X_norm)
    dists, indices = nbrs.kneighbors(X_norm)
    dists, indices = dists[:, 1:], indices[:, 1:]
    sigma  = float(dists.mean()) or 1.0
    rows   = np.repeat(np.arange(n), k_safe)
    cols   = indices.ravel()
    vals   = np.exp(-(dists.ravel() ** 2) / sigma ** 2)
    W      = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return (W + W.T) / 2


def _lbo_eigenvectors(X: np.ndarray, n_eigs: int, k: int) -> np.ndarray:
    """Retourne les vecteurs propres LBO (N_zones × n_eigs), vecteur trivial exclu."""
    W      = _build_knn_kernel(_normalize_rows(X), k)
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-10
    L      = sparse.diags(1.0 / degree) @ W
    n_safe = min(n_eigs + 1, W.shape[0] - 1)
    eigvals, eigvecs = eigsh(L, k=n_safe, which="LM")
    idx    = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx][:, 1:]   # skip trivial φ₀


# ── Scores de sélection (continus, pour le classement par zone) ──────────────

def _score_lbo(X: np.ndarray, n_eigenvectors: int, k_neighbors: int) -> np.ndarray:
    """
    Score LBO de chaque zone = déviation maximale par rapport à la moyenne
    sur tous les vecteurs propres non-triviaux (extremeness dans la variété).
    """
    phi = _lbo_eigenvectors(X, n_eigenvectors, k_neighbors)
    return np.abs(phi - phi.mean(axis=0)).max(axis=1)


def _score_nmf(X: np.ndarray, n_components: int) -> np.ndarray:
    """Score NMF = activation maximale d'une zone sur toutes les composantes."""
    n_comp = max(1, min(n_components, min(X.shape) - 1))
    W = NMF(n_components=n_comp, init="random", random_state=42, max_iter=1000).fit_transform(X + 1e-10)
    return W.max(axis=1)


def _score_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Score PCA = amplitude maximale dans l'espace des composantes principales."""
    n_comp = max(1, min(n_components, min(X.shape) - 1))
    scores = PCA(n_components=n_comp).fit_transform(X)
    return np.abs(scores).max(axis=1)


# ── Détection des extremums locaux (pour Test 2) ──────────────────────────────

def _find_extremums(components: np.ndarray, k: int) -> np.ndarray:
    """
    Détecte les extremums locaux dans l'espace réduit (masque booléen).
    Un point i est un extremum si φ_c(i) est strictement max ou min
    de tous ses k voisins sur au moins une composante c.
    """
    n      = components.shape[0]
    k_safe = min(k, n - 1)
    nbrs   = NearestNeighbors(n_neighbors=k_safe + 1, algorithm="kd_tree").fit(components)
    _, indices = nbrs.kneighbors(components)

    is_extremum = np.zeros(n, dtype=bool)
    for i in range(n):
        nbr = indices[i, 1:]
        for c in range(components.shape[1]):
            v = components[nbr, c]
            if components[i, c] > v.max() or components[i, c] < v.min():
                is_extremum[i] = True
                break
    return is_extremum


# ── MSE ───────────────────────────────────────────────────────────────────────

def _mse(X: np.ndarray, idx: np.ndarray) -> float:
    mean_full    = X.mean(axis=0)
    mean_reduced = X[idx].mean(axis=0)
    return float(np.mean((mean_full - mean_reduced) ** 2))


# ============================================================================
# TEST 1 — MSE vs k zones (comparaison équitable à nombre de zones fixé)
# ============================================================================

def run_approximation_test(
    X: np.ndarray,
    n_eigenvectors: int = 8,
    n_components: int   = 5,
    k_neighbors: int    = 5,
    n_random_trials: int = 15,
    max_zones: int | None = None,
) -> dict[str, list[float]]:
    """
    Pour chaque méthode, classe les zones par score décroissant.
    Pour k ∈ [1, max_zones], sélectionne les top-k zones et calcule le MSE.

    Retourne {méthode: [mse_k=1, mse_k=2, ...]}.
    """
    n_zones = X.shape[0]
    max_k   = min(max_zones or n_zones, n_zones)

    log.info("Calcul des scores de zone (%d zones × %d jours)...", *X.shape)

    rank = {
        "LBO": np.argsort(_score_lbo(X, n_eigenvectors, k_neighbors))[::-1],
        "NMF": np.argsort(_score_nmf(X, n_components))[::-1],
        "PCA": np.argsort(_score_pca(X, n_components))[::-1],
    }

    rng          = np.random.RandomState(42)
    random_perms = [rng.permutation(n_zones) for _ in range(n_random_trials)]

    mse: dict[str, list[float]] = {m: [] for m in ["LBO", "NMF", "PCA", "Random"]}

    for k in range(1, max_k + 1):
        for method, r in rank.items():
            mse[method].append(_mse(X, r[:k]))
        mse["Random"].append(
            float(np.mean([_mse(X, p[:k]) for p in random_perms]))
        )

    # Log résumé
    for method, values in mse.items():
        k10 = min(9, len(values) - 1)
        log.info("  %-8s MSE@k=1 %.5f  MSE@k=10 %.5f", method, values[0], values[k10])

    return mse


# ============================================================================
# TEST 2 — MSE vs n_components (version stricte Xu et al.)
# ============================================================================

def run_component_test(
    X: np.ndarray,
    k_neighbors: int  = 5,
    n_comp_range: list[int] | None = None,
) -> dict[str, list[float]]:
    """
    Pour chaque n_comp, extrait les zones constituantes (extremums) de chaque méthode
    et calcule le MSE entre leur courbe de congestion moyenne et la courbe globale.

    Retourne {méthode: [mse_ncomp=2, mse_ncomp=3, ...]}.
    """
    n_comp_range = n_comp_range or list(range(2, min(11, X.shape[0])))
    mse: dict[str, list[float]] = {m: [] for m in ["LBO", "NMF", "PCA"]}

    for n_comp in n_comp_range:
        n_safe = max(1, min(n_comp, min(X.shape) - 1))

        # LBO — extremums dans l'espace des n_comp premiers vecteurs propres
        phi_lbo    = _lbo_eigenvectors(X, n_comp, k_neighbors)[:, :n_safe]
        idx_lbo    = np.where(_find_extremums(phi_lbo,  k_neighbors))[0]

        # NMF — extremums dans la matrice W (n_comp facteurs)
        W_nmf      = NMF(n_components=n_safe, init="random", random_state=42, max_iter=1000).fit_transform(X + 1e-10)
        idx_nmf    = np.where(_find_extremums(W_nmf, k_neighbors))[0]

        # PCA — extremums dans les scores PCA (n_comp composantes)
        scores_pca = PCA(n_components=n_safe).fit_transform(X)
        idx_pca    = np.where(_find_extremums(scores_pca, k_neighbors))[0]

        for method, idx in [("LBO", idx_lbo), ("NMF", idx_nmf), ("PCA", idx_pca)]:
            if len(idx) == 0:
                mse[method].append(float(_mse(X, np.arange(len(X)))))
            else:
                mse[method].append(_mse(X, idx))

        log.info("n_comp=%d : LBO=%d zones NMF=%d zones PCA=%d zones",
                 n_comp, len(idx_lbo), len(idx_nmf), len(idx_pca))

    return mse, n_comp_range


# ============================================================================
# Visualisation
# ============================================================================

def plot_results(
    mse_k: dict[str, list[float]],
    mse_comp: dict[str, list[float]],
    n_comp_range: list[int],
    location: str,
    start: date,
    end: date,
    out_path: Path,
) -> None:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Test 1 — MSE vs nombre de zones sélectionnées (k)",
            "Test 2 — MSE vs nombre de composantes (Xu et al.)",
        ],
    )

    # ── Test 1 ────────────────────────────────────────────────────────────────
    k_range = list(range(1, len(next(iter(mse_k.values()))) + 1))
    for method, values in mse_k.items():
        s = METHOD_STYLE[method]
        fig.add_trace(go.Scatter(
            x=k_range, y=values, name=method,
            mode="lines",
            line=dict(color=s["color"], dash=s["dash"], width=s["width"]),
            legendgroup=method,
        ), row=1, col=1)

    # ── Test 2 ────────────────────────────────────────────────────────────────
    for method, values in mse_comp.items():
        s = METHOD_STYLE[method]
        fig.add_trace(go.Scatter(
            x=n_comp_range, y=values, name=method,
            mode="lines+markers",
            line=dict(color=s["color"], dash=s["dash"], width=s["width"]),
            marker=dict(size=6, color=s["color"]),
            legendgroup=method,
            showlegend=False,
        ), row=1, col=2)

    fig.update_xaxes(title_text="k zones sélectionnées", row=1, col=1)
    fig.update_xaxes(title_text="n_components",           row=1, col=2)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"{location.upper()} — Test d'approximation Xu et al. ({start} → {end})",
            x=0.5,
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
        height=480,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    log.info("Graphique → %s", out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test d'approximation Xu et al. — LBO vs NMF vs PCA vs Random"
    )
    parser.add_argument("--location",       default="la", choices=list(LOCATIONS.keys()))
    parser.add_argument("--start",          required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end",            required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--n-eigenvectors", type=int, default=8,
                        help="Vecteurs propres LBO (défaut 8)")
    parser.add_argument("--n-components",   type=int, default=5,
                        help="Composantes NMF/PCA (défaut 5)")
    parser.add_argument("--k-neighbors",    type=int, default=5,
                        help="Voisins KNN (défaut 5)")
    parser.add_argument("--max-zones",      type=int, default=None,
                        help="Limite le Test 1 aux max-zones premières zones (défaut : toutes)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    config = _make_config(args.location)

    log.info("Construction de la matrice zones × jours...")
    zone_df, X, dates = build_zone_matrix(start, end, args.location, config)

    if len(X) == 0:
        log.error("Aucune zone trouvée — vérifier les fichiers Parquet.")
        return

    log.info("Matrice : %d zones × %d jours", *X.shape)

    # ── Test 1 : MSE vs k zones ───────────────────────────────────────────────
    log.info("=== Test 1 : MSE vs k zones ===")
    mse_k = run_approximation_test(
        X,
        n_eigenvectors=args.n_eigenvectors,
        n_components=args.n_components,
        k_neighbors=args.k_neighbors,
        max_zones=args.max_zones,
    )

    # ── Test 2 : MSE vs n_components ─────────────────────────────────────────
    log.info("=== Test 2 : MSE vs n_components ===")
    n_comp_max   = min(10, min(X.shape) - 1)
    n_comp_range = list(range(2, n_comp_max + 1))
    mse_comp, n_comp_range = run_component_test(X, k_neighbors=args.k_neighbors, n_comp_range=n_comp_range)

    # ── Résumé console ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {args.location.upper()} : {X.shape[0]} zones × {X.shape[1]} jours")
    print(f"{'='*55}")
    print(f"{'Méthode':<10} {'MSE@k=1':>10} {'MSE@k=5':>10} {'MSE@k=10':>10}")
    print(f"{'-'*55}")
    for method, values in mse_k.items():
        k5  = values[min(4, len(values)-1)]
        k10 = values[min(9, len(values)-1)]
        print(f"  {method:<8} {values[0]:>10.5f} {k5:>10.5f} {k10:>10.5f}")

    print(f"\n{'Méthode':<10} {'MSE@ncomp=2':>12} {'MSE@ncomp=5':>12}")
    print(f"{'-'*40}")
    for method, values in mse_comp.items():
        v2 = values[0]
        v5 = values[min(3, len(values)-1)]
        print(f"  {method:<8} {v2:>12.5f} {v5:>12.5f}")

    # ── Graphique ─────────────────────────────────────────────────────────────
    out_path = Path(f"outputs/figures/{args.location}_approximation_test.html")
    plot_results(mse_k, mse_comp, n_comp_range, args.location, start, end, out_path)
    print(f"\nGraphique → {out_path.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
