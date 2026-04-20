"""
Phase 4 — Cross-correlation and Granger causality analysis.

Tests whether gravity_score (LA congestion) Granger-causes SCFI variations.

Cross-correlation: gravity[t] vs ΔSCFI[t → t+k], lags 0–21 days.
Granger: weekly subsampled data (52 obs) to respect SCFI publication frequency.

No statsmodels dependency — F-test implemented via numpy.linalg.lstsq.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from scipy import stats

log = logging.getLogger(__name__)


def compute_cross_correlation(
    gravity: np.ndarray,
    scfi: np.ndarray,
    max_lag: int = 21,
) -> pl.DataFrame:
    """
    Pearson and Spearman correlation between gravity[i] and ΔSCFI[i → i+k]
    for k in 0..max_lag.

    ΔSCFI[i → i+k] = scfi[i+k] - scfi[i]  (cumulative change over k days).
    At lag 0: uses contemporaneous ΔSCFI (1-day) as a degenerate case.
    """
    rows = []
    for lag in range(max_lag + 1):
        if lag == 0:
            g = gravity[:-1] if len(gravity) > 1 else gravity
            s = np.diff(scfi)
        else:
            g = gravity[:-lag]
            s = scfi[lag:] - scfi[:-lag]

        n = min(len(g), len(s))
        g, s = g[:n], s[:n]

        valid = np.isfinite(g) & np.isfinite(s)
        g_v, s_v = g[valid], s[valid]

        if len(g_v) < 5 or s_v.std() == 0:
            rows.append({
                "lag_days": lag, "pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n_obs": int(valid.sum()),
            })
            continue

        pr, pp = stats.pearsonr(g_v, s_v)
        sr, sp = stats.spearmanr(g_v, s_v)
        rows.append({
            "lag_days": lag,
            "pearson_r":  float(pr), "pearson_p":  float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "n_obs": int(valid.sum()),
        })

    return pl.DataFrame(rows)


def run_granger_causality(
    gravity: np.ndarray,
    scfi: np.ndarray,
    max_lag_weeks: int = 8,
    weekly_dates: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Manual Granger causality test: does gravity_score Granger-cause ΔSCFI?

    Uses weekly observations only (every 7th point or provided weekly_dates mask)
    to avoid artificial autocorrelation from forward-filled SCFI.

    Restricted model   : ΔSCFI[t] ~ intercept + ΔSCFI[t-1..t-p]
    Unrestricted model : ΔSCFI[t] ~ intercept + ΔSCFI[t-1..t-p] + gravity[t-1..t-p]

    F = ((RSS_r - RSS_ur) / p) / (RSS_ur / (T - 2p - 1))
    """
    # Subsample to weekly to avoid forward-fill autocorrelation
    step = 7
    scfi_w    = scfi[::step]
    gravity_w = gravity[::step]

    delta_scfi = np.diff(scfi_w)           # first-difference: stationary
    gravity_dm = gravity_w[:-1] - gravity_w[:-1].mean()  # demean

    rows = []
    T = len(delta_scfi)

    for p in range(1, max_lag_weeks + 1):
        usable = T - p
        if usable <= 2 * p + 1:
            log.debug("Lag %d: not enough observations (%d) — skipping", p, usable)
            continue

        y = delta_scfi[p:]  # (usable,)

        # Build restricted design matrix: intercept + p lags of ΔSCFI
        X_r = np.column_stack([
            np.ones(usable),
            *[delta_scfi[p - k - 1: T - k - 1] for k in range(p)],
        ])

        # Build unrestricted: add p lags of gravity
        X_ur = np.column_stack([
            X_r,
            *[gravity_dm[p - k - 1: T - k - 1] for k in range(p)],
        ])

        try:
            beta_r,  rss_r_arr,  _, _ = np.linalg.lstsq(X_r,  y, rcond=None)
            beta_ur, rss_ur_arr, _, _ = np.linalg.lstsq(X_ur, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        rss_r  = float(np.sum((y - X_r  @ beta_r)  ** 2))
        rss_ur = float(np.sum((y - X_ur @ beta_ur) ** 2))

        df1 = p
        df2 = usable - 2 * p - 1

        if df2 <= 0 or rss_ur == 0:
            continue

        F     = ((rss_r - rss_ur) / df1) / (rss_ur / df2)
        p_val = float(1.0 - stats.f.cdf(F, df1, df2))

        rows.append({
            "lag_weeks":    p,
            "lag_days":     p * 7,
            "F_stat":       float(F),
            "p_value":      p_val,
            "significant_05": p_val < 0.05,
            "n_obs":        usable,
        })
        log.debug("Granger lag=%dw  F=%.3f  p=%.4f", p, F, p_val)

    return pl.DataFrame(rows)


def summarize_analysis(
    cross_corr_df: pl.DataFrame,
    granger_df: pl.DataFrame,
) -> dict:
    """Extract key findings for logging and figure annotations."""
    cc = cross_corr_df.drop_nulls(subset=["pearson_r"])
    if len(cc) == 0:
        return {"optimal_lag": 0, "max_pearson_r": 0.0, "min_granger_p": 1.0}

    abs_r = cc["pearson_r"].abs()
    best_idx   = int(abs_r.arg_max())
    optimal_lag = int(cc["lag_days"][best_idx])
    max_r       = float(cc["pearson_r"][best_idx])

    min_granger_p = float(granger_df["p_value"].min()) if len(granger_df) > 0 else 1.0
    best_granger_lag = (
        int(granger_df.filter(pl.col("p_value") == min_granger_p)["lag_days"][0])
        if len(granger_df) > 0 else 0
    )

    log.info("Optimal Pearson lag : %d days  (r=%.4f)", optimal_lag, max_r)
    log.info("Best Granger lag    : %d days  (p=%.4f)", best_granger_lag, min_granger_p)

    return {
        "optimal_lag":       optimal_lag,
        "max_pearson_r":     max_r,
        "min_granger_p":     min_granger_p,
        "best_granger_lag":  best_granger_lag,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    rng = np.random.default_rng(42)
    g = rng.uniform(0, 1, 364)
    s = np.cumsum(rng.normal(0, 5, 364)) + 800
    cc = compute_cross_correlation(g, s)
    gr = run_granger_causality(g, s)
    print(cc)
    print(gr)
