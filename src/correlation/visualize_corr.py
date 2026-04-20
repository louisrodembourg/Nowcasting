"""
Phase 4 — Visualisation des résultats de corrélation.

Génère 3 figures :
  1. la_scfi_timeseries.png    — dual-axis gravity_score + SCFI
  2. la_scfi_cross_correlation.png — corrélation par lag + Granger
  3. la_scfi_scatter.png       — scatter au lag optimal avec OLS
"""
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Trade war event dates for annotations
_TARIFF_EVENTS = [
    (date(2019, 5, 10),  "US tariffs\n10%→25%"),
    (date(2019, 8,  1),  "New\n$300B tariff"),
    (date(2019, 8, 23),  "China\nretaliates"),
    (date(2019, 12, 13), "Phase 1\ndeal"),
]


def plot_timeseries(df: pl.DataFrame, output_path: Path) -> Path:
    """Dual-axis time series: gravity_score (left) + SCFI (right)."""
    dates    = df["date"].to_list()
    gravity  = df["gravity_score"].to_numpy()
    scfi     = df["scfi"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    # Congestion window shading (June–September)
    ax1.axvspan(date(2019, 6, 1), date(2019, 9, 30), color="#FF6F00", alpha=0.06, label="Congestion window")

    # Tariff event lines
    for ev_date, label in _TARIFF_EVENTS:
        ax1.axvline(ev_date, color="#B71C1C", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.text(ev_date, gravity.max() * 0.92, label, fontsize=6.5,
                 color="#B71C1C", ha="center", va="top", rotation=0,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    # Gravity score — filled area
    ax1.fill_between(dates, gravity, alpha=0.35, color="#1565C0", label="Gravity score")
    ax1.plot(dates, gravity, color="#1565C0", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Gravity score (LA congestion)", color="#1565C0", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_ylim(bottom=0)

    # SCFI line
    ax2.plot(dates, scfi, color="#E53935", linewidth=1.8, label="SCFI composite")
    ax2.set_ylabel("SCFI (USD/TEU)", color="#E53935", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#E53935")

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(
        "LA/Long Beach Port — Gravity Score vs SCFI 2019\nUS-China Trade War Context",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_path)
    return output_path


def plot_cross_correlation(
    cross_corr_df: pl.DataFrame,
    granger_df: pl.DataFrame,
    output_path: Path,
) -> Path:
    """Two-panel: Pearson r by lag (top) + Granger F-stat (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # ── Top: Pearson correlation ──
    cc = cross_corr_df.drop_nulls(subset=["pearson_r"])
    lags = cc["lag_days"].to_numpy()
    r    = cc["pearson_r"].to_numpy()
    n    = cc["n_obs"].to_numpy()

    colors = ["#2E7D32" if v >= 0 else "#B71C1C" for v in r]
    ax1.bar(lags, r, color=colors, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)

    # 95% CI under H0
    ci = 1.96 / np.sqrt(n)
    ax1.fill_between(lags, -ci, ci, alpha=0.12, color="gray", label="95% CI (H₀)")
    ax1.axhline(0.2,  color="#1B5E20", linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.axhline(-0.2, color="#1B5E20", linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.axhline(0.4,  color="#1B5E20", linewidth=0.9, linestyle=":", alpha=0.7)
    ax1.axhline(-0.4, color="#1B5E20", linewidth=0.9, linestyle=":", alpha=0.7)

    # Annotate optimal lag
    best_idx = int(np.argmax(np.abs(r)))
    ax1.annotate(
        f"lag={lags[best_idx]}d\nr={r[best_idx]:.3f}",
        xy=(lags[best_idx], r[best_idx]),
        xytext=(lags[best_idx] + 1.5, r[best_idx] + 0.03),
        fontsize=8, color="#1A237E",
        arrowprops=dict(arrowstyle="->", color="#1A237E", lw=0.8),
    )

    ax1.set_ylabel("Pearson r", fontsize=10)
    ax1.set_title("Cross-Correlation: LA Gravity Score leads SCFI", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-0.5, lags.max() + 0.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Bottom: Granger F-statistic ──
    if len(granger_df) > 0:
        gl = granger_df["lag_days"].to_numpy()
        gf = granger_df["F_stat"].to_numpy()
        gp = granger_df["p_value"].to_numpy()

        g_colors = []
        for p_val in gp:
            if p_val < 0.01:
                g_colors.append("#B71C1C")
            elif p_val < 0.05:
                g_colors.append("#E53935")
            elif p_val < 0.10:
                g_colors.append("#FF8F00")
            else:
                g_colors.append("#9E9E9E")

        bars = ax2.bar(gl, gf, color=g_colors, alpha=0.8, edgecolor="white", linewidth=0.5, width=5)

        # p-value on secondary y-axis (log scale)
        ax2b = ax2.twinx()
        ax2b.plot(gl, gp, "o--", color="#37474F", markersize=4, linewidth=0.8, alpha=0.7, label="p-value")
        ax2b.axhline(0.05, color="#37474F", linewidth=0.8, linestyle=":", alpha=0.6)
        ax2b.set_yscale("log")
        ax2b.set_ylabel("p-value (log)", fontsize=9, color="#37474F")
        ax2b.tick_params(axis="y", labelcolor="#37474F")
        ax2b.legend(fontsize=8, loc="upper right")

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#9E9E9E", label="p > 0.10"),
            Patch(facecolor="#FF8F00", label="p < 0.10"),
            Patch(facecolor="#E53935", label="p < 0.05"),
            Patch(facecolor="#B71C1C", label="p < 0.01"),
        ]
        ax2.legend(handles=legend_els, fontsize=8, loc="upper left")
        ax2.set_xlabel("Lag (days — gravity leads SCFI)", fontsize=10)
        ax2.set_ylabel("Granger F-statistic", fontsize=10)
        ax2.set_title("Granger Causality: gravity_score → ΔSCFI (weekly subsampled)", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for Granger test",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)

    fig.tight_layout(h_pad=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_path)
    return output_path


def plot_scatter(
    df: pl.DataFrame,
    optimal_lag: int,
    output_path: Path,
) -> Path:
    """Scatter: gravity_score[t] vs ΔSCFI[t → t+lag], colored by month + OLS."""
    from scipy import stats as scipy_stats

    gravity = df["gravity_score"].to_numpy().astype(float)
    scfi    = df["scfi"].to_numpy().astype(float)
    dates   = df["date"].to_list()

    lag = max(optimal_lag, 1)
    g   = gravity[:-lag]
    s   = scfi[lag:] - scfi[:-lag]
    d   = dates[:-lag]

    valid = np.isfinite(g) & np.isfinite(s)
    g, s, d = g[valid], s[valid], [dates[i] for i, v in enumerate(valid) if v]

    months  = np.array([dt.month for dt in d])
    cmap    = plt.get_cmap("tab20", 12)
    m_colors = cmap(months - 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(g, s, c=months, cmap="tab20", vmin=1, vmax=12,
                    alpha=0.55, s=22, edgecolors="none")

    # OLS line + 95% CI
    slope, intercept, r_val, p_val, _ = scipy_stats.linregress(g, s)
    x_line = np.linspace(g.min(), g.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#B71C1C", linewidth=1.8, label=f"OLS (slope={slope:.1f})")

    # CI band
    n   = len(g)
    se  = np.sqrt(np.sum((s - (slope * g + intercept)) ** 2) / (n - 2))
    x_m = g.mean()
    ci  = 1.96 * se * np.sqrt(1/n + (x_line - x_m)**2 / np.sum((g - x_m)**2))
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color="#B71C1C")

    # Highlight top-3 gravity days with labels
    top3 = np.argsort(g)[-3:][::-1]
    for idx in top3:
        ax.annotate(
            str(d[idx]),
            xy=(g[idx], s[idx]),
            xytext=(g[idx] + 0.01, s[idx] + 1),
            fontsize=7, color="#1A237E",
            arrowprops=dict(arrowstyle="->", color="#1A237E", lw=0.6),
        )

    # Month colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="tab20", norm=Normalize(vmin=1, vmax=12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(1, 13), pad=0.01)
    cbar.set_ticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    cbar.set_label("Month", fontsize=9)

    pr, pp = scipy_stats.pearsonr(g, s)
    sr, sp = scipy_stats.spearmanr(g, s)
    ax.set_xlabel("Gravity score (LA congestion at day t)", fontsize=11)
    ax.set_ylabel(f"ΔSCFI (t → t+{lag}d, USD/TEU)", fontsize=11)
    ax.set_title(
        f"LA gravity_score vs ΔSCFI (+{lag}d) — 2019\n"
        f"Pearson r={pr:.3f} (p={pp:.3f}) | Spearman ρ={sr:.3f} (p={sp:.3f}) | n={n}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_path)
    return output_path
