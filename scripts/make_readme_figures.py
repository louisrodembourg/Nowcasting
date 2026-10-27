"""
Regenerate the figures shown in the README (docs/img/) from pipeline outputs.

Inputs (produced by the pipeline, not versioned):
    data/features/la_gravity_all_years.parquet        Phase 2 — daily gravity score
    data/features/pinn_la_episodes_profiles.parquet   Phase 3 — PINN fields ρ, v, q
    outputs/financial/two_regime_results.csv          Phase 4 — global OOS ICs
    outputs/financial/two_regime_by_regime.csv        Phase 4 — ICs per regime

Usage (from the repository root):
    python scripts/make_readme_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("docs/img")

# Validated categorical pair (blue / orange) + recessive ink for text and grid
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def congestion_overview() -> Path:
    """Waiting vessels and gravity score, LA/Long Beach 2017–2024 (two panels, shared x)."""
    df = pd.read_parquet("data/features/la_gravity_all_years.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    waiting_ma = df["waiting_vessels"].rolling(30, min_periods=10).mean()
    gravity_ma = df["gravity_score"].rolling(30, min_periods=10).mean()

    fig, (ax_wait, ax_grav) = plt.subplots(
        2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    for ax in (ax_wait, ax_grav):
        ax.axvspan(pd.Timestamp("2021-06-01"), pd.Timestamp("2022-07-31"), color=ORANGE, alpha=0.08, lw=0)

    ax_wait.plot(df.index, df["waiting_vessels"], color=BLUE, lw=0.6, alpha=0.25)
    ax_wait.plot(waiting_ma.index, waiting_ma, color=BLUE, lw=2)
    ax_wait.set_title("Vessels waiting at anchor (daily, 30-day mean)", loc="left")
    ax_wait.text(pd.Timestamp("2021-06-15"), ax_wait.get_ylim()[1] * 0.92,
                 "2021–22 West Coast backlog", color=INK_MUTED, fontsize=9, va="top")

    ax_grav.plot(gravity_ma.index, gravity_ma, color=ORANGE, lw=2)
    ax_grav.set_title("Manifold gravity score (30-day mean, unsupervised)", loc="left")
    ax_grav.xaxis.set_major_locator(mdates.YearLocator())
    ax_grav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_grav.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))

    fig.suptitle("Port of Los Angeles / Long Beach — congestion seen from AIS, 2017–2024",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    path = OUT_DIR / "la_congestion_overview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def pinn_fields() -> Path:
    """PINN-reconstructed density, speed and flow fields for one LA congestion episode."""
    df = pd.read_parquet("data/features/pinn_la_episodes_profiles.parquet")
    panels = [
        ("rho_hat", "Density ρ(x, t)  [vessels/km]", "Blues"),
        ("v_hat", "Speed v(x, t)  [knots]", "Greens"),
        ("flux_hat", "Flow q = ρ·v  [vessels·kn/km]", "Oranges"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (col, title, cmap) in zip(axes, panels):
        grid = df.pivot_table(index="x_km", columns="t_days", values=col)
        mesh = ax.pcolormesh(grid.columns, grid.index, grid.values, cmap=cmap, shading="auto")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Days since episode onset")
        ax.grid(False)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    axes[0].set_ylabel("Distance along the channel [km]")
    fig.suptitle("Physics-informed neural network (LWR) — reconstructed traffic fields",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    path = OUT_DIR / "la_pinn_fields.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def results_ic() -> Path:
    """Out-of-sample Spearman IC: AIS baseline vs two-regime (PINN) model, plus episode days."""
    res = pd.read_csv("outputs/financial/two_regime_results.csv")
    reg = pd.read_csv("outputs/financial/two_regime_by_regime.csv")
    cases = [("MATX", 10), ("MATX", 21), ("BDRY", 21), ("SBLK", 10)]

    def ic(frame, target, horizon, **filters):
        sub = frame[(frame["target"] == target) & (frame["horizon"] == horizon)]
        for key, value in filters.items():
            sub = sub[sub[key] == value]
        return float(sub["IC_spearman"].iloc[0])

    labels = [f"{t}\n{h}-day" for t, h in cases]
    baseline = [ic(res, t, h, model="Baseline") for t, h in cases]
    two_regime = [ic(res, t, h, model="2Regime") for t, h in cases]
    episode = [ic(reg, t, h, regime="episode") for t, h in cases]

    x = np.arange(len(cases))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    series = [
        (baseline, "AIS baseline (all days)", BLUE, 1.0, None),
        (two_regime, "Two-regime + PINN (all days)", ORANGE, 1.0, None),
        (episode, "Two-regime + PINN (congestion episodes only)", ORANGE, 0.45, "////"),
    ]
    for i, (values, name, color, alpha, hatch) in enumerate(series):
        bars = ax.bar(x + (i - 1) * width, values, width * 0.92, color=color, alpha=alpha,
                      hatch=hatch, edgecolor=SURFACE if hatch is None else color, linewidth=0, label=name)
        for bar, value in zip(bars, values):
            y = value + (0.006 if value >= 0 else -0.006)
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value + 0.0:.2f}".replace("-0.00", "0.00"), ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8, color=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman IC (out-of-sample)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Predicting forward returns of shipping equities from LA port congestion",
                 loc="left")
    fig.tight_layout()
    path = OUT_DIR / "results_ic.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for make in (congestion_overview, pinn_fields, results_ic):
        print("saved", make())
