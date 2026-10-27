import numpy as np
import pandas as pd
import pytest

from src.pinns.lwr_pinn import LWRPINN
from src.pinns.pinn_pipeline import extract_physical_profiles, extract_pinn_summaries

GLOBAL_META = {"channel_len_km": 20.0, "global_rho_max": 10.0, "global_v_max": 12.0}


def _profiles(rho_of_t) -> pd.DataFrame:
    """Synthetic PINN output on a 5 × 11 grid where density only depends on time."""
    x, t = np.meshgrid(np.linspace(0, 10, 5), np.arange(11, dtype=float))
    rho = rho_of_t(t)
    v = np.full_like(rho, 2.0)
    return pd.DataFrame({
        "x_km": x.ravel(), "t_days": t.ravel(),
        "rho_hat": rho.ravel(), "v_hat": v.ravel(), "flux_hat": (rho * v).ravel(),
    })


def test_summaries_capture_peak_and_decay():
    # Triangle: density rises to 10 at day 4, then clears at 1 vessel/km/day
    summaries = extract_pinn_summaries(_profiles(lambda t: np.where(t <= 4, 2.5 * t, 10 - (t - 4))))
    assert summaries["peak_rho"] == pytest.approx(10.0)
    assert summaries["t_to_peak_days"] == pytest.approx(4.0)
    assert summaries["decay_rate"] == pytest.approx(-1.0)
    assert summaries["v_mean_global"] == pytest.approx(2.0)
    assert summaries["area_under_rho"] > 0


def test_physical_profiles_are_denormalised_on_a_dense_grid():
    episode = pd.DataFrame({"x_km": [1.0, 5.0, 12.0], "t_days": [0.0, 3.0, 9.0]})
    profiles = extract_physical_profiles(
        model=LWRPINN(), episode_data=episode, global_meta=GLOBAL_META,
        x_grid_points=12, t_grid_points=8, t_max_days=30,
    )
    assert len(profiles) == 12 * 8
    assert {"x_km", "t_days", "rho_hat", "v_hat", "flux_hat"} <= set(profiles.columns)
    assert profiles["x_km"].max() == pytest.approx(GLOBAL_META["channel_len_km"])
    assert profiles["rho_hat"].between(0, GLOBAL_META["global_rho_max"]).all()
    assert np.allclose(profiles["flux_hat"], profiles["rho_hat"] * profiles["v_hat"], rtol=1e-4)
