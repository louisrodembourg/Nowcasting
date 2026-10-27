from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.ingestion.kinematic_filter import _haversine_nm, prepare_kinematics


def test_haversine_one_degree_of_latitude_is_sixty_nautical_miles():
    distance = _haversine_nm(np.array([29.0]), np.array([-95.0]), np.array([30.0]), np.array([-95.0]))
    assert distance[0] == pytest.approx(60.0, rel=1e-3)


def test_trajectories_split_on_gaps_and_aberrant_sog_is_corrected():
    t0 = datetime(2017, 8, 25, 12, 0)
    df = pl.DataFrame({
        "MMSI": [366_000_001] * 4 + [366_000_002],
        "BaseDateTime": [
            t0,
            t0 + timedelta(minutes=10),
            t0 + timedelta(minutes=20),
            t0 + timedelta(minutes=80),   # 60 min gap → new trajectory
            t0,
        ],
        "LAT": [29.70, 29.71, 29.72, 29.73, 29.60],
        "LON": [-95.00, -95.00, -95.00, -95.00, -95.10],
        "SOG": [5.0, 99.0, 5.0, 5.0, 0.0],   # 99 kt is a sensor error
    })

    out = prepare_kinematics(df).sort(["MMSI", "BaseDateTime"])

    assert out["traj_id"].to_list() == [1, 1, 1, 2, 3]
    # 0.01° of latitude in 10 min ≈ 3.6 kt: the aberrant 99 kt is replaced
    assert out["SOG_corr"][1] == pytest.approx(3.6, abs=0.1)
    assert out["SOG_corr"][0] == pytest.approx(5.0)
