from __future__ import annotations

import math

import pandas as pd

from statpad_cbb.features.box import finalize_player_features
from statpad_cbb.model.design import standardize_box_features


def test_finalize_nan_rates_below_min_possessions() -> None:
    box = {1: {"ftm": 1.0, "fta": 1.0}}
    usage = {
        1: {
            "off_possessions": 0.0,
            "def_possessions": 1.0,
            "possessions": 1.0,
            "minutes": 0.25,
            "team_id": 10.0,
            "team_off_possessions": 80.0,
        }
    }
    rows = finalize_player_features(box, usage, season=2027, roster={1: {"name": "Tiny"}})
    assert len(rows) == 1
    assert math.isnan(rows[0]["usage"])
    assert math.isnan(rows[0]["ftm_100"])


def test_finalize_caps_extreme_rates() -> None:
    box = {1: {"fg3m": 40.0, "fg3a": 40.0, "tov": 0.0}}
    usage = {
        1: {
            "off_possessions": 25.0,
            "def_possessions": 25.0,
            "possessions": 50.0,
            "minutes": 12.0,
            "team_id": 10.0,
            "team_off_possessions": 80.0,
        }
    }
    rows = finalize_player_features(box, usage, season=2027, roster={1: {"name": "Hot"}})
    assert rows[0]["fg3m_100"] == 100.0
    assert rows[0]["usage"] <= 1.0


def test_standardize_clips_and_rejects_pathological_prior_std() -> None:
    features = pd.DataFrame(
        [
            {"player_id": 1, "usage": 0.2, "fg2m_100": 5.0, "fg3m_100": 2.0, "ftm_100": 3.0,
             "ast_100": 4.0, "tov_100": 2.0, "orb_100": 1.0, "drb_100": 5.0, "stl_100": 1.0, "blk_100": 1.0},
            {"player_id": 2, "usage": 0.3, "fg2m_100": 6.0, "fg3m_100": 1.0, "ftm_100": 4.0,
             "ast_100": 3.0, "tov_100": 3.0, "orb_100": 2.0, "drb_100": 4.0, "stl_100": 2.0, "blk_100": 0.0},
        ]
    )
    box_by, names, means, stds = standardize_box_features(
        features,
        mean={"usage": 198.0},
        std={"usage": 14073.0},
    )
    assert "usage" in names
    assert stds["usage"] < 10.0  # fell back to sample std
    assert abs(box_by[1][names.index("usage")]) <= 5.0
