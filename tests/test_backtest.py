import pandas as pd

from statpad_cbb.model.backtest import default_cut_dates, predict_possession_points


def test_predict_possession_points_uses_zero_for_unseen_players():
    frame = pd.DataFrame(
        [
            {
                "home_offense": 1,
                "off_team_id": 100,
                "def_team_id": 200,
                "off_player_1": 1,
                "off_player_2": 2,
                "off_player_3": 3,
                "off_player_4": 4,
                "off_player_5": 5,
                "def_player_1": 6,
                "def_player_2": 7,
                "def_player_3": 8,
                "def_player_4": 9,
                "def_player_5": 99,  # unseen
            }
        ]
    )
    pred = predict_possession_points(
        frame,
        intercept=1.0,
        home=0.1,
        player_off_coef={1: 0.05, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        player_def_coef={6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0},
        team_off_coef={100: 0.02},
        team_def_coef={200: 0.01},
    )
    # Once-per-possession: 1.0 + 0.1 + 0.05 + 0.02 - 0.01 = 1.16; unseen def → 0
    assert abs(float(pred[0]) - 1.16) < 1e-9


def test_default_cut_dates_are_season_aware():
    assert default_cut_dates(2027) == (
        "2026-12-01",
        "2027-01-01",
        "2027-02-01",
        "2027-03-01",
    )
