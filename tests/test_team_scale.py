"""Team/conf encoding must be PPP-scale and match unit forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statpad_cbb.model.design import build_design_matrix, build_player_maps
from statpad_cbb.model.rollup import team_rollup


def test_team_indicator_is_once_per_possession():
    poss = pd.DataFrame(
        [
            {
                "game_id": 1,
                "points": 2,
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
                "def_player_5": 10,
            }
        ]
    )
    roster = pd.DataFrame(
        [{"player_id": i, "team_id": 100 if i <= 5 else 200} for i in range(1, 11)]
    )
    teams = pd.DataFrame(
        [
            {"team_id": 100, "conference": "A"},
            {"team_id": 200, "conference": "B"},
        ]
    )
    conf_map, team_map = build_player_maps(list(range(1, 11)), roster=roster, teams=teams)
    design = build_design_matrix(
        poss,
        player_conference=conf_map,
        player_team=team_map,
        include_conference=True,
        include_team=True,
    )
    # Team off column for team 100 should be 1.0 once, not 5.0.
    assert design.team_off0 is not None
    team100_col = design.team_off0 + design.team_ids.index(100)
    assert float(design.x[0, team100_col]) == 1.0


def test_centering_absorbs_five_times_player_mean_into_team():
    from statpad_cbb.model.rapm import center_player_effects_within_team
    from statpad_cbb.model.rollup import team_rollup

    player_ids = [1, 2]
    player_team = {1: 10, 2: 10}
    player_off = np.array([3.0, 1.0])
    player_def = np.array([0.0, 0.0])
    team_off = np.array([2.0])
    team_def = np.array([0.0])
    off, deff, t_off, t_def = center_player_effects_within_team(
        player_ids=player_ids,
        player_team=player_team,
        player_off=player_off,
        player_def=player_def,
        team_ids=[10],
        team_off=team_off,
        team_def=team_def,
        minutes_by_player={1: 20.0, 2: 20.0},
    )
    # mean β_off = 2; absorb 5*2 into τ
    assert abs(float(np.mean(off)) ) < 1e-9
    assert abs(float(t_off[0]) - (2.0 + 5.0 * 2.0)) < 1e-9

    # Unit rating invariant to the reparameterization.
    before = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "off_player_effect": 300.0,
                "def_player_effect": 0.0,
                "off_team_effect": 200.0,
                "def_team_effect": 0.0,
                "off_conference_effect": 0.0,
                "def_conference_effect": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "minutes": 20.0,
            },
            {
                "player_id": 2,
                "team_id": 10,
                "off_player_effect": 100.0,
                "def_player_effect": 0.0,
                "off_team_effect": 200.0,
                "def_team_effect": 0.0,
                "off_conference_effect": 0.0,
                "def_conference_effect": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "minutes": 20.0,
            },
        ]
    )
    after = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "off_player_effect": float(off[0] * 100.0),
                "def_player_effect": 0.0,
                "off_team_effect": float(t_off[0] * 100.0),
                "def_team_effect": 0.0,
                "off_conference_effect": 0.0,
                "def_conference_effect": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "minutes": 20.0,
            },
            {
                "player_id": 2,
                "team_id": 10,
                "off_player_effect": float(off[1] * 100.0),
                "def_player_effect": 0.0,
                "off_team_effect": float(t_off[0] * 100.0),
                "def_team_effect": 0.0,
                "off_conference_effect": 0.0,
                "def_conference_effect": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "minutes": 20.0,
            },
        ]
    )
    poss = pd.DataFrame([{"off_team_id": 10, "def_team_id": 99, "points": 1}])
    lookup = {10: {"school": "A", "conference": "X"}}
    u0 = float(team_rollup(before, poss, pd.DataFrame(), team_lookup=lookup).iloc[0].off_rating)
    u1 = float(team_rollup(after, poss, pd.DataFrame(), team_lookup=lookup).iloc[0].off_rating)
    assert abs(u0 - u1) < 1e-6


def test_unit_rating_matches_five_man_plus_group_effects():
    ratings = pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "off_player_effect": 1.0,
                "def_player_effect": 0.0,
                "off_team_effect": 2.0,
                "def_team_effect": 0.0,
                "off_conference_effect": 0.5,
                "def_conference_effect": 0.0,
                "minutes": 20.0,
            },
            {
                "player_id": 2,
                "team_id": 10,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "off_player_effect": 1.0,
                "def_player_effect": 0.0,
                "off_team_effect": 2.0,
                "def_team_effect": 0.0,
                "off_conference_effect": 0.5,
                "def_conference_effect": 0.0,
                "minutes": 20.0,
            },
        ]
    )
    possessions = pd.DataFrame(
        [{"off_team_id": 10, "def_team_id": 99, "points": 1}]
    )
    teams = team_rollup(
        ratings,
        possessions,
        pd.DataFrame(),
        team_lookup={10: {"school": "A", "conference": "X"}},
    )
    # 5 * 1.0 + 2.0 + 0.5 = 7.5
    assert abs(float(teams.loc[teams.team_id == 10, "off_rating"].iloc[0]) - 7.5) < 1e-9
