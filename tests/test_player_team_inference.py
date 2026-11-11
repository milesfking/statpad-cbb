from __future__ import annotations

import pandas as pd

from statpad_cbb.model.design import (
    UNKNOWN_CONFERENCE,
    build_design_matrix,
    build_player_maps,
    build_team_conference_lookup,
    infer_player_teams_from_possessions,
    is_ranked_conference,
)

OFF = [f"off_player_{i}" for i in range(1, 6)]
DEF = [f"def_player_{i}" for i in range(1, 6)]


def test_infer_player_team_from_possessions_when_missing_from_roster():
    poss = pd.DataFrame(
        [
            {
                "game_id": 1,
                "points": 1,
                "home_offense": 1,
                "off_team_id": 100,
                "def_team_id": 200,
                **{c: 0 for c in OFF + DEF},
                "off_player_1": 999,
            }
        ]
    )
    poss.loc[0, "def_player_1"] = 6
    inferred = infer_player_teams_from_possessions(poss, roster_team={6: 200})
    assert inferred[999] == 100
    assert inferred[6] == 200


def test_build_player_maps_uses_possession_team_for_unlisted_player():
    poss = pd.DataFrame(
        [
            {
                "game_id": 1,
                "points": 1,
                "home_offense": 1,
                "off_team_id": 100,
                "def_team_id": 200,
                **{c: 0 for c in OFF + DEF},
                "off_player_1": 999,
                "def_player_1": 6,
            }
        ]
    )
    roster = pd.DataFrame([{"player_id": 6, "team_id": 200}])
    teams = pd.DataFrame(
        [
            {"team_id": 100, "conference": "ACC"},
            {"team_id": 200, "conference": "SEC"},
        ]
    )
    conf_map, team_map = build_player_maps(
        [6, 999],
        roster=roster,
        teams=teams,
        possessions=poss,
    )
    assert team_map[999] == 100
    assert conf_map[999] == "ACC"
    assert conf_map[6] == "SEC"
    assert conf_map[999] != UNKNOWN_CONFERENCE


def test_design_matrix_omits_unknown_conference_level():
    poss = pd.DataFrame(
        [
            {
                "game_id": i,
                "points": 1,
                "home_offense": 1,
                "off_team_id": 100,
                "def_team_id": 200,
                **{f"off_player_{j}": j for j in range(1, 6)},
                **{f"def_player_{j}": j + 5 for j in range(1, 6)},
            }
            for i in range(5)
        ]
    )
    roster = pd.DataFrame([{"player_id": i, "team_id": 100 if i <= 5 else 200} for i in range(1, 11)])
    teams = pd.DataFrame(
        [
            {"team_id": 100, "conference": "ACC"},
            {"team_id": 200, "conference": "SEC"},
        ]
    )
    team_conf = build_team_conference_lookup(teams)
    conf_map, team_map = build_player_maps(
        list(range(1, 11)),
        roster=roster,
        teams=teams,
        possessions=poss,
    )
    design = build_design_matrix(
        poss,
        player_conference=conf_map,
        player_team=team_map,
        team_conference=team_conf,
        include_conference=True,
        include_team=True,
    )
    assert "Unknown" not in design.conference_names
    assert is_ranked_conference("ACC")
