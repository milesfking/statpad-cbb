from statpad_cbb.possessions.builder import GameContext, build_game_possessions
from statpad_cbb.possessions.lineups import parse_on_floor


def _ctx() -> GameContext:
    return GameContext(
        game_id=1,
        season=2026,
        home_team_id=10,
        away_team_id=20,
        home_school="Home U",
        away_school="Away U",
        neutral_site=False,
    )


def _floor(home_ids, away_ids):
    return [{"id": i, "name": str(i), "team": "Home U"} for i in home_ids] + [
        {"id": i, "name": str(i), "team": "Away U"} for i in away_ids
    ]


def test_parse_on_floor_rejects_invalid_lineups():
    kwargs = dict(
        home_team_id=10,
        away_team_id=20,
        home_school="Home U",
        away_school="Away U",
    )
    assert parse_on_floor({"onFloor": _floor([1, 2, 3, 4], [6, 7, 8, 9, 10])}, **kwargs) is None
    assert parse_on_floor({"onFloor": _floor([1, 1, 3, 4, 5], [6, 7, 8, 9, 10])}, **kwargs) is None
    assert parse_on_floor({"onFloor": []}, **kwargs) is None
    valid = parse_on_floor({"onFloor": _floor([5, 4, 3, 2, 1], [10, 9, 8, 7, 6])}, **kwargs)
    assert valid == ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])


def test_made_fg_creates_possession_with_points():
    home = [1, 2, 3, 4, 5]
    away = [6, 7, 8, 9, 10]
    plays = [
        {
            "id": 1,
            "playType": "JumpShot",
            "teamId": 10,
            "period": 1,
            "secondsRemaining": 1100,
            "homeScore": 3,
            "awayScore": 0,
            "scoringPlay": True,
            "shotInfo": {"made": True, "range": "three_pointer"},
            "playText": "Home player made Three Point Jumper.",
            "onFloor": _floor(home, away),
        },
        {
            "id": 2,
            "playType": "End Period",
            "period": 1,
            "secondsRemaining": 0,
            "homeScore": 3,
            "awayScore": 0,
            "onFloor": _floor(home, away),
        },
    ]
    rows, stats = build_game_possessions(_ctx(), plays)
    assert stats["possessions_valid"] >= 1
    assert rows[0]["points"] == 3
    assert rows[0]["off_team_id"] == 10
    assert rows[0]["home_offense"] == 1


def test_turnover_zero_point_possession():
    home = [1, 2, 3, 4, 5]
    away = [6, 7, 8, 9, 10]
    plays = [
        {
            "id": 1,
            "playType": "Lost Ball Turnover",
            "teamId": 20,
            "period": 1,
            "secondsRemaining": 1000,
            "homeScore": 0,
            "awayScore": 0,
            "playText": "Away Turnover.",
            "onFloor": _floor(home, away),
        },
        {
            "id": 2,
            "playType": "End Period",
            "period": 1,
            "secondsRemaining": 0,
            "homeScore": 0,
            "awayScore": 0,
            "onFloor": _floor(home, away),
        },
    ]
    rows, _ = build_game_possessions(_ctx(), plays)
    assert any(row["points"] == 0 and row["off_team_id"] == 20 for row in rows)
