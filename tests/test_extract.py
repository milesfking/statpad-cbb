from statpad_cbb.ingest.extract import (
    canonicalize_game,
    directory_team_rows,
    games_for_di_teams,
    is_division_i,
    iter_unique_plays,
    unique_players,
    unique_teams,
)


def test_canonicalize_game_from_flat_and_nested_shapes():
    nested = {
        "id": "401234",
        "season": 2026,
        "startDate": "2025-11-04T00:00:00.000Z",
        "seasonType": "regular",
        "neutralSite": False,
        "venue": {"name": "Cameron Indoor Stadium", "city": "Durham", "state": "NC"},
        "homeTeam": {"id": 150, "points": 80, "winner": True},
        "awayTeam": {"id": 52, "points": 72, "winner": False},
        "status": "final",
    }
    row = canonicalize_game(nested, 2026)
    assert row is not None
    assert row["game_id"] == 401234
    assert row["home_team_id"] == 150
    assert row["away_team_id"] == 52
    assert row["start_date_utc"] == "2025-11-04"
    assert row["venue"] == "Cameron Indoor Stadium"

    flat = {
        "id": 209786,
        "startDate": "2025-11-04T00:00:00.000Z",
        "status": "final",
        "homeTeamId": 44,
        "homeTeam": "Central Connecticut",
        "homePoints": 117,
        "homeWinner": True,
        "awayTeamId": 634,
        "awayTeam": "Vermont State - Johnson",
        "awayPoints": 55,
        "awayWinner": False,
        "venue": "Detrick Gymnasium",
        "city": "New Britain",
        "state": "CT",
    }
    flat_row = canonicalize_game(flat, 2026)
    assert flat_row is not None
    assert flat_row["home_team_id"] == 44
    assert flat_row["away_points"] == 55
    assert flat_row["venue"] == "Detrick Gymnasium"


def test_play_dedup_by_play_and_game_id():
    plays = [
        {"id": "p1", "gameId": 1, "playText": "Make"},
        {"id": "p1", "gameId": 1, "playText": "duplicate"},
        {"id": "p1", "gameId": 2, "playText": "same play id, other game"},
        {"playId": "p2", "game": {"id": 1}},
    ]
    unique = list(iter_unique_plays(plays))
    assert [(row["play_id"], row["game_id"]) for row in unique] == [
        ("p1", 1),
        ("p1", 2),
        ("p2", 1),
    ]


def test_division_i_filter_and_roster_ids():
    teams = [
        {"id": 1, "school": "Duke", "division": "I", "conference": "ACC"},
        {"id": 2, "school": "Other", "division": "II"},
    ]
    canonical = unique_teams(teams, 2026)
    assert [row["team_id"] for row in canonical] == [1, 2]
    assert is_division_i(teams[0])
    assert not is_division_i(teams[1])

    games = [
        {"game_id": 10, "home_team_id": 1, "away_team_id": 99},
        {"game_id": 11, "home_team_id": 2, "away_team_id": 3},
    ]
    assert [g["game_id"] for g in games_for_di_teams(games, {1})] == [10]

    players = unique_players(
        [
            {
                "teamId": 1,
                "players": [
                    {"id": 7, "name": "A"},
                    {"id": 7, "name": "A dup"},
                ],
            },
            {"teamId": 2, "players": [{"id": 8, "name": "B"}]},
        ],
        2026,
    )
    assert [(p["player_id"], p["team_id"]) for p in players] == [(7, 1), (8, 2)]


def test_directory_envelope_resolves_conference_names():
    teams, conferences = directory_team_rows(
        {
            "teams": [{"id": 1, "school": "Duke", "conferenceId": 10}],
            "conferences": [{"id": 10, "name": "ACC"}],
        }
    )
    canonical = unique_teams(teams, 2026, conferences=conferences)
    assert canonical[0]["conference"] == "ACC"
    assert canonical[0]["conference_id"] == 10
