from datetime import date
from pathlib import Path

from statpad_cbb.config import GAMES_RESPONSE_LIMIT, Settings
from statpad_cbb.ingest.coverage import build_coverage_report
from statpad_cbb.ingest.pipeline import _ingest_games_range, ingest_season
from statpad_cbb.ingest.storage import SeasonPaths, read_json, read_jsonl


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def teams_directory(self, season: int):
        self.calls.append(("directory", (season,)))
        return {
            "season": season,
            "seasonLabel": "2025–26",
            "teams": [
                {"id": 1, "school": "Duke", "division": "I", "conferenceId": 10},
                {"id": 2, "school": "D2 School", "division": "II", "conferenceId": 99},
            ],
            "conferences": [
                {"id": 10, "name": "ACC", "abbreviation": "ACC"},
                {"id": 99, "name": "Other", "abbreviation": "OTH"},
            ],
        }

    def teams_roster(self, season: int):
        self.calls.append(("roster", (season,)))
        return [
            {
                "teamId": 1,
                "team": "Duke",
                "conference": "ACC",
                "season": season,
                "players": [{"id": 10, "name": "Player", "position": "G"}],
            }
        ]

    def games(self, season: int, start: date, end: date):
        self.calls.append(("games", (season, start, end)))
        rows = []
        if start <= date(2025, 11, 4) <= end:
            rows.append(
                {
                    "id": 100,
                    "season": season,
                    "startDate": "2025-11-04T18:00:00.000Z",
                    "status": "final",
                    "homeTeamId": 1,
                    "homeTeam": "Duke",
                    "homePoints": 80,
                    "homeWinner": True,
                    "awayTeamId": 3,
                    "awayTeam": "Opponent",
                    "awayPoints": 70,
                    "awayWinner": False,
                    "venue": "Cameron Indoor Stadium",
                    "city": "Durham",
                    "state": "NC",
                }
            )
        if start <= date(2025, 11, 5) <= end:
            rows.append(
                {
                    "id": 101,
                    "season": season,
                    "startDate": "2025-11-05T00:00:00.000Z",
                    "status": "scheduled",
                    "homeTeamId": 1,
                    "homeTeam": "Duke",
                    "homePoints": None,
                    "awayTeamId": 4,
                    "awayTeam": "Next",
                    "awayPoints": None,
                }
            )
        return rows

    def plays_date(self, game_date: date):
        self.calls.append(("plays", (game_date,)))
        if game_date == date(2025, 11, 4):
            return [
                {"id": "p1", "gameId": 100, "playText": "Jump ball"},
                {"id": "p1", "gameId": 100, "playText": "duplicate"},
            ]
        return []


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://api.collegebasketballdata.com",
        data_dir=tmp_path,
        min_request_interval_seconds=0.0,
        default_season=2026,
    )


def test_ingest_season_writes_raw_canonical_and_coverage(tmp_path: Path):
    client = FakeClient()
    report = ingest_season(2026, settings=settings_for(tmp_path), client=client)

    raw = tmp_path / "raw" / "cbbd" / "2026"
    assert (raw / "teams" / "directory.json").exists()
    assert (raw / "teams" / "roster.json").exists()
    assert (raw / "plays" / "2025-11-04.json").exists()

    directory = read_json(raw / "teams" / "directory.json")
    assert isinstance(directory, dict)
    assert "teams" in directory

    teams = list(read_jsonl(tmp_path / "processed" / "cbbd" / "2026" / "teams.jsonl"))
    assert teams[0]["conference"] == "ACC"
    players = list(read_jsonl(tmp_path / "processed" / "cbbd" / "2026" / "players.jsonl"))
    assert [(p["player_id"], p["team_id"], p["name"]) for p in players] == [(10, 1, "Player")]

    games = list(read_jsonl(tmp_path / "processed" / "cbbd" / "2026" / "games.jsonl"))
    assert [g["game_id"] for g in games] == [100, 101]
    assert games[0]["home_team_id"] == 1
    plays = list(read_jsonl(tmp_path / "processed" / "cbbd" / "2026" / "plays.jsonl"))
    assert len(plays) == 1
    assert report["games_missing_play_by_play_count"] == 0
    assert report["games_pending_play_by_play_count"] == 1
    assert "2025-11-05" in report["empty_play_dates"]

    first_calls = len(client.calls)
    ingest_season(2026, settings=settings_for(tmp_path), client=client)
    assert len(client.calls) == first_calls


def test_force_refetches(tmp_path: Path):
    client = FakeClient()
    ingest_season(2026, settings=settings_for(tmp_path), client=client)
    n = len(client.calls)
    ingest_season(2026, settings=settings_for(tmp_path), client=client, force=True)
    assert len(client.calls) > n


def test_coverage_tracks_missing_finals():
    report = build_coverage_report(
        season=2026,
        games=[
            {
                "game_id": 1,
                "status": "final",
                "home_points": 70,
                "away_points": 60,
                "start_date_utc": "2025-11-04",
            }
        ],
        play_game_ids=set(),
        requested_dates=[date(2025, 11, 4)],
        empty_play_dates=[date(2025, 11, 4)],
        failed_play_dates=[],
        skipped_network=False,
    )
    assert report["games_missing_play_by_play_count"] == 1
    assert report["season_label"] == "2025-26"


def test_cli_without_key_exits(tmp_path: Path, monkeypatch):
    from statpad_cbb.ingest import cli

    # Keep an empty key in the environment so load_dotenv() does not reload .env.
    monkeypatch.setenv("CBBD_API_KEY", "")
    monkeypatch.setenv("STATPAD_DATA_DIR", str(tmp_path))
    assert cli.main(["ingest", "--season", "2026", "--data-dir", str(tmp_path)]) == 1
    assert not list((tmp_path / "raw").glob("**/*.json"))


def test_games_range_splits_when_response_hits_limit(tmp_path: Path):
    class SplitClient:
        def __init__(self) -> None:
            self.calls: list[tuple[date, date]] = []

        def games(self, season: int, start: date, end: date):
            self.calls.append((start, end))
            if start == end:
                return [
                    {
                        "id": int(start.strftime("%Y%m%d")),
                        "startDate": f"{start.isoformat()}T00:00:00.000Z",
                        "homeTeam": {"id": 1},
                        "awayTeam": {"id": 2},
                    }
                ]
            return [{"id": i} for i in range(GAMES_RESPONSE_LIMIT)]

    paths = SeasonPaths(tmp_path, 2026)
    paths.ensure()
    client = SplitClient()
    rows = _ingest_games_range(
        lambda: client,
        paths,
        2026,
        date(2025, 11, 1),
        date(2025, 11, 2),
        force=True,
    )
    assert client.calls[0] == (date(2025, 11, 1), date(2025, 11, 2))
    assert (date(2025, 11, 1), date(2025, 11, 1)) in client.calls
    assert (date(2025, 11, 2), date(2025, 11, 2)) in client.calls
    assert [row["id"] for row in rows] == [20251101, 20251102]
    assert not (paths.games_chunk(date(2025, 11, 1), date(2025, 11, 2))).exists()

