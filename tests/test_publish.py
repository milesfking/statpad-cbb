from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from statpad_cbb.ingest.storage import SeasonPaths, write_json, write_jsonl
from statpad_cbb.publish.site import (
    publish_site,
    season_as_of,
    season_label,
)


def test_season_label():
    assert season_label(2026) == "2025-26"


def test_season_as_of_uses_games_in_possessions(tmp_path: Path):
    data_dir = tmp_path / "data"
    paths = SeasonPaths(data_dir, 2026)
    paths.ensure()

    write_jsonl(
        paths.processed_games,
        [
            {
                "game_id": 1,
                "start_date_utc": "2026-11-09T00:00:00Z",
            },
            {
                "game_id": 2,
                "start_date_utc": "2026-11-15T00:00:00Z",
            },
        ],
    )
    pd.DataFrame({"game_id": [1]}).to_parquet(paths.possessions, index=False)

    assert season_as_of(paths) == date(2026, 11, 9)


def test_publish_site_exports_manifest_and_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    paths = SeasonPaths(data_dir, 2025)
    paths.ensure()

    teams = pd.DataFrame(
        [
            {
                "rank": 1,
                "off_rank": 1,
                "def_rank": 2,
                "team_id": 10,
                "school": "Alpha",
                "conference": "A",
                "off_rating": 12.0,
                "def_rating": 8.0,
                "net_rating": 20.0,
            }
        ]
    )
    teams.to_parquet(paths.team_ratings, index=False)
    write_json(
        paths.processed_season / "conference_ratings.json",
        [
            {
                "conference": "A",
                "off_rating": 5.0,
                "def_rating": 3.0,
                "net_rating": 8.0,
            }
        ],
    )
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "team_id": 10,
                "name": "Player One",
                "conference": "A",
                "minutes": 20.0,
                "possessions": 500.0,
                "off_rating": 4.0,
                "def_rating": 2.0,
                "net_rating": 6.0,
            }
        ]
    ).to_parquet(paths.player_ratings, index=False)
    write_jsonl(paths.processed_teams, [{"team_id": 10, "school": "Alpha"}])
    write_jsonl(paths.processed_games, [{"game_id": 1, "start_date_utc": "2025-03-01T00:00:00Z"}])
    pd.DataFrame({"game_id": [1]}).to_parquet(paths.possessions, index=False)

    out = tmp_path / "out"
    monkeypatch.setenv("STATPAD_DATA_DIR", str(data_dir))

    report = publish_site(output_dir=out, seasons=[2025], min_player_possessions=100.0)

    assert (out / "rankings_meta.json").exists()
    assert (out / "2025" / "teams.json").exists()
    assert (out / "2025" / "conferences.json").exists()
    assert (out / "2025" / "players.json").exists()
    assert report["manifest"]["seasons"][0]["status"] == "final"
    assert report["manifest"]["seasons"][0]["as_of"] == "2025-03-01"
