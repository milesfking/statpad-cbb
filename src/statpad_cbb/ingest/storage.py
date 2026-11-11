from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator


class SeasonPaths:
    def __init__(self, data_dir: Path, season: int) -> None:
        self.data_dir = data_dir
        self.season = season
        self.raw_season = data_dir / "raw" / "cbbd" / str(season)
        self.processed_season = data_dir / "processed" / "cbbd" / str(season)
        self.coverage_dir = data_dir / "coverage" / "cbbd" / str(season)

    def ensure(self) -> None:
        (self.raw_season / "teams").mkdir(parents=True, exist_ok=True)
        (self.raw_season / "games").mkdir(parents=True, exist_ok=True)
        (self.raw_season / "plays").mkdir(parents=True, exist_ok=True)
        self.processed_season.mkdir(parents=True, exist_ok=True)
        self.coverage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def teams_directory(self) -> Path:
        return self.raw_season / "teams" / "directory.json"

    @property
    def teams_roster(self) -> Path:
        return self.raw_season / "teams" / "roster.json"

    def games_chunk(self, start: date, end: date) -> Path:
        return self.raw_season / "games" / f"{start.isoformat()}_{end.isoformat()}.json"

    def plays_date(self, game_date: date) -> Path:
        return self.raw_season / "plays" / f"{game_date.isoformat()}.json"

    @property
    def processed_teams(self) -> Path:
        return self.processed_season / "teams.jsonl"

    @property
    def processed_players(self) -> Path:
        return self.processed_season / "players.jsonl"

    @property
    def processed_games(self) -> Path:
        return self.processed_season / "games.jsonl"

    @property
    def processed_plays(self) -> Path:
        return self.processed_season / "plays.jsonl"

    @property
    def coverage_report(self) -> Path:
        return self.coverage_dir / "ingest_coverage.json"

    @property
    def possessions(self) -> Path:
        return self.processed_season / "possessions.parquet"

    @property
    def possession_coverage(self) -> Path:
        return self.coverage_dir / "possession_coverage.json"

    @property
    def player_features(self) -> Path:
        return self.processed_season / "player_features.parquet"

    @property
    def player_ratings(self) -> Path:
        return self.processed_season / "player_ratings.parquet"

    @property
    def team_ratings(self) -> Path:
        return self.processed_season / "team_ratings.parquet"

    @property
    def model_metrics(self) -> Path:
        return self.coverage_dir / "rapm_metrics.json"

    @property
    def team_ratings_json(self) -> Path:
        return self.processed_season / "team_ratings.json"

    @property
    def eda_dir(self) -> Path:
        return self.coverage_dir / "eda"

    @property
    def backtest_metrics(self) -> Path:
        return self.coverage_dir / "temporal_backtest.json"

    @property
    def backtest_game_predictions(self) -> Path:
        return self.processed_season / "backtest_game_predictions.parquet"

    @property
    def season_priors(self) -> Path:
        return self.processed_season / "season_priors.json"


def write_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json_lists(paths: Iterable[Path]) -> list[Any]:
    items: list[Any] = []
    for path in sorted(paths):
        payload = read_json(path)
        if payload is None:
            continue
        if isinstance(payload, list):
            items.extend(payload)
        else:
            items.append(payload)
    return items
