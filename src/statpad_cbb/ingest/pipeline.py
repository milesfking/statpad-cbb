from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Callable

from statpad_cbb.config import GAMES_RESPONSE_LIMIT, Settings, load_settings
from statpad_cbb.ingest.client import CbbdApi, CbbdClient
from statpad_cbb.ingest.coverage import build_coverage_report
from statpad_cbb.ingest.dates import season_game_chunks, split_range, unique_utc_game_dates
from statpad_cbb.ingest.extract import (
    di_team_ids,
    directory_team_rows,
    games_for_di_teams,
    iter_unique_plays,
    unique_games,
    unique_players,
    unique_teams,
)
from statpad_cbb.ingest.storage import SeasonPaths, load_json_lists, read_json, write_json, write_jsonl

logger = logging.getLogger(__name__)

Fetcher = Callable[[], Any]


class _LazyClient:
    def __init__(self, settings: Settings, provided: CbbdApi | None) -> None:
        self._settings = settings
        self._client = provided
        self._owns = provided is None

    def get(self) -> CbbdApi:
        if self._client is None:
            if not self._settings.api_key:
                raise ValueError("CBBD_API_KEY is required to call the API")
            self._client = CbbdClient(
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
                min_request_interval_seconds=self._settings.min_request_interval_seconds,
            )
            self._owns = True
        return self._client

    def close(self) -> None:
        if self._owns and isinstance(self._client, CbbdClient):
            self._client.close()


def ingest_season(
    season: int,
    *,
    settings: Settings | None = None,
    client: CbbdApi | None = None,
    force: bool = False,
    skip_plays: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    paths.ensure()
    lazy = _LazyClient(settings, client)

    try:
        directory = _fetch_raw(
            paths.teams_directory,
            lambda: lazy.get().teams_directory(season),
            force=force,
            label=f"teams/directory season={season}",
        )
        roster = _fetch_raw(
            paths.teams_roster,
            lambda: lazy.get().teams_roster(season),
            force=force,
            label=f"teams/roster season={season}",
        )

        raw_games = _ingest_games(lazy.get, paths, season, force=force)
        team_rows, conferences = directory_team_rows(directory)
        teams = unique_teams(team_rows, season, conferences=conferences)
        players = unique_players(roster if isinstance(roster, list) else [], season)
        games = unique_games(raw_games, season)
        di_ids = di_team_ids(team_rows)
        games = games_for_di_teams(games, di_ids)

        write_jsonl(paths.processed_teams, teams)
        write_jsonl(paths.processed_players, players)
        write_jsonl(paths.processed_games, games)
        logger.info(
            "Canonicalized %s teams, %s players, %s games for season %s",
            len(teams),
            len(players),
            len(games),
            season,
        )

        play_dates = unique_utc_game_dates(games)
        empty_play_dates: list[date] = []
        failed_play_dates: list[date] = []
        play_game_ids: set[int] = set()

        if skip_plays:
            write_jsonl(paths.processed_plays, [])
        else:
            for game_date in play_dates:
                try:
                    payload = _fetch_raw(
                        paths.plays_date(game_date),
                        lambda d=game_date: lazy.get().plays_date(d),
                        force=force,
                        label=f"plays/date {game_date.isoformat()}",
                        indent=None,
                    )
                except Exception:
                    logger.exception("Failed to ingest plays for %s", game_date.isoformat())
                    failed_play_dates.append(game_date)
                    continue
                if not payload:
                    empty_play_dates.append(game_date)

            play_files = sorted(paths.raw_season.joinpath("plays").glob("*.json"))
            logger.info(
                "Canonicalizing plays for season %s from %s raw date files...",
                season,
                len(play_files),
            )
            raw_plays = load_json_lists(play_files)
            logger.info("Loaded %s raw plays; deduplicating...", len(raw_plays))
            plays = list(iter_unique_plays(raw_plays))
            logger.info(
                "Writing %s unique plays to %s...",
                len(plays),
                paths.processed_plays,
            )
            write_jsonl(paths.processed_plays, plays)
            for play in plays:
                play_game_ids.add(play["game_id"])
            logger.info("Wrote %s unique plays for season %s", len(plays), season)

        report = build_coverage_report(
            season=season,
            games=games,
            play_game_ids=play_game_ids,
            requested_dates=play_dates,
            empty_play_dates=empty_play_dates,
            failed_play_dates=failed_play_dates,
            skipped_network=False,
        )
        write_json(paths.coverage_report, report)
        logger.info(
            "Coverage season=%s games=%s missing_pbp=%s pending_pbp=%s empty_dates=%s failed_dates=%s",
            season,
            report["games_count"],
            report["games_missing_play_by_play_count"],
            report["games_pending_play_by_play_count"],
            len(empty_play_dates),
            len(failed_play_dates),
        )
        return report
    finally:
        lazy.close()


def _ingest_games(
    get_client: Callable[[], CbbdApi],
    paths: SeasonPaths,
    season: int,
    *,
    force: bool,
) -> list[Any]:
    collected: list[Any] = []
    for start, end in season_game_chunks(season):
        collected.extend(_ingest_games_range(get_client, paths, season, start, end, force=force))
    return collected


def _ingest_games_range(
    get_client: Callable[[], CbbdApi],
    paths: SeasonPaths,
    season: int,
    start: date,
    end: date,
    *,
    force: bool,
) -> list[Any]:
    path = paths.games_chunk(start, end)
    if path.exists() and not force:
        payload = read_json(path)
        games = payload if isinstance(payload, list) else []
        if len(games) >= GAMES_RESPONSE_LIMIT and start != end:
            logger.warning(
                "Existing games file %s hit the %s-game cap; splitting range",
                path.name,
                GAMES_RESPONSE_LIMIT,
            )
            left, right = split_range(start, end)
            return _ingest_games_range(
                get_client, paths, season, *left, force=force
            ) + _ingest_games_range(get_client, paths, season, *right, force=force)
        return games

    games = get_client().games(season, start, end)
    if len(games) >= GAMES_RESPONSE_LIMIT and start != end:
        logger.warning(
            "Games %s to %s returned %s rows (limit %s); splitting range",
            start.isoformat(),
            end.isoformat(),
            len(games),
            GAMES_RESPONSE_LIMIT,
        )
        left, right = split_range(start, end)
        return _ingest_games_range(
            get_client, paths, season, *left, force=True
        ) + _ingest_games_range(get_client, paths, season, *right, force=True)

    write_json(path, games)
    logger.info("Wrote %s games for %s to %s", len(games), path.stem, path)
    return games


def _fetch_raw(
    path: Path,
    fetcher: Fetcher,
    *,
    force: bool,
    label: str,
    indent: int | None = 2,
) -> Any:
    if path.exists() and not force:
        logger.info("Reusing raw %s (%s)", label, path)
        return read_json(path)
    payload = fetcher()
    write_json(path, payload, indent=indent)
    count = len(payload) if isinstance(payload, list) else 1
    logger.info("Wrote raw %s (%s records) to %s", label, count, path)
    return payload
