from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from statpad_cbb.config import load_settings
from statpad_cbb.ingest.extract import game_is_final
from statpad_cbb.ingest.storage import SeasonPaths, read_jsonl, write_json
from statpad_cbb.possessions.builder import GameContext, build_game_possessions, iter_play_groups

logger = logging.getLogger(__name__)


def _load_teams(paths: SeasonPaths) -> dict[int, dict[str, Any]]:
    return {int(row["team_id"]): row for row in read_jsonl(paths.processed_teams)}


def _load_games(paths: SeasonPaths) -> dict[int, dict[str, Any]]:
    return {int(row["game_id"]): row for row in read_jsonl(paths.processed_games)}


def build_possessions(
    season: int,
    *,
    data_dir: Path | None = None,
    max_games: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    paths.ensure()

    teams = _load_teams(paths)
    games = _load_games(paths)
    directory_ids = set(teams)

    rows: list[dict[str, Any]] = []
    coverage = {
        "season": season,
        "games_considered": 0,
        "games_skipped_non_directory": 0,
        "games_skipped_not_final": 0,
        "games_skipped_missing_meta": 0,
        "games_processed": 0,
        "games_with_possessions": 0,
        "plays_seen": 0,
        "possessions": 0,
        "skipped_bad_open": 0,
        "mean_ppp": None,
        "end_reason_counts": {},
    }
    end_reasons: dict[str, int] = {}

    for game_id, plays in iter_play_groups(read_jsonl(paths.processed_plays)):
        coverage["plays_seen"] += len(plays)
        game = games.get(game_id)
        if game is None:
            coverage["games_skipped_missing_meta"] += 1
            continue
        coverage["games_considered"] += 1
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")
        if home_id not in directory_ids or away_id not in directory_ids:
            coverage["games_skipped_non_directory"] += 1
            continue
        if not game_is_final(game):
            coverage["games_skipped_not_final"] += 1
            continue
        home = teams[int(home_id)]
        away = teams[int(away_id)]
        home_school = home.get("school")
        away_school = away.get("school")
        if not home_school or not away_school:
            coverage["games_skipped_missing_meta"] += 1
            continue

        ctx = GameContext(
            game_id=game_id,
            season=int(game.get("season") or season),
            home_team_id=int(home_id),
            away_team_id=int(away_id),
            home_school=str(home_school),
            away_school=str(away_school),
            neutral_site=bool(game.get("neutral_site")),
        )
        poss_rows, stats = build_game_possessions(ctx, plays)
        coverage["games_processed"] += 1
        coverage["skipped_bad_open"] += stats.get("skipped_bad_open", 0)
        if poss_rows:
            coverage["games_with_possessions"] += 1
            rows.extend(poss_rows)
            for row in poss_rows:
                reason = str(row.get("end_reason") or "unknown")
                end_reasons[reason] = end_reasons.get(reason, 0) + 1

        if max_games is not None and coverage["games_processed"] >= max_games:
            break

        if coverage["games_processed"] % 250 == 0:
            logger.info(
                "possession build progress: games=%s possessions=%s",
                coverage["games_processed"],
                len(rows),
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        coverage["possessions"] = int(len(frame))
        coverage["mean_ppp"] = float(frame["points"].mean())
        coverage["end_reason_counts"] = dict(sorted(end_reasons.items(), key=lambda kv: (-kv[1], kv[0])))
        frame.to_parquet(paths.possessions, index=False)
    else:
        # Still write an empty parquet with expected columns for downstream stability.
        empty = pd.DataFrame(
            columns=[
                "season",
                "game_id",
                "poss_id",
                "poss_index",
                "period",
                "off_team_id",
                "def_team_id",
                "off_player_1",
                "off_player_2",
                "off_player_3",
                "off_player_4",
                "off_player_5",
                "def_player_1",
                "def_player_2",
                "def_player_3",
                "def_player_4",
                "def_player_5",
                "points",
                "home_offense",
                "neutral_site",
                "end_reason",
                "lineup_source",
                "valid_lineup",
            ]
        )
        empty.to_parquet(paths.possessions, index=False)
        coverage["possessions"] = 0
        coverage["end_reason_counts"] = end_reasons

    write_json(paths.possession_coverage, coverage)
    logger.info(
        "wrote %s possessions to %s (games_processed=%s)",
        coverage["possessions"],
        paths.possessions,
        coverage["games_processed"],
    )
    return coverage
