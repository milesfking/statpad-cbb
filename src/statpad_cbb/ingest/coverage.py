from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from statpad_cbb.ingest.extract import game_is_final


def build_coverage_report(
    *,
    season: int,
    games: Iterable[dict[str, Any]],
    play_game_ids: set[int],
    requested_dates: list[date],
    empty_play_dates: list[date],
    failed_play_dates: list[date],
    skipped_network: bool,
) -> dict[str, Any]:
    games_list = list(games)
    game_ids = [game["game_id"] for game in games_list]
    missing = []
    pending = []
    for game in games_list:
        game_id = game["game_id"]
        if game_id in play_game_ids:
            continue
        if game_is_final(game):
            missing.append(_game_coverage_row(game))
        else:
            pending.append(_game_coverage_row(game))

    return {
        "season": season,
        "season_label": f"{season - 1}-{str(season)[2:]}",
        "skipped_network": skipped_network,
        "games_count": len(games_list),
        "unique_game_ids": len(set(game_ids)),
        "plays_game_ids_count": len(play_game_ids),
        "requested_play_dates_count": len(requested_dates),
        "requested_play_dates": [d.isoformat() for d in requested_dates],
        "empty_play_dates": [d.isoformat() for d in empty_play_dates],
        "failed_play_dates": [d.isoformat() for d in failed_play_dates],
        "games_missing_play_by_play": missing,
        "games_pending_play_by_play": pending,
        "games_missing_play_by_play_count": len(missing),
        "games_pending_play_by_play_count": len(pending),
    }


def _game_coverage_row(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": game.get("game_id"),
        "start_date_utc": game.get("start_date_utc"),
        "home_team_id": game.get("home_team_id"),
        "away_team_id": game.get("away_team_id"),
        "status": game.get("status"),
        "home_points": game.get("home_points"),
        "away_points": game.get("away_points"),
    }
