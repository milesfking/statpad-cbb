from __future__ import annotations

import re
from typing import Any

SUB_IN_RE = re.compile(r"subbing in", re.IGNORECASE)
SUB_OUT_RE = re.compile(r"subbing out", re.IGNORECASE)


def parse_on_floor(
    play: dict[str, Any],
    *,
    home_team_id: int,
    away_team_id: int,
    home_school: str,
    away_school: str,
) -> tuple[list[int], list[int]] | None:
    """Return (home_player_ids, away_player_ids) when onFloor is a valid 5-5."""
    on_floor = play.get("onFloor") or []
    if len(on_floor) != 10:
        return None
    by_team: dict[str, list[int]] = {}
    for entry in on_floor:
        if not isinstance(entry, dict):
            return None
        team = entry.get("team")
        player_id = entry.get("id")
        if team is None or player_id is None:
            return None
        by_team.setdefault(str(team), []).append(int(player_id))
    if set(by_team) != {home_school, away_school}:
        return None
    home_ids = by_team[home_school]
    away_ids = by_team[away_school]
    if len(home_ids) != 5 or len(away_ids) != 5:
        return None
    if len(set(home_ids)) != 5 or len(set(away_ids)) != 5:
        return None
    return sorted(home_ids), sorted(away_ids)


def apply_substitution(
    play: dict[str, Any],
    home_ids: set[int],
    away_ids: set[int],
    *,
    home_team_id: int,
    away_team_id: int,
) -> bool:
    """Mutate on-court sets from a Substitution play when possible.

    Returns True when a mutation was applied.
    """
    text = str(play.get("playText") or "")
    participants = play.get("participants") or []
    if not participants or not isinstance(participants[0], dict):
        return False
    player_id = participants[0].get("id")
    if player_id is None:
        return False
    player_id = int(player_id)
    team_id = play.get("teamId")
    if team_id is None:
        return False
    team_id = int(team_id)
    if team_id == home_team_id:
        target = home_ids
    elif team_id == away_team_id:
        target = away_ids
    else:
        return False
    if SUB_OUT_RE.search(text):
        target.discard(player_id)
        return True
    if SUB_IN_RE.search(text):
        target.add(player_id)
        return True
    return False


def lineup_sides(
    home_ids: list[int] | set[int],
    away_ids: list[int] | set[int],
    *,
    offense_team_id: int,
    home_team_id: int,
    away_team_id: int,
) -> tuple[list[int], list[int]] | None:
    """Split home/away lineups into offense/defense player id lists."""
    home = sorted(home_ids)
    away = sorted(away_ids)
    if len(home) != 5 or len(away) != 5:
        return None
    if offense_team_id == home_team_id:
        return home, away
    if offense_team_id == away_team_id:
        return away, home
    return None


def balanced_counts(home_ids: set[int], away_ids: set[int]) -> bool:
    return len(home_ids) == 5 and len(away_ids) == 5
