from __future__ import annotations

from typing import Any, Iterable, Iterator

from statpad_cbb.ingest.dates import utc_date_from_value


def first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def is_division_i(team: dict[str, Any]) -> bool:
    division = first_present(team, "division", "classification", "level")
    if division is None:
        return True
    text = str(division).strip().lower().replace(" ", "")
    return text in {"i", "di", "d1", "divisioni", "division1"}


def team_side(game: dict[str, Any], side: str) -> dict[str, Any]:
    nested = first_present(game, f"{side}Team", f"{side}_team", side)
    if isinstance(nested, dict):
        return nested
    teams = game.get("teams")
    if isinstance(teams, list):
        for team in teams:
            if not isinstance(team, dict):
                continue
            home_away = str(first_present(team, "homeAway", "home_away", "side") or "").lower()
            if home_away == side:
                return team
    return {}


def conference_lookup(conferences: Iterable[dict[str, Any]] | None) -> dict[int, str]:
    lookup: dict[int, str] = {}
    if not conferences:
        return lookup
    for conference in conferences:
        if not isinstance(conference, dict):
            continue
        conference_id = as_int(first_present(conference, "id", "conferenceId", "conference_id"))
        name = first_present(conference, "name", "abbreviation")
        if conference_id is not None and name is not None:
            lookup[conference_id] = str(name)
    return lookup


def directory_team_rows(payload: Any) -> tuple[list[dict[str, Any]], dict[int, str]]:
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        return rows, {}
    if isinstance(payload, dict):
        teams = payload.get("teams")
        conferences = payload.get("conferences")
        rows = [row for row in teams if isinstance(row, dict)] if isinstance(teams, list) else []
        conf_rows = [row for row in conferences if isinstance(row, dict)] if isinstance(conferences, list) else []
        return rows, conference_lookup(conf_rows)
    return [], {}


def flatten_roster(payload: Any) -> list[dict[str, Any]]:
    """CBBD /teams/roster returns one object per team with nested players."""
    if not isinstance(payload, list):
        return []
    players: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        nested = row.get("players")
        if isinstance(nested, list):
            team_id = as_int(first_present(row, "teamId", "team_id", "id"))
            for player in nested:
                if not isinstance(player, dict):
                    continue
                record = dict(player)
                if team_id is not None and record.get("teamId") is None and record.get("team_id") is None:
                    record["teamId"] = team_id
                players.append(record)
        else:
            players.append(row)
    return players


def canonicalize_team(
    team: dict[str, Any],
    season: int,
    *,
    conferences: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    team_id = as_int(first_present(team, "id", "teamId", "team_id"))
    if team_id is None:
        return None
    conference = first_present(team, "conference", "conferenceName", "conference_name")
    if isinstance(conference, dict):
        conference = first_present(conference, "name", "abbreviation")
    if conference is None and conferences:
        conference_id = as_int(first_present(team, "conferenceId", "conference_id"))
        if conference_id is not None:
            conference = conferences.get(conference_id)
    return {
        "team_id": team_id,
        "season": season,
        "school": first_present(team, "school", "name", "displayName", "display_name"),
        "abbreviation": first_present(team, "abbreviation", "abbrev"),
        "conference": conference,
        "conference_id": as_int(first_present(team, "conferenceId", "conference_id")),
        "division": first_present(team, "division", "classification"),
        "source_id": first_present(team, "sourceId", "source_id"),
    }


def canonicalize_player(player: dict[str, Any], season: int) -> dict[str, Any] | None:
    player_id = as_int(first_present(player, "id", "athleteId", "athlete_id", "playerId", "player_id"))
    if player_id is None:
        return None
    team = player.get("team") if isinstance(player.get("team"), dict) else {}
    team_id = as_int(
        first_present(player, "teamId", "team_id")
        or (first_present(team, "id", "teamId") if team else None)
    )
    return {
        "player_id": player_id,
        "season": season,
        "team_id": team_id,
        "name": first_present(player, "name", "displayName", "display_name"),
        "first_name": first_present(player, "firstName", "first_name"),
        "last_name": first_present(player, "lastName", "last_name"),
        "position": first_present(player, "position"),
        "jersey": first_present(player, "jersey", "number"),
        "year": first_present(player, "year", "class", "experience"),
        "height": first_present(player, "height"),
        "weight": first_present(player, "weight"),
        "source_id": first_present(player, "sourceId", "source_id"),
    }


def canonicalize_game(game: dict[str, Any], season: int) -> dict[str, Any] | None:
    game_id = as_int(first_present(game, "id", "gameId", "game_id"))
    if game_id is None:
        return None
    home = team_side(game, "home")
    away = team_side(game, "away")
    start_raw = first_present(game, "startDate", "start_date", "date")
    start_utc = utc_date_from_value(start_raw)
    venue = first_present(game, "venue", "venueName", "venue_name")
    venue_name = venue.get("name") if isinstance(venue, dict) else venue
    city = first_present(game, "city") or (venue.get("city") if isinstance(venue, dict) else None)
    state = first_present(game, "state") or (venue.get("state") if isinstance(venue, dict) else None)
    return {
        "game_id": game_id,
        "season": as_int(first_present(game, "season")) or season,
        "season_type": first_present(game, "seasonType", "season_type"),
        "tournament": first_present(game, "tournament", "tournamentName"),
        "start_date": start_raw,
        "start_date_utc": start_utc.isoformat() if start_utc else None,
        "status": first_present(game, "status", "gameStatus", "game_status"),
        "neutral_site": first_present(game, "neutralSite", "neutral_site"),
        "conference_game": first_present(game, "conferenceGame", "conference_game"),
        "venue": venue_name,
        "city": city,
        "state": state,
        "home_team_id": as_int(
            first_present(game, "homeTeamId", "home_team_id")
            or first_present(home, "id", "teamId", "team_id")
        ),
        "away_team_id": as_int(
            first_present(game, "awayTeamId", "away_team_id")
            or first_present(away, "id", "teamId", "team_id")
        ),
        "home_points": as_int(
            first_present(game, "homePoints", "home_points")
            or first_present(home, "points", "score")
        ),
        "away_points": as_int(
            first_present(game, "awayPoints", "away_points")
            or first_present(away, "points", "score")
        ),
        "home_winner": first_present(game, "homeWinner", "home_winner")
        if any(key in game for key in ("homeWinner", "home_winner"))
        else first_present(home, "winner"),
        "away_winner": first_present(game, "awayWinner", "away_winner")
        if any(key in game for key in ("awayWinner", "away_winner"))
        else first_present(away, "winner"),
    }


def canonicalize_play(play: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(play, dict):
        return None
    game = play.get("game") if isinstance(play.get("game"), dict) else {}
    play_id = first_present(play, "id", "playId", "play_id")
    game_id = as_int(
        first_present(play, "gameId", "game_id")
        or (first_present(game, "id", "gameId") if game else None)
    )
    if play_id is None or game_id is None:
        return None
    record = dict(play)
    record["play_id"] = play_id
    record["game_id"] = game_id
    return record


def dedupe_by_keys(
    rows: Iterable[dict[str, Any]],
    *keys: str,
) -> Iterator[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        if identity in seen:
            continue
        seen.add(identity)
        yield row


def unique_teams(
    teams: Iterable[dict[str, Any]],
    season: int,
    *,
    conferences: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    canonical = [
        row
        for row in (
            canonicalize_team(team, season, conferences=conferences) for team in teams
        )
        if row
    ]
    return list(dedupe_by_keys(canonical, "team_id", "season"))


def unique_players(players: Iterable[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    flat = flatten_roster(list(players))
    canonical = [row for row in (canonicalize_player(player, season) for player in flat) if row]
    return list(dedupe_by_keys(canonical, "player_id", "team_id", "season"))


def unique_games(games: Iterable[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    canonical = [row for row in (canonicalize_game(game, season) for game in games) if row]
    return list(dedupe_by_keys(canonical, "game_id"))


def iter_unique_plays(plays: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    canonical = (row for row in (canonicalize_play(play) for play in plays) if row)
    yield from dedupe_by_keys(canonical, "play_id", "game_id")


def game_is_final(game: dict[str, Any]) -> bool:
    status = str(game.get("status") or "").lower()
    if status in {"final", "completed", "complete", "status_final"}:
        return True
    return game.get("home_points") is not None and game.get("away_points") is not None


def di_team_ids(teams: Iterable[dict[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for team in teams:
        if not is_division_i(team):
            continue
        team_id = as_int(first_present(team, "id", "teamId", "team_id") or team.get("team_id"))
        if team_id is not None:
            ids.add(team_id)
    return ids


def games_for_di_teams(
    games: Iterable[dict[str, Any]],
    di_ids: set[int],
) -> list[dict[str, Any]]:
    if not di_ids:
        return list(games)
    selected: list[dict[str, Any]] = []
    for game in games:
        home = game.get("home_team_id")
        away = game.get("away_team_id")
        if home in di_ids or away in di_ids:
            selected.append(game)
    return selected
