from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from statpad_cbb.config import DEFAULT_SEASON, load_settings, repo_root
from statpad_cbb.model.design import is_ranked_conference
from statpad_cbb.ingest.storage import SeasonPaths, read_jsonl, write_json


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


def discover_seasons(data_dir: Path) -> list[int]:
    root = data_dir / "processed" / "cbbd"
    if not root.is_dir():
        return []
    seasons: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            season = int(child.name)
        except ValueError:
            continue
        if (child / "team_ratings.parquet").exists() or (child / "team_ratings.json").exists():
            seasons.append(season)
    return sorted(seasons)


def _parse_utc_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
    if m:
        return date.fromisoformat(m.group(1))
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def possession_game_ids(paths: SeasonPaths) -> set[int]:
    if not paths.possessions.exists():
        return set()
    poss = pd.read_parquet(paths.possessions, columns=["game_id"])
    return {int(x) for x in poss["game_id"].dropna().unique()}


def season_as_of(paths: SeasonPaths) -> date | None:
    """Latest game date in possessions, falling back to processed games."""
    game_ids = possession_game_ids(paths)
    dates: list[date] = []
    if paths.processed_games.exists():
        for row in read_jsonl(paths.processed_games):
            gid = row.get("game_id")
            if game_ids and gid is not None and int(gid) not in game_ids:
                continue
            d = _parse_utc_date(row.get("start_date_utc") or row.get("start_date"))
            if d is not None:
                dates.append(d)
    if dates:
        return max(dates)
    return None


def _add_rating_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, rank_col in (
        ("net_rating", "rank"),
        ("off_rating", "off_rank"),
        ("def_rating", "def_rank"),
    ):
        if col in out.columns:
            out[rank_col] = out[col].rank(method="min", ascending=False).astype(int)
    out = out.sort_values("net_rating", ascending=False).reset_index(drop=True)
    return out


def _round_records(rows: list[dict[str, Any]], float_cols: set[str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in float_cols:
            val = item.get(key)
            if isinstance(val, float):
                item[key] = round(val, 2)
        cleaned.append(item)
    return cleaned


def export_teams(paths: SeasonPaths) -> list[dict[str, Any]]:
    if not paths.team_ratings.exists() and not paths.team_ratings_json.exists():
        raise FileNotFoundError(f"Missing team ratings under {paths.processed_season}")
    if paths.team_ratings.exists():
        teams = pd.read_parquet(paths.team_ratings)
    else:
        from statpad_cbb.ingest.storage import read_json

        teams = pd.DataFrame(read_json(paths.team_ratings_json))
    if "off_rank" not in teams.columns or "def_rank" not in teams.columns:
        teams = _add_rating_ranks(teams)
    float_cols = {
        "off_rating",
        "def_rating",
        "net_rating",
        "off_eff",
        "def_eff",
        "eff_margin",
        "player_minutes",
    }
    from statpad_cbb.model.rollup import teams_records

    return _round_records(teams_records(teams), float_cols)


def export_conferences(paths: SeasonPaths) -> list[dict[str, Any]]:
    conf_path = paths.processed_season / "conference_ratings.json"
    if not conf_path.exists():
        return []
    import json

    payload = json.loads(conf_path.read_text(encoding="utf-8"))
    if not payload:
        return []
    frame = pd.DataFrame(payload)
    if "conference" in frame.columns:
        frame = frame.loc[frame["conference"].map(is_ranked_conference)]
    frame = _add_rating_ranks(frame)
    float_cols = {"off_rating", "def_rating", "net_rating"}
    rows = frame.to_dict(orient="records")
    return _round_records(rows, float_cols)


def export_players(
    paths: SeasonPaths,
    *,
    min_possessions: float = 200.0,
) -> list[dict[str, Any]]:
    if not paths.player_ratings.exists():
        return []
    players = pd.read_parquet(paths.player_ratings)
    if players.empty:
        return []
    if "school" not in players.columns and paths.processed_teams.exists():
        lookup = {
            int(row["team_id"]): row.get("school")
            for row in read_jsonl(paths.processed_teams)
            if row.get("team_id") is not None
        }
        players = players.copy()
        players["school"] = players["team_id"].map(lookup)

    if min_possessions > 0:
        if "possessions" in players.columns:
            players = players.loc[players["possessions"].fillna(0) >= min_possessions].copy()
        elif "minutes" in players.columns:
            players = players.loc[players["minutes"].fillna(0) >= min_possessions].copy()
    if "conference" in players.columns:
        players = players.loc[players["conference"].map(is_ranked_conference)]

    players = _add_rating_ranks(players)
    cols = [
        "rank",
        "off_rank",
        "def_rank",
        "player_id",
        "name",
        "team_id",
        "school",
        "conference",
        "minutes",
        "off_rating",
        "def_rating",
        "net_rating",
    ]
    present = [c for c in cols if c in players.columns]
    players = players.sort_values("rank").reset_index(drop=True)
    float_cols = {"minutes", "off_rating", "def_rating", "net_rating"}
    rows = players[present].to_dict(orient="records")
    return _round_records(rows, float_cols)


def season_manifest_entry(
    paths: SeasonPaths,
    *,
    default_season: int = DEFAULT_SEASON,
) -> dict[str, Any]:
    as_of = season_as_of(paths)
    status = "live" if paths.season == default_season else "final"
    entry: dict[str, Any] = {
        "season": paths.season,
        "label": season_label(paths.season),
        "status": status,
    }
    if as_of is not None:
        entry["as_of"] = as_of.isoformat()
    return entry


def publish_site(
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    seasons: list[int] | None = None,
    min_player_possessions: float = 200.0,
) -> dict[str, Any]:
    settings = load_settings(data_dir=data_dir)
    out = output_dir or (repo_root() / "web" / "public" / "data")
    out = out if out.is_absolute() else repo_root() / out
    out.mkdir(parents=True, exist_ok=True)

    available = discover_seasons(settings.data_dir)
    if not available:
        raise FileNotFoundError(
            f"No team_ratings artifacts under {settings.data_dir / 'processed' / 'cbbd'}"
        )

    if seasons:
        unknown = [s for s in seasons if s not in available]
        if unknown:
            raise FileNotFoundError(f"Season(s) not found in data dir: {unknown}")
        target = sorted(seasons)
    else:
        target = available

    published: list[dict[str, Any]] = []
    for season in target:
        paths = SeasonPaths(settings.data_dir, season)
        season_dir = out / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)

        teams = export_teams(paths)
        conferences = export_conferences(paths)
        players = export_players(paths, min_possessions=min_player_possessions)

        write_json(season_dir / "teams.json", teams)
        write_json(season_dir / "conferences.json", conferences)
        write_json(season_dir / "players.json", players)
        published.append(season_manifest_entry(paths, default_season=settings.default_season))

    default = settings.default_season if settings.default_season in target else target[-1]
    manifest = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "default_season": default,
        "seasons": sorted(published, key=lambda r: r["season"], reverse=True),
    }
    write_json(out / "rankings_meta.json", manifest)

    return {
        "output_dir": str(out),
        "seasons": target,
        "manifest": manifest,
        "min_player_possessions": min_player_possessions,
    }
