from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from statpad_cbb.config import DEFAULT_GAME_CHUNK_DAYS


def season_date_bounds(season: int) -> tuple[date, date]:
    """Inclusive calendar window for a season-ending year (2026 = 2025–26)."""
    return date(season - 1, 10, 1), date(season, 4, 20)


def iter_date_chunks(
    start: date,
    end: date,
    chunk_days: int = DEFAULT_GAME_CHUNK_DAYS,
) -> list[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if end < start:
        raise ValueError("end must be on or after start")

    chunks: list[tuple[date, date]] = []
    cursor = start
    delta = timedelta(days=chunk_days - 1)
    while cursor <= end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def season_game_chunks(
    season: int,
    chunk_days: int = DEFAULT_GAME_CHUNK_DAYS,
) -> list[tuple[date, date]]:
    start, end = season_date_bounds(season)
    return iter_date_chunks(start, end, chunk_days=chunk_days)


def utc_date_from_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).date()
    return parsed.date()


def unique_utc_game_dates(games: Iterable[dict[str, Any]]) -> list[date]:
    dates: set[date] = set()
    for game in games:
        raw = (
            game.get("start_date")
            or game.get("startDate")
            or game.get("start_date_utc")
            or game.get("date")
        )
        parsed = utc_date_from_value(raw)
        if parsed is not None:
            dates.add(parsed)
    return sorted(dates)


def split_range(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    if start == end:
        raise ValueError("cannot split a single-day range")
    span = (end - start).days
    mid = start + timedelta(days=span // 2)
    return (start, mid), (mid + timedelta(days=1), end)
