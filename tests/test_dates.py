from datetime import date, datetime, timezone

from statpad_cbb.ingest.dates import (
    iter_date_chunks,
    season_date_bounds,
    season_game_chunks,
    split_range,
    unique_utc_game_dates,
    utc_date_from_value,
)


def test_season_2026_is_2025_26_window():
    start, end = season_date_bounds(2026)
    assert start == date(2025, 10, 1)
    assert end == date(2026, 4, 20)


def test_date_chunks_are_inclusive_and_contiguous():
    chunks = iter_date_chunks(date(2025, 11, 1), date(2025, 11, 20), chunk_days=7)
    assert chunks[0] == (date(2025, 11, 1), date(2025, 11, 7))
    assert chunks[-1][1] == date(2025, 11, 20)
    flattened = []
    for start, end in chunks:
        cursor = start
        while cursor <= end:
            flattened.append(cursor)
            cursor = cursor.fromordinal(cursor.toordinal() + 1)
    assert flattened == [date(2025, 11, d) for d in range(1, 21)]


def test_utc_date_uses_utc_not_local_clock():
    parsed = utc_date_from_value("2025-11-05T01:30:00.000Z")
    assert parsed == date(2025, 11, 5)
    parsed_offset = utc_date_from_value("2025-11-05T01:30:00-05:00")
    assert parsed_offset == date(2025, 11, 5)
    parsed_dt = utc_date_from_value(datetime(2025, 11, 5, 23, 0, tzinfo=timezone.utc))
    assert parsed_dt == date(2025, 11, 5)


def test_unique_utc_game_dates_sorted_and_deduped():
    games = [
        {"startDate": "2025-11-04T23:00:00Z"},
        {"start_date": "2025-11-04T01:00:00Z"},
        {"startDate": "2025-11-06T18:00:00Z"},
    ]
    assert unique_utc_game_dates(games) == [date(2025, 11, 4), date(2025, 11, 6)]


def test_split_range_and_season_chunks():
    left, right = split_range(date(2025, 11, 1), date(2025, 11, 4))
    assert left == (date(2025, 11, 1), date(2025, 11, 2))
    assert right == (date(2025, 11, 3), date(2025, 11, 4))
    chunks = season_game_chunks(2026, chunk_days=14)
    assert chunks[0][0] == date(2025, 10, 1)
    assert chunks[-1][1] == date(2026, 4, 20)
