# statpad-cbb

College basketball ranking model. The first implemented piece is a minimal CBBD ingestion pipeline: teams, rosters, games, and play-by-play for one season. Player and team statistics are reconstructed later from plays. Box scores and substitutions are not requested.

## Season convention

Use the season-ending year. `2026` means 2025–26.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put a College Basketball Data API bearer token in `.env` as `CBBD_API_KEY`. The live pull is not run until that key is present.

## Ingest 2025–26

```bash
python -m statpad_cbb ingest --season 2026
```

Re-runs are idempotent: existing raw JSON is reused unless you pass `--force`.

## What is pulled

| Step | Endpoint | Cadence |
| --- | --- | --- |
| 1 | `GET /teams/directory?season={season}` | once per season |
| 2 | `GET /teams/roster?season={season}` | once per season |
| 3 | `GET /games?season={season}&startDateRange={start}&endDateRange={end}` | date-partitioned windows (14 days, split if a response hits 3,000 games) |
| 4 | `GET /plays/date?date={YYYY-MM-DD}` | once per unique UTC game date |

Expected request count: `2 + N_game_date_ranges + N_unique_UTC_game_dates`.

## Layout

```
src/statpad_cbb/ingest/   ingestion client, extractors, coverage
data/raw/cbbd/{season}/   untouched API payloads
data/processed/cbbd/{season}/  canonical teams, players, games, plays (JSONL)
data/coverage/cbbd/{season}/   missing games and missing play-by-play
tests/                    unit tests that do not call the live API
```

Canonical IDs are the CBBD team, player, and game IDs.

## Tests

```bash
pytest
```
