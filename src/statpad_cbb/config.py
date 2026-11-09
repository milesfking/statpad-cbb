from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.collegebasketballdata.com"
DEFAULT_SEASON = 2026  # 2025–26, season-ending year convention
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.5
GAMES_RESPONSE_LIMIT = 3000
DEFAULT_GAME_CHUNK_DAYS = 14


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    data_dir: Path
    min_request_interval_seconds: float
    default_season: int


def load_settings(data_dir: Path | None = None) -> Settings:
    load_dotenv()
    root = repo_root()
    configured_dir = data_dir or Path(os.getenv("STATPAD_DATA_DIR", "data"))
    resolved_dir = configured_dir if configured_dir.is_absolute() else root / configured_dir
    interval_raw = os.getenv(
        "CBBD_MIN_REQUEST_INTERVAL_SECONDS",
        str(DEFAULT_MIN_REQUEST_INTERVAL_SECONDS),
    )
    api_key = os.getenv("CBBD_API_KEY") or None
    return Settings(
        api_key=api_key,
        base_url=os.getenv("CBBD_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        data_dir=resolved_dir,
        min_request_interval_seconds=float(interval_raw),
        default_season=DEFAULT_SEASON,
    )
