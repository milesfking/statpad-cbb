from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
LIST_ENVELOPE_KEYS = (
    "teams",
    "games",
    "plays",
    "players",
    "roster",
    "data",
    "items",
    "results",
)


def unwrap_list(payload: Any, *, path: str = "response") -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in LIST_ENVELOPE_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise RuntimeError(f"Expected a JSON list from {path}, got {type(payload).__name__}")


class CbbdApi(Protocol):
    def teams_directory(self, season: int) -> Any: ...
    def teams_roster(self, season: int) -> list[Any]: ...
    def games(self, season: int, start: date, end: date) -> list[Any]: ...
    def plays_date(self, game_date: date) -> list[Any]: ...


class CbbdClient:
    """Thin authenticated client for the College Basketball Data API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        min_request_interval_seconds: float = 0.5,
        timeout: httpx.Timeout | float = httpx.Timeout(30.0, read=120.0),
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("CBBD_API_KEY is required to call the API")
        self._min_interval = min_request_interval_seconds
        self._max_attempts = max_attempts
        self._last_request_at = 0.0
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "statpad-cbb/0.1",
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CbbdClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def teams_directory(self, season: int) -> Any:
        """Raw directory payload (object with teams/conferences, or a list)."""
        return self._request_json("/teams/directory", {"season": season})

    def teams_roster(self, season: int) -> list[Any]:
        return self._get_list("/teams/roster", {"season": season})

    def games(self, season: int, start: date, end: date) -> list[Any]:
        return self._get_list(
            "/games",
            {
                "season": season,
                "startDateRange": start.isoformat(),
                "endDateRange": end.isoformat(),
            },
        )

    def plays_date(self, game_date: date) -> list[Any]:
        return self._get_list("/plays/date", {"date": game_date.isoformat()})

    def _get_list(self, path: str, params: dict[str, Any]) -> list[Any]:
        return unwrap_list(self._request_json(path, params), path=path)

    def _request_json(self, path: str, params: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            self._wait_for_rate_limit()
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if response.status_code in RETRY_STATUSES:
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after and retry_after.isdigit() else None
                logger.warning(
                    "CBBD %s %s returned %s (attempt %s/%s)",
                    path,
                    params,
                    response.status_code,
                    attempt,
                    self._max_attempts,
                )
                self._backoff(attempt, wait_s)
                last_error = httpx.HTTPStatusError(
                    f"{response.status_code} for {path}",
                    request=response.request,
                    response=response,
                )
                continue

            if response.status_code == 404:
                logger.info("CBBD %s %s returned 404; treating as empty", path, params)
                return []

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.error("CBBD %s failed: %s %s", path, response.status_code, response.text[:500])
                raise

            if not response.content:
                return []
            return response.json()

        assert last_error is not None
        raise last_error

    def _wait_for_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = retry_after if retry_after is not None else min(2 ** attempt, 16)
        time.sleep(delay)
