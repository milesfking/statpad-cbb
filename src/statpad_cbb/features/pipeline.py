from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from statpad_cbb.config import load_settings
from statpad_cbb.features.box import (
    accumulate_box_from_play,
    finalize_player_features,
    iter_relevant_plays,
    minutes_from_possessions,
)
from statpad_cbb.ingest.storage import SeasonPaths, read_jsonl

logger = logging.getLogger(__name__)


def build_player_features_frame(
    possessions: pd.DataFrame,
    *,
    season: int,
    roster: dict[int, dict[str, Any]],
    plays: Any | None = None,
    play_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Build player-season features from a possession slice (and optional plays).

    When ``plays`` is omitted, counting rates are left missing and only minutes /
    possession shares are populated — enough for rollup weights without leaking
    post-cut box rates into an early temporal cut.
    """
    if possessions.empty:
        return pd.DataFrame(), 0
    valid_game_ids = play_game_ids or {int(x) for x in possessions["game_id"].unique()}
    box: dict[int, dict[str, float]] = {}
    n_plays = 0
    if plays is not None:
        for play in iter_relevant_plays(plays, valid_game_ids):
            accumulate_box_from_play(play, box)
            n_plays += 1
            if n_plays % 500_000 == 0:
                logger.info("feature box progress: plays=%s players=%s", n_plays, len(box))
    usage = minutes_from_possessions(possessions.to_dict(orient="records"))
    rows = finalize_player_features(box, usage, season=season, roster=roster)
    return pd.DataFrame(rows), n_plays


def build_player_features(
    season: int,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    paths.ensure()

    if not paths.possessions.exists():
        raise FileNotFoundError(
            f"Missing possessions at {paths.possessions}; run build-possessions first"
        )

    possessions = pd.read_parquet(paths.possessions)
    if possessions.empty:
        raise ValueError("Possessions table is empty")

    roster = {
        int(row["player_id"]): row
        for row in read_jsonl(paths.processed_players)
        if row.get("player_id") is not None
    }
    frame, n_plays = build_player_features_frame(
        possessions,
        season=season,
        roster=roster,
        plays=read_jsonl(paths.processed_plays),
    )
    frame.to_parquet(paths.player_features, index=False)

    report = {
        "season": season,
        "players": int(len(frame)),
        "plays_scanned": n_plays,
        "mean_minutes": float(frame["minutes"].mean()) if not frame.empty else None,
        "mean_usage": float(frame["usage"].mean()) if not frame.empty else None,
    }
    logger.info("wrote player features for %s players to %s", report["players"], paths.player_features)
    return report
