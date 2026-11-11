from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator

from statpad_cbb.possessions.builder import FG_TYPES, FT_TYPE


def _player_id(participant: dict[str, Any] | None) -> int | None:
    if not isinstance(participant, dict):
        return None
    value = participant.get("id")
    return int(value) if value is not None else None


def accumulate_box_from_play(play: dict[str, Any], box: dict[int, dict[str, float]]) -> None:
    """Update per-player raw counting stats from one play."""
    play_type = str(play.get("playType") or "")
    shot = play.get("shotInfo") or {}
    participants = play.get("participants") or []

    def row(pid: int) -> dict[str, float]:
        if pid not in box:
            box[pid] = defaultdict(float)
        return box[pid]

    if play_type in FG_TYPES:
        shooter = _player_id(shot.get("shooter")) or (
            _player_id(participants[0]) if participants else None
        )
        if shooter is None:
            return
        stats = row(shooter)
        made = shot.get("made")
        if made is None:
            made = bool(play.get("scoringPlay"))
        rng = str(shot.get("range") or "")
        if rng == "three_pointer":
            stats["fg3a"] += 1
            stats["fg3m"] += 1 if made else 0
        else:
            stats["fg2a"] += 1
            stats["fg2m"] += 1 if made else 0
        if made and shot.get("assisted"):
            assister = _player_id(shot.get("assistedBy"))
            if assister is None and len(participants) > 1:
                assister = _player_id(participants[1])
            if assister is not None:
                row(assister)["ast"] += 1
        return

    if play_type == FT_TYPE:
        shooter = _player_id(shot.get("shooter")) or (
            _player_id(participants[0]) if participants else None
        )
        if shooter is None:
            return
        stats = row(shooter)
        stats["fta"] += 1
        made = shot.get("made")
        if made is None:
            text = str(play.get("playText") or "").lower()
            made = "miss" not in text
        stats["ftm"] += 1 if made else 0
        return

    if play_type == "Lost Ball Turnover":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["tov"] += 1
        return

    if play_type == "Offensive Rebound":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["orb"] += 1
        return

    if play_type == "Defensive Rebound":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["drb"] += 1
        return

    if play_type == "Steal":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["stl"] += 1
        return

    if play_type == "Block Shot":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["blk"] += 1
        return

    if play_type == "PersonalFoul":
        pid = _player_id(participants[0]) if participants else None
        if pid is not None:
            row(pid)["pf"] += 1


def minutes_from_possessions(possessions) -> dict[int, dict[str, float]]:
    """Approximate player minutes/possessions from possession table rows.

    Each possession counts as one offensive and one defensive possession for
    the ten on-court players. Minutes use a simple  sec/poss heuristic later
    replaced by clock-based estimates when available; here we track possession
    counts and a minutes proxy = possessions * (40 game-minutes / team poss).
    """
    off_poss: dict[int, float] = defaultdict(float)
    def_poss: dict[int, float] = defaultdict(float)
    team_off_poss: dict[int, float] = defaultdict(float)
    player_team: dict[int, int] = {}

    off_cols = [f"off_player_{i}" for i in range(1, 6)]
    def_cols = [f"def_player_{i}" for i in range(1, 6)]

    for row in possessions:
        off_team = int(row["off_team_id"])
        def_team = int(row["def_team_id"])
        team_off_poss[off_team] += 1
        for col in off_cols:
            pid = int(row[col])
            off_poss[pid] += 1
            player_team[pid] = off_team
        for col in def_cols:
            pid = int(row[col])
            def_poss[pid] += 1
            player_team[pid] = def_team

    out: dict[int, dict[str, float]] = {}
    for pid in set(off_poss) | set(def_poss):
        team_id = player_team.get(pid)
        team_poss = team_off_poss.get(team_id, 0.0) if team_id is not None else 0.0
        poss = off_poss[pid] + def_poss[pid]
        # Rough minutes share: each team possession ~ possession for both sides counted once
        # Player on-court possessions / (2 * team offensive possessions) * 40 minutes.
        minutes = 0.0
        if team_poss > 0:
            minutes = (poss / (2.0 * team_poss)) * 40.0
        out[pid] = {
            "off_possessions": float(off_poss[pid]),
            "def_possessions": float(def_poss[pid]),
            "possessions": float(poss),
            "minutes": float(minutes),
            "team_id": float(team_id) if team_id is not None else float("nan"),
            "team_off_possessions": float(team_poss),
        }
    return out


# Below this, per-possession rates are too noisy / unstable for box priors.
MIN_OFF_POSS_FOR_RATES = 20.0
# Hard caps: one make over a tiny sample must not become 1e8-scale features.
USAGE_RATE_CAP = 1.0
PER_100_RATE_CAP = 100.0


def _capped_rate(value: float, *, cap: float) -> float:
    if value != value:  # NaN
        return value
    if value < 0.0:
        return 0.0
    return float(min(value, cap))


def finalize_player_features(
    box: dict[int, dict[str, float]],
    usage: dict[int, dict[str, float]],
    *,
    season: int,
    roster: dict[int, dict[str, Any]],
    min_off_possessions: float = MIN_OFF_POSS_FOR_RATES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, mins in usage.items():
        stats = box.get(pid, {})
        off_poss = float(mins.get("off_possessions", 0.0) or 0.0)
        fg2a = float(stats.get("fg2a", 0.0))
        fg2m = float(stats.get("fg2m", 0.0))
        fg3a = float(stats.get("fg3a", 0.0))
        fg3m = float(stats.get("fg3m", 0.0))
        fta = float(stats.get("fta", 0.0))
        ftm = float(stats.get("ftm", 0.0))
        # Tiny samples: leave rates missing so standardization falls back to the mean
        # (player still gets team/conf RAPM; box mean contribution is ~0).
        if off_poss < min_off_possessions:
            usage_rate = float("nan")
            scale = float("nan")
        else:
            scale = 100.0 / off_poss
            usage_rate = _capped_rate(
                (fg2a + fg3a + 0.44 * fta + float(stats.get("tov", 0.0))) / off_poss,
                cap=USAGE_RATE_CAP,
            )
        team_id = mins.get("team_id")
        roster_row = roster.get(pid, {})

        def per_100(count: float) -> float:
            if scale != scale:  # NaN
                return float("nan")
            return _capped_rate(float(count) * scale, cap=PER_100_RATE_CAP)

        rows.append(
            {
                "season": season,
                "player_id": pid,
                "team_id": int(team_id) if team_id == team_id else roster_row.get("team_id"),
                "name": roster_row.get("name"),
                "minutes": mins.get("minutes", 0.0),
                "off_possessions": mins.get("off_possessions", 0.0),
                "def_possessions": mins.get("def_possessions", 0.0),
                "possessions": mins.get("possessions", 0.0),
                "minutes_share": (
                    mins.get("minutes", 0.0) / 40.0 if mins.get("team_off_possessions", 0) else 0.0
                ),
                "usage": usage_rate,
                "fg2a_100": per_100(fg2a),
                "fg2m_100": per_100(fg2m),
                "fg3a_100": per_100(fg3a),
                "fg3m_100": per_100(fg3m),
                "fta_100": per_100(fta),
                "ftm_100": per_100(ftm),
                "ast_100": per_100(float(stats.get("ast", 0.0))),
                "tov_100": per_100(float(stats.get("tov", 0.0))),
                "orb_100": per_100(float(stats.get("orb", 0.0))),
                "drb_100": per_100(float(stats.get("drb", 0.0))),
                "stl_100": per_100(float(stats.get("stl", 0.0))),
                "blk_100": per_100(float(stats.get("blk", 0.0))),
                "pf_100": per_100(float(stats.get("pf", 0.0))),
            }
        )
    return rows


def iter_relevant_plays(
    plays: Iterator[dict[str, Any]],
    valid_game_ids: set[int],
) -> Iterator[dict[str, Any]]:
    for play in plays:
        game_id = play.get("game_id") or play.get("gameId")
        if game_id is None:
            continue
        if int(game_id) in valid_game_ids:
            yield play
