from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from statpad_cbb.possessions.lineups import (
    apply_substitution,
    lineup_sides,
    parse_on_floor,
)

IGNORE_TYPES = {
    "Substitution",
    "OfficialTVTimeOut",
    "ShortTimeOut",
    "RegularTimeOut",
    "Coach's Challenge Accepted",
    "Coach's Challenge Overturned",
    "Coach's Challenge Rejected",
    "Coach's Challenge Stands",
}

FG_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot", "Shot"}
FT_TYPE = "MadeFreeThrow"
TO_TYPE = "Lost Ball Turnover"
DREB_TYPE = "Defensive Rebound"
OREB_TYPE = "Offensive Rebound"
DEAD_REB_TYPE = "Dead Ball Rebound"
STEAL_TYPE = "Steal"
FOUL_TYPES = {"PersonalFoul", "Technical Foul"}
END_TYPES = {"End Period", "End Game"}
JUMPBALL_TYPE = "Jumpball"

FT_TRIP_RE = re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE)


@dataclass
class GameContext:
    game_id: int
    season: int
    home_team_id: int
    away_team_id: int
    home_school: str
    away_school: str
    neutral_site: bool


@dataclass
class LineupState:
    home_ids: set[int] = field(default_factory=set)
    away_ids: set[int] = field(default_factory=set)
    source: str = "none"

    def snapshot(
        self,
        *,
        offense_team_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> tuple[list[int], list[int]] | None:
        return lineup_sides(
            self.home_ids,
            self.away_ids,
            offense_team_id=offense_team_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )


@dataclass
class OpenPossession:
    offense_team_id: int
    start_home_score: int
    start_away_score: int
    period: int
    off_players: list[int]
    def_players: list[int]
    lineup_source: str


def other_team(team_id: int, ctx: GameContext) -> int:
    if team_id == ctx.home_team_id:
        return ctx.away_team_id
    return ctx.home_team_id


def score_pair(play: dict[str, Any]) -> tuple[int, int]:
    return int(play.get("homeScore") or 0), int(play.get("awayScore") or 0)


def offense_points(
    ctx: GameContext,
    start: tuple[int, int],
    end: tuple[int, int],
    offense_team_id: int,
) -> int:
    sh, sa = start
    eh, ea = end
    if offense_team_id == ctx.home_team_id:
        return max(0, eh - sh)
    return max(0, ea - sa)


def parse_ft_trip(play: dict[str, Any]) -> tuple[int, int] | None:
    text = str(play.get("playText") or "")
    match = FT_TRIP_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_made_fg(play: dict[str, Any]) -> bool:
    if play.get("playType") not in FG_TYPES:
        return False
    shot = play.get("shotInfo") or {}
    if shot.get("made") is True:
        return True
    return bool(play.get("scoringPlay"))


def is_missed_fg(play: dict[str, Any]) -> bool:
    if play.get("playType") not in FG_TYPES:
        return False
    shot = play.get("shotInfo") or {}
    if shot.get("made") is False:
        return True
    return not bool(play.get("scoringPlay"))


def is_made_ft(play: dict[str, Any]) -> bool:
    if play.get("playType") != FT_TYPE:
        return False
    shot = play.get("shotInfo") or {}
    if shot.get("made") is not None:
        return bool(shot.get("made"))
    text = str(play.get("playText") or "").lower()
    return "miss" not in text


def sort_key(play: dict[str, Any]) -> tuple[Any, ...]:
    period = play.get("period") or 0
    seconds = play.get("secondsRemaining")
    if seconds is None:
        seconds = -1
    play_id = play.get("play_id") or play.get("id") or 0
    return (int(period), -int(seconds), str(play_id))


def update_lineup_from_play(play: dict[str, Any], state: LineupState, ctx: GameContext) -> None:
    play_type = play.get("playType")
    if play_type == "Substitution":
        if state.home_ids or state.away_ids:
            apply_substitution(
                play,
                state.home_ids,
                state.away_ids,
                home_team_id=ctx.home_team_id,
                away_team_id=ctx.away_team_id,
            )
            state.source = "subs"
        return

    parsed = parse_on_floor(
        play,
        home_team_id=ctx.home_team_id,
        away_team_id=ctx.away_team_id,
        home_school=ctx.home_school,
        away_school=ctx.away_school,
    )
    if parsed is None:
        return
    home_ids, away_ids = parsed
    state.home_ids = set(home_ids)
    state.away_ids = set(away_ids)
    state.source = "onFloor"


def emit_possession(
    *,
    ctx: GameContext,
    open_poss: OpenPossession,
    end_play: dict[str, Any],
    end_reason: str,
    poss_index: int,
) -> dict[str, Any]:
    end_scores = score_pair(end_play)
    points = offense_points(
        ctx,
        (open_poss.start_home_score, open_poss.start_away_score),
        end_scores,
        open_poss.offense_team_id,
    )
    home_offense = 0 if ctx.neutral_site else int(open_poss.offense_team_id == ctx.home_team_id)
    return {
        "season": ctx.season,
        "game_id": ctx.game_id,
        "poss_id": f"{ctx.game_id}-{poss_index}",
        "poss_index": poss_index,
        "period": open_poss.period,
        "off_team_id": open_poss.offense_team_id,
        "def_team_id": other_team(open_poss.offense_team_id, ctx),
        "off_player_1": open_poss.off_players[0],
        "off_player_2": open_poss.off_players[1],
        "off_player_3": open_poss.off_players[2],
        "off_player_4": open_poss.off_players[3],
        "off_player_5": open_poss.off_players[4],
        "def_player_1": open_poss.def_players[0],
        "def_player_2": open_poss.def_players[1],
        "def_player_3": open_poss.def_players[2],
        "def_player_4": open_poss.def_players[3],
        "def_player_5": open_poss.def_players[4],
        "points": int(points),
        "home_offense": int(home_offense),
        "neutral_site": bool(ctx.neutral_site),
        "end_reason": end_reason,
        "lineup_source": open_poss.lineup_source,
        "valid_lineup": True,
    }


def try_open(
    *,
    ctx: GameContext,
    lineup: LineupState,
    offense_team_id: int,
    start_scores: tuple[int, int],
    play: dict[str, Any],
) -> OpenPossession | None:
    sides = lineup.snapshot(
        offense_team_id=offense_team_id,
        home_team_id=ctx.home_team_id,
        away_team_id=ctx.away_team_id,
    )
    if sides is None:
        # Recover from the current play's onFloor when sub tracking drifted.
        parsed = parse_on_floor(
            play,
            home_team_id=ctx.home_team_id,
            away_team_id=ctx.away_team_id,
            home_school=ctx.home_school,
            away_school=ctx.away_school,
        )
        if parsed is not None:
            home_ids, away_ids = parsed
            lineup.home_ids = set(home_ids)
            lineup.away_ids = set(away_ids)
            lineup.source = "onFloor"
            sides = lineup.snapshot(
                offense_team_id=offense_team_id,
                home_team_id=ctx.home_team_id,
                away_team_id=ctx.away_team_id,
            )
    if sides is None:
        return None
    off_players, def_players = sides
    return OpenPossession(
        offense_team_id=offense_team_id,
        start_home_score=start_scores[0],
        start_away_score=start_scores[1],
        period=int(play.get("period") or 1),
        off_players=off_players,
        def_players=def_players,
        lineup_source=lineup.source,
    )


def _lookahead_and_one(ordered: list[dict[str, Any]], start_idx: int, team_id: int) -> bool:
    j = start_idx + 1
    while j < len(ordered):
        nxt = ordered[j]
        nxt_type = str(nxt.get("playType") or "")
        if nxt_type in IGNORE_TYPES or nxt_type in FOUL_TYPES:
            j += 1
            continue
        return nxt_type == FT_TYPE and nxt.get("teamId") == team_id
    return False


def build_game_possessions(
    ctx: GameContext,
    plays: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Segment one game into possession rows with valid 5v5 lineups."""
    ordered = sorted(plays, key=sort_key)
    lineup = LineupState()
    open_poss: OpenPossession | None = None
    waiting_rebound = False
    pending_and_one = False
    and_one_fg_play: dict[str, Any] | None = None
    pre_score = (0, 0)
    possessions: list[dict[str, Any]] = []
    stats = {
        "plays": len(ordered),
        "possessions_raw": 0,
        "possessions_valid": 0,
        "skipped_bad_open": 0,
    }

    def close(end_play: dict[str, Any], reason: str) -> None:
        nonlocal open_poss, waiting_rebound, pending_and_one, and_one_fg_play
        if open_poss is None:
            waiting_rebound = False
            pending_and_one = False
            and_one_fg_play = None
            return
        stats["possessions_raw"] += 1
        possessions.append(
            emit_possession(
                ctx=ctx,
                open_poss=open_poss,
                end_play=end_play,
                end_reason=reason,
                poss_index=stats["possessions_raw"],
            )
        )
        stats["possessions_valid"] += 1
        open_poss = None
        waiting_rebound = False
        pending_and_one = False
        and_one_fg_play = None

    def open_for(team_id: int, play: dict[str, Any], start_scores: tuple[int, int] | None = None) -> None:
        nonlocal open_poss
        opened = try_open(
            ctx=ctx,
            lineup=lineup,
            offense_team_id=team_id,
            start_scores=start_scores if start_scores is not None else pre_score,
            play=play,
        )
        if opened is None:
            stats["skipped_bad_open"] += 1
            open_poss = None
            return
        open_poss = opened

    def ensure_offense(team_id: int, play: dict[str, Any]) -> None:
        if open_poss is None:
            open_for(team_id, play, start_scores=pre_score)
        elif open_poss.offense_team_id != team_id:
            close(play, "possession_mismatch")
            open_for(team_id, play, start_scores=score_pair(play))

    i = 0
    while i < len(ordered):
        play = ordered[i]
        update_lineup_from_play(play, lineup, ctx)
        play_type = str(play.get("playType") or "")
        team_id = play.get("teamId")
        team_id_int = int(team_id) if team_id is not None else None
        current_score = score_pair(play)

        if play_type in IGNORE_TYPES:
            i += 1
            continue

        if play_type in END_TYPES:
            if open_poss is not None:
                close(play, "period_end")
            waiting_rebound = False
            pending_and_one = False
            and_one_fg_play = None
            pre_score = current_score
            i += 1
            continue

        if play_type == JUMPBALL_TYPE:
            pre_score = current_score
            i += 1
            continue

        if play_type == FT_TYPE and team_id_int is not None:
            if pending_and_one and open_poss is not None and open_poss.offense_team_id == team_id_int:
                pending_and_one = False
                and_one_fg_play = None
            elif open_poss is None or open_poss.offense_team_id != team_id_int:
                if open_poss is not None:
                    close(play, "pre_ft")
                open_for(team_id_int, play, start_scores=pre_score)
                pending_and_one = False
                and_one_fg_play = None

            trip = parse_ft_trip(play)
            made = is_made_ft(play)
            end_play = play
            if trip is None:
                j = i + 1
                while j < len(ordered):
                    nxt = ordered[j]
                    update_lineup_from_play(nxt, lineup, ctx)
                    nxt_type = str(nxt.get("playType") or "")
                    if nxt_type in IGNORE_TYPES or nxt_type == DEAD_REB_TYPE:
                        j += 1
                        continue
                    if nxt_type == FT_TYPE and nxt.get("teamId") == team_id_int:
                        end_play = nxt
                        made = is_made_ft(nxt)
                        i = j
                        j += 1
                        continue
                    break
                end_score = score_pair(end_play)
                close(end_play, "ft_trip")
                if made:
                    open_for(other_team(team_id_int, ctx), end_play, start_scores=end_score)
                else:
                    waiting_rebound = True
                pre_score = end_score
            elif trip[0] == trip[1]:
                end_score = score_pair(end_play)
                close(end_play, "ft_trip")
                if made:
                    open_for(other_team(team_id_int, ctx), end_play, start_scores=end_score)
                else:
                    waiting_rebound = True
                pre_score = end_score
            else:
                pre_score = current_score
            i += 1
            continue

        if pending_and_one and open_poss is not None and and_one_fg_play is not None:
            fg_play = and_one_fg_play
            fg_team = fg_play.get("teamId")
            fg_score = score_pair(fg_play)
            close(fg_play, "made_fg")
            if fg_team is not None:
                open_for(other_team(int(fg_team), ctx), fg_play, start_scores=fg_score)

        if play_type in FG_TYPES and team_id_int is not None:
            ensure_offense(team_id_int, play)
            if is_made_fg(play):
                if _lookahead_and_one(ordered, i, team_id_int):
                    pending_and_one = True
                    and_one_fg_play = play
                else:
                    close(play, "made_fg")
                    open_for(other_team(team_id_int, ctx), play, start_scores=current_score)
            elif is_missed_fg(play):
                waiting_rebound = True
            pre_score = current_score
            i += 1
            continue

        if play_type == OREB_TYPE and team_id_int is not None:
            ensure_offense(team_id_int, play)
            waiting_rebound = False
            pre_score = current_score
            i += 1
            continue

        if play_type == DREB_TYPE and team_id_int is not None:
            if open_poss is not None and open_poss.offense_team_id != team_id_int:
                close(play, "def_reb")
            waiting_rebound = False
            if open_poss is None:
                open_for(team_id_int, play, start_scores=current_score)
            pre_score = current_score
            i += 1
            continue

        if play_type == DEAD_REB_TYPE and waiting_rebound and team_id_int is not None:
            if open_poss is not None and open_poss.offense_team_id != team_id_int:
                close(play, "dead_ball_reb")
                open_for(team_id_int, play, start_scores=current_score)
            elif open_poss is None:
                open_for(team_id_int, play, start_scores=current_score)
            waiting_rebound = False
            pre_score = current_score
            i += 1
            continue

        if play_type == TO_TYPE and team_id_int is not None:
            ensure_offense(team_id_int, play)
            close(play, "turnover")
            open_for(other_team(team_id_int, ctx), play, start_scores=current_score)
            pre_score = current_score
            i += 1
            continue

        if play_type == STEAL_TYPE and team_id_int is not None:
            if open_poss is not None and open_poss.offense_team_id != team_id_int:
                close(play, "steal")
            if open_poss is None:
                open_for(team_id_int, play, start_scores=current_score)
            pre_score = current_score
            i += 1
            continue

        pre_score = current_score
        i += 1

    if open_poss is not None and ordered:
        close(ordered[-1], "game_end")

    return possessions, stats


def iter_play_groups(plays: Iterator[dict[str, Any]]) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Group a game-sorted play stream into (game_id, plays)."""
    current_id: int | None = None
    bucket: list[dict[str, Any]] = []
    for play in plays:
        game_id = play.get("game_id") or play.get("gameId")
        if game_id is None:
            continue
        game_id = int(game_id)
        if current_id is None:
            current_id = game_id
        if game_id != current_id:
            yield current_id, bucket
            current_id = game_id
            bucket = []
        bucket.append(play)
    if current_id is not None and bucket:
        yield current_id, bucket
