from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statpad_cbb.config import load_settings
from statpad_cbb.ingest.storage import SeasonPaths, read_jsonl

logger = logging.getLogger(__name__)


def _load_teams(paths: SeasonPaths) -> pd.DataFrame:
    if not paths.team_ratings.exists():
        raise FileNotFoundError(f"Missing team ratings at {paths.team_ratings}; run evaluate or fit-rapm")
    teams = pd.read_parquet(paths.team_ratings)
    if "school" not in teams.columns or teams["school"].isna().all():
        lookup = {
            int(row["team_id"]): row.get("school")
            for row in read_jsonl(paths.processed_teams)
            if row.get("team_id") is not None
        }
        teams = teams.copy()
        teams["school"] = teams["team_id"].map(lookup)
    return teams.sort_values("net_rating", ascending=False).reset_index(drop=True)


def _load_players(paths: SeasonPaths) -> pd.DataFrame:
    if not paths.player_ratings.exists():
        raise FileNotFoundError(
            f"Missing player ratings at {paths.player_ratings}; run fit-rapm first"
        )
    players = pd.read_parquet(paths.player_ratings)
    if "school" not in players.columns:
        lookup = {
            int(row["team_id"]): row.get("school")
            for row in read_jsonl(paths.processed_teams)
            if row.get("team_id") is not None
        }
        players = players.copy()
        players["school"] = players["team_id"].map(lookup)
    return players


def _resolve_team(
    teams: pd.DataFrame,
    *,
    team_id: int | None = None,
    school: str | None = None,
) -> pd.Series:
    if team_id is not None:
        hit = teams.loc[teams["team_id"] == int(team_id)]
        if hit.empty:
            raise ValueError(f"No team_id={team_id} in ratings")
        return hit.iloc[0]
    if school is not None:
        needle = school.strip().lower()
        hit = teams.loc[teams["school"].fillna("").str.lower() == needle]
        if hit.empty:
            # fuzzy contains
            hit = teams.loc[teams["school"].fillna("").str.lower().str.contains(needle, regex=False)]
        if hit.empty:
            raise ValueError(f"No school matching {school!r}")
        if len(hit) > 1:
            names = ", ".join(hit["school"].astype(str).head(8).tolist())
            raise ValueError(f"Ambiguous school {school!r}; matches: {names}")
        return hit.iloc[0]
    raise ValueError("Provide team_id or school")


def _resolve_player(
    players: pd.DataFrame,
    *,
    player_id: int | None = None,
    name: str | None = None,
) -> pd.Series:
    if player_id is not None:
        hit = players.loc[players["player_id"] == int(player_id)]
        if hit.empty:
            raise ValueError(f"No player_id={player_id} in ratings")
        return hit.iloc[0]
    if name is not None:
        needle = name.strip().lower()
        hit = players.loc[players["name"].fillna("").str.lower() == needle]
        if hit.empty:
            hit = players.loc[players["name"].fillna("").str.lower().str.contains(needle, regex=False)]
        if hit.empty:
            raise ValueError(f"No player matching {name!r}")
        if len(hit) > 1:
            # Prefer higher-minute matches when ambiguous.
            if "minutes" in hit.columns:
                hit = hit.sort_values("minutes", ascending=False)
            names = ", ".join(
                f"{r['name']} ({r['player_id']}, {r['school']})"
                for _, r in hit.head(5).iterrows()
            )
            if len(hit) > 5:
                raise ValueError(f"Ambiguous player {name!r}; top matches: {names}")
            # If few matches and user used a partial, take the top minutes one but warn via log.
            logger.warning("Multiple players matched %r; using %s", name, hit.iloc[0]["name"])
        return hit.iloc[0]
    raise ValueError("Provide player_id or name")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_teams(
    season: int,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = 40,
) -> dict[str, Path]:
    """Team EDA: net ranking bars, off/def scatter, ratings vs observed margin."""
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    out_dir = output_dir or paths.eda_dir
    teams = _load_teams(paths)

    written: dict[str, Path] = {}

    # 1) Top-N net rating bars
    top = teams.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.22)))
    ax.barh(top["school"].astype(str), top["net_rating"], color="#1f4e79")
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Net rating (pts / 100 poss)")
    ax.set_title(f"{season} team net ratings (top {min(top_n, len(teams))})")
    written["team_net_bars"] = _save(fig, out_dir / "team_net_bars.png")

    # 2) Offense vs defense scatter
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(teams["off_rating"], teams["def_rating"], s=28, alpha=0.75, c="#1f4e79")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.axhline(0, color="#999999", linewidth=0.8)
    # Label extremes
    label_ids = set(teams.head(8)["team_id"]).union(set(teams.tail(5)["team_id"]))
    for row in teams.itertuples():
        if row.team_id in label_ids:
            ax.annotate(
                str(row.school),
                (row.off_rating, row.def_rating),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_xlabel("Offensive rating")
    ax.set_ylabel("Defensive rating (higher = better)")
    ax.set_title(f"{season} team offense vs defense")
    written["team_off_def"] = _save(fig, out_dir / "team_off_def.png")

    # 3) Model net vs observed efficiency margin
    fig, ax = plt.subplots(figsize=(8, 8))
    valid = teams.dropna(subset=["eff_margin", "net_rating"])
    ax.scatter(valid["eff_margin"], valid["net_rating"], s=28, alpha=0.75, c="#1f4e79")
    if len(valid) >= 2:
        corr = float(valid["net_rating"].corr(valid["eff_margin"]))
        lims = [
            min(valid["eff_margin"].min(), valid["net_rating"].min()),
            max(valid["eff_margin"].max(), valid["net_rating"].max()),
        ]
        ax.plot(lims, lims, color="#aaaaaa", linestyle="--", linewidth=1, label="y = x")
        ax.legend(loc="best")
        ax.set_title(f"{season} team net vs observed eff margin (r={corr:.3f})")
    else:
        ax.set_title(f"{season} team net vs observed eff margin")
    for row in valid.head(8).itertuples():
        ax.annotate(
            str(row.school),
            (row.eff_margin, row.net_rating),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("Observed efficiency margin")
    ax.set_ylabel("Minutes-weighted RAPM net")
    written["team_vs_eff"] = _save(fig, out_dir / "team_vs_eff.png")

    return written


def plot_players_for_team(
    season: int,
    *,
    team_id: int | None = None,
    school: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    min_minutes: float = 8.0,
) -> dict[str, Path]:
    """Roster EDA for one team: off/def scatter and net bars."""
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    out_dir = output_dir or paths.eda_dir
    teams = _load_teams(paths)
    players = _load_players(paths)
    team = _resolve_team(teams, team_id=team_id, school=school)
    roster = players.loc[players["team_id"] == int(team["team_id"])].copy()
    if roster.empty:
        raise ValueError(f"No players found for {team.get('school')}")
    roster = roster.sort_values("net_rating", ascending=False)
    filtered = roster.loc[roster["minutes"].fillna(0) >= min_minutes]
    if filtered.empty:
        filtered = roster

    school_name = str(team.get("school") or team["team_id"])
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in school_name).strip("-")
    written: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(filtered))))
    plot_df = filtered.iloc[::-1]
    ax.barh(plot_df["name"].astype(str), plot_df["net_rating"], color="#1f4e79")
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Net rating (pts / 100 poss)")
    ax.set_title(f"{school_name} player net ratings (≥ {min_minutes:.0f} min)")
    written["roster_net"] = _save(fig, out_dir / f"players_{slug}_net.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = 40 + 8 * filtered["minutes"].fillna(10).to_numpy()
    ax.scatter(filtered["off_rating"], filtered["def_rating"], s=sizes, alpha=0.8, c="#1f4e79")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.axhline(0, color="#999999", linewidth=0.8)
    for _, row in filtered.iterrows():
        ax.annotate(
            str(row["name"]),
            (row["off_rating"], row["def_rating"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("Offensive rating")
    ax.set_ylabel("Defensive rating (higher = better)")
    ax.set_title(f"{school_name} players: offense vs defense")
    written["roster_off_def"] = _save(fig, out_dir / f"players_{slug}_off_def.png")
    return written


def plot_player(
    season: int,
    *,
    player_id: int | None = None,
    name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    peer_minutes: float = 15.0,
) -> dict[str, Path]:
    """Single-player EDA against peers: off/def location and rating bars."""
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    out_dir = output_dir or paths.eda_dir
    players = _load_players(paths)
    player = _resolve_player(players, player_id=player_id, name=name)

    peers = players.loc[players["minutes"].fillna(0) >= peer_minutes].copy()
    if peers.empty:
        peers = players.copy()

    label = str(player.get("name") or player["player_id"])
    school = str(player.get("school") or "")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in label).strip("-")
    written: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(peers["off_rating"], peers["def_rating"], s=18, alpha=0.25, c="#7a7a7a", label="peers")
    ax.scatter(
        [player["off_rating"]],
        [player["def_rating"]],
        s=120,
        c="#c45c26",
        zorder=5,
        label=label,
    )
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.annotate(
        f"{label}\n{school}",
        (player["off_rating"], player["def_rating"]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color="#c45c26",
    )
    ax.set_xlabel("Offensive rating")
    ax.set_ylabel("Defensive rating (higher = better)")
    ax.set_title(f"{label} vs peers (≥ {peer_minutes:.0f} min)")
    ax.legend(loc="best")
    written["player_off_def"] = _save(fig, out_dir / f"player_{slug}_off_def.png")

    # Component bars for the player
    fig, ax = plt.subplots(figsize=(6, 4))
    components = {
        "Off": float(player["off_rating"]),
        "Def": float(player["def_rating"]),
        "Net": float(player["net_rating"]),
    }
    colors = ["#1f4e79" if v >= 0 else "#8b3a3a" for v in components.values()]
    ax.bar(list(components), list(components.values()), color=colors)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_ylabel("Points / 100 possessions")
    minutes = player.get("minutes")
    minutes_txt = f", {minutes:.1f} min" if isinstance(minutes, (int, float)) and minutes == minutes else ""
    ax.set_title(f"{label} ({school}){minutes_txt}")
    written["player_components"] = _save(fig, out_dir / f"player_{slug}_components.png")
    return written


def run_eda(
    season: int,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    team_id: int | None = None,
    school: str | None = None,
    player_id: int | None = None,
    player_name: str | None = None,
    top_n: int = 40,
    min_minutes: float = 8.0,
) -> dict[str, Any]:
    """Generate team plots and optional team-roster / player plots."""
    written: dict[str, Path] = {}
    errors: list[str] = []
    written.update(plot_teams(season, data_dir=data_dir, output_dir=output_dir, top_n=top_n))
    if team_id is not None or school is not None:
        try:
            written.update(
                plot_players_for_team(
                    season,
                    team_id=team_id,
                    school=school,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    min_minutes=min_minutes,
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
            logger.error("%s", exc)
    if player_id is not None or player_name is not None:
        try:
            written.update(
                plot_player(
                    season,
                    player_id=player_id,
                    name=player_name,
                    data_dir=data_dir,
                    output_dir=output_dir,
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
            logger.error("%s", exc)
    result: dict[str, Any] = {key: str(path) for key, path in written.items()}
    if errors:
        result["errors"] = errors
    return result
