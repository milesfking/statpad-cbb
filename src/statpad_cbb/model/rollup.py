from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def player_ratings_frame(
    *,
    player_ids: list[int],
    off: np.ndarray,
    defense: np.ndarray,
    features: pd.DataFrame | None = None,
    player_conference: dict[int, str] | None = None,
    player_team: dict[int, int] | None = None,
    off_raw: np.ndarray | None = None,
    def_raw: np.ndarray | None = None,
    conf_off: np.ndarray | None = None,
    conf_def: np.ndarray | None = None,
    conference_names: list[str] | None = None,
    team_off: np.ndarray | None = None,
    team_def: np.ndarray | None = None,
    team_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Convert coefficients to points-per-100 ratings."""
    conf_index = {name: i for i, name in enumerate(conference_names or [])}
    team_index = {tid: i for i, tid in enumerate(team_ids or [])}
    rows = []
    for i, pid in enumerate(player_ids):
        conf = (player_conference or {}).get(int(pid))
        tid = (player_team or {}).get(int(pid))
        row = {
            "player_id": int(pid),
            "conference": conf,
            "off_rating": float(off[i] * 100.0),
            "def_rating": float(defense[i] * 100.0),
            "net_rating": float((off[i] + defense[i]) * 100.0),
        }
        if off_raw is not None and def_raw is not None:
            row["off_player_effect"] = float(off_raw[i] * 100.0)
            row["def_player_effect"] = float(def_raw[i] * 100.0)
        if conf is not None and conf_off is not None and conf_def is not None and conf in conf_index:
            j = conf_index[conf]
            row["off_conference_effect"] = float(conf_off[j] * 100.0)
            row["def_conference_effect"] = float(conf_def[j] * 100.0)
        if tid is not None and team_off is not None and team_def is not None and tid in team_index:
            j = team_index[tid]
            row["off_team_effect"] = float(team_off[j] * 100.0)
            row["def_team_effect"] = float(team_def[j] * 100.0)
        rows.append(row)
    ratings = pd.DataFrame(rows)
    if features is not None and not features.empty:
        cols = [
            "player_id",
            "team_id",
            "name",
            "minutes",
            "off_possessions",
            "def_possessions",
            "possessions",
            "minutes_share",
        ]
        present = [c for c in cols if c in features.columns]
        ratings = ratings.merge(features[present], on="player_id", how="left")
    return ratings


def team_rollup(
    ratings: pd.DataFrame,
    possessions: pd.DataFrame,
    games: pd.DataFrame,
    *,
    team_lookup: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Lineup-scale team ratings vs observed efficiency margin.

    With once-per-possession team/conf encoding, a five-man unit contributes
    ``5 * mean(player_effect) + team_effect + conference_effect`` in PPP.
    Ratings are reported per 100 possessions. Falls back to minutes-weighted
    player totals when effect components are unavailable.
    """
    if ratings.empty:
        return pd.DataFrame()

    # Observed team offensive/defensive efficiency from possessions.
    off_eff = (
        possessions.groupby("off_team_id", as_index=False)
        .agg(points=("points", "sum"), off_poss=("points", "count"))
        .rename(columns={"off_team_id": "team_id"})
    )
    off_eff["off_eff"] = 100.0 * off_eff["points"] / off_eff["off_poss"]
    def_eff = (
        possessions.groupby("def_team_id", as_index=False)
        .agg(points_allowed=("points", "sum"), def_poss=("points", "count"))
        .rename(columns={"def_team_id": "team_id"})
    )
    def_eff["def_eff"] = 100.0 * def_eff["points_allowed"] / def_eff["def_poss"]
    observed = off_eff.merge(def_eff, on="team_id", how="outer")
    observed["eff_margin"] = observed["off_eff"] - observed["def_eff"]

    rated = ratings.dropna(subset=["team_id"]).copy()
    rated["team_id"] = rated["team_id"].astype(int)
    rated["minutes"] = rated["minutes"].fillna(0.0)
    has_components = {
        "off_player_effect",
        "def_player_effect",
    }.issubset(rated.columns)

    def weighted_avg(group: pd.DataFrame, col: str) -> float:
        if col not in group.columns:
            return 0.0
        w = group["minutes"].to_numpy()
        vals = group[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            return 0.0
        w = w[mask]
        vals = vals[mask]
        if w.sum() <= 0:
            return float(np.mean(vals))
        return float(np.average(vals, weights=w))

    def unit_side(group: pd.DataFrame, side: str) -> float:
        """PPP*100 contribution of a typical five-man unit on one side."""
        if has_components:
            player = weighted_avg(group, f"{side}_player_effect")
            team = weighted_avg(group, f"{side}_team_effect")
            conf = weighted_avg(group, f"{side}_conference_effect")
            return 5.0 * player + team + conf
        # Legacy fallback: totals already mixed scales — treat as per-player.
        return 5.0 * weighted_avg(group, f"{side}_rating")

    rows: list[dict[str, Any]] = []
    for team_id, group in rated.groupby("team_id"):
        meta = (team_lookup or {}).get(int(team_id), {})
        off_rating = unit_side(group, "off")
        def_rating = unit_side(group, "def")
        rows.append(
            {
                "team_id": int(team_id),
                "school": meta.get("school"),
                "conference": meta.get("conference"),
                "off_rating": off_rating,
                "def_rating": def_rating,
                "net_rating": off_rating + def_rating,
                "player_minutes": float(group["minutes"].sum()),
                "n_players": int(len(group)),
            }
        )
    teams = pd.DataFrame(rows)
    out = teams.merge(observed, on="team_id", how="left")
    out = out.sort_values("net_rating", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def teams_records(teams: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize all team ratings for metrics / JSON export."""
    if teams.empty:
        return []
    cols = [
        "rank",
        "team_id",
        "school",
        "conference",
        "off_rating",
        "def_rating",
        "net_rating",
        "off_eff",
        "def_eff",
        "eff_margin",
        "off_poss",
        "def_poss",
        "player_minutes",
        "n_players",
    ]
    present = [c for c in cols if c in teams.columns]
    frame = teams[present].copy()
    return json_safe_records(frame)


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def rating_dispersion_by_minutes(ratings: pd.DataFrame) -> dict[str, Any]:
    if ratings.empty or "minutes" not in ratings.columns:
        return {}
    frame = ratings.copy()
    frame["bucket"] = pd.cut(
        frame["minutes"].fillna(0),
        bins=[-0.01, 5, 15, 25, 40, 1000],
        labels=["0-5", "5-15", "15-25", "25-40", "40+"],
    )
    summary = (
        frame.groupby("bucket", observed=False)
        .agg(
            n=("player_id", "count"),
            net_sd=("net_rating", "std"),
            net_mean=("net_rating", "mean"),
            mean_minutes=("minutes", "mean"),
        )
        .reset_index()
    )
    return {
        str(row["bucket"]): {
            "n": int(row["n"]),
            "net_sd": None if pd.isna(row["net_sd"]) else float(row["net_sd"]),
            "net_mean": None if pd.isna(row["net_mean"]) else float(row["net_mean"]),
            "mean_minutes": None if pd.isna(row["mean_minutes"]) else float(row["mean_minutes"]),
        }
        for _, row in summary.iterrows()
    }
