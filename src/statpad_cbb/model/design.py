from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


OFF_COLS = [f"off_player_{i}" for i in range(1, 6)]
DEF_COLS = [f"def_player_{i}" for i in range(1, 6)]

# Small, stable box feature set for hierarchical player means.
BOX_FEATURE_COLS = [
    "usage",
    "fg2m_100",
    "fg3m_100",
    "ftm_100",
    "ast_100",
    "tov_100",
    "orb_100",
    "drb_100",
    "stl_100",
    "blk_100",
]


@dataclass
class DesignMatrix:
    x: sparse.csr_matrix
    y: np.ndarray
    game_ids: np.ndarray
    player_ids: list[int]
    n_off: int
    feature_names: list[str]
    conference_names: list[str] = field(default_factory=list)
    team_ids: list[int] = field(default_factory=list)
    n_conf: int = 0
    n_team: int = 0
    n_box: int = 0
    player_conference: dict[int, str] = field(default_factory=dict)
    player_team: dict[int, int] = field(default_factory=dict)
    box_feature_names: list[str] = field(default_factory=list)
    box_mean: dict[str, float] = field(default_factory=dict)
    box_std: dict[str, float] = field(default_factory=dict)
    # Column offsets for blocks after intercept/home.
    player_off0: int = 2
    player_def0: int = 2
    team_off0: int | None = None
    team_def0: int | None = None
    conf_off0: int | None = None
    conf_def0: int | None = None
    box_off0: int | None = None
    box_def0: int | None = None

    @property
    def has_conference(self) -> bool:
        return self.n_conf > 0

    @property
    def has_team(self) -> bool:
        return self.n_team > 0

    @property
    def has_box(self) -> bool:
        return self.n_box > 0


def build_player_maps(
    player_ids: list[int],
    *,
    roster: pd.DataFrame,
    teams: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, int]]:
    """Return (player->conference, player->team_id)."""
    team_conf = {
        int(row.team_id): str(row.conference) if pd.notna(row.conference) else "Unknown"
        for row in teams.itertuples(index=False)
        if pd.notna(row.team_id)
    }
    player_team = {
        int(row.player_id): int(row.team_id)
        for row in roster.itertuples(index=False)
        if pd.notna(row.player_id) and pd.notna(row.team_id)
    }
    conf_map: dict[int, str] = {}
    team_map: dict[int, int] = {}
    for pid in player_ids:
        team_id = player_team.get(int(pid))
        if team_id is None:
            conf_map[int(pid)] = "Unknown"
            continue
        team_map[int(pid)] = team_id
        conf_map[int(pid)] = team_conf.get(team_id, "Unknown")
    return conf_map, team_map


def build_player_conference_map(
    player_ids: list[int],
    *,
    roster: pd.DataFrame,
    teams: pd.DataFrame,
) -> dict[int, str]:
    conf_map, _ = build_player_maps(player_ids, roster=roster, teams=teams)
    return conf_map


def _majority_key(player_ids: np.ndarray, mapping: dict[int, object], default: object) -> object:
    """Fallback group key when possessions lack team ids."""
    counts: dict[object, int] = {}
    for pid in player_ids:
        key = mapping.get(int(pid), default)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _majority_team(player_ids: np.ndarray, team_map: dict[int, int]) -> int:
    """Fallback team id when possessions lack off_team_id / def_team_id."""
    value = _majority_key(player_ids, team_map, default=-1)
    return int(value) if value is not None else -1


# Guard against residual rate explosions after feature finalization.
BOX_Z_CLIP = 5.0
# Prior-season moments from unguarded rates can be pathological; fall back.
_PRIOR_STD_MAX = {
    "usage": 1.0,
    "fg2m_100": 50.0,
    "fg3m_100": 50.0,
    "ftm_100": 50.0,
    "ast_100": 50.0,
    "tov_100": 50.0,
    "orb_100": 50.0,
    "drb_100": 50.0,
    "stl_100": 50.0,
    "blk_100": 50.0,
}


def standardize_box_features(
    features: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    mean: dict[str, float] | None = None,
    std: dict[str, float] | None = None,
    z_clip: float = BOX_Z_CLIP,
) -> tuple[dict[int, np.ndarray], list[str], dict[str, float], dict[str, float]]:
    """Return player_id -> z-scored box vector, plus names/mean/std used."""
    use_cols = [c for c in (cols or BOX_FEATURE_COLS) if c in features.columns]
    if not use_cols or features.empty:
        return {}, [], {}, {}

    frame = features[["player_id", *use_cols]].copy()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    z = pd.DataFrame({"player_id": frame["player_id"]})
    for col in use_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        sample_mu = float(series.mean(skipna=True)) if series.notna().any() else 0.0
        sample_sigma = float(series.std(ddof=0, skipna=True)) if series.notna().sum() > 1 else 1.0
        use_prior = bool(mean and col in mean and std and col in std)
        if use_prior:
            mu = float(mean[col])
            sigma = float(std[col])
            # Pathological prior moments (tiny-sample explosions): fall back.
            prior_cap = _PRIOR_STD_MAX.get(col)
            if prior_cap is not None and (not np.isfinite(sigma) or sigma > prior_cap):
                mu = sample_mu
                sigma = sample_sigma
        else:
            mu = sample_mu
            sigma = sample_sigma
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        means[col] = mu
        stds[col] = sigma
        z_col = ((series.fillna(mu) - mu) / sigma).astype(float)
        if z_clip is not None and z_clip > 0:
            z_col = z_col.clip(lower=-z_clip, upper=z_clip)
        z[col] = z_col

    out: dict[int, np.ndarray] = {}
    for row in z.itertuples(index=False):
        out[int(row.player_id)] = np.asarray([getattr(row, c) for c in use_cols], dtype=np.float64)
    return out, use_cols, means, stds


def build_design_matrix(
    possessions: pd.DataFrame,
    *,
    player_conference: dict[int, str] | None = None,
    player_team: dict[int, int] | None = None,
    include_conference: bool = False,
    include_team: bool = False,
    box_by_player: dict[int, np.ndarray] | None = None,
    box_feature_names: list[str] | None = None,
    box_mean: dict[str, float] | None = None,
    box_std: dict[str, float] | None = None,
) -> DesignMatrix:
    """Build hierarchical RAPM design.

    Layout after intercept/home:
      player off/def,
      optional team off/def,
      optional conference off/def,
      optional box off/def feature sums.

    Team and conference indicators are encoded once per possession (from
    off_team_id / def_team_id), not once per on-court player. That keeps τ and
    κ on the same PPP scale as the intercept and as team unit forecasts.
    """
    if possessions.empty:
        raise ValueError("Cannot build design matrix from empty possessions")

    player_ids = sorted(
        {
            int(pid)
            for col in OFF_COLS + DEF_COLS
            for pid in possessions[col].tolist()
        }
    )
    player_index = {pid: i for i, pid in enumerate(player_ids)}
    n_players = len(player_ids)

    conf_map = player_conference or {}
    team_map = player_team or {}
    team_to_conf = {
        tid: conf_map.get(pid, "Unknown")
        for pid, tid in team_map.items()
    }

    conference_names: list[str] = []
    conf_index: dict[str, int] = {}
    if include_conference:
        conference_names = sorted({conf_map.get(pid, "Unknown") for pid in player_ids})
        conf_index = {name: i for i, name in enumerate(conference_names)}
    n_conf = len(conference_names)

    team_ids: list[int] = []
    team_index: dict[int, int] = {}
    if include_team:
        team_ids = sorted({team_map[pid] for pid in player_ids if pid in team_map})
        team_index = {tid: i for i, tid in enumerate(team_ids)}
    n_team = len(team_ids)

    box_names = list(box_feature_names or [])
    n_box = len(box_names) if box_by_player else 0
    if n_box and box_by_player is None:
        n_box = 0
        box_names = []

    # Column offsets
    player_off0 = 2
    player_def0 = 2 + n_players
    cursor = 2 + 2 * n_players
    team_off0 = team_def0 = None
    if include_team and n_team:
        team_off0 = cursor
        team_def0 = cursor + n_team
        cursor += 2 * n_team
    conf_off0 = conf_def0 = None
    if include_conference and n_conf:
        conf_off0 = cursor
        conf_def0 = cursor + n_conf
        cursor += 2 * n_conf
    box_off0 = box_def0 = None
    if n_box:
        box_off0 = cursor
        box_def0 = cursor + n_box
        cursor += 2 * n_box
    n_cols = cursor

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    off_values = possessions[OFF_COLS].to_numpy(dtype=np.int64)
    def_values = possessions[DEF_COLS].to_numpy(dtype=np.int64)
    home = possessions["home_offense"].to_numpy(dtype=np.float64)
    y = possessions["points"].to_numpy(dtype=np.float64)
    game_ids = possessions["game_id"].to_numpy(dtype=np.int64)
    if "off_team_id" in possessions.columns and "def_team_id" in possessions.columns:
        off_team_ids = possessions["off_team_id"].to_numpy(dtype=np.int64)
        def_team_ids = possessions["def_team_id"].to_numpy(dtype=np.int64)
    else:
        # Fallback for sparse test fixtures: majority team among on-court players.
        off_team_ids = np.array(
            [_majority_team(row, team_map) for row in off_values], dtype=np.int64
        )
        def_team_ids = np.array(
            [_majority_team(row, team_map) for row in def_values], dtype=np.int64
        )
    zero_box = np.zeros(n_box, dtype=np.float64) if n_box else None

    for i in range(len(possessions)):
        indices.append(0)
        data.append(1.0)
        indices.append(1)
        data.append(float(home[i]))

        box_off_sum = zero_box.copy() if zero_box is not None else None
        box_def_sum = zero_box.copy() if zero_box is not None else None

        for pid in off_values[i]:
            pid_i = int(pid)
            indices.append(player_off0 + player_index[pid_i])
            data.append(1.0)
            if box_off_sum is not None:
                vec = box_by_player.get(pid_i) if box_by_player else None
                if vec is not None:
                    box_off_sum += vec

        for pid in def_values[i]:
            pid_i = int(pid)
            indices.append(player_def0 + player_index[pid_i])
            data.append(-1.0)
            if box_def_sum is not None:
                vec = box_by_player.get(pid_i) if box_by_player else None
                if vec is not None:
                    box_def_sum += vec

        # One team / conference indicator per possession (PPP-scale effects).
        off_tid = int(off_team_ids[i])
        def_tid = int(def_team_ids[i])
        if include_team and team_off0 is not None and team_def0 is not None:
            if off_tid in team_index:
                indices.append(team_off0 + team_index[off_tid])
                data.append(1.0)
            if def_tid in team_index:
                indices.append(team_def0 + team_index[def_tid])
                data.append(-1.0)
        if include_conference and conf_off0 is not None and conf_def0 is not None:
            if off_tid in team_to_conf:
                off_conf = team_to_conf[off_tid]
            else:
                off_conf = str(_majority_key(off_values[i], conf_map, default="Unknown"))
            if def_tid in team_to_conf:
                def_conf = team_to_conf[def_tid]
            else:
                def_conf = str(_majority_key(def_values[i], conf_map, default="Unknown"))
            if off_conf in conf_index:
                indices.append(conf_off0 + conf_index[off_conf])
                data.append(1.0)
            if def_conf in conf_index:
                indices.append(conf_def0 + conf_index[def_conf])
                data.append(-1.0)

        if box_off_sum is not None and box_off0 is not None:
            for k, value in enumerate(box_off_sum):
                if value != 0.0:
                    indices.append(box_off0 + k)
                    data.append(float(value))
        if box_def_sum is not None and box_def0 is not None:
            for k, value in enumerate(box_def_sum):
                if value != 0.0:
                    indices.append(box_def0 + k)
                    data.append(float(value))

        indptr.append(len(indices))

    x = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(possessions), n_cols),
    )

    feature_names = (
        ["intercept", "home_offense"]
        + [f"off_{pid}" for pid in player_ids]
        + [f"def_{pid}" for pid in player_ids]
    )
    if include_team and n_team:
        feature_names.extend([f"team_off_{tid}" for tid in team_ids])
        feature_names.extend([f"team_def_{tid}" for tid in team_ids])
    if include_conference and n_conf:
        feature_names.extend([f"conf_off_{c}" for c in conference_names])
        feature_names.extend([f"conf_def_{c}" for c in conference_names])
    if n_box:
        feature_names.extend([f"box_off_{c}" for c in box_names])
        feature_names.extend([f"box_def_{c}" for c in box_names])

    return DesignMatrix(
        x=x,
        y=y,
        game_ids=game_ids,
        player_ids=player_ids,
        n_off=n_players,
        feature_names=feature_names,
        conference_names=conference_names,
        team_ids=team_ids,
        n_conf=n_conf,
        n_team=n_team,
        n_box=n_box,
        player_conference={pid: conf_map.get(pid, "Unknown") for pid in player_ids},
        player_team={pid: team_map[pid] for pid in player_ids if pid in team_map},
        box_feature_names=box_names,
        box_mean=dict(box_mean or {}),
        box_std=dict(box_std or {}),
        player_off0=player_off0,
        player_def0=player_def0,
        team_off0=team_off0,
        team_def0=team_def0,
        conf_off0=conf_off0,
        conf_def0=conf_def0,
        box_off0=box_off0,
        box_def0=box_def0,
    )
