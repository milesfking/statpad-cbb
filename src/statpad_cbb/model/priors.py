from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from statpad_cbb.model.design import DesignMatrix, standardize_box_features


@dataclass
class BoxPriorSpec:
    """How box features enter a hierarchical fit."""

    mode: str  # "prior_mean" | "design" | "none"
    box_by_player: dict[int, np.ndarray]
    feature_names: list[str]
    mean: dict[str, float]
    std: dict[str, float]
    gamma_off: np.ndarray | None = None
    gamma_def: np.ndarray | None = None

    @property
    def include_in_design(self) -> bool:
        return self.mode == "design"

    @property
    def has_player_means(self) -> bool:
        return (
            self.mode == "prior_mean"
            and self.gamma_off is not None
            and self.gamma_def is not None
        )


def _gamma_arrays(box_prior: dict[str, Any] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not box_prior:
        return None, None
    gamma_off = box_prior.get("gamma_off")
    gamma_def = box_prior.get("gamma_def")
    if not gamma_off or not gamma_def:
        return None, None
    return (
        np.asarray(gamma_off, dtype=np.float64),
        np.asarray(gamma_def, dtype=np.float64),
    )


def resolve_box_prior(
    *,
    features,
    prior_artifact: dict[str, Any] | None,
    use_box: bool,
    box_mode: str = "prior_mean",
) -> BoxPriorSpec:
    """Resolve box features for a fit.

    prior_mean (default): freeze γ from the prior season and shrink player
    effects toward μ_i = x_i'γ. Box is not a design column, so early-season
    counting stats cannot outvote team priors.

    design: legacy path — box sums enter X and γ is estimated (seed / ablation).
    """
    if box_mode not in {"prior_mean", "design"}:
        raise ValueError("box_mode must be 'prior_mean' or 'design'")
    if not use_box or features is None or getattr(features, "empty", True):
        return BoxPriorSpec(mode="none", box_by_player={}, feature_names=[], mean={}, std={})

    prior_box = (prior_artifact or {}).get("box") if prior_artifact else None
    box_by_player, names, mean, std = standardize_box_features(
        features,
        mean=(prior_box or {}).get("mean"),
        std=(prior_box or {}).get("std"),
    )
    if not names:
        return BoxPriorSpec(mode="none", box_by_player={}, feature_names=[], mean={}, std={})

    gamma_off, gamma_def = _gamma_arrays(prior_box)
    if (
        box_mode == "prior_mean"
        and gamma_off is not None
        and gamma_def is not None
        and list((prior_box or {}).get("feature_names") or []) == names
        and len(gamma_off) == len(names)
        and len(gamma_def) == len(names)
    ):
        return BoxPriorSpec(
            mode="prior_mean",
            box_by_player=box_by_player,
            feature_names=names,
            mean=mean,
            std=std,
            gamma_off=gamma_off,
            gamma_def=gamma_def,
        )

    # No usable frozen γ: fall back to design estimation to seed / continue.
    return BoxPriorSpec(
        mode="design",
        box_by_player=box_by_player,
        feature_names=names,
        mean=mean,
        std=std,
        gamma_off=gamma_off,
        gamma_def=gamma_def,
    )


def residualize_box_by_team(
    box_by_player: dict[int, np.ndarray],
    player_team: dict[int, int],
) -> dict[int, np.ndarray]:
    """Subtract within-team mean of box z-features.

    Team/conference priors already carry level; residual box μ should only
    differentiate players *within* a roster so skill is not double-counted.
    """
    if not box_by_player:
        return {}
    by_team: dict[int, list[int]] = {}
    for pid, tid in player_team.items():
        if int(pid) in box_by_player:
            by_team.setdefault(int(tid), []).append(int(pid))
    out = {int(pid): np.asarray(vec, dtype=np.float64).copy() for pid, vec in box_by_player.items()}
    for pids in by_team.values():
        if not pids:
            continue
        stacked = np.vstack([out[pid] for pid in pids])
        mean = stacked.mean(axis=0)
        for pid in pids:
            out[pid] = out[pid] - mean
    # Players without a team mapping keep original (usually league-mean zeros).
    return out


def player_box_prior_means(
    player_ids: list[int],
    *,
    box_by_player: dict[int, np.ndarray],
    gamma_off: np.ndarray,
    gamma_def: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """μ_i = x_i'γ for each player; missing box rows get 0 (league-mean z)."""
    off = np.zeros(len(player_ids), dtype=np.float64)
    deff = np.zeros(len(player_ids), dtype=np.float64)
    for i, pid in enumerate(player_ids):
        x = box_by_player.get(int(pid))
        if x is None:
            continue
        off[i] = float(np.dot(x, gamma_off))
        deff[i] = float(np.dot(x, gamma_def))
    return off, deff


def extract_group_prior(
    fit,
    design: DesignMatrix,
    *,
    box_spec: BoxPriorSpec | None = None,
) -> dict[str, Any]:
    """Serialize group-level / box priors for reuse in a later season."""
    conf = {}
    if fit.conf_off is not None and fit.conf_def is not None:
        for i, name in enumerate(fit.conference_names):
            conf[name] = {
                "off": float(fit.conf_off[i]),
                "def": float(fit.conf_def[i]),
            }
    teams = {}
    if fit.team_off is not None and fit.team_def is not None:
        for i, tid in enumerate(fit.team_ids):
            teams[str(int(tid))] = {
                "off": float(fit.team_off[i]),
                "def": float(fit.team_def[i]),
            }

    # Prefer newly estimated design γ; otherwise pass through frozen γ.
    gamma_off = fit.box_gamma_off
    gamma_def = fit.box_gamma_def
    feature_names = list(fit.box_feature_names or [])
    box_mean = dict(design.box_mean)
    box_std = dict(design.box_std)
    if box_spec is not None and box_spec.mode != "none":
        if not feature_names:
            feature_names = list(box_spec.feature_names)
        if not box_mean:
            box_mean = dict(box_spec.mean)
        if not box_std:
            box_std = dict(box_spec.std)
        if gamma_off is None and box_spec.gamma_off is not None:
            gamma_off = box_spec.gamma_off
        if gamma_def is None and box_spec.gamma_def is not None:
            gamma_def = box_spec.gamma_def

    box = {
        "feature_names": feature_names,
        "gamma_off": None if gamma_off is None else [float(x) for x in gamma_off],
        "gamma_def": None if gamma_def is None else [float(x) for x in gamma_def],
        "mean": box_mean,
        "std": box_std,
        "mode": None if box_spec is None else box_spec.mode,
    }
    return {
        "conferences": conf,
        "teams": teams,
        "box": box,
        "intercept": float(fit.intercept),
        "home": float(fit.home),
    }


def build_prior_mean_vector(
    design: DesignMatrix,
    prior: dict[str, Any] | None,
    *,
    player_off_mean: np.ndarray | None = None,
    player_def_mean: np.ndarray | None = None,
    include_box_gamma_prior: bool = True,
) -> np.ndarray | None:
    """Map a saved prior artifact onto the current design column space.

    player_off_mean / player_def_mean: optional per-player ridge targets
    (used for frozen box-as-prior-mean).
    include_box_gamma_prior: when False, skip γ columns (box not in design).
    """
    if (
        not prior
        and player_off_mean is None
        and player_def_mean is None
    ):
        return None
    mean = np.zeros(design.x.shape[1], dtype=np.float64)
    prior = prior or {}

    if player_off_mean is not None:
        if len(player_off_mean) != design.n_off:
            raise ValueError("player_off_mean length must match n_off")
        mean[design.player_off0 : design.player_off0 + design.n_off] = player_off_mean
    if player_def_mean is not None:
        if len(player_def_mean) != design.n_off:
            raise ValueError("player_def_mean length must match n_off")
        mean[design.player_def0 : design.player_def0 + design.n_off] = player_def_mean

    conf_prior = prior.get("conferences") or {}
    if design.has_conference and design.conf_off0 is not None and design.conf_def0 is not None:
        for i, name in enumerate(design.conference_names):
            vals = conf_prior.get(name)
            if not vals:
                continue
            mean[design.conf_off0 + i] = float(vals.get("off", 0.0))
            mean[design.conf_def0 + i] = float(vals.get("def", 0.0))

    team_prior = prior.get("teams") or {}
    if design.has_team and design.team_off0 is not None and design.team_def0 is not None:
        for i, tid in enumerate(design.team_ids):
            vals = team_prior.get(str(int(tid)))
            if not vals:
                continue
            mean[design.team_off0 + i] = float(vals.get("off", 0.0))
            mean[design.team_def0 + i] = float(vals.get("def", 0.0))

    if include_box_gamma_prior:
        box_prior = prior.get("box") or {}
        names = list(box_prior.get("feature_names") or [])
        gamma_off = box_prior.get("gamma_off")
        gamma_def = box_prior.get("gamma_def")
        if (
            design.has_box
            and design.box_off0 is not None
            and design.box_def0 is not None
            and gamma_off
            and gamma_def
            and names == design.box_feature_names
        ):
            for i, _ in enumerate(design.box_feature_names):
                mean[design.box_off0 + i] = float(gamma_off[i])
                mean[design.box_def0 + i] = float(gamma_def[i])

    # Keep intercept/home weakly centered on prior season if present.
    if "intercept" in prior:
        mean[0] = float(prior["intercept"])
    if "home" in prior:
        mean[1] = float(prior["home"])
    return mean
