from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from statpad_cbb.model.design import DesignMatrix


@dataclass
class RapmResult:
    coef: np.ndarray
    intercept: float
    home: float
    player_ids: list[int]
    off: np.ndarray
    defense: np.ndarray
    lambda_off: float
    lambda_def: float
    rmse: float
    lambda_team_off: float | None = None
    lambda_team_def: float | None = None
    lambda_conf_off: float | None = None
    lambda_conf_def: float | None = None
    lambda_box: float | None = None
    conference_names: list[str] = field(default_factory=list)
    team_ids: list[int] = field(default_factory=list)
    conf_off: np.ndarray | None = None
    conf_def: np.ndarray | None = None
    team_off: np.ndarray | None = None
    team_def: np.ndarray | None = None
    box_gamma_off: np.ndarray | None = None
    box_gamma_def: np.ndarray | None = None
    box_feature_names: list[str] = field(default_factory=list)
    player_off_raw: np.ndarray | None = None
    player_def_raw: np.ndarray | None = None
    prior: str = "league_mean"


# Weak Gaussian priors on intercept / home so prior_mean[0:2] is not a no-op.
DEFAULT_LAMBDA_INTERCEPT = 1.0
DEFAULT_LAMBDA_HOME = 25.0


def _penalty_diag(
    design: DesignMatrix,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_team_off: float | None,
    lambda_team_def: float | None,
    lambda_conf_off: float | None,
    lambda_conf_def: float | None,
    lambda_box: float | None,
    lambda_intercept: float | None = None,
    lambda_home: float | None = None,
) -> np.ndarray:
    diag = np.zeros(design.x.shape[1], dtype=np.float64)
    diag[0] = float(DEFAULT_LAMBDA_INTERCEPT if lambda_intercept is None else lambda_intercept)
    diag[1] = float(DEFAULT_LAMBDA_HOME if lambda_home is None else lambda_home)
    n_players = design.n_off
    diag[design.player_off0 : design.player_off0 + n_players] = lambda_off
    diag[design.player_def0 : design.player_def0 + n_players] = lambda_def
    if design.has_team and design.team_off0 is not None and design.team_def0 is not None:
        diag[design.team_off0 : design.team_off0 + design.n_team] = float(
            lambda_team_off if lambda_team_off is not None else max(lambda_off / 10.0, 1.0)
        )
        diag[design.team_def0 : design.team_def0 + design.n_team] = float(
            lambda_team_def if lambda_team_def is not None else max(lambda_def / 10.0, 1.0)
        )
    if design.has_conference and design.conf_off0 is not None and design.conf_def0 is not None:
        diag[design.conf_off0 : design.conf_off0 + design.n_conf] = float(
            lambda_conf_off if lambda_conf_off is not None else max(lambda_off / 20.0, 1.0)
        )
        diag[design.conf_def0 : design.conf_def0 + design.n_conf] = float(
            lambda_conf_def if lambda_conf_def is not None else max(lambda_def / 20.0, 1.0)
        )
    if design.has_box and design.box_off0 is not None and design.box_def0 is not None:
        lb = float(lambda_box if lambda_box is not None else 10.0)
        diag[design.box_off0 : design.box_off0 + design.n_box] = lb
        diag[design.box_def0 : design.box_def0 + design.n_box] = lb
    return diag


def _total_player_effects(
    design: DesignMatrix,
    coef: np.ndarray,
) -> dict[str, Any]:
    n_players = design.n_off
    player_off = coef[design.player_off0 : design.player_off0 + n_players].copy()
    player_def = coef[design.player_def0 : design.player_def0 + n_players].copy()
    total_off = player_off.copy()
    total_def = player_def.copy()

    team_off = team_def = None
    if design.has_team and design.team_off0 is not None and design.team_def0 is not None:
        team_off = coef[design.team_off0 : design.team_off0 + design.n_team]
        team_def = coef[design.team_def0 : design.team_def0 + design.n_team]
        team_index = {tid: i for i, tid in enumerate(design.team_ids)}
        for i, pid in enumerate(design.player_ids):
            tid = design.player_team.get(pid)
            if tid is not None and tid in team_index:
                j = team_index[tid]
                total_off[i] += team_off[j]
                total_def[i] += team_def[j]

    conf_off = conf_def = None
    if design.has_conference and design.conf_off0 is not None and design.conf_def0 is not None:
        conf_off = coef[design.conf_off0 : design.conf_off0 + design.n_conf]
        conf_def = coef[design.conf_def0 : design.conf_def0 + design.n_conf]
        conf_index = {name: i for i, name in enumerate(design.conference_names)}
        for i, pid in enumerate(design.player_ids):
            conf = design.player_conference.get(pid, "Unknown")
            j = conf_index[conf]
            total_off[i] += conf_off[j]
            total_def[i] += conf_def[j]

    box_off = box_def = None
    if design.has_box and design.box_off0 is not None and design.box_def0 is not None:
        box_off = coef[design.box_off0 : design.box_off0 + design.n_box]
        box_def = coef[design.box_def0 : design.box_def0 + design.n_box]
        # Box contribution is encoded via on-court feature sums; for player rating
        # report x_i'γ using standardized features stored externally by caller if needed.
        # Here we only return γ; pipeline adds x'γ when features are available.

    return {
        "total_off": total_off,
        "total_def": total_def,
        "player_off": player_off,
        "player_def": player_def,
        "team_off": team_off,
        "team_def": team_def,
        "conf_off": conf_off,
        "conf_def": conf_def,
        "box_off": box_off,
        "box_def": box_def,
    }


def apply_box_to_player_totals(
    *,
    player_ids: list[int],
    total_off: np.ndarray,
    total_def: np.ndarray,
    box_by_player: dict[int, np.ndarray],
    box_gamma_off: np.ndarray | None,
    box_gamma_def: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if box_gamma_off is None or box_gamma_def is None:
        return total_off, total_def
    off = total_off.copy()
    deff = total_def.copy()
    for i, pid in enumerate(player_ids):
        x = box_by_player.get(int(pid))
        if x is None:
            continue
        off[i] += float(np.dot(x, box_gamma_off))
        deff[i] += float(np.dot(x, box_gamma_def))
    return off, deff


# Five on-court players: shifting each β by m shifts unit PPP by 5m.
LINEUP_SIZE = 5


def center_player_effects_within_team(
    *,
    player_ids: list[int],
    player_team: dict[int, int],
    player_off: np.ndarray,
    player_def: np.ndarray,
    team_ids: list[int],
    team_off: np.ndarray | None,
    team_def: np.ndarray | None,
    minutes_by_player: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Absorb minutes-weighted within-team player means into team effects.

    Soft identifiability: player residuals sum (weighted) to 0 within each team.
    Because team indicators are once-per-possession while five players are on
    the floor, the unit contribution is ``5·mean(β) + τ``. Absorbing level into
    τ therefore requires ``τ += 5 · mean(β)`` (not ``+ mean(β)``).
    """
    if team_off is None or team_def is None or not team_ids:
        return player_off, player_def, team_off, team_def
    off = player_off.copy()
    deff = player_def.copy()
    t_off = team_off.copy()
    t_def = team_def.copy()
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    for tid, j in team_index.items():
        idxs = [i for i, pid in enumerate(player_ids) if player_team.get(pid) == tid]
        if not idxs:
            continue
        if minutes_by_player:
            weights = np.asarray(
                [max(float(minutes_by_player.get(player_ids[i], 0.0)), 0.0) for i in idxs],
                dtype=np.float64,
            )
        else:
            weights = np.ones(len(idxs), dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(idxs), dtype=np.float64)
        mean_off = float(np.average(off[idxs], weights=weights))
        mean_def = float(np.average(deff[idxs], weights=weights))
        for i in idxs:
            off[i] -= mean_off
            deff[i] -= mean_def
        t_off[j] += LINEUP_SIZE * mean_off
        t_def[j] += LINEUP_SIZE * mean_def
    return off, deff, t_off, t_def


def center_teams_within_conference(
    *,
    team_ids: list[int],
    team_conference: dict[int, str],
    team_off: np.ndarray | None,
    team_def: np.ndarray | None,
    conference_names: list[str],
    conf_off: np.ndarray | None,
    conf_def: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Absorb mean team effects within conference into κ (both once-per-possession)."""
    if (
        team_off is None
        or team_def is None
        or conf_off is None
        or conf_def is None
        or not team_ids
        or not conference_names
    ):
        return team_off, team_def, conf_off, conf_def
    t_off = team_off.copy()
    t_def = team_def.copy()
    c_off = conf_off.copy()
    c_def = conf_def.copy()
    conf_index = {name: i for i, name in enumerate(conference_names)}
    by_conf: dict[str, list[int]] = {}
    for i, tid in enumerate(team_ids):
        conf = team_conference.get(tid, "Unknown")
        by_conf.setdefault(conf, []).append(i)
    for conf, idxs in by_conf.items():
        if conf not in conf_index or not idxs:
            continue
        mean_off = float(np.mean(t_off[idxs]))
        mean_def = float(np.mean(t_def[idxs]))
        for i in idxs:
            t_off[i] -= mean_off
            t_def[i] -= mean_def
        j = conf_index[conf]
        c_off[j] += mean_off
        c_def[j] += mean_def
    return t_off, t_def, c_off, c_def


def fit_hierarchical_rapm(
    design: DesignMatrix,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_team_off: float | None = None,
    lambda_team_def: float | None = None,
    lambda_conf_off: float | None = None,
    lambda_conf_def: float | None = None,
    lambda_box: float | None = None,
    lambda_intercept: float | None = None,
    lambda_home: float | None = None,
    prior_mean: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
    box_by_player: dict[int, np.ndarray] | None = None,
    frozen_box_gamma_off: np.ndarray | None = None,
    frozen_box_gamma_def: np.ndarray | None = None,
    frozen_box_feature_names: list[str] | None = None,
    minutes_by_player: dict[int, float] | None = None,
    center_within_team: bool = True,
) -> RapmResult:
    """MAP hierarchical RAPM with optional team/conference/box structure and prior means.

    box_by_player: when box columns are in the design, add x'γ onto player totals
    after the fit (legacy design mode). Do not pass this when box is encoded only
    as player prior means (μ_i = x_i'γ) — those means already live in prior_mean.

    frozen_box_*: pass-through γ for prior export when box is not in the design.
    """
    if row_mask is None:
        x = design.x
        y = design.y
    else:
        x = design.x[row_mask]
        y = design.y[row_mask]

    diag = _penalty_diag(
        design,
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_team_off=lambda_team_off,
        lambda_team_def=lambda_team_def,
        lambda_conf_off=lambda_conf_off,
        lambda_conf_def=lambda_conf_def,
        lambda_box=lambda_box,
        lambda_intercept=lambda_intercept,
        lambda_home=lambda_home,
    )
    penalty = sparse.diags(np.sqrt(diag))
    target = np.zeros(design.x.shape[1], dtype=np.float64)
    if prior_mean is not None:
        if len(prior_mean) != design.x.shape[1]:
            raise ValueError("prior_mean length must match design columns")
        target = np.asarray(prior_mean, dtype=np.float64) * np.sqrt(diag)

    x_aug = sparse.vstack([x, penalty], format="csr")
    y_aug = np.concatenate([y, target])
    solution = lsqr(x_aug, y_aug, atol=1e-6, btol=1e-6, iter_lim=max(2 * design.x.shape[1], 1000))
    coef = solution[0]
    pred = x @ coef
    rmse = float(np.sqrt(np.mean((y - pred) ** 2))) if len(y) else float("nan")

    parts = _total_player_effects(design, coef)
    player_off = parts["player_off"]
    player_def = parts["player_def"]
    team_off = parts["team_off"]
    team_def = parts["team_def"]
    conf_off = parts["conf_off"]
    conf_def = parts["conf_def"]
    if center_within_team and design.has_team:
        player_off, player_def, team_off, team_def = center_player_effects_within_team(
            player_ids=design.player_ids,
            player_team=design.player_team,
            player_off=player_off,
            player_def=player_def,
            team_ids=design.team_ids,
            team_off=team_off,
            team_def=team_def,
            minutes_by_player=minutes_by_player,
        )
        if design.has_conference:
            team_conference = {
                tid: design.player_conference.get(pid, "Unknown")
                for pid, tid in design.player_team.items()
            }
            team_off, team_def, conf_off, conf_def = center_teams_within_conference(
                team_ids=design.team_ids,
                team_conference=team_conference,
                team_off=team_off,
                team_def=team_def,
                conference_names=design.conference_names,
                conf_off=conf_off,
                conf_def=conf_def,
            )

    # Rebuild per-player display totals after optional centering.
    # Note: these are NOT lineup-summable under once-per-possession τ/κ —
    # possession predictions must add τ/κ once from team ids (see backtest).
    total_off = player_off.copy()
    total_def = player_def.copy()
    if team_off is not None and team_def is not None:
        team_index = {tid: i for i, tid in enumerate(design.team_ids)}
        for i, pid in enumerate(design.player_ids):
            tid = design.player_team.get(pid)
            if tid is not None and tid in team_index:
                j = team_index[tid]
                total_off[i] += team_off[j]
                total_def[i] += team_def[j]
    if conf_off is not None and conf_def is not None:
        conf_index = {name: i for i, name in enumerate(design.conference_names)}
        for i, pid in enumerate(design.player_ids):
            conf = design.player_conference.get(pid, "Unknown")
            j = conf_index[conf]
            total_off[i] += conf_off[j]
            total_def[i] += conf_def[j]

    # Design-mode only: γ is estimated in-coef and applied onto player totals.
    # Prior-mean mode encodes μ=x'γ in prior_mean instead — do not double-count.
    if box_by_player is not None and parts["box_off"] is not None:
        total_off, total_def = apply_box_to_player_totals(
            player_ids=design.player_ids,
            total_off=total_off,
            total_def=total_def,
            box_by_player=box_by_player,
            box_gamma_off=parts["box_off"],
            box_gamma_def=parts["box_def"],
        )

    prior = "league_mean"
    if design.has_team or design.has_conference or design.has_box:
        prior = "hierarchical"
    elif frozen_box_gamma_off is not None:
        prior = "hierarchical"

    box_gamma_off = parts["box_off"]
    box_gamma_def = parts["box_def"]
    box_names = list(design.box_feature_names)
    if box_gamma_off is None and frozen_box_gamma_off is not None:
        box_gamma_off = np.asarray(frozen_box_gamma_off, dtype=np.float64)
        box_gamma_def = (
            None
            if frozen_box_gamma_def is None
            else np.asarray(frozen_box_gamma_def, dtype=np.float64)
        )
        box_names = list(frozen_box_feature_names or [])

    return RapmResult(
        coef=coef,
        intercept=float(coef[0]),
        home=float(coef[1]),
        player_ids=list(design.player_ids),
        off=total_off,
        defense=total_def,
        lambda_off=float(lambda_off),
        lambda_def=float(lambda_def),
        rmse=rmse,
        lambda_team_off=None if lambda_team_off is None else float(lambda_team_off),
        lambda_team_def=None if lambda_team_def is None else float(lambda_team_def),
        lambda_conf_off=None if lambda_conf_off is None else float(lambda_conf_off),
        lambda_conf_def=None if lambda_conf_def is None else float(lambda_conf_def),
        lambda_box=None if lambda_box is None else float(lambda_box),
        conference_names=list(design.conference_names),
        team_ids=list(design.team_ids),
        conf_off=conf_off,
        conf_def=conf_def,
        team_off=team_off,
        team_def=team_def,
        box_gamma_off=box_gamma_off,
        box_gamma_def=box_gamma_def,
        box_feature_names=box_names,
        player_off_raw=player_off,
        player_def_raw=player_def,
        prior=prior,
    )


def predict_possessions(design: DesignMatrix, coef: np.ndarray, row_mask: np.ndarray | None = None) -> np.ndarray:
    x = design.x if row_mask is None else design.x[row_mask]
    return x @ coef


def game_blocked_folds(game_ids: np.ndarray, n_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_games = np.unique(game_ids)
    rng.shuffle(unique_games)
    folds: list[np.ndarray] = []
    for fold in range(n_folds):
        holdout_games = set(unique_games[fold::n_folds].tolist())
        mask = np.isin(game_ids, list(holdout_games))
        folds.append(mask)
    return folds


def _game_margin_rmse_from_possessions(
    *,
    design: DesignMatrix,
    coef: np.ndarray,
    val_mask: np.ndarray,
    possessions: pd.DataFrame,
) -> float:
    """Approximate held-out game margin RMSE using possession predictions."""
    pred = predict_possessions(design, coef, row_mask=val_mask)
    y = design.y[val_mask]
    games = design.game_ids[val_mask]
    # Map each val possession to offense team via original frame alignment.
    # design rows correspond 1:1 with possessions rows used to build it.
    val_idx = np.where(val_mask)[0]
    off_team = possessions["off_team_id"].to_numpy()[val_idx]
    # Aggregate predicted and actual points by (game, off_team), then margin.
    actual_pts: dict[tuple[int, int], float] = {}
    pred_pts: dict[tuple[int, int], float] = {}
    for g, team, a, p in zip(games, off_team, y, pred):
        key = (int(g), int(team))
        actual_pts[key] = actual_pts.get(key, 0.0) + float(a)
        pred_pts[key] = pred_pts.get(key, 0.0) + float(p)

    # Need home/away from possessions
    game_home = {}
    if "home_team_id" in possessions.columns:
        for g, h in zip(possessions["game_id"], possessions["home_team_id"]):
            if pd.notna(h):
                game_home[int(g)] = int(h)

    margins_a = []
    margins_p = []
    for gid in set(int(g) for g in games):
        home = game_home.get(gid)
        if home is None:
            continue
        # away is the other team appearing
        teams = {t for (g, t) in actual_pts if g == gid}
        if home not in teams or len(teams) != 2:
            continue
        away = next(t for t in teams if t != home)
        a_margin = actual_pts.get((gid, home), 0.0) - actual_pts.get((gid, away), 0.0)
        p_margin = pred_pts.get((gid, home), 0.0) - pred_pts.get((gid, away), 0.0)
        margins_a.append(a_margin)
        margins_p.append(p_margin)
    if not margins_a:
        return float("inf")
    err = np.asarray(margins_a) - np.asarray(margins_p)
    return float(np.sqrt(np.mean(err**2)))


def cross_validate_lambdas(
    design: DesignMatrix,
    lambda_grid: list[tuple],
    *,
    n_folds: int = 5,
    seed: int = 42,
    possessions: pd.DataFrame | None = None,
    metric: str = "game_margin_rmse",
    box_by_player: dict[int, np.ndarray] | None = None,
    prior_mean: np.ndarray | None = None,
) -> dict:
    """Cross-validate penalties.

    metric='game_margin_rmse' uses held-out game margins (preferred).
    metric='possession_rmse' uses possession-level RMSE.
    """
    folds = game_blocked_folds(design.game_ids, n_folds=n_folds, seed=seed)
    records = []
    best = None
    for item in lambda_grid:
        # Support legacy 2-tuple and richer tuples.
        lambda_off = float(item[0])
        lambda_def = float(item[1])
        lambda_team_off = float(item[2]) if len(item) > 2 and item[2] is not None else None
        lambda_team_def = float(item[3]) if len(item) > 3 and item[3] is not None else None
        lambda_conf_off = float(item[4]) if len(item) > 4 and item[4] is not None else None
        lambda_conf_def = float(item[5]) if len(item) > 5 and item[5] is not None else None
        lambda_box = float(item[6]) if len(item) > 6 and item[6] is not None else None

        # Back-compat with old 4-tuple conference grid (off, def, conf_off, conf_def)
        if len(item) == 4 and not design.has_team:
            lambda_team_off = None
            lambda_team_def = None
            lambda_conf_off = float(item[2])
            lambda_conf_def = float(item[3])
            lambda_box = None

        fold_scores = []
        fold_poss_rmse = []
        for val_mask in folds:
            train_mask = ~val_mask
            if not train_mask.any() or not val_mask.any():
                continue
            fit = fit_hierarchical_rapm(
                design,
                lambda_off=lambda_off,
                lambda_def=lambda_def,
                lambda_team_off=lambda_team_off,
                lambda_team_def=lambda_team_def,
                lambda_conf_off=lambda_conf_off,
                lambda_conf_def=lambda_conf_def,
                lambda_box=lambda_box,
                prior_mean=prior_mean,
                row_mask=train_mask,
                box_by_player=box_by_player,
                center_within_team=False,
            )
            pred = predict_possessions(design, fit.coef, row_mask=val_mask)
            y_val = design.y[val_mask]
            fold_poss_rmse.append(float(np.sqrt(np.mean((y_val - pred) ** 2))))
            if metric == "game_margin_rmse" and possessions is not None:
                fold_scores.append(
                    _game_margin_rmse_from_possessions(
                        design=design,
                        coef=fit.coef,
                        val_mask=val_mask,
                        possessions=possessions,
                    )
                )
            else:
                fold_scores.append(fold_poss_rmse[-1])

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
        mean_poss = float(np.mean(fold_poss_rmse)) if fold_poss_rmse else float("inf")
        record = {
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_team_off": lambda_team_off,
            "lambda_team_def": lambda_team_def,
            "lambda_conf_off": lambda_conf_off,
            "lambda_conf_def": lambda_conf_def,
            "lambda_box": lambda_box,
            "score": mean_score,
            "rmse": mean_poss,
            "metric": metric,
        }
        records.append(record)
        if best is None or mean_score < best["score"]:
            best = record
    return {"grid": records, "best": best}
