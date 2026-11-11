from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from statpad_cbb.config import load_settings
from statpad_cbb.ingest.storage import SeasonPaths, read_jsonl, write_json
from statpad_cbb.model.design import (
    DEF_COLS,
    OFF_COLS,
    build_design_matrix,
    build_player_maps,
    build_team_conference_lookup,
)
from statpad_cbb.features.pipeline import build_player_features_frame
from statpad_cbb.model.priors import (
    build_prior_mean_vector,
    player_box_prior_means,
    residualize_box_by_team,
    resolve_box_prior,
)
from statpad_cbb.model.rapm import fit_hierarchical_rapm
from statpad_cbb.model.rollup import player_ratings_frame, team_rollup

logger = logging.getLogger(__name__)

def default_cut_dates(season: int) -> tuple[str, ...]:
    """Season-ending-year cuts: Dec 1 / Jan 1 / Feb 1 / Mar 1."""
    return (
        f"{season - 1}-12-01",
        f"{season}-01-01",
        f"{season}-02-01",
        f"{season}-03-01",
    )


# Back-compat alias for older imports/tests (2025–26 season).
DEFAULT_CUT_DATES = default_cut_dates(2026)

BLEND_WEIGHTS = np.linspace(0.0, 1.0, 21)


def _load_games(paths: SeasonPaths) -> pd.DataFrame:
    games = pd.DataFrame(list(read_jsonl(paths.processed_games)))
    if games.empty:
        raise ValueError("No games found")
    games["game_id"] = games["game_id"].astype(int)
    games["game_date"] = pd.to_datetime(games["start_date_utc"]).dt.strftime("%Y-%m-%d")
    return games


def _load_team_lookup(paths: SeasonPaths) -> dict[int, dict[str, Any]]:
    return {
        int(row["team_id"]): row
        for row in read_jsonl(paths.processed_teams)
        if row.get("team_id") is not None
    }


def attach_game_dates(possessions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    meta = games[
        [
            "game_id",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_points",
            "away_points",
            "neutral_site",
        ]
    ]
    out = possessions.merge(meta, on="game_id", how="left")
    missing = out["game_date"].isna().mean()
    if missing > 0:
        logger.warning("%.1f%% possessions missing game_date", 100 * missing)
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 3:
        return None
    value = float(a.corr(b))
    return None if np.isnan(value) else value


def predict_possession_points(
    possessions: pd.DataFrame,
    *,
    intercept: float,
    home: float,
    player_off_coef: dict[int, float],
    player_def_coef: dict[int, float],
    team_off_coef: dict[int, float] | None = None,
    team_def_coef: dict[int, float] | None = None,
    conf_off_coef: dict[str, float] | None = None,
    conf_def_coef: dict[str, float] | None = None,
    team_conference: dict[int, str] | None = None,
) -> np.ndarray:
    """Predict PPP matching once-per-possession team/conf encoding.

    ``player_*_coef`` must be player residuals (β), not β+τ+κ. Team and
    conference effects are added once from possession team ids.
    """
    preds = intercept + home * possessions["home_offense"].to_numpy(dtype=np.float64)
    for col in OFF_COLS:
        preds = preds + possessions[col].map(player_off_coef).fillna(0.0).to_numpy(dtype=np.float64)
    for col in DEF_COLS:
        preds = preds - possessions[col].map(player_def_coef).fillna(0.0).to_numpy(dtype=np.float64)

    if team_off_coef is not None and "off_team_id" in possessions.columns:
        preds = preds + possessions["off_team_id"].map(team_off_coef).fillna(0.0).to_numpy(
            dtype=np.float64
        )
    if team_def_coef is not None and "def_team_id" in possessions.columns:
        preds = preds - possessions["def_team_id"].map(team_def_coef).fillna(0.0).to_numpy(
            dtype=np.float64
        )

    if (
        conf_off_coef is not None
        and conf_def_coef is not None
        and team_conference is not None
        and "off_team_id" in possessions.columns
    ):
        off_conf = possessions["off_team_id"].map(
            lambda tid: team_conference.get(int(tid), "Unknown") if pd.notna(tid) else "Unknown"
        )
        def_conf = possessions["def_team_id"].map(
            lambda tid: team_conference.get(int(tid), "Unknown") if pd.notna(tid) else "Unknown"
        )
        preds = preds + off_conf.map(conf_off_coef).fillna(0.0).to_numpy(dtype=np.float64)
        preds = preds - def_conf.map(conf_def_coef).fillna(0.0).to_numpy(dtype=np.float64)
    return preds


def _team_rate_lookup(teams: pd.DataFrame) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in teams.itertuples(index=False):
        out[int(row.team_id)] = {
            "off_rating": float(row.off_rating),
            "def_rating": float(row.def_rating),
            "net_rating": float(row.net_rating),
            "off_eff": float(row.off_eff) if pd.notna(row.off_eff) else float("nan"),
            "def_eff": float(row.def_eff) if pd.notna(row.def_eff) else float("nan"),
            "eff_margin": float(row.eff_margin) if pd.notna(row.eff_margin) else float("nan"),
        }
    return out


def predict_game_margins_from_team_ratings(
    future_games: pd.DataFrame,
    *,
    team_rates: dict[int, dict[str, float]],
    intercept: float,
    home_advantage: float,
    possessions_per_team: float,
) -> pd.DataFrame:
    """True prospective team prediction (no future lineup knowledge)."""
    rows = []
    for game in future_games.itertuples(index=False):
        home_id = int(game.home_team_id)
        away_id = int(game.away_team_id)
        home = team_rates.get(home_id)
        away = team_rates.get(away_id)
        if home is None or away is None:
            continue
        if pd.isna(game.home_points) or pd.isna(game.away_points):
            continue

        neutral = bool(game.neutral_site)
        home_term = 0.0 if neutral else home_advantage
        pred_home_ppp = intercept + home_term + home["off_rating"] / 100.0 - away["def_rating"] / 100.0
        pred_away_ppp = intercept + away["off_rating"] / 100.0 - home["def_rating"] / 100.0
        pred_margin = (pred_home_ppp - pred_away_ppp) * possessions_per_team

        if np.isfinite(home["eff_margin"]) and np.isfinite(away["eff_margin"]):
            baseline_margin = (home["eff_margin"] - away["eff_margin"]) / 100.0 * possessions_per_team
            if not neutral:
                baseline_margin += home_advantage * possessions_per_team
        else:
            baseline_margin = float("nan")

        actual_margin = float(game.home_points) - float(game.away_points)
        rows.append(
            {
                "game_id": int(game.game_id),
                "game_date": game.game_date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "neutral_site": neutral,
                "actual_margin": actual_margin,
                "pred_margin_rapm": float(pred_margin),
                "pred_margin_eff": None if not np.isfinite(baseline_margin) else float(baseline_margin),
                "home_win_actual": int(actual_margin > 0),
                "home_win_pred_rapm": int(pred_margin > 0),
                "abs_err_rapm": abs(actual_margin - pred_margin),
                "abs_err_eff": None
                if not np.isfinite(baseline_margin)
                else abs(actual_margin - baseline_margin),
            }
        )
    return pd.DataFrame(rows)


def tune_blend_weight(preds: pd.DataFrame) -> dict[str, Any]:
    """Choose w for blend = w*RAPM + (1-w)*eff minimizing margin RMSE."""
    valid = preds.dropna(subset=["pred_margin_rapm", "pred_margin_eff", "actual_margin"])
    if valid.empty:
        return {"blend_weight_rapm": 1.0, "margin_rmse_blend": None, "grid": []}
    actual = valid["actual_margin"].to_numpy(dtype=np.float64)
    rapm = valid["pred_margin_rapm"].to_numpy(dtype=np.float64)
    eff = valid["pred_margin_eff"].to_numpy(dtype=np.float64)
    grid = []
    best_w = 1.0
    best_rmse = float("inf")
    for w in BLEND_WEIGHTS:
        blend = w * rapm + (1.0 - w) * eff
        rmse = _rmse(actual, blend)
        grid.append({"w_rapm": float(w), "margin_rmse": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)
    return {
        "blend_weight_rapm": best_w,
        "margin_rmse_blend": best_rmse,
        "grid": grid,
    }


def apply_blend(preds: pd.DataFrame, w_rapm: float) -> pd.DataFrame:
    out = preds.copy()
    if out.empty:
        return out
    rapm = out["pred_margin_rapm"].to_numpy(dtype=np.float64)
    eff = out["pred_margin_eff"].to_numpy(dtype=np.float64)
    blend = np.where(np.isfinite(eff), w_rapm * rapm + (1.0 - w_rapm) * eff, rapm)
    out["pred_margin_blend"] = blend
    out["home_win_pred_blend"] = (blend > 0).astype(int)
    out["abs_err_blend"] = np.abs(out["actual_margin"].to_numpy(dtype=np.float64) - blend)
    return out


def _summarize_game_preds(preds: pd.DataFrame) -> dict[str, Any]:
    if preds.empty:
        return {
            "n_games": 0,
            "margin_rmse_rapm": None,
            "margin_mae_rapm": None,
            "margin_rmse_eff": None,
            "margin_mae_eff": None,
            "margin_rmse_blend": None,
            "margin_mae_blend": None,
            "win_accuracy_rapm": None,
            "win_accuracy_eff": None,
            "win_accuracy_blend": None,
            "margin_corr_rapm": None,
            "margin_corr_eff": None,
            "margin_corr_blend": None,
            "blend_weight_rapm": None,
        }
    out: dict[str, Any] = {
        "n_games": int(len(preds)),
        "margin_rmse_rapm": _rmse(preds["actual_margin"], preds["pred_margin_rapm"]),
        "margin_mae_rapm": _mae(preds["actual_margin"], preds["pred_margin_rapm"]),
        "win_accuracy_rapm": float((preds["home_win_actual"] == preds["home_win_pred_rapm"]).mean()),
        "margin_corr_rapm": _corr(preds["actual_margin"], preds["pred_margin_rapm"]),
    }
    eff = preds.dropna(subset=["pred_margin_eff"])
    if not eff.empty:
        out["margin_rmse_eff"] = _rmse(eff["actual_margin"], eff["pred_margin_eff"])
        out["margin_mae_eff"] = _mae(eff["actual_margin"], eff["pred_margin_eff"])
        out["win_accuracy_eff"] = float(
            (eff["home_win_actual"] == (eff["pred_margin_eff"] > 0).astype(int)).mean()
        )
        out["margin_corr_eff"] = _corr(eff["actual_margin"], eff["pred_margin_eff"])
    else:
        out["margin_rmse_eff"] = None
        out["margin_mae_eff"] = None
        out["win_accuracy_eff"] = None
        out["margin_corr_eff"] = None

    if "pred_margin_blend" in preds.columns:
        out["margin_rmse_blend"] = _rmse(preds["actual_margin"], preds["pred_margin_blend"])
        out["margin_mae_blend"] = _mae(preds["actual_margin"], preds["pred_margin_blend"])
        out["win_accuracy_blend"] = float(
            (preds["home_win_actual"] == preds["home_win_pred_blend"]).mean()
        )
        out["margin_corr_blend"] = _corr(preds["actual_margin"], preds["pred_margin_blend"])
        out["blend_weight_rapm"] = (
            float(preds["blend_weight_rapm"].iloc[0])
            if "blend_weight_rapm" in preds.columns
            else None
        )
    else:
        out["margin_rmse_blend"] = None
        out["margin_mae_blend"] = None
        out["win_accuracy_blend"] = None
        out["margin_corr_blend"] = None
        out["blend_weight_rapm"] = None
    return out


def _player_stability(
    train_ratings: pd.DataFrame,
    final_ratings: pd.DataFrame | None,
    *,
    min_minutes: float = 10.0,
) -> dict[str, Any]:
    if final_ratings is None or final_ratings.empty or train_ratings.empty:
        return {}
    merged = train_ratings.merge(
        final_ratings[["player_id", "off_rating", "def_rating", "net_rating"]].rename(
            columns={
                "off_rating": "final_off",
                "def_rating": "final_def",
                "net_rating": "final_net",
            }
        ),
        on="player_id",
        how="inner",
    )
    if "minutes" in merged.columns:
        eligible = merged.loc[merged["minutes"].fillna(0) >= min_minutes]
    else:
        eligible = merged
    if eligible.empty:
        return {"n_players": 0}
    return {
        "n_players": int(len(eligible)),
        "min_minutes": min_minutes,
        "corr_off": _corr(eligible["off_rating"], eligible["final_off"]),
        "corr_def": _corr(eligible["def_rating"], eligible["final_def"]),
        "corr_net": _corr(eligible["net_rating"], eligible["final_net"]),
    }


def _resolve_prior_flags(prior: str) -> tuple[bool, bool, bool]:
    if prior not in {"hierarchical", "conference", "league_mean"}:
        raise ValueError("prior must be hierarchical, conference, or league_mean")
    include_team = prior == "hierarchical"
    include_conference = prior in {"hierarchical", "conference"}
    include_box = prior == "hierarchical"
    return include_team, include_conference, include_box


def run_temporal_backtest(
    season: int,
    *,
    data_dir: Path | None = None,
    cut_dates: list[str] | None = None,
    lambda_off: float | None = None,
    lambda_def: float | None = None,
    lambda_team_off: float | None = None,
    lambda_team_def: float | None = None,
    lambda_conf_off: float | None = None,
    lambda_conf_def: float | None = None,
    lambda_box: float | None = None,
    possessions_per_team: float | None = None,
    prior: str = "hierarchical",
    prior_season: int | None = None,
    use_box: bool = True,
    box_mode: str = "prior_mean",
    blend_weight: float | None = None,
) -> dict[str, Any]:
    """Fit RAPM through each cut date and score later games / rating stability."""
    include_team, include_conference, want_box = _resolve_prior_flags(prior)
    include_box = want_box and use_box
    if box_mode not in {"prior_mean", "design"}:
        raise ValueError("box_mode must be 'prior_mean' or 'design'")

    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    paths.ensure()

    if not paths.possessions.exists():
        raise FileNotFoundError(f"Missing possessions at {paths.possessions}")

    possessions = pd.read_parquet(paths.possessions)
    games = _load_games(paths)
    possessions = attach_game_dates(possessions, games)
    possessions = possessions.dropna(subset=["game_date"]).copy()

    final_ratings = (
        pd.read_parquet(paths.player_ratings)
        if paths.player_ratings.exists()
        else None
    )
    team_lookup = _load_team_lookup(paths)
    roster = pd.DataFrame(list(read_jsonl(paths.processed_players)))
    teams_frame = pd.DataFrame(list(read_jsonl(paths.processed_teams)))
    roster_lookup = {
        int(row["player_id"]): row
        for row in read_jsonl(paths.processed_players)
        if row.get("player_id") is not None
    }
    # Plays for cut-dated box rates (avoid leaking post-cut counting stats).
    plays_iter_factory = lambda: read_jsonl(paths.processed_plays)

    prior_artifact = None
    if prior_season is not None:
        prior_paths = SeasonPaths(settings.data_dir, prior_season)
        if prior_paths.season_priors.exists():
            prior_artifact = json.loads(prior_paths.season_priors.read_text())
            logger.info("backtest using priors from season %s", prior_season)

    # Prefer prior-season λ (no same-season full-fit lookahead into early cuts).
    lambda_source = None
    if prior_season is not None:
        prior_metrics_path = SeasonPaths(settings.data_dir, prior_season).model_metrics
        if prior_metrics_path.exists():
            lambda_source = json.loads(prior_metrics_path.read_text())
            logger.info("backtest λ defaults from prior season %s metrics", prior_season)
    if lambda_source is None:
        lambda_source = {}

    if lambda_off is None:
        lambda_off = float(lambda_source.get("lambda_off", 250.0))
    if lambda_def is None:
        lambda_def = float(lambda_source.get("lambda_def", 250.0))
    if include_team:
        if lambda_team_off is None:
            lambda_team_off = float(lambda_source.get("lambda_team_off") or 50.0)
        if lambda_team_def is None:
            lambda_team_def = float(lambda_source.get("lambda_team_def") or 50.0)
    if include_conference:
        if lambda_conf_off is None:
            lambda_conf_off = float(lambda_source.get("lambda_conf_off") or 25.0)
        if lambda_conf_def is None:
            lambda_conf_def = float(lambda_source.get("lambda_conf_def") or 25.0)
    if include_box and lambda_box is None:
        lambda_box = float(lambda_source.get("lambda_box") or 10.0)

    cuts = list(cut_dates or default_cut_dates(season))
    cut_reports: list[dict[str, Any]] = []
    all_game_preds: list[pd.DataFrame] = []
    rolling_blend_w = blend_weight  # if None, learn from previous cut then apply forward

    for cut in cuts:
        train_poss = possessions.loc[possessions["game_date"] <= cut]
        test_poss = possessions.loc[possessions["game_date"] > cut]
        train_game_ids = set(train_poss["game_id"].unique())
        test_games = games.loc[
            (~games["game_id"].isin(train_game_ids))
            & (games["game_date"] > cut)
            & games["home_points"].notna()
            & games["away_points"].notna()
        ].copy()

        logger.info(
            "cut=%s prior=%s train_poss=%s test_poss=%s test_games=%s",
            cut,
            prior,
            len(train_poss),
            len(test_poss),
            len(test_games),
        )
        if train_poss.empty or test_games.empty:
            cut_reports.append(
                {
                    "cut_date": cut,
                    "skipped": True,
                    "reason": "insufficient train or test games",
                    "n_train_possessions": int(len(train_poss)),
                    "n_test_games": int(len(test_games)),
                }
            )
            continue

        train_players = sorted(
            {
                int(pid)
                for col in OFF_COLS + DEF_COLS
                for pid in train_poss[col].tolist()
            }
        )
        team_conference = build_team_conference_lookup(teams_frame)
        conf_map, team_map = build_player_maps(
            train_players,
            roster=roster,
            teams=teams_frame,
            possessions=train_poss,
        )

        train_game_ids = set(int(x) for x in train_poss["game_id"].unique())
        features, _ = build_player_features_frame(
            train_poss,
            season=season,
            roster=roster_lookup,
            plays=plays_iter_factory(),
            play_game_ids=train_game_ids,
        )
        box_spec = resolve_box_prior(
            features=features,
            prior_artifact=prior_artifact,
            use_box=include_box,
            box_mode=box_mode,
        )
        design = build_design_matrix(
            train_poss,
            player_conference=conf_map,
            player_team=team_map,
            team_conference=team_conference,
            include_conference=include_conference,
            include_team=include_team,
            box_by_player=box_spec.box_by_player if box_spec.include_in_design else None,
            box_feature_names=box_spec.feature_names if box_spec.include_in_design else None,
            box_mean=box_spec.mean if box_spec.include_in_design else None,
            box_std=box_spec.std if box_spec.include_in_design else None,
        )
        player_off_mean = player_def_mean = None
        if box_spec.has_player_means:
            box_for_prior = residualize_box_by_team(box_spec.box_by_player, team_map)
            player_off_mean, player_def_mean = player_box_prior_means(
                design.player_ids,
                box_by_player=box_for_prior,
                gamma_off=box_spec.gamma_off,
                gamma_def=box_spec.gamma_def,
            )
        prior_mean = build_prior_mean_vector(
            design,
            prior_artifact,
            player_off_mean=player_off_mean,
            player_def_mean=player_def_mean,
            include_box_gamma_prior=box_spec.include_in_design,
        )
        minutes_by_player = {}
        if not features.empty and "minutes" in features.columns:
            minutes_by_player = {
                int(r.player_id): float(r.minutes)
                for r in features[["player_id", "minutes"]].itertuples(index=False)
                if pd.notna(r.player_id)
            }
        fit = fit_hierarchical_rapm(
            design,
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_team_off=lambda_team_off if include_team else None,
            lambda_team_def=lambda_team_def if include_team else None,
            lambda_conf_off=lambda_conf_off if include_conference else None,
            lambda_conf_def=lambda_conf_def if include_conference else None,
            lambda_box=lambda_box if box_spec.include_in_design else None,
            prior_mean=prior_mean,
            box_by_player=box_spec.box_by_player if box_spec.include_in_design else None,
            frozen_box_gamma_off=box_spec.gamma_off if box_spec.has_player_means else None,
            frozen_box_gamma_def=box_spec.gamma_def if box_spec.has_player_means else None,
            frozen_box_feature_names=box_spec.feature_names if box_spec.has_player_means else None,
            minutes_by_player=minutes_by_player or None,
            center_within_team=True,
        )
        ratings = player_ratings_frame(
            player_ids=fit.player_ids,
            off=fit.off,
            defense=fit.defense,
            features=features,
            player_conference=design.player_conference,
            player_team=design.player_team,
            off_raw=fit.player_off_raw,
            def_raw=fit.player_def_raw,
            conf_off=fit.conf_off,
            conf_def=fit.conf_def,
            conference_names=fit.conference_names,
            team_off=fit.team_off,
            team_def=fit.team_def,
            team_ids=fit.team_ids,
        )
        teams = team_rollup(ratings, train_poss, games, team_lookup=team_lookup)
        team_rates = _team_rate_lookup(teams)

        if possessions_per_team is None:
            by_game_team = (
                train_poss.groupby(["game_id", "off_team_id"], as_index=False)
                .size()
                .rename(columns={"size": "n"})
            )
            ppt = float(by_game_team["n"].mean()) if not by_game_team.empty else 70.0
        else:
            ppt = float(possessions_per_team)

        game_preds = predict_game_margins_from_team_ratings(
            test_games,
            team_rates=team_rates,
            intercept=fit.intercept,
            home_advantage=fit.home,
            possessions_per_team=ppt,
        )
        # Oracular in-cut blend (upper bound) + causal rolling blend from prior cut.
        oracle_blend = tune_blend_weight(game_preds)
        if rolling_blend_w is None:
            applied_w = float(oracle_blend["blend_weight_rapm"])
        else:
            applied_w = float(rolling_blend_w)
        game_preds = apply_blend(game_preds, applied_w)
        game_preds["blend_weight_rapm"] = applied_w
        game_preds["oracle_blend_weight_rapm"] = float(oracle_blend["blend_weight_rapm"])
        game_preds["cut_date"] = cut
        all_game_preds.append(game_preds)
        team_summary = _summarize_game_preds(game_preds)
        team_summary["oracle_blend"] = {
            "blend_weight_rapm": oracle_blend["blend_weight_rapm"],
            "margin_rmse_blend": oracle_blend["margin_rmse_blend"],
        }
        # Update rolling weight for next cut from this cut's oracle (causal for future cuts).
        if blend_weight is None and oracle_blend["blend_weight_rapm"] is not None:
            rolling_blend_w = float(oracle_blend["blend_weight_rapm"])

        poss_summary: dict[str, Any] = {"n_possessions": int(len(test_poss))}
        if not test_poss.empty:
            # Residuals only — team/conf added once from possession team ids.
            player_off = {
                pid: float(coef)
                for pid, coef in zip(fit.player_ids, fit.player_off_raw or fit.off)
            }
            player_def = {
                pid: float(coef)
                for pid, coef in zip(fit.player_ids, fit.player_def_raw or fit.defense)
            }
            team_off = (
                {tid: float(v) for tid, v in zip(fit.team_ids, fit.team_off)}
                if fit.team_off is not None
                else None
            )
            team_def = (
                {tid: float(v) for tid, v in zip(fit.team_ids, fit.team_def)}
                if fit.team_def is not None
                else None
            )
            conf_off = (
                {name: float(v) for name, v in zip(fit.conference_names, fit.conf_off)}
                if fit.conf_off is not None
                else None
            )
            conf_def = (
                {name: float(v) for name, v in zip(fit.conference_names, fit.conf_def)}
                if fit.conf_def is not None
                else None
            )
            team_conference = {
                tid: design.player_conference.get(pid, "Unknown")
                for pid, tid in design.player_team.items()
            }
            pred = predict_possession_points(
                test_poss,
                intercept=fit.intercept,
                home=fit.home,
                player_off_coef=player_off,
                player_def_coef=player_def,
                team_off_coef=team_off,
                team_def_coef=team_def,
                conf_off_coef=conf_off,
                conf_def_coef=conf_def,
                team_conference=team_conference,
            )
            y = test_poss["points"].to_numpy(dtype=np.float64)
            poss_summary["possession_rmse"] = _rmse(y, pred)
            train_mean = float(train_poss["points"].mean())
            poss_summary["possession_rmse_mean_baseline"] = _rmse(
                y, np.full_like(y, train_mean)
            )

        stability = _player_stability(ratings, final_ratings)

        cut_reports.append(
            {
                "cut_date": cut,
                "skipped": False,
                "prior": prior,
                "n_train_possessions": int(len(train_poss)),
                "n_train_games": int(train_poss["game_id"].nunique()),
                "n_train_players": int(len(fit.player_ids)),
                "n_test_possessions": int(len(test_poss)),
                "possessions_per_team": ppt,
                "intercept_ppp": fit.intercept,
                "home_advantage_ppp": fit.home,
                "lambda_off": lambda_off,
                "lambda_def": lambda_def,
                "lambda_team_off": lambda_team_off if include_team else None,
                "lambda_team_def": lambda_team_def if include_team else None,
                "lambda_conf_off": lambda_conf_off if include_conference else None,
                "lambda_conf_def": lambda_conf_def if include_conference else None,
                "lambda_box": lambda_box if box_spec.include_in_design else None,
                "box_mode": box_spec.mode,
                "team": team_summary,
                "possessions": poss_summary,
                "player_stability_vs_final": stability,
            }
        )

    game_pred_frame = pd.concat(all_game_preds, ignore_index=True) if all_game_preds else pd.DataFrame()
    if not game_pred_frame.empty:
        game_pred_frame.to_parquet(paths.backtest_game_predictions, index=False)
    else:
        pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
                "neutral_site",
                "actual_margin",
                "pred_margin_rapm",
                "pred_margin_eff",
                "pred_margin_blend",
                "home_win_actual",
                "home_win_pred_rapm",
                "home_win_pred_blend",
                "abs_err_rapm",
                "abs_err_eff",
                "abs_err_blend",
                "blend_weight_rapm",
                "oracle_blend_weight_rapm",
                "cut_date",
            ]
        ).to_parquet(paths.backtest_game_predictions, index=False)

    overall_blend = (
        tune_blend_weight(game_pred_frame) if not game_pred_frame.empty else {"blend_weight_rapm": None}
    )
    report = {
        "season": season,
        "prior": prior,
        "prior_season": prior_season,
        "use_box": include_box,
        "box_mode": box_mode,
        "lambda_off": lambda_off,
        "lambda_def": lambda_def,
        "lambda_team_off": lambda_team_off if include_team else None,
        "lambda_team_def": lambda_team_def if include_team else None,
        "lambda_conf_off": lambda_conf_off if include_conference else None,
        "lambda_conf_def": lambda_conf_def if include_conference else None,
        "lambda_box": lambda_box,
        "overall_oracle_blend": overall_blend,
        "cut_dates": cuts,
        "cuts": cut_reports,
    }
    write_json(paths.backtest_metrics, report)
    logger.info("wrote temporal backtest metrics to %s", paths.backtest_metrics)
    return report
