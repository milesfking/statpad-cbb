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
)
from statpad_cbb.model.priors import (
    build_prior_mean_vector,
    extract_group_prior,
    player_box_prior_means,
    residualize_box_by_team,
    resolve_box_prior,
)
from statpad_cbb.model.rapm import cross_validate_lambdas, fit_hierarchical_rapm
from statpad_cbb.model.rollup import (
    player_ratings_frame,
    rating_dispersion_by_minutes,
    team_rollup,
    teams_records,
)

logger = logging.getLogger(__name__)

# Tuples: off, def, team_off, team_def, conf_off, conf_def, box
DEFAULT_HIER_DESIGN_BOX_GRID = [
    (250.0, 250.0, 25.0, 25.0, 10.0, 10.0, 10.0),
    (250.0, 250.0, 50.0, 50.0, 25.0, 25.0, 10.0),
    (500.0, 500.0, 50.0, 50.0, 25.0, 25.0, 10.0),
    (500.0, 500.0, 50.0, 50.0, 50.0, 50.0, 25.0),
    (100.0, 100.0, 25.0, 25.0, 10.0, 10.0, 5.0),
    (500.0, 500.0, 25.0, 25.0, 10.0, 10.0, 5.0),
]

# Box encoded as frozen player prior means — no λ_box in the design.
DEFAULT_HIER_PRIOR_MEAN_GRID = [
    (250.0, 250.0, 25.0, 25.0, 10.0, 10.0),
    (250.0, 250.0, 50.0, 50.0, 25.0, 25.0),
    (500.0, 500.0, 50.0, 50.0, 25.0, 25.0),
    (500.0, 500.0, 50.0, 50.0, 50.0, 50.0),
    (100.0, 100.0, 25.0, 25.0, 10.0, 10.0),
    (500.0, 500.0, 25.0, 25.0, 10.0, 10.0),
    # Stronger team/conf shrink: early season stays closer to priors.
    (500.0, 500.0, 100.0, 100.0, 50.0, 50.0),
    (250.0, 250.0, 100.0, 100.0, 50.0, 50.0),
]

DEFAULT_HIER_GRID = DEFAULT_HIER_DESIGN_BOX_GRID  # back-compat alias

DEFAULT_LEAGUE_GRID = [
    (1.0, 1.0),
    (25.0, 25.0),
    (100.0, 100.0),
    (250.0, 250.0),
    (500.0, 500.0),
]


def _load_games_frame(paths: SeasonPaths) -> pd.DataFrame:
    return pd.DataFrame(list(read_jsonl(paths.processed_games)))


def _load_team_lookup(paths: SeasonPaths) -> dict[int, dict[str, Any]]:
    return {
        int(row["team_id"]): row
        for row in read_jsonl(paths.processed_teams)
        if row.get("team_id") is not None
    }


def _load_roster_frame(paths: SeasonPaths) -> pd.DataFrame:
    return pd.DataFrame(list(read_jsonl(paths.processed_players)))


def _load_teams_frame(paths: SeasonPaths) -> pd.DataFrame:
    return pd.DataFrame(list(read_jsonl(paths.processed_teams)))


def _write_team_outputs(paths: SeasonPaths, teams: pd.DataFrame) -> list[dict[str, Any]]:
    teams.to_parquet(paths.team_ratings, index=False)
    records = teams_records(teams)
    write_json(paths.team_ratings_json, records)
    return records


def _attach_home_team(possessions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    meta = games[["game_id", "home_team_id", "away_team_id", "neutral_site"]].copy()
    out = possessions.merge(meta, on="game_id", how="left", suffixes=("", "_game"))
    if "home_team_id_game" in out.columns and "home_team_id" not in possessions.columns:
        out["home_team_id"] = out["home_team_id_game"]
    return out


def _group_tables(fit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conferences = []
    if fit.conf_off is not None and fit.conf_def is not None:
        for i, name in enumerate(fit.conference_names):
            conferences.append(
                {
                    "conference": name,
                    "off_rating": float(fit.conf_off[i] * 100.0),
                    "def_rating": float(fit.conf_def[i] * 100.0),
                    "net_rating": float((fit.conf_off[i] + fit.conf_def[i]) * 100.0),
                }
            )
        conferences.sort(key=lambda r: r["net_rating"], reverse=True)

    teams = []
    if fit.team_off is not None and fit.team_def is not None:
        for i, tid in enumerate(fit.team_ids):
            teams.append(
                {
                    "team_id": int(tid),
                    "off_rating": float(fit.team_off[i] * 100.0),
                    "def_rating": float(fit.team_def[i] * 100.0),
                    "net_rating": float((fit.team_off[i] + fit.team_def[i]) * 100.0),
                }
            )
        teams.sort(key=lambda r: r["net_rating"], reverse=True)
    return conferences, teams


def fit_rapm(
    season: int,
    *,
    data_dir: Path | None = None,
    lambda_off: float | None = None,
    lambda_def: float | None = None,
    lambda_team_off: float | None = None,
    lambda_team_def: float | None = None,
    lambda_conf_off: float | None = None,
    lambda_conf_def: float | None = None,
    lambda_box: float | None = None,
    n_folds: int = 5,
    prior: str = "hierarchical",
    prior_season: int | None = None,
    use_box: bool = True,
    box_mode: str = "prior_mean",
) -> dict[str, Any]:
    """Fit hierarchical RAPM.

    prior='hierarchical' => player + team + conference (+ optional box)
    prior='league_mean' => classic zero-mean player RAPM
    prior_season => load group/box prior means from that season's season_priors.json

    box_mode='prior_mean' (default): freeze γ from prior_season and shrink
    player effects toward μ_i = x_i'γ. Team/conf beliefs update from the
    possession likelihood toward their priors — box does not enter X.
    box_mode='design': legacy; box sums in X and γ is estimated.
    """
    if prior not in {"hierarchical", "conference", "league_mean"}:
        raise ValueError("prior must be hierarchical, conference, or league_mean")
    if box_mode not in {"prior_mean", "design"}:
        raise ValueError("box_mode must be 'prior_mean' or 'design'")
    # Treat legacy 'conference' as conf nesting without team/box.
    include_team = prior == "hierarchical"
    include_conference = prior in {"hierarchical", "conference"}
    want_box = use_box and prior == "hierarchical"

    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    paths.ensure()

    possessions = pd.read_parquet(paths.possessions)
    if possessions.empty:
        raise ValueError("Possessions table is empty")

    features = (
        pd.read_parquet(paths.player_features)
        if paths.player_features.exists()
        else pd.DataFrame()
    )
    roster = _load_roster_frame(paths)
    teams_frame = _load_teams_frame(paths)
    games_df = _load_games_frame(paths)
    possessions = _attach_home_team(possessions, games_df)

    player_ids = sorted(
        {int(pid) for col in OFF_COLS + DEF_COLS for pid in possessions[col].tolist()}
    )
    conf_map, team_map = build_player_maps(player_ids, roster=roster, teams=teams_frame)

    prior_artifact = None
    if prior_season is not None:
        prior_paths = SeasonPaths(settings.data_dir, prior_season)
        if prior_paths.season_priors.exists():
            prior_artifact = json.loads(prior_paths.season_priors.read_text())
            logger.info("loaded priors from season %s", prior_season)
        else:
            logger.warning("prior_season=%s has no season_priors.json; continuing without", prior_season)

    box_spec = resolve_box_prior(
        features=features,
        prior_artifact=prior_artifact,
        use_box=want_box,
        box_mode=box_mode,
    )
    if want_box and box_mode == "prior_mean" and box_spec.mode == "design":
        logger.warning(
            "box_mode=prior_mean requested but prior γ unavailable; falling back to design box"
        )
    elif box_spec.mode == "prior_mean":
        logger.info("box as frozen player prior means (%s features)", len(box_spec.feature_names))

    design = build_design_matrix(
        possessions,
        player_conference=conf_map,
        player_team=team_map,
        include_conference=include_conference,
        include_team=include_team,
        box_by_player=box_spec.box_by_player if box_spec.include_in_design else None,
        box_feature_names=box_spec.feature_names if box_spec.include_in_design else None,
        box_mean=box_spec.mean if box_spec.include_in_design else None,
        box_std=box_spec.std if box_spec.include_in_design else None,
    )

    player_off_mean = player_def_mean = None
    box_for_prior = box_spec.box_by_player
    if box_spec.has_player_means:
        # Residualize within team so box μ does not restack team/conf priors.
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
    logger.info(
        "design: prior=%s box_mode=%s rows=%s cols=%s players=%s teams=%s confs=%s box_cols=%s",
        prior,
        box_spec.mode,
        design.x.shape[0],
        design.x.shape[1],
        design.n_off,
        design.n_team,
        design.n_conf,
        design.n_box,
    )

    need_cv = lambda_off is None or lambda_def is None
    if include_conference and not need_cv:
        if include_team and (lambda_team_off is None or lambda_team_def is None):
            need_cv = True
        elif lambda_conf_off is None or lambda_conf_def is None:
            need_cv = True
        elif box_spec.include_in_design and lambda_box is None:
            need_cv = True

    if need_cv:
        if not include_conference:
            grid = DEFAULT_LEAGUE_GRID
        elif box_spec.include_in_design:
            grid = DEFAULT_HIER_DESIGN_BOX_GRID
        else:
            grid = DEFAULT_HIER_PRIOR_MEAN_GRID
        logger.info("CV on game_margin_rmse (%s candidates, %s folds)", len(grid), n_folds)
        cv_report = cross_validate_lambdas(
            design,
            grid,
            n_folds=n_folds,
            possessions=possessions,
            metric="game_margin_rmse",
            # Design-mode only: post-hoc x'γ on player totals for CV metrics parity.
            box_by_player=box_spec.box_by_player if box_spec.include_in_design else None,
            prior_mean=prior_mean,
        )
        best = cv_report["best"]
        if best is None:
            raise RuntimeError("Cross-validation failed")
        lambda_off = float(best["lambda_off"])
        lambda_def = float(best["lambda_def"])
        lambda_team_off = best.get("lambda_team_off")
        lambda_team_def = best.get("lambda_team_def")
        lambda_conf_off = best.get("lambda_conf_off")
        lambda_conf_def = best.get("lambda_conf_def")
        lambda_box = best.get("lambda_box")
        logger.info("selected %s (score=%.4f)", best, best["score"])
    else:
        cv_report = {
            "grid": [],
            "best": {
                "lambda_off": lambda_off,
                "lambda_def": lambda_def,
                "lambda_team_off": lambda_team_off,
                "lambda_team_def": lambda_team_def,
                "lambda_conf_off": lambda_conf_off,
                "lambda_conf_def": lambda_conf_def,
                "lambda_box": lambda_box,
                "score": None,
            },
        }

    minutes_by_player = {}
    if not features.empty and "minutes" in features.columns:
        minutes_by_player = {
            int(r.player_id): float(r.minutes)
            for r in features[["player_id", "minutes"]].itertuples(index=False)
            if pd.notna(r.player_id)
        }
    final = fit_hierarchical_rapm(
        design,
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_team_off=lambda_team_off,
        lambda_team_def=lambda_team_def,
        lambda_conf_off=lambda_conf_off,
        lambda_conf_def=lambda_conf_def,
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
        player_ids=final.player_ids,
        off=final.off,
        defense=final.defense,
        features=features,
        player_conference=design.player_conference,
        player_team=design.player_team,
        off_raw=final.player_off_raw,
        def_raw=final.player_def_raw,
        conf_off=final.conf_off,
        conf_def=final.conf_def,
        conference_names=final.conference_names,
        team_off=final.team_off,
        team_def=final.team_def,
        team_ids=final.team_ids,
    )
    if player_off_mean is not None and player_def_mean is not None:
        prior_frame = pd.DataFrame(
            {
                "player_id": design.player_ids,
                "off_box_prior": player_off_mean * 100.0,
                "def_box_prior": player_def_mean * 100.0,
            }
        )
        ratings = ratings.merge(prior_frame, on="player_id", how="left")
    ratings.to_parquet(paths.player_ratings, index=False)

    team_lookup = _load_team_lookup(paths)
    teams = team_rollup(ratings, possessions, games_df, team_lookup=team_lookup)
    team_records = _write_team_outputs(paths, teams)
    conferences, team_effects = _group_tables(final)
    if conferences:
        write_json(paths.processed_season / "conference_ratings.json", conferences)

    # Save priors for future seasons (pass through frozen γ when not re-estimated).
    prior_payload = extract_group_prior(final, design, box_spec=box_spec)
    prior_payload["season"] = season
    write_json(paths.season_priors, prior_payload)

    corr = None
    if not teams.empty and teams["eff_margin"].notna().any() and teams["net_rating"].notna().any():
        corr = float(teams["net_rating"].corr(teams["eff_margin"]))

    metrics = {
        "season": season,
        "prior": prior,
        "prior_season": prior_season,
        "use_box": want_box and box_spec.mode != "none",
        "box_mode": box_spec.mode,
        "n_possessions": int(len(possessions)),
        "n_players": int(design.n_off),
        "n_teams_effects": int(design.n_team),
        "n_conferences": int(design.n_conf),
        "n_box_features": int(len(box_spec.feature_names) if box_spec.mode != "none" else 0),
        "n_box_design_cols": int(design.n_box),
        "n_teams": int(len(teams)),
        "intercept_ppp": final.intercept,
        "home_advantage_ppp": final.home,
        "lambda_off": lambda_off,
        "lambda_def": lambda_def,
        "lambda_team_off": lambda_team_off,
        "lambda_team_def": lambda_team_def,
        "lambda_conf_off": lambda_conf_off,
        "lambda_conf_def": lambda_conf_def,
        "lambda_box": lambda_box if box_spec.include_in_design else None,
        "full_sample_rmse": final.rmse,
        "cv": cv_report,
        "team_net_vs_eff_margin_corr": corr,
        "rating_dispersion_by_minutes": rating_dispersion_by_minutes(ratings),
        "conferences": conferences,
        "team_effects_top": team_effects[:25],
        "teams": team_records,
    }
    write_json(paths.model_metrics, metrics)
    logger.info(
        "fit complete prior=%s box_mode=%s intercept=%.3f team_corr=%s rmse=%.4f",
        prior,
        box_spec.mode,
        final.intercept,
        corr,
        final.rmse,
    )
    return metrics


def evaluate_rapm(
    season: int,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    if not paths.player_ratings.exists() or not paths.possessions.exists():
        raise FileNotFoundError("Need player_ratings and possessions; run fit-rapm first")

    ratings = pd.read_parquet(paths.player_ratings)
    possessions = pd.read_parquet(paths.possessions)
    games_df = _load_games_frame(paths)
    team_lookup = _load_team_lookup(paths)
    teams = team_rollup(ratings, possessions, games_df, team_lookup=team_lookup)
    team_records = _write_team_outputs(paths, teams)

    corr = None
    if not teams.empty and teams["eff_margin"].notna().any():
        corr = float(teams["net_rating"].corr(teams["eff_margin"]))

    report = {
        "season": season,
        "n_players": int(len(ratings)),
        "n_teams": int(len(teams)),
        "team_net_vs_eff_margin_corr": corr,
        "rating_dispersion_by_minutes": rating_dispersion_by_minutes(ratings),
        "teams": team_records,
    }
    existing: dict[str, Any] = {}
    if paths.model_metrics.exists():
        existing = json.loads(paths.model_metrics.read_text())
    existing.update(
        {
            "n_teams": report["n_teams"],
            "team_net_vs_eff_margin_corr": corr,
            "rating_dispersion_by_minutes": report["rating_dispersion_by_minutes"],
            "teams": team_records,
        }
    )
    write_json(paths.model_metrics, existing)
    return report
