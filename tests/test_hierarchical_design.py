import numpy as np
import pandas as pd

from statpad_cbb.model.backtest import apply_blend, tune_blend_weight
from statpad_cbb.model.design import build_design_matrix, build_player_maps
from statpad_cbb.model.priors import (
    build_prior_mean_vector,
    player_box_prior_means,
    resolve_box_prior,
)
from statpad_cbb.model.rapm import fit_hierarchical_rapm


def _toy_possessions() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append(
            {
                "game_id": i % 8,
                "points": 2 if i % 2 == 0 else 0,
                "home_offense": i % 2,
                "off_team_id": 100,
                "def_team_id": 200,
                "home_team_id": 100,
                "off_player_1": 1,
                "off_player_2": 2,
                "off_player_3": 3,
                "off_player_4": 4,
                "off_player_5": 5,
                "def_player_1": 6,
                "def_player_2": 7,
                "def_player_3": 8,
                "def_player_4": 9,
                "def_player_5": 10,
            }
        )
    return pd.DataFrame(rows)


def test_blend_weight_prefers_better_source():
    preds = pd.DataFrame(
        {
            "actual_margin": [10.0, -8.0, 4.0, -2.0],
            "pred_margin_rapm": [1.0, -1.0, 0.5, 0.0],
            "pred_margin_eff": [9.0, -7.0, 3.5, -1.5],
        }
    )
    tuned = tune_blend_weight(preds)
    assert tuned["blend_weight_rapm"] < 0.5
    blended = apply_blend(preds, tuned["blend_weight_rapm"])
    assert "pred_margin_blend" in blended.columns


def test_box_as_frozen_player_prior_mean():
    """Box should shrink players toward μ=x'γ without entering the design."""
    poss = _toy_possessions()
    roster = pd.DataFrame(
        [{"player_id": i, "team_id": 100 if i <= 5 else 200} for i in range(1, 11)]
    )
    teams = pd.DataFrame(
        [
            {"team_id": 100, "conference": "Big Ten Conference"},
            {"team_id": 200, "conference": "SEC"},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "player_id": i,
                "usage": 0.1 + 0.01 * i,
                "fg2m_100": 5 + i,
                "fg3m_100": 2,
                "ftm_100": 3,
                "ast_100": 4,
                "tov_100": 2,
                "orb_100": 1,
                "drb_100": 5,
                "stl_100": 1,
                "blk_100": 1,
            }
            for i in range(1, 11)
        ]
    )
    names = [
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
    prior_artifact = {
        "teams": {"100": {"off": 0.02, "def": 0.01}, "200": {"off": -0.01, "def": 0.0}},
        "conferences": {
            "Big Ten Conference": {"off": 0.01, "def": 0.0},
            "SEC": {"off": 0.0, "def": 0.01},
        },
        "box": {
            "feature_names": names,
            "gamma_off": [0.01] * 10,
            "gamma_def": [0.005] * 10,
            "mean": {n: 0.0 for n in names},
            "std": {n: 1.0 for n in names},
        },
        "intercept": 1.0,
        "home": 0.03,
    }
    box_spec = resolve_box_prior(
        features=features,
        prior_artifact=prior_artifact,
        use_box=True,
        box_mode="prior_mean",
    )
    assert box_spec.mode == "prior_mean"
    assert box_spec.include_in_design is False

    conf_map, team_map = build_player_maps(list(range(1, 11)), roster=roster, teams=teams)
    design = build_design_matrix(
        poss,
        player_conference=conf_map,
        player_team=team_map,
        include_conference=True,
        include_team=True,
    )
    assert design.n_box == 0

    off_mu, def_mu = player_box_prior_means(
        design.player_ids,
        box_by_player=box_spec.box_by_player,
        gamma_off=box_spec.gamma_off,
        gamma_def=box_spec.gamma_def,
    )
    prior_mean = build_prior_mean_vector(
        design,
        prior_artifact,
        player_off_mean=off_mu,
        player_def_mean=def_mu,
        include_box_gamma_prior=False,
    )
    assert prior_mean is not None
    assert np.any(prior_mean[design.player_off0 : design.player_off0 + design.n_off] != 0)

    fit = fit_hierarchical_rapm(
        design,
        lambda_off=50.0,
        lambda_def=50.0,
        lambda_team_off=10.0,
        lambda_team_def=10.0,
        lambda_conf_off=5.0,
        lambda_conf_def=5.0,
        prior_mean=prior_mean,
        frozen_box_gamma_off=box_spec.gamma_off,
        frozen_box_gamma_def=box_spec.gamma_def,
        frozen_box_feature_names=box_spec.feature_names,
    )
    # Frozen γ is preserved; design has no box columns.
    assert fit.box_gamma_off is not None
    assert np.allclose(fit.box_gamma_off, box_spec.gamma_off)
