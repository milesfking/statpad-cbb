from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from statpad_cbb.config import DEFAULT_SEASON, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="statpad-cbb: ingest and RAPM modeling")
    subparsers = parser.add_subparsers(dest="command")

    def add_season_data(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--season",
            type=int,
            default=DEFAULT_SEASON,
            help="Season-ending year (2026 = 2025-26). Default: %(default)s",
        )
        p.add_argument(
            "--data-dir",
            type=Path,
            default=None,
            help="Override data directory (default: STATPAD_DATA_DIR or ./data)",
        )

    ingest = subparsers.add_parser("ingest", help="Pull one season of CBBD data")
    add_season_data(ingest)
    ingest.add_argument("--force", action="store_true", help="Re-download even when raw files exist")
    ingest.add_argument("--skip-plays", action="store_true", help="Stop after teams/rosters/games")

    poss = subparsers.add_parser("build-possessions", help="Build possession table from PBP")
    add_season_data(poss)
    poss.add_argument("--max-games", type=int, default=None, help="Optional cap for debugging")

    feats = subparsers.add_parser("build-features", help="Build player-season box/minutes features")
    add_season_data(feats)

    fit = subparsers.add_parser("fit-rapm", help="Fit hierarchical RAPM and write ratings")
    add_season_data(fit)
    fit.add_argument(
        "--prior",
        choices=["hierarchical", "conference", "league_mean"],
        default="hierarchical",
        help="Shrinkage structure: hierarchical (player→team→conf+box), conference, or league_mean",
    )
    fit.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Optional prior season for group/box means (e.g. 2025 for a 2026 fit)",
    )
    fit.add_argument("--no-box", action="store_true", help="Disable box-feature player means")
    fit.add_argument(
        "--box-mode",
        choices=["prior_mean", "design"],
        default="prior_mean",
        help=(
            "How box enters hierarchical fits: prior_mean freezes γ from --prior-season "
            "and shrinks players toward μ=x'γ (default); design puts box in X (legacy)"
        ),
    )
    fit.add_argument("--lambda-off", type=float, default=None, help="Override offensive ridge penalty")
    fit.add_argument("--lambda-def", type=float, default=None, help="Override defensive ridge penalty")
    fit.add_argument("--lambda-team-off", type=float, default=None, help="Team offense penalty")
    fit.add_argument("--lambda-team-def", type=float, default=None, help="Team defense penalty")
    fit.add_argument("--lambda-conf-off", type=float, default=None, help="Conference offense penalty")
    fit.add_argument("--lambda-conf-def", type=float, default=None, help="Conference defense penalty")
    fit.add_argument("--lambda-box", type=float, default=None, help="Box feature ridge penalty (design mode)")
    fit.add_argument("--folds", type=int, default=5, help="Game-blocked CV folds")

    evaluate = subparsers.add_parser("evaluate", help="Refresh full team roll-up diagnostics")
    add_season_data(evaluate)

    eda = subparsers.add_parser("eda", help="Plot team and player rating diagnostics")
    add_season_data(eda)
    eda.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs")
    eda.add_argument("--top-n", type=int, default=40, help="Teams to show in net ranking bars")
    eda.add_argument("--team-id", type=int, default=None, help="Optional team_id for roster plots")
    eda.add_argument("--school", type=str, default=None, help="Optional school name for roster plots")
    eda.add_argument("--player-id", type=int, default=None, help="Optional player_id for player plots")
    eda.add_argument("--player", type=str, default=None, help="Optional player name for player plots")
    eda.add_argument(
        "--min-minutes",
        type=float,
        default=8.0,
        help="Minutes filter for roster player plots",
    )

    backtest = subparsers.add_parser(
        "backtest",
        help="Temporal RAPM backtest: fit through cut dates, predict later games",
    )
    add_season_data(backtest)
    backtest.add_argument(
        "--cut-date",
        action="append",
        dest="cut_dates",
        default=None,
        help="Cut date YYYY-MM-DD (repeatable). Default: Dec 1 / Jan 1 / Feb 1 / Mar 1",
    )
    backtest.add_argument(
        "--prior",
        choices=["hierarchical", "conference", "league_mean"],
        default="hierarchical",
        help="Player prior used in each cut-date fit",
    )
    backtest.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Optional prior season for group/box means",
    )
    backtest.add_argument("--no-box", action="store_true", help="Disable box-feature player means")
    backtest.add_argument(
        "--box-mode",
        choices=["prior_mean", "design"],
        default="prior_mean",
        help="Box encoding: prior_mean (frozen γ → player means) or design (legacy)",
    )
    backtest.add_argument("--lambda-off", type=float, default=None, help="Override offensive ridge penalty")
    backtest.add_argument("--lambda-def", type=float, default=None, help="Override defensive ridge penalty")
    backtest.add_argument("--lambda-team-off", type=float, default=None, help="Team offense penalty")
    backtest.add_argument("--lambda-team-def", type=float, default=None, help="Team defense penalty")
    backtest.add_argument("--lambda-conf-off", type=float, default=None, help="Conference offense penalty")
    backtest.add_argument("--lambda-conf-def", type=float, default=None, help="Conference defense penalty")
    backtest.add_argument("--lambda-box", type=float, default=None, help="Box feature ridge penalty (design mode)")
    backtest.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        help="Fixed RAPM weight in RAPM/efficiency blend (default: rolling oracle from prior cut)",
    )
    backtest.add_argument(
        "--possessions-per-team",
        type=float,
        default=None,
        help="Override possessions/team used to scale PPP margins to points",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        # Backward compatible: bare `python -m statpad_cbb` behaves like ingest.
        return main(["ingest", *(argv or [])])

    if args.command == "ingest":
        from statpad_cbb.ingest.pipeline import ingest_season

        settings = load_settings(data_dir=args.data_dir)
        if not settings.api_key:
            raw_dir = settings.data_dir / "raw" / "cbbd" / str(args.season)
            if not args.force and raw_dir.exists() and any(raw_dir.rglob("*.json")):
                logging.getLogger(__name__).warning(
                    "CBBD_API_KEY is unset; rebuilding canonical tables from existing raw files"
                )
            else:
                logging.error("CBBD_API_KEY is not set. Copy .env.example to .env and add your token.")
                return 1
        try:
            report = ingest_season(
                args.season,
                settings=settings,
                force=args.force,
                skip_plays=args.skip_plays,
                data_dir=args.data_dir,
            )
        except ValueError as exc:
            logging.error("%s", exc)
            return 1
        json.dump(
            {
                "season": report["season"],
                "games_count": report["games_count"],
                "games_missing_play_by_play_count": report["games_missing_play_by_play_count"],
                "games_pending_play_by_play_count": report["games_pending_play_by_play_count"],
                "requested_play_dates_count": report["requested_play_dates_count"],
                "empty_play_dates": report["empty_play_dates"],
                "failed_play_dates": report["failed_play_dates"],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    if args.command == "build-possessions":
        from statpad_cbb.possessions.pipeline import build_possessions

        report = build_possessions(args.season, data_dir=args.data_dir, max_games=args.max_games)
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "build-features":
        from statpad_cbb.features.pipeline import build_player_features

        report = build_player_features(args.season, data_dir=args.data_dir)
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "fit-rapm":
        from statpad_cbb.model.pipeline import fit_rapm

        report = fit_rapm(
            args.season,
            data_dir=args.data_dir,
            lambda_off=args.lambda_off,
            lambda_def=args.lambda_def,
            lambda_team_off=args.lambda_team_off,
            lambda_team_def=args.lambda_team_def,
            lambda_conf_off=args.lambda_conf_off,
            lambda_conf_def=args.lambda_conf_def,
            lambda_box=args.lambda_box,
            n_folds=args.folds,
            prior=args.prior,
            prior_season=args.prior_season,
            use_box=not args.no_box,
            box_mode=args.box_mode,
        )
        summary = {
            "season": report["season"],
            "prior": report["prior"],
            "prior_season": report.get("prior_season"),
            "use_box": report.get("use_box"),
            "box_mode": report.get("box_mode"),
            "n_possessions": report["n_possessions"],
            "n_players": report["n_players"],
            "n_teams_effects": report.get("n_teams_effects"),
            "n_conferences": report.get("n_conferences"),
            "n_box_features": report.get("n_box_features"),
            "n_box_design_cols": report.get("n_box_design_cols"),
            "n_teams": report["n_teams"],
            "intercept_ppp": report["intercept_ppp"],
            "home_advantage_ppp": report["home_advantage_ppp"],
            "lambda_off": report["lambda_off"],
            "lambda_def": report["lambda_def"],
            "lambda_team_off": report.get("lambda_team_off"),
            "lambda_team_def": report.get("lambda_team_def"),
            "lambda_conf_off": report.get("lambda_conf_off"),
            "lambda_conf_def": report.get("lambda_conf_def"),
            "lambda_box": report.get("lambda_box"),
            "full_sample_rmse": report["full_sample_rmse"],
            "team_net_vs_eff_margin_corr": report["team_net_vs_eff_margin_corr"],
            "top_conferences": (report.get("conferences") or [])[:5],
            "top_teams": report["teams"][:10],
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "evaluate":
        from statpad_cbb.model.pipeline import evaluate_rapm

        report = evaluate_rapm(args.season, data_dir=args.data_dir)
        summary = {
            "season": report["season"],
            "n_players": report["n_players"],
            "n_teams": report["n_teams"],
            "team_net_vs_eff_margin_corr": report["team_net_vs_eff_margin_corr"],
            "top_teams": report["teams"][:10],
            "team_ratings_path": "data/processed/cbbd/{}/team_ratings.json".format(args.season),
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "eda":
        from statpad_cbb.eda.plots import run_eda

        try:
            report = run_eda(
                args.season,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                team_id=args.team_id,
                school=args.school,
                player_id=args.player_id,
                player_name=args.player,
                top_n=args.top_n,
                min_minutes=args.min_minutes,
            )
        except FileNotFoundError as exc:
            logging.error("%s", exc)
            return 1
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if report.get("errors") else 0

    if args.command == "backtest":
        from statpad_cbb.eda.backtest_plots import plot_backtest
        from statpad_cbb.model.backtest import run_temporal_backtest

        try:
            report = run_temporal_backtest(
                args.season,
                data_dir=args.data_dir,
                cut_dates=args.cut_dates,
                lambda_off=args.lambda_off,
                lambda_def=args.lambda_def,
                lambda_team_off=args.lambda_team_off,
                lambda_team_def=args.lambda_team_def,
                lambda_conf_off=args.lambda_conf_off,
                lambda_conf_def=args.lambda_conf_def,
                lambda_box=args.lambda_box,
                possessions_per_team=args.possessions_per_team,
                prior=args.prior,
                prior_season=args.prior_season,
                use_box=not args.no_box,
                box_mode=args.box_mode,
                blend_weight=args.blend_weight,
            )
        except (FileNotFoundError, ValueError) as exc:
            logging.error("%s", exc)
            return 1
        try:
            plots = plot_backtest(args.season, data_dir=args.data_dir)
        except Exception as exc:  # plotting should not fail the backtest
            logging.warning("backtest plots skipped: %s", exc)
            plots = {}
        summary = {
            "season": report["season"],
            "prior": report.get("prior"),
            "prior_season": report.get("prior_season"),
            "lambda_off": report["lambda_off"],
            "lambda_def": report["lambda_def"],
            "lambda_team_off": report.get("lambda_team_off"),
            "lambda_team_def": report.get("lambda_team_def"),
            "lambda_conf_off": report.get("lambda_conf_off"),
            "lambda_conf_def": report.get("lambda_conf_def"),
            "lambda_box": report.get("lambda_box"),
            "overall_oracle_blend": report.get("overall_oracle_blend"),
            "cut_dates": report["cut_dates"],
            "cuts": [
                {
                    "cut_date": c["cut_date"],
                    "skipped": c.get("skipped", False),
                    "n_train_games": c.get("n_train_games"),
                    "team": c.get("team"),
                    "possessions": c.get("possessions"),
                    "player_stability_vs_final": c.get("player_stability_vs_final"),
                }
                for c in report["cuts"]
            ],
            "plots": plots,
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
