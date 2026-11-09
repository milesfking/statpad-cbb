from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from statpad_cbb.config import DEFAULT_SEASON, load_settings
from statpad_cbb.ingest.pipeline import ingest_season


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest College Basketball Data games, rosters, and play-by-play.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Pull one season of CBBD data")
    ingest.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help="Season-ending year (2026 = 2025-26). Default: %(default)s",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when raw files already exist",
    )
    ingest.add_argument(
        "--skip-plays",
        action="store_true",
        help="Stop after teams, rosters, and games",
    )
    ingest.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: STATPAD_DATA_DIR or ./data)",
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
        args = parser.parse_args(["ingest", *(argv or [])])

    if args.command != "ingest":
        parser.print_help()
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())
