"""Backward-compatible ingest CLI entrypoint."""

from __future__ import annotations

import sys

from statpad_cbb.cli import main as _main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "ingest":
        args = ["ingest", *args]
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
