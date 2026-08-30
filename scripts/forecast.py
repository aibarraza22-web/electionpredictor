"""Build the current forecast: train, freeze snapshots, simulate control,
and run validation backtests. Requires ingested historical data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.forecast import build_forecasts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-backtests", action="store_true",
                        help="Use for frequent data-only refreshes; run full validation separately.")
    parser.add_argument("--force", action="store_true",
                        help="Publish even when the input fingerprint is unchanged.")
    args = parser.parse_args()
    store.init_db()
    summary = build_forecasts(with_backtests=not args.skip_backtests, force=args.force)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
