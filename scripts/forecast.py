"""Build the current forecast: train, freeze snapshots, simulate control,
and run validation backtests. Requires ingested historical data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.forecast import build_forecasts  # noqa: E402


def main() -> int:
    store.init_db()
    summary = build_forecasts()
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
