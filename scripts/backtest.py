"""Run expanding-window backtests against ingested data and persist metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.backtest import run_backtests  # noqa: E402
from app.forecast import MODEL_VERSION  # noqa: E402


def main() -> int:
    store.init_db()
    runs = run_backtests(MODEL_VERSION)
    if not runs:
        print("no backtests produced: ingest historical results first")
        return 1
    for run in runs:
        printable = {k: v for k, v in run.items() if k not in ("calibration", "by_cycle", "config")}
        print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
