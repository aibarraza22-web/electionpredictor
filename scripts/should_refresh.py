"""Gate an hourly cron into an election-stage-aware refresh cadence."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ELECTION = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)


def refresh_decision(now: datetime, manual: bool = False) -> dict:
    now = now.astimezone(timezone.utc)
    days = (ELECTION.date() - now.date()).days
    if manual:
        run, cadence = True, "manual"
    elif days > 180 or days < 0:
        run, cadence = now.hour == 9, "daily"
    elif days > 60:
        run, cadence = now.hour % 6 == 3, "every-6-hours"
    elif days > 14:
        run, cadence = now.hour % 3 == 0, "every-3-hours"
    else:
        run, cadence = now.hour % 2 == 0, "every-2-hours"
    full_backtest = manual or (run and now.weekday() == 6 and now.hour == 9)
    return {"run": run, "full_backtest": full_backtest,
            "cadence": cadence, "days_until_election": days}


def main() -> int:
    decision = refresh_decision(
        datetime.now(timezone.utc),
        manual=os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch")
    print(json.dumps(decision))
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            for key, value in decision.items():
                rendered = str(value).lower() if isinstance(value, bool) else value
                handle.write(f"{key}={rendered}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
