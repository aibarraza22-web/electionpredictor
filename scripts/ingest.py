"""Run data ingestion adapters.

Usage: python scripts/ingest.py [--sources fte_polls,legislators,medsl,fec]

Adapters that need unavailable credentials/network report their status and
are skipped without failing the whole run (exit code 1 only if *nothing*
succeeded).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.ingest import ADAPTERS  # noqa: E402

DEFAULT = "fte_polls,legislators,medsl,fec,votehub,polls_feed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=DEFAULT)
    args = parser.parse_args()
    store.init_db()
    succeeded = 0
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        if name not in ADAPTERS:
            print(f"[skip] unknown source {name!r}; known: {sorted(ADAPTERS)}")
            continue
        try:
            summary = ADAPTERS[name]()
            print(f"[ok] {name}: {json.dumps(summary, default=str)}")
            if "skipped" not in summary:
                succeeded += 1
        except Exception as exc:
            print(f"[fail] {name}: {exc}")
    print(f"table counts: {store.counts()}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
