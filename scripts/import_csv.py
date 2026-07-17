"""Import certified official results from a CSV file.

Usage: python scripts/import_csv.py path/to/results.csv [--url SOURCE_URL]

Expected header: cycle,chamber,state,district,dem_votes,rep_votes[,special]
See app/ingest/csv_results.py for the exact schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.ingest import csv_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--url", default=None, help="where these records were published")
    args = parser.parse_args()
    store.init_db()
    summary = csv_results.ingest_file(args.path, source_url=args.url)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
