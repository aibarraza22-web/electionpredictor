"""Data ingestion adapters.

Every adapter records provenance (source, url, license, retrieved_at,
available_at, sha256, record count) in ``raw_sources`` and writes normalized
rows with idempotent inserts, so re-running an ingest never duplicates data.
"""
import os

from . import fte_polls, legislators, medsl, fec, csv_results  # noqa: F401


def polls_feed() -> dict:
    """Live polling feed: any CSV in the 538 raw-polls schema, configured
    with ``POLLS_FEED_URL`` (e.g. a maintained 2026 polls aggregation)."""
    url = os.getenv("POLLS_FEED_URL")
    if not url:
        return {"source": "polls-feed", "skipped": "POLLS_FEED_URL not configured"}
    return fte_polls.ingest(url)


ADAPTERS = {
    "fte_polls": fte_polls.ingest,
    "legislators": legislators.ingest,
    "medsl": medsl.ingest,
    "fec": fec.ingest,
    "polls_feed": polls_feed,
}
