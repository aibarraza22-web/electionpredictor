"""FiveThirtyEight pollster-ratings dataset adapter.

Ingests real historical House/Senate general-election polls (1998-2022) and
the certified election margins attached to each polled race. Licensed
CC-BY-4.0 (https://github.com/fivethirtyeight/data).

Polls carry their median field date, so downstream feature construction can
enforce as-of cutoffs; race outcomes are keyed by cycle so they only enter
training sets for later cycles.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict

from .. import store
from .base import STATES, fetch, house_seat_key, senate_seat_key, sha256

URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
LICENSE = "CC-BY-4.0 (FiveThirtyEight / ABC News)"
SOURCE = "fivethirtyeight-raw-polls"


def _orient(row: dict) -> tuple[float, float, float, float] | None:
    """Return (dem_pct, rep_pct, dem_actual, rep_actual) or None if not D-vs-R."""
    parties = {row["cand1_party"], row["cand2_party"]}
    if parties != {"DEM", "REP"}:
        return None
    if row["cand1_party"] == "DEM":
        d, r = ("cand1", "cand2")
    else:
        d, r = ("cand2", "cand1")
    try:
        return (float(row[f"{d}_pct"]), float(row[f"{r}_pct"]),
                float(row[f"{d}_actual"]), float(row[f"{r}_actual"]))
    except (TypeError, ValueError):
        return None


def _seat_key(row: dict, extra_races: dict) -> tuple[str, str, str | None] | None:
    """Return (seat_key, chamber, district) for a raw poll row."""
    kind, location = row["type_simple"], row["location"]
    if kind == "House-G-US":
        # National generic congressional ballot: a shared environment input.
        return "us-generic", "national", None
    if kind == "Sen-G":
        if location not in STATES:
            return None
        # A state can host two Senate contests in one cycle (regular +
        # special). The dataset does not label specials, so the second
        # race_id observed for a (cycle, state) is keyed separately.
        key = (row["cycle"], location)
        first_race = extra_races.setdefault(key, row["race_id"])
        special = first_race != row["race_id"]
        return senate_seat_key(location, special), "senate", None
    if kind == "House-G":
        state, _, district = location.partition("-")
        if state not in STATES or not district.isdigit():
            return None
        return house_seat_key(state, district), "house", f"{int(district):02d}"
    return None


def parse(payload: bytes) -> tuple[list[dict], list[dict]]:
    """Normalize raw CSV bytes into (poll_rows, result_rows)."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    poll_rows: list[dict] = []
    results: dict[tuple[int, str], dict] = {}
    senate_race_ids: dict = {}
    latest_election: dict[tuple[int, str], str] = defaultdict(str)
    for row in reader:
        seat = _seat_key(row, senate_race_ids)
        if not seat:
            continue
        oriented = _orient(row)
        if not oriented:
            continue
        seat_key, chamber, district = seat
        dem_pct, rep_pct, dem_actual, rep_actual = oriented
        cycle = int(row["cycle"])
        state = row["location"].split("-")[0]
        poll_rows.append({
            "external_id": row["question_id"], "cycle": cycle, "chamber": chamber,
            "state": state, "district": district, "seat_key": seat_key,
            "pollster": row["pollster"], "methodology": row["methodology"] or None,
            "sample_size": float(row["samplesize"]) if row["samplesize"] not in ("", "NA") else None,
            "partisan": row["partisan"] if row["partisan"] not in ("", "NA") else None,
            "poll_date": row["polldate"], "election_date": row["electiondate"],
            "dem_pct": dem_pct, "rep_pct": rep_pct,
            "dem_margin": dem_pct - rep_pct, "source": SOURCE,
        })
        # Keep the decisive (latest election date, e.g. runoff) outcome.
        rk = (cycle, seat_key)
        if row["electiondate"] >= latest_election[rk]:
            latest_election[rk] = row["electiondate"]
            results[rk] = {
                "cycle": cycle, "chamber": chamber, "state": state,
                "district": district, "seat_key": seat_key,
                "dem_pct": dem_actual, "rep_pct": rep_actual,
                "dem_margin": dem_actual - rep_actual,
                "winner_party": "D" if dem_actual > rep_actual else "R",
                "source": SOURCE,
            }
    return poll_rows, list(results.values())


def ingest(url: str = URL) -> dict:
    payload = fetch(url)
    poll_rows, result_rows = parse(payload)
    inserted_polls = store.insert_rows("polls", poll_rows)
    inserted_results = store.insert_rows("election_results", result_rows)
    source_id = store.record_source(
        SOURCE, url, LICENSE, available_at=store.now(), sha256=sha256(payload),
        record_count=inserted_polls + inserted_results,
        note=f"{len(poll_rows)} polls parsed, {len(result_rows)} race outcomes")
    return {"source": SOURCE, "source_id": source_id, "polls": inserted_polls,
            "results": inserted_results}
