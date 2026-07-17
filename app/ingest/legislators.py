"""Incumbency adapter: the @unitedstates ``congress-legislators`` dataset.

Public-domain (CC0) community-maintained roster of current members of
Congress, used to build the real 2026 race universe: House incumbents per
district, Senate seats up in 2026 (class 2 regular plus class 3 specials for
appointed seats), and the count of Democratic-caucus Senate seats not up.
"""
from __future__ import annotations

import json

from .. import store
from .base import STATES, fetch, house_seat_key, senate_seat_key, sha256

URL = "https://raw.githubusercontent.com/unitedstates/congress-legislators/gh-pages/legislators-current.json"
LICENSE = "CC0-1.0 (public domain)"
SOURCE = "unitedstates-congress-legislators"

PARTY = {"Democrat": "D", "Republican": "R", "Independent": "I"}
# Independents currently caucusing with the Democratic conference.
DEM_CAUCUS_PARTIES = ("Democrat", "Independent")
CYCLE = 2026
NEXT_CONGRESS_START = "2027-06-01"  # terms ending before this are up in 2026


def parse(payload: bytes) -> tuple[list[dict], int]:
    members = json.loads(payload.decode("utf-8"))
    rows: list[dict] = []
    dem_not_up = 0
    for member in members:
        term = member["terms"][-1]
        state = term["state"]
        if state not in STATES:
            continue  # delegates and resident commissioners have no vote
        name = member["name"].get("official_full") or \
            f"{member['name'].get('first', '')} {member['name'].get('last', '')}".strip()
        party = PARTY.get(term["party"], term["party"])
        if term["type"] == "rep":
            rows.append({
                "cycle": CYCLE, "chamber": "house", "state": state,
                "district": f"{max(int(term.get('district') or 1), 1):02d}",
                "seat_key": house_seat_key(state, term.get("district") or 1),
                "party": party, "name": name, "source": SOURCE,
            })
        elif term["type"] == "sen":
            if term["end"] <= NEXT_CONGRESS_START:
                # Class 2 seats are regularly scheduled in 2026; any other
                # class ending now is an appointed seat facing a special.
                special = term.get("class") != 2
                rows.append({
                    "cycle": CYCLE, "chamber": "senate", "state": state,
                    "district": None,
                    "seat_key": senate_seat_key(state, special),
                    "party": party, "name": name, "source": SOURCE,
                })
            elif term["party"] in DEM_CAUCUS_PARTIES:
                dem_not_up += 1
    return rows, dem_not_up


def ingest(url: str = URL) -> dict:
    payload = fetch(url)
    rows, dem_not_up = parse(payload)
    inserted = store.insert_rows("incumbents", rows)
    store.set_meta("senate_dem_seats_not_up", str(dem_not_up))
    source_id = store.record_source(
        SOURCE, url, LICENSE, available_at=store.now(), sha256=sha256(payload),
        record_count=inserted, note=f"{len(rows)} incumbents for cycle {CYCLE}")
    return {"source": SOURCE, "source_id": source_id, "incumbents": inserted,
            "senate_dem_seats_not_up": dem_not_up}
