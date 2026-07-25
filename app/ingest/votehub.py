"""VoteHub live polling adapter (2026 cycle).

FiveThirtyEight stopped publishing in 2025; VoteHub (votehub.com) maintains a
free public polling API covering the generic congressional ballot and named
2026 races. This adapter ingests whatever it can parse defensively — schema
drift in a third-party API must degrade to a reported skip, never a crash or
an invented value. Runs where api.votehub.com is reachable (the scheduled
GitHub Actions pipeline; some sandboxes block it).
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from .. import store
from .base import STATES, house_seat_key, senate_seat_key, sha256

API = "https://api.votehub.com/polls"
LICENSE = "VoteHub public API; verify terms before redistribution"
SOURCE = "votehub-polls"
CYCLE = 2026

DEM_TOKENS = ("dem", "democrat")
REP_TOKENS = ("rep", "republican", "gop")

# VoteHub labels congressional races "us-senator"/"us-representative" (NOT
# "senate"/"house"), and its candidate polls carry only candidate names and
# percentages -- no party. Until this was handled, every one of those polls was
# silently discarded and the 2026 races had no polling at all: only the
# generic ballot survived, which attaches to no seat. Party is resolved by
# matching candidate surnames against FEC's candidate registry for that seat.
FEC_CANDIDATES = "https://api.open.fec.gov/v1/candidates/"
STATE_NAMES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
# A subject like "2026 Texas Democratic" is a PRIMARY poll; only party-less
# subjects ("2026 Texas") are general-election matchups.
PRIMARY_SUFFIXES = ("democratic", "republican", "primary")


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z ]", "", str(value or "").lower())


FEC_INDEX_BUDGET_SECONDS = 180.0


def _fec_party_index(cycle: int = CYCLE,
                     budget_seconds: float = FEC_INDEX_BUDGET_SECONDS) -> dict[str, list[tuple[str, str]]]:
    """{seat_key: [(surname, "DEM"|"REP"), ...]} from the FEC candidate registry.

    Best-effort: any failure yields an empty index, which simply means candidate
    polls are skipped rather than mis-attributed.
    """
    api_key = os.getenv("FEC_API_KEY") or "DEMO_KEY"
    index: dict[str, list[tuple[str, str]]] = {}
    deadline = time.monotonic() + budget_seconds
    # Senate first: it is the smaller half and the better-polled one, so a
    # truncated run still yields the most valuable attributions. Without an
    # FEC_API_KEY the shared DEMO_KEY is heavily throttled and this budget will
    # cut the walk short -- which degrades to FEWER attached polls, never to
    # wrongly attributed ones.
    for office in ("S", "H"):
        page = 1
        while page <= 40:
            if time.monotonic() > deadline:
                return index
            payload = None
            for attempt in range(5):
                try:
                    response = httpx.get(FEC_CANDIDATES, timeout=30, params={
                        "api_key": api_key, "cycle": cycle, "office": office,
                        "per_page": 100, "page": page})
                except httpx.HTTPError:
                    payload = None
                    break
                if response.status_code == 429:
                    time.sleep(3 + attempt * 2)
                    continue
                if response.status_code != 200:
                    payload = None
                    break
                payload = response.json()
                break
            if payload is None:
                # Keep whatever is already indexed (Senate is fetched first and
                # is the higher-value half) and move on rather than discarding.
                break
            for candidate in payload.get("results", []):
                party, state = candidate.get("party"), candidate.get("state")
                if party not in ("DEM", "REP") or state not in STATES:
                    continue
                if office == "S":
                    seat_key = senate_seat_key(state)
                else:
                    district = candidate.get("district")
                    if not district or not str(district).isdigit():
                        continue
                    seat_key = house_seat_key(state, district)
                surname = _norm_name(str(candidate.get("name", "")).split(",")[0])
                if surname:
                    index.setdefault(seat_key, []).append((surname, party))
            if page >= payload.get("pagination", {}).get("pages", 1):
                break
            page += 1
            time.sleep(1.1)
    return index


def _candidate_pcts(answers: list[dict], seat_key: str,
                    index: dict) -> tuple[float, float] | None:
    """(dem_pct, rep_pct) for a named-candidate poll, or None if unresolvable.

    Requires exactly one Democrat and one Republican to match, which also
    excludes primaries and ambiguous multi-candidate fields rather than
    guessing.
    """
    roster = index.get(seat_key) or []
    if not roster:
        return None
    found: dict[str, list[float]] = {}
    for answer in answers or []:
        tokens = set(_norm_name(answer.get("choice")).split())
        if not tokens:
            continue
        parties = {party for surname, party in roster if surname in tokens}
        if len(parties) != 1:
            continue
        try:
            pct = float(answer.get("pct"))
        except (TypeError, ValueError):
            continue
        found.setdefault(parties.pop(), []).append(pct)
    if sorted(found) != ["DEM", "REP"] or len(found["DEM"]) != 1 or len(found["REP"]) != 1:
        return None
    return found["DEM"][0], found["REP"][0]


def _party_pcts(answers: list[dict]) -> tuple[float, float] | None:
    dem = rep = None
    for answer in answers or []:
        label = str(answer.get("party") or answer.get("choice") or "").lower()
        try:
            pct = float(answer.get("pct"))
        except (TypeError, ValueError):
            continue
        if any(t in label for t in DEM_TOKENS) and dem is None:
            dem = pct
        elif any(t in label for t in REP_TOKENS) and rep is None:
            rep = pct
    if dem is None or rep is None:
        return None
    return dem, rep


def _seat_of(poll: dict, poll_type: str) -> tuple[str, str, str, str | None] | None:
    """(chamber, seat_key, state, district) for a congressional poll."""
    if "generic" in poll_type:
        return "national", "us-generic", "US", None
    subject = str(poll.get("subject") or "")
    if any(subject.strip().lower().endswith(s) for s in PRIMARY_SUFFIXES):
        return None  # primary matchup, not a general election
    if "senator" in poll_type or "senate" in poll_type:
        state = str(poll.get("state") or "").strip().upper()[:2]
        if state not in STATES:
            # VoteHub leaves `state` null for Senate polls; the state is in the
            # subject line, e.g. "2026 Michigan".
            state = next((ab for name, ab in STATE_NAMES.items() if name in subject), "")
        if state not in STATES:
            return None
        return "senate", senate_seat_key(state), state, None
    if "representative" in poll_type or "house" in poll_type:
        seat_name = str(poll.get("seat_name") or "").strip().upper()
        match = re.fullmatch(r"([A-Z]{2})-(\d{1,2})", seat_name)
        if not match or match.group(1) not in STATES:
            return None
        state, district = match.group(1), f"{int(match.group(2)):02d}"
        return "house", house_seat_key(state, district), state, district
    return None


def _normalize(poll: dict, party_index: dict | None = None) -> dict | None:
    poll_type = str(poll.get("poll_type") or poll.get("type") or "").lower()
    seat = _seat_of(poll, poll_type)
    if not seat:
        return None
    chamber, seat_key, state_out, district = seat
    date = str(poll.get("end_date") or poll.get("median_date")
               or poll.get("created_at") or "")[:10]
    if len(date) != 10:
        return None
    # Party-labelled answers (the generic ballot) resolve directly; named
    # candidate fields need the FEC registry to say who is which party.
    pcts = _party_pcts(poll.get("answers"))
    if not pcts and chamber != "national":
        pcts = _candidate_pcts(poll.get("answers"), seat_key, party_index or {})
    if not pcts:
        return None
    dem, rep = pcts
    state = state_out
    return {
        "external_id": str(poll.get("id") or f"{seat_key}-{date}-{dem}-{rep}"),
        "cycle": CYCLE, "chamber": chamber, "state": state_out,
        "district": district, "seat_key": seat_key,
        "pollster": poll.get("pollster") or poll.get("pollster_name"),
        "methodology": poll.get("methodology"),
        "sample_size": poll.get("sample_size"),
        "partisan": poll.get("partisan") or None,
        "poll_date": date,
        "election_date": f"{CYCLE}-11-03",
        "dem_pct": dem, "rep_pct": rep, "dem_margin": dem - rep,
        "source": SOURCE,
    }


def ingest(url: str = API) -> dict:
    rows: list[dict] = []
    skipped = 0
    # One unfiltered fetch: the previous version queried made-up poll_type
    # values ("senate-general"), so only the generic ballot ever came back.
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
        polls = payload if isinstance(payload, list) else payload.get("polls", [])
    except Exception:
        polls, skipped = [], 1
    congressional = [p for p in polls
                     if "senator" in str(p.get("poll_type") or "").lower()
                     or "representative" in str(p.get("poll_type") or "").lower()]
    # Only pay for the FEC registry when there are candidate polls to resolve.
    party_index = _fec_party_index() if congressional else {}
    for poll in polls:
        normalized = _normalize(poll, party_index)
        if normalized:
            rows.append(normalized)
    if not rows:
        return {"source": SOURCE,
                "skipped": f"no parseable polls ({skipped} endpoint failures); "
                           "schema may have changed - inspect api.votehub.com"}
    inserted = store.insert_rows("polls", rows)
    store.record_source(SOURCE, url, LICENSE, available_at=store.now(),
                        sha256=sha256(json.dumps([r["external_id"] for r in rows]).encode()),
                        record_count=inserted,
                        note=f"{len(rows)} live {CYCLE} polls parsed "
                             f"({sum(1 for r in rows if r['chamber'] != 'national')} attached to seats)")
    return {"source": SOURCE, "polls": inserted}
