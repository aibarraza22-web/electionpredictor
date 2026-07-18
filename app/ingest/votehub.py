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

import httpx

from .. import store
from .base import STATES, senate_seat_key, sha256

API = "https://api.votehub.com/polls"
LICENSE = "VoteHub public API; verify terms before redistribution"
SOURCE = "votehub-polls"
CYCLE = 2026

DEM_TOKENS = ("dem", "democrat")
REP_TOKENS = ("rep", "republican", "gop")


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


def _normalize(poll: dict) -> dict | None:
    pcts = _party_pcts(poll.get("answers"))
    if not pcts:
        return None
    poll_type = str(poll.get("poll_type") or poll.get("type") or "").lower()
    state = str(poll.get("state") or "").strip().upper()[:2]
    date = str(poll.get("end_date") or poll.get("median_date")
               or poll.get("created_at") or "")[:10]
    if len(date) != 10:
        return None
    if "generic" in poll_type:
        chamber, seat_key, state_out, district = "national", "us-generic", "US", None
    elif "senate" in poll_type and state in STATES:
        chamber, seat_key, state_out, district = "senate", senate_seat_key(state), state, None
    else:
        return None  # only shapes we can attribute to a seat with confidence
    dem, rep = pcts
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
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for poll_type in ("generic-ballot", "senate-general", "approval"):
            try:
                response = client.get(url, params={"poll_type": poll_type})
                response.raise_for_status()
                payload = response.json()
            except Exception:
                skipped += 1
                continue
            polls = payload if isinstance(payload, list) else payload.get("polls", [])
            for poll in polls:
                normalized = _normalize(poll)
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
                        note=f"{len(rows)} live {CYCLE} polls parsed")
    return {"source": SOURCE, "polls": inserted}
