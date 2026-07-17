"""Federal Election Commission campaign-finance adapter.

Pulls live 2026-cycle candidate financial totals from the official FEC API
(https://api.open.fec.gov). Requires ``FEC_API_KEY`` (free from
https://api.data.gov/signup/). Finance rows are displayed per race and stored
with provenance; they are deliberately NOT a model input until a finance term
has passed vintage-correct backtesting.
"""
from __future__ import annotations

import json
import os

import httpx

from .. import store
from .base import STATES, house_seat_key, senate_seat_key

API = "https://api.open.fec.gov/v1/candidates/totals/"
LICENSE = "Public domain (U.S. federal government work)"
SOURCE = "fec-candidate-totals"

PARTY = {"DEMOCRATIC PARTY": "D", "REPUBLICAN PARTY": "R"}


def ingest(cycle: int = 2026, api_key: str | None = None) -> dict:
    api_key = api_key or os.getenv("FEC_API_KEY")
    if not api_key:
        return {"source": SOURCE, "skipped": "FEC_API_KEY not configured"}
    rows: list[dict] = []
    with httpx.Client(timeout=60.0) as client:
        for office in ("H", "S"):
            page = 1
            while True:
                response = client.get(API, params={
                    "api_key": api_key, "cycle": cycle, "office": office,
                    "election_full": "true", "per_page": 100, "page": page,
                    "is_active_candidate": "true"})
                response.raise_for_status()
                data = response.json()
                for item in data.get("results", []):
                    state = item.get("state")
                    if state not in STATES:
                        continue
                    if office == "H":
                        seat_key = house_seat_key(state, item.get("district") or 1)
                    else:
                        seat_key = senate_seat_key(state)
                    rows.append({
                        "cycle": cycle, "seat_key": seat_key,
                        "candidate": item.get("name"),
                        "party": PARTY.get((item.get("party_full") or "").upper(),
                                           item.get("party")),
                        "receipts": item.get("receipts"),
                        "disbursements": item.get("disbursements"),
                        "cash_on_hand": item.get("cash_on_hand_end_period"),
                        "as_of": item.get("coverage_end_date"),
                        "source": SOURCE,
                    })
                pages = data.get("pagination", {}).get("pages", 1)
                if page >= pages:
                    break
                page += 1
    inserted = store.insert_rows("finance", rows)
    store.record_source(
        SOURCE, API, LICENSE, available_at=store.now(), sha256=None,
        record_count=inserted, note=f"cycle {cycle} candidate totals")
    return {"source": SOURCE, "finance_rows": inserted}
