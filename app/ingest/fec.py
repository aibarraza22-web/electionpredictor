"""Federal Election Commission campaign-finance adapter.

Pulls live 2026-cycle candidate financial totals from the official FEC API
(https://api.open.fec.gov). Requires ``FEC_API_KEY`` (free from
https://api.data.gov/signup/). Finance rows are displayed per race and stored
with provenance. Model 2026.18 uses the reporting vintages in a bounded
provisional capacity overlay while retaining the historical rejection of raw
receipts disparity.
"""
from __future__ import annotations

import hashlib
import json
import os

import httpx

from .. import store
from .base import STATES, house_seat_key, senate_seat_key

API = "https://api.open.fec.gov/v1/candidates/totals/"
LICENSE = "Public domain (U.S. federal government work)"
SOURCE = "fec-candidate-totals"

PARTY = {"DEMOCRATIC PARTY": "D", "REPUBLICAN PARTY": "R"}
STATUS_PROFILE = {"I": "incumbent", "C": "challenger", "O": "open_seat_candidate"}


def ingest(cycle: int = 2026, api_key: str | None = None) -> dict:
    api_key = api_key or os.getenv("FEC_API_KEY")
    if not api_key:
        return {"source": SOURCE, "skipped": "FEC_API_KEY not configured"}
    rows: list[dict] = []
    snapshots: list[dict] = []
    profiles: list[dict] = []
    retrieved_at = store.now()
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
                    candidate_id = item.get("candidate_id") or (
                        f"unresolved:{seat_key}:{item.get('name') or 'unknown'}")
                    coverage_end = (item.get("coverage_end_date") or retrieved_at)[:10]
                    party = PARTY.get((item.get("party_full") or "").upper(),
                                      item.get("party"))
                    committee_id = (item.get("committee_id") or
                                    item.get("principal_committee_id"))
                    if committee_id is not None and not isinstance(committee_id, str):
                        committee_id = json.dumps(committee_id, sort_keys=True, default=str)
                    rows.append({
                        "cycle": cycle, "seat_key": seat_key,
                        "candidate": item.get("name"),
                        "party": party,
                        "receipts": item.get("receipts"),
                        "disbursements": item.get("disbursements"),
                        "cash_on_hand": item.get("cash_on_hand_end_period"),
                        "as_of": coverage_end,
                        "source": SOURCE,
                    })
                    snapshots.append({
                        "cycle": cycle, "seat_key": seat_key,
                        "candidate_id": candidate_id,
                        "committee_id": committee_id,
                        "candidate": item.get("name"),
                        "party": party,
                        "receipts": item.get("receipts"),
                        "disbursements": item.get("disbursements"),
                        "cash_on_hand": item.get("cash_on_hand_end_period"),
                        "individual_contributions": (
                            item.get("individual_contributions")
                            if item.get("individual_contributions") is not None
                            else float(item.get("individual_itemized_contributions") or 0)
                            + float(item.get("individual_unitemized_contributions") or 0)),
                        "other_committee_contributions": item.get(
                            "other_political_committee_contributions"),
                        "candidate_contributions": item.get("candidate_contribution"),
                        "debts_owed": item.get("debts_owed_by_committee"),
                        "coverage_start": item.get("coverage_start_date"),
                        "coverage_end": coverage_end,
                        "retrieved_at": retrieved_at,
                        "payload_hash": hashlib.sha256(json.dumps(
                            item, sort_keys=True, default=str).encode()).hexdigest(),
                        "source": SOURCE,
                    })
                    profile_type = STATUS_PROFILE.get(str(
                        item.get("incumbent_challenge") or "").upper())
                    if profile_type and party in {"D", "R"}:
                        profiles.append({
                            "cycle": cycle, "seat_key": seat_key,
                            "candidate_id": candidate_id, "candidate": item.get("name"),
                            "party": party, "profile_type": profile_type, "value": 1.0,
                            "observed_at": f"{coverage_end}T00:00:00+00:00",
                            "available_at": item.get("load_date") or retrieved_at,
                            "source": SOURCE,
                            "source_url": (f"https://www.fec.gov/data/candidate/{candidate_id}/"
                                           f"?cycle={cycle}&election_full=true"),
                        })
                pages = data.get("pagination", {}).get("pages", 1)
                if page >= pages:
                    break
                page += 1
    # Keep the backwards-compatible latest view current and separately retain
    # immutable report vintages for timing/velocity research.
    latest_changed = store.upsert_finance_latest(rows)
    inserted = store.insert_rows("finance_snapshots", snapshots)
    source_id = store.record_source(
        SOURCE, API, LICENSE, available_at=store.now(), sha256=None,
        record_count=len(snapshots),
        note=f"cycle {cycle} candidate totals; {inserted} new content vintages")
    for profile in profiles:
        profile["source_id"] = source_id
    profile_inserted = store.insert_rows("candidate_profiles", profiles)
    return {"source": SOURCE, "finance_rows": latest_changed,
            "finance_snapshot_rows": inserted,
            "candidate_status_profile_rows": profile_inserted,
            "records_seen": len(snapshots)}
