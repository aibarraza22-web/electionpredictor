"""Auditable candidate-profile and campaign-event CSV adapters.

These feeds are opt-in because there is no complete, license-safe federal
source for candidate experience, withdrawals, endorsements, and scandals.
Every observation must carry its original public URL and the timestamp at
which it first became available.  The adapters never translate prose into a
numerical margin adjustment.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime

import httpx

from .. import store

LICENSE = "As declared by configured feed"
PROFILE_SOURCE = "candidate-profile-feed"
EVENT_SOURCE = "campaign-event-feed"

PROFILE_TYPES = {
    "incumbent", "challenger", "open_seat_candidate", "former_officeholder",
    "elected_official", "veteran",
    "business_executive", "first_time_candidate", "previous_overperformance",
    "major_party_support", "competitive_primary", "uncontested_primary",
}
EVENT_TYPES = {
    "announcement", "withdrawal", "replacement", "primary_result",
    "endorsement", "party_intervention", "independent_expenditure_surge",
    "scandal", "indictment", "legal_event", "health_event",
    "ballot_access", "election_system", "redistricting",
}


def _download(url: str) -> tuple[str, list[dict]]:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    text = response.text
    return text, list(csv.DictReader(io.StringIO(text)))


def _required(row: dict, fields: tuple[str, ...], row_number: int) -> None:
    missing = [name for name in fields if not str(row.get(name) or "").strip()]
    if missing:
        raise ValueError(f"row {row_number}: missing required fields {missing}")


def _validate_timestamp(value: str, field: str, row_number: int) -> str:
    rendered = value.strip()
    try:
        datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be ISO-8601") from exc
    return rendered


def _source_url(value: str, row_number: int) -> str:
    rendered = value.strip()
    if not rendered.startswith(("https://", "http://")):
        raise ValueError(f"row {row_number}: source_url must be an HTTP(S) URL")
    return rendered


def ingest_profiles(cycle: int = 2026, url: str | None = None) -> dict:
    url = url or os.getenv("CANDIDATE_PROFILES_URL")
    if not url:
        return {"source": PROFILE_SOURCE,
                "skipped": "CANDIDATE_PROFILES_URL not configured"}
    text, raw = _download(url)
    rows = []
    for number, item in enumerate(raw, 2):
        _required(item, ("seat_key", "candidate_id", "profile_type",
                         "observed_at", "available_at", "source_url"), number)
        kind = item["profile_type"].strip()
        if kind not in PROFILE_TYPES:
            raise ValueError(f"row {number}: unsupported profile_type {kind!r}")
        rows.append({
            "cycle": int(item.get("cycle") or cycle),
            "seat_key": item["seat_key"].strip(),
            "candidate_id": item["candidate_id"].strip(),
            "candidate": item.get("candidate") or None,
            "party": item.get("party") or None,
            "profile_type": kind,
            "value": float(item.get("value") or 1.0),
            "observed_at": _validate_timestamp(item["observed_at"], "observed_at", number),
            "available_at": _validate_timestamp(item["available_at"], "available_at", number),
            "source": PROFILE_SOURCE,
            "source_url": _source_url(item["source_url"], number),
        })
    source_id = store.record_source(
        PROFILE_SOURCE, url, LICENSE, available_at=store.now(),
        sha256=hashlib.sha256(text.encode()).hexdigest(), record_count=len(rows),
        note="structured candidate observations; never inferred from protected traits")
    for row in rows:
        row["source_id"] = source_id
    inserted = store.insert_rows("candidate_profiles", rows)
    return {"source": PROFILE_SOURCE, "candidate_profile_rows": inserted,
            "records_seen": len(rows)}


def ingest_events(cycle: int = 2026, url: str | None = None) -> dict:
    url = url or os.getenv("CAMPAIGN_EVENTS_URL")
    if not url:
        return {"source": EVENT_SOURCE,
                "skipped": "CAMPAIGN_EVENTS_URL not configured"}
    text, raw = _download(url)
    rows = []
    for number, item in enumerate(raw, 2):
        _required(item, ("external_id", "seat_key", "event_type", "event_date",
                         "available_at", "reliability", "source_url"), number)
        kind = item["event_type"].strip()
        if kind not in EVENT_TYPES:
            raise ValueError(f"row {number}: unsupported event_type {kind!r}")
        reliability = item["reliability"].strip().lower()
        if reliability not in {"primary", "high", "medium", "low"}:
            raise ValueError(f"row {number}: invalid reliability {reliability!r}")
        eligible = str(item.get("model_eligible") or "").lower() in {"1", "true", "yes"}
        details = item.get("details") or "{}"
        try:
            json.loads(details)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {number}: details must be valid JSON") from exc
        rows.append({
            "external_id": item["external_id"].strip(),
            "cycle": int(item.get("cycle") or cycle),
            "seat_key": item["seat_key"].strip(),
            "candidate_id": item.get("candidate_id") or None,
            "event_type": kind,
            "event_date": _validate_timestamp(item["event_date"], "event_date", number),
            "available_at": _validate_timestamp(item["available_at"], "available_at", number),
            "reliability": reliability,
            "model_eligible": eligible,
            "details": details,
            "source": EVENT_SOURCE,
            "source_url": _source_url(item["source_url"], number),
        })
    source_id = store.record_source(
        EVENT_SOURCE, url, LICENSE, available_at=store.now(),
        sha256=hashlib.sha256(text.encode()).hexdigest(), record_count=len(rows),
        note="source-backed event ledger; model eligibility is explicit")
    for row in rows:
        row["source_id"] = source_id
    inserted = store.insert_rows("campaign_events", rows)
    return {"source": EVENT_SOURCE, "campaign_event_rows": inserted,
            "records_seen": len(rows)}
