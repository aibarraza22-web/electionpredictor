"""MIT Election Data + Science Lab (MEDSL) constituency returns adapter.

Downloads the canonical candidate-level general-election returns from Harvard
Dataverse (CC0 public domain):

* U.S. House 1976-2022  — doi:10.7910/DVN/IG0UN2
* U.S. Senate 1976-2020 — doi:10.7910/DVN/PEJ5QU

and normalizes them to per-seat two-party margins. This is the authoritative
full-coverage historical results source; without it the model only sees
seat history for districts that happened to be polled (the FiveThirtyEight
adapter's coverage), which is most House districts.

The House file is guestbook-gated: Dataverse returns 400 with "You may not
download this file without the required Guestbook response" for
unauthenticated requests, and — confirmed empirically — a ``DATAVERSE_API_KEY``
authenticated request alone was not sufficient to satisfy it either, even
from an account that had already agreed to the guestbook via the web UI.
Rather than depend on a fragile, unverified authentication path for data
that barely changes (results are final once certified; there's no reason to
re-fetch daily), this adapter ships a **bundled vintage snapshot**:
``data/vintage/medsl_us_house_1976_2024.tab``, downloaded through the
Dataverse web UI after satisfying the guestbook (see DATA_SOURCES.md for
provenance), used directly rather than fetched live. It is the primary and
default path for House results.

``DATAVERSE_API_KEY`` (see ``_auth_headers``) remains supported as a live
fallback/refresh mechanism, used automatically when the bundled snapshot is
absent (e.g. a checkout that stripped ``data/vintage``); it may or may not
work depending on Dataverse's guestbook semantics, so it is not relied on.

The Senate file (``doi:10.7910/DVN/PEJ5QU``) is not guestbook-gated and is
always fetched live.
"""
from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import httpx

from .. import store
from .base import STATES, fetch, house_seat_key, senate_seat_key, sha256, two_party_margin

DATAVERSE = "https://dataverse.harvard.edu"
DATASETS = {
    "house": "doi:10.7910/DVN/IG0UN2",
    "senate": "doi:10.7910/DVN/PEJ5QU",
}
LICENSE = "CC0-1.0 (MIT Election Data + Science Lab)"
SOURCE = "medsl-constituency-returns"

DEM_PARTIES = {"DEMOCRAT", "DEMOCRATIC-FARMER-LABOR", "DEMOCRATIC-NPL"}
REP_PARTIES = {"REPUBLICAN"}

BUNDLED_HOUSE_FILE = Path(__file__).resolve().parents[2] / "data" / "vintage" / "medsl_us_house_1976_2024.tab"
BUNDLED_HOUSE_PROVENANCE_URL = f"{DATAVERSE}/dataset.xhtml?persistentId={DATASETS['house']}"


def _auth_headers() -> dict | None:
    key = os.getenv("DATAVERSE_API_KEY")
    return {"X-Dataverse-key": key} if key else None


def _dataset_file_urls(doi: str) -> list[tuple[str, str]]:
    listing = json.loads(fetch(
        f"{DATAVERSE}/api/datasets/:persistentId?persistentId={doi}",
        headers=_auth_headers()).decode("utf-8"))
    files = listing["data"]["latestVersion"]["files"]
    urls = []
    for f in files:
        df = f["dataFile"]
        label = (df.get("originalFileName") or df.get("filename") or "").lower()
        if label.endswith((".csv", ".tab")):
            urls.append((label, df["id"]))
    return urls


def _fetch_datafile(file_id: int) -> bytes:
    """Try the ingested/archival copy, then the distinct original upload
    (some files have no archival copy and 400 on the plain endpoint) —
    both authenticated with DATAVERSE_API_KEY when configured, required
    for guestbook-gated files (see module docstring)."""
    headers = _auth_headers()
    urls = [
        f"{DATAVERSE}/api/access/datafile/{file_id}",
        f"{DATAVERSE}/api/access/datafile/{file_id}?format=original",
    ]
    last_error: httpx.HTTPStatusError | None = None
    for url in urls:
        try:
            return fetch(url, headers=headers)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (400, 403):
                raise
            last_error = exc
    raise last_error


def parse(payload: bytes, chamber: str) -> list[dict]:
    text = payload.decode("utf-8", errors="replace")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    votes: dict[tuple, dict] = defaultdict(lambda: {"D": 0.0, "R": 0.0, "other": 0.0})
    for row in reader:
        row = {k.lower(): (v or "").strip().strip('"') for k, v in row.items() if k}
        stage = row.get("stage", "GEN").upper()
        if stage not in ("GEN", "GENERAL"):
            continue
        state = row.get("state_po", "")
        if state not in STATES:
            continue
        year = int(row["year"])
        special = row.get("special", "FALSE").upper() in ("TRUE", "1")
        party = (row.get("party_simplified") or row.get("party") or "").upper()
        try:
            cand_votes = float(row.get("candidatevotes") or 0)
        except ValueError:
            continue
        if chamber == "house":
            key = (year, house_seat_key(state, row.get("district") or 0), state,
                   f"{max(int(row.get('district') or 1), 1):02d}", special)
        else:
            key = (year, senate_seat_key(state, special), state, None, special)
        bucket = votes[key]
        if party in DEM_PARTIES:
            bucket["D"] += cand_votes
        elif party in REP_PARTIES:
            bucket["R"] += cand_votes
        else:
            bucket["other"] += cand_votes
    rows = []
    for (year, seat_key, state, district, _special), bucket in votes.items():
        margin = two_party_margin(bucket["D"], bucket["R"])
        if margin is None:
            continue  # uncontested by one major party: no two-party margin
        total = bucket["D"] + bucket["R"]
        rows.append({
            "cycle": year, "chamber": chamber, "state": state, "district": district,
            "seat_key": seat_key,
            "dem_pct": bucket["D"] / total * 100.0, "rep_pct": bucket["R"] / total * 100.0,
            "dem_margin": margin,
            "winner_party": "D" if margin > 0 else "R",
            "source": SOURCE,
        })
    return rows


def _ingest_bundled_house() -> dict:
    payload = BUNDLED_HOUSE_FILE.read_bytes()
    rows = parse(payload, "house")
    inserted = store.insert_rows("election_results", rows)
    store.record_source(
        SOURCE, BUNDLED_HOUSE_PROVENANCE_URL, LICENSE, available_at=store.now(),
        sha256=sha256(payload), record_count=inserted,
        note=(f"house: bundled vintage snapshot ({BUNDLED_HOUSE_FILE.name}), "
              f"{len(rows)} seat-cycle margins, 1976-2024. Guestbook-gated at "
              "Dataverse; a maintainer satisfied it via the web UI and downloaded "
              "this file directly (see DATA_SOURCES.md for full provenance). "
              "Refresh by re-downloading after a new cycle is certified."))
    return {"results": inserted, "files": [BUNDLED_HOUSE_FILE.name], "failed_files": []}


def ingest(chambers: tuple[str, ...] = ("house", "senate")) -> dict:
    summary: dict = {"source": SOURCE, "results": 0, "files": [], "failed_files": []}
    for chamber in chambers:
        if chamber == "house" and BUNDLED_HOUSE_FILE.exists():
            bundled = _ingest_bundled_house()
            summary["results"] += bundled["results"]
            summary["files"] += bundled["files"]
            continue
        try:
            file_urls = _dataset_file_urls(DATASETS[chamber])
        except httpx.HTTPError as exc:
            # A dataset-listing failure (network error, rate limit, transient
            # 403, ...) must not discard results already collected for other
            # chambers.
            summary["failed_files"].append(f"{chamber}: dataset listing: {exc}")
            continue
        for label, file_id in file_urls:
            try:
                payload = _fetch_datafile(file_id)
            except httpx.HTTPError as exc:
                # One unreadable file (e.g. a codebook, an unusual format, or
                # a network error) must not block every other file.
                summary["failed_files"].append(f"{chamber}: {label}: {exc}")
                continue
            rows = parse(payload, chamber)
            if not rows:
                continue
            inserted = store.insert_rows("election_results", rows)
            store.record_source(
                SOURCE, f"{DATAVERSE}/api/access/datafile/{file_id}", LICENSE,
                available_at=store.now(), sha256=sha256(payload),
                record_count=inserted,
                note=f"{chamber}: {label}, {len(rows)} seat-cycle margins")
            summary["results"] += inserted
            summary["files"].append(label)
    return summary
