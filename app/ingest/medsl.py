"""MIT Election Data + Science Lab (MEDSL) constituency returns adapter.

Downloads the canonical candidate-level general-election returns from Harvard
Dataverse (CC0 public domain):

* U.S. House 1976-2022  — doi:10.7910/DVN/IG0UN2
* U.S. Senate 1976-2020 — doi:10.7910/DVN/PEJ5QU

and normalizes them to per-seat two-party margins. This is the authoritative
full-coverage historical results source; it requires outbound access to
``dataverse.harvard.edu`` (available in the scheduled GitHub Actions pipeline
and local runs; some sandboxed environments block it, in which case the
polled-race outcomes from the FiveThirtyEight adapter remain available).
"""
from __future__ import annotations

import csv
import io
import json
from collections import defaultdict

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


def _dataset_file_urls(doi: str) -> list[tuple[str, str]]:
    listing = json.loads(fetch(
        f"{DATAVERSE}/api/datasets/:persistentId?persistentId={doi}").decode("utf-8"))
    files = listing["data"]["latestVersion"]["files"]
    urls = []
    for f in files:
        df = f["dataFile"]
        label = (df.get("originalFileName") or df.get("filename") or "").lower()
        if label.endswith((".csv", ".tab")):
            urls.append((label, df["id"]))
    return urls


def _fetch_datafile(file_id: int) -> bytes:
    """Three-tier fallback for Dataverse's file-access quirks, each a
    documented cause of an HTTP 400/403 on this endpoint:

    1. ``format=original`` - fails when Dataverse stored no distinct
       original upload separate from its ingested/archival copy.
    2. plain access - the archival copy; fails for guestbook-gated files.
    3. ``gbrecs=true`` - explicitly answers "yes" to a dataset's guestbook
       requirement so the download proceeds without an interactive form.
    """
    urls = [
        f"{DATAVERSE}/api/access/datafile/{file_id}?format=original",
        f"{DATAVERSE}/api/access/datafile/{file_id}",
        f"{DATAVERSE}/api/access/datafile/{file_id}?gbrecs=true",
    ]
    last_error: httpx.HTTPStatusError | None = None
    for url in urls:
        try:
            return fetch(url)
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


def ingest(chambers: tuple[str, ...] = ("house", "senate")) -> dict:
    summary: dict = {"source": SOURCE, "results": 0, "files": [], "failed_files": []}
    for chamber in chambers:
        for label, file_id in _dataset_file_urls(DATASETS[chamber]):
            try:
                payload = _fetch_datafile(file_id)
            except httpx.HTTPStatusError as exc:
                # One unreadable file (e.g. a codebook or an unusual format)
                # must not block every other file in the dataset.
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
