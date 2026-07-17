"""Official certified-results CSV importer.

For loading certified results published by state election authorities (or a
canonical aggregation of them, e.g. the 2024 cycle before MEDSL's release).
Expected header::

    cycle,chamber,state,district,dem_votes,rep_votes

``district`` is blank for Senate (append ``-special`` semantics with a
``special`` truthy column if needed). Rows without both major parties are
skipped — no values are fabricated.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .. import store
from .base import STATES, house_seat_key, senate_seat_key, sha256, two_party_margin

SOURCE = "official-results-csv"


def ingest_file(path: str | Path, source_url: str | None = None,
                license_: str = "official public records") -> dict:
    payload = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    rows = []
    for row in reader:
        state = row["state"].strip().upper()
        chamber = row["chamber"].strip().lower()
        if state not in STATES or chamber not in ("house", "senate"):
            continue
        try:
            dem, rep = float(row["dem_votes"]), float(row["rep_votes"])
        except (KeyError, ValueError):
            continue
        margin = two_party_margin(dem, rep)
        if margin is None:
            continue
        special = str(row.get("special", "")).strip().lower() in ("1", "true", "yes")
        if chamber == "house":
            seat_key = house_seat_key(state, row.get("district") or 1)
            district = f"{max(int(row.get('district') or 1), 1):02d}"
        else:
            seat_key, district = senate_seat_key(state, special), None
        total = dem + rep
        rows.append({
            "cycle": int(row["cycle"]), "chamber": chamber, "state": state,
            "district": district, "seat_key": seat_key,
            "dem_pct": dem / total * 100.0, "rep_pct": rep / total * 100.0,
            "dem_margin": margin, "winner_party": "D" if margin > 0 else "R",
            "source": SOURCE,
        })
    inserted = store.insert_rows("election_results", rows)
    store.record_source(SOURCE, source_url or str(path), license_,
                        available_at=store.now(), sha256=sha256(payload),
                        record_count=inserted, note=f"{len(rows)} certified rows parsed")
    return {"source": SOURCE, "results": inserted}
