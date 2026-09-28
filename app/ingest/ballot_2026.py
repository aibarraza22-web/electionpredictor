"""2026 ballot status: is the sitting member on the November ballot?

Builds ``data/vintage/incumbency_2026.csv`` (read by ``app.incumbency``) from
the status column of Wikipedia's "2026 United States House of
Representatives elections" and "2026 United States Senate elections"
articles -- a documented fact per seat ("Incumbent renominated", "Incumbent
retiring", "Incumbent lost renomination", ...), not a poll. Rebuild with
``python -m app.ingest.ballot_2026`` after primaries or retirements change.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .base import house_seat_key
from .cook_pvi import WIKI, fetch_wiki
from .votehub import STATE_NAMES

CYCLE = 2026
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "vintage" / "incumbency_2026.csv"
HOUSE_PAGE = "2026_United_States_House_of_Representatives_elections"
SENATE_PAGE = "2026_United_States_Senate_elections"
FIELDS = ["cycle", "seat_key", "incumbent_party", "incumbent_running", "status", "source"]

# Status phrases that mean the sitting member IS on the November ballot.
RUNNING = ("incumbent renominated", "incumbent advanced", "incumbent running",
           "incumbent re-elected")
# Appointed senators on the ballot ("Interim appointee nominated") are recorded
# as NOT running incumbents: the historical flags in app.incumbency only count
# members who previously WON the seat, and the fitted incumbency effect must
# mean the same thing at prediction time as it did in training.


def _party(row: str) -> str:
    """Party of the sitting member, read from the row BEFORE its candidate
    list (the candidate list names every party on the ballot)."""
    head = re.split(r"\{\{\s*plainlist", row, flags=re.IGNORECASE)[0]
    if re.search(r"Party shading/(?:Text/)?Republican|\|\s*Republican\b", head):
        return "R"
    if re.search(r"Party shading/(?:Text/)?(?:Democratic|DFL)|\|\s*(?:Democratic|DFL)\b", head):
        return "D"
    return ""


def _status(row: str) -> str | None:
    match = re.search(r"\|\s*(?:[^|\n]*\|\s*)?((?:Incumbent|New representative|Vacant)[^<\n{|]*)", row)
    if match:
        return match.group(1).strip().rstrip(".")
    if re.search(r"lost the initial nomination", row):
        return "Incumbent lost renomination"
    return None


def parse_house(text: str) -> list[dict]:
    rows = []
    for chunk in text.split("\n|-"):
        head = re.search(r"^!\s*(?:rowspan=\d+\s*\|\s*)?\{\{ushr\|([A-Z]{2})\|(\w+)\|X\}\}",
                         chunk.strip(), re.M)
        if not head or ("Party stripe" not in chunk and "plainlist" not in chunk.lower()):
            continue
        status = _status(chunk)
        if status is None:
            continue
        district = head.group(2)
        seat = house_seat_key(head.group(1), 1 if district.upper() == "AL" else district)
        rows.append({"cycle": CYCLE, "seat_key": seat, "incumbent_party": _party(chunk),
                     "incumbent_running": int(status.lower().startswith(RUNNING)),
                     "status": status, "source": f"wikipedia:{HOUSE_PAGE}"})
    return _dedupe(rows)


def parse_senate(text: str) -> list[dict]:
    rows = []
    for chunk in text.split("\n|-"):
        head = re.search(r"^!\s*\[\[2026 United States Senate (special )?election in ([A-Za-z ]+)\|",
                         chunk.strip(), re.M)
        if not head:
            continue
        status = re.search(r"\|\s*((?:Incumbent|Interim appointee|Appointee)[^<\n{|]*)", chunk)
        if not status:
            continue
        state = STATE_NAMES.get(head.group(2).strip())
        if not state:
            continue
        seat = f"senate-{state}" + ("-special" if head.group(1) else "")
        text_status = status.group(1).strip().rstrip(".")
        rows.append({"cycle": CYCLE, "seat_key": seat, "incumbent_party": _party(chunk),
                     "incumbent_running": int(text_status.lower().startswith(RUNNING)),
                     "status": text_status, "source": f"wikipedia:{SENATE_PAGE}"})
    return _dedupe(rows)


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["seat_key"], row)
    return list(seen.values())


def build(path: Path = OUTPUT) -> dict:
    house = parse_house(fetch_wiki(f"{WIKI}?title={HOUSE_PAGE}&action=raw").decode("utf-8", "replace"))
    senate = parse_senate(fetch_wiki(f"{WIKI}?title={SENATE_PAGE}&action=raw").decode("utf-8", "replace"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(house + senate, key=lambda r: r["seat_key"]))
    return {"house": len(house), "senate": len(senate), "path": str(path)}


if __name__ == "__main__":
    print(build())
