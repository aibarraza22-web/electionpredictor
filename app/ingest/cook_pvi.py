"""Cook Partisan Voting Index (PVI), by district and by state, per cycle.

PVI measures how a district (or state) voted for PRESIDENT in the previous
two presidential elections relative to the nation. It is built entirely from
certified presidential returns -- no polls -- which is exactly why the
poll-free forecast (``app.lean_model``) leans on it: it is the standard
district-partisanship measure, and unlike a district's last House result it
is recomputed on NEW lines whenever a map is redrawn (e.g. the 2025-26 Texas,
California, Tennessee maps), so a redrawn seat gets a real partisan baseline
instead of a stale one.

Source: the Wikipedia "Cook Partisan Voting Index" article, which republishes
every Cook release. Backtests must only use the PVI that existed BEFORE each
election, so each cycle's values come from the article revision in force on
September 15 of that election year (identified by its permanent ``oldid``).
The parsed values are bundled in ``data/vintage/cook_pvi_vintages.csv`` --
PVI changes once every two years, so there is no reason to hit Wikipedia on
every forecast run -- and ``build_vintages`` regenerates the file.

Sign convention here: positive = Democratic lean, in PVI points (D+5 -> +5,
R+5 -> -5, EVEN -> 0). One PVI point is roughly two points of two-party
MARGIN; the model fits that scale itself.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import urllib.request

from .votehub import STATE_NAMES

PAGE = "Cook_Partisan_Voting_Index"
WIKI = "https://en.wikipedia.org/w/index.php"
# Same identifying agent as the ratings adapter (Wikimedia rejects generic ones).
from .race_ratings import FETCH_TIMEOUT, USER_AGENT  # noqa: E402


def fetch_wiki(url: str) -> bytes:
    """urllib, not httpx: Wikimedia answers httpx's client signature with 403
    (see app.ingest.race_ratings._fetch)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        return response.read()
VINTAGE_FILE = Path(__file__).resolve().parents[2] / "data" / "vintage" / "cook_pvi_vintages.csv"
FIELDS = ["cycle", "level", "state", "district", "pvi", "wiki_oldid", "revision_as_of"]

_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)")
_PVI_PATTERNS = (
    re.compile(r"Shading PVI\|(R|D)\|(?:value=)?(\d+(?:\.\d+)?)"),
    re.compile(r"\b(R|D)\s?\+\s?(\d+(?:\.\d+)?)"),
)
_EVEN = re.compile(r"Shading PVI\|EVEN|\bEVEN\b", re.IGNORECASE)
_NAME_BY_LENGTH = sorted(STATE_NAMES, key=len, reverse=True)


def _pvi_value(cell_text: str) -> float | None:
    for pattern in _PVI_PATTERNS:
        match = pattern.search(cell_text)
        if match:
            value = float(match.group(2))
            return value if match.group(1) == "D" else -value
    if _EVEN.search(cell_text):
        return 0.0
    return None


def _state_in(text: str) -> str | None:
    for name in _NAME_BY_LENGTH:  # longest first: "West Virginia" before "Virginia"
        if name in text:
            return STATE_NAMES[name]
    return None


def _district_of(row: str) -> tuple[str, int] | None:
    """(state, district number; 1 for at-large) for one wikitable row."""
    ushr = re.search(r"\{\{ushr\|([^|}]+)\|([^|}]+)", row)
    if ushr:
        state = STATE_NAMES.get(ushr.group(1).strip())
        token = ushr.group(2).strip().upper()
        if not state:
            return None
        return state, 1 if token in {"AL", "AT-LARGE", "0"} else int(token) if token.isdigit() else None
    link = re.search(r"\[\[([A-Za-z .]+?)'s? (?:(\d+)(?:st|nd|rd|th)|at-large) congressional district",
                     row, re.IGNORECASE)
    if link:
        state = STATE_NAMES.get(link.group(1).strip())
        if not state:
            return None
        return state, int(link.group(2)) if link.group(2) else 1
    return None


def _section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker)) if end_marker else -1
    return text[start:end] if end > 0 else text[start:]


def parse_districts(text: str) -> dict[tuple[str, int], float]:
    """{(state, district): pvi} from one revision of the article."""
    body = _section(text, "congressional district", "By state")
    values: dict[tuple[str, int], float] = {}
    for row in body.split("\n|-"):
        seat = _district_of(row)
        if not seat or seat[1] is None:
            continue
        # The PVI cell is the first cell carrying a lean; the rest of the row
        # is representative/party shading, which never contains "R+"/"D+".
        value = _pvi_value(row)
        if value is not None and seat not in values:
            values[seat] = value
    return values


def parse_states(text: str) -> dict[str, float]:
    """{state: pvi} from the article's by-state table."""
    start = max(text.find("==By state=="), text.find("===By state==="))
    if start < 0:
        return {}
    body = text[start:]
    table = body.find("{|")
    stop = body.find("\n|}", table if table >= 0 else 0)
    if stop > 0:
        body = body[:stop]
    values: dict[str, float] = {}
    for row in body.split("\n|-"):
        if "congressional district" in row:
            continue
        head = row.strip().split("\n")[0] if row.strip() else ""
        state = _state_in(head) or _state_in(row[:120])
        value = _pvi_value(row)
        if state and value is not None and state not in values:
            values[state] = value
    return values


def revision_before(cycle: int) -> str:
    """oldid of the article revision in force on Sept 15 of ``cycle``."""
    html = fetch_wiki(f"{WIKI}?title={PAGE}&action=history&offset={cycle}0915000000&limit=1"
                      ).decode("utf-8", "replace")
    match = re.search(r"oldid=(\d+)", html)
    if not match:
        raise RuntimeError(f"no {PAGE} revision found before {cycle}-09-15")
    return match.group(1)


def build_vintages(cycles: list[int], path: Path = VINTAGE_FILE) -> dict:
    """Fetch each cycle's pre-election revision, parse it, write the CSV."""
    rows = []
    for cycle in cycles:
        oldid = revision_before(cycle)
        text = fetch_wiki(f"{WIKI}?oldid={oldid}&action=raw").decode("utf-8", "replace")
        for (state, district), value in sorted(parse_districts(text).items()):
            rows.append({"cycle": cycle, "level": "house", "state": state,
                         "district": district, "pvi": value, "wiki_oldid": oldid,
                         "revision_as_of": f"{cycle}-09-15"})
        for state, value in sorted(parse_states(text).items()):
            rows.append({"cycle": cycle, "level": "state", "state": state,
                         "district": "", "pvi": value, "wiki_oldid": oldid,
                         "revision_as_of": f"{cycle}-09-15"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "path": str(path)}


def load_vintages(path: Path = VINTAGE_FILE) -> tuple[dict, dict]:
    """({(cycle, state, district): pvi}, {(cycle, state): pvi}) from the bundle."""
    districts: dict[tuple[int, str, int], float] = {}
    states: dict[tuple[int, str], float] = {}
    if not path.exists():
        return districts, states
    with path.open() as handle:
        for row in csv.DictReader(handle):
            cycle, value = int(row["cycle"]), float(row["pvi"])
            if row["level"] == "house":
                districts[(cycle, row["state"], int(row["district"]))] = value
            else:
                states[(cycle, row["state"])] = value
    return districts, states
