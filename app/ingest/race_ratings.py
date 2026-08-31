"""Expert race-ratings adapter (Wikipedia election-ratings tables).

Why this source exists
----------------------
Before this adapter the 2026 forecast had **no seat-level current-cycle
signal for 427 of 470 races**: only 43 seats carried any 2026 polling, and
the campaign-finance overlay only moves a race when both major-party
candidates have filed comparable FEC totals.  Every other race was predicted
from history and the national environment alone, which is why new research
could not move the toplines.

Wikipedia maintains, for every recent cycle, a structured table of the
*published* race ratings of the major handicappers and models (Cook Political
Report, Inside Elections, Sabato's Crystal Ball, DDHQ, The Economist, Split
Ticket, Silver Bulletin, RealClearPolitics, Fox, Race to the WH, ...) for
every seat any of them considers competitive, plus the full Senate map.  Each
column carries the publication date of that rater's snapshot, which is what
makes the feed vintage-safe.

The handicappers' own sites are not machine-readable from a server: as of this
change ``cookpolitical.com`` and ``centerforpolitics.org`` both answer
automated clients with HTTP 403, and Inside Elections does the same.  The
Wikipedia mirror is retrievable (``index.php?action=raw`` — the ``api.php``
endpoint is aggressively rate-limited for shared egress IPs, so the raw
article path is used deliberately), carries per-column publication dates, and
cites every underlying source.  It is CC BY-SA 4.0; ``LICENSE`` records that
and every row keeps the page URL it came from.

Nothing here is scored or interpreted: the adapter records *what each rater
published, and when*.  ``app.ratings`` turns those observations into a
consensus number and ``app.features`` exposes it to the model, whose
coefficient is fitted on historical cycles like every other feature — see
research claim R-001.
"""
from __future__ import annotations

import re
import time
import urllib.request
from datetime import date, datetime, timezone

from .. import store
from .base import STATES, house_seat_key, senate_seat_key, sha256

SOURCE = "wikipedia-race-ratings"
LICENSE = "CC BY-SA 4.0 (Wikipedia); underlying ratings cited per column"
BASE = "https://en.wikipedia.org/w/index.php"
# Wikimedia asks automated clients to identify themselves; an anonymous
# default user-agent is throttled hard.
USER_AGENT = ("CongressionalForecastLab/1.0 "
              "(https://github.com/aibarraza22-web/electionpredictor)")
FETCH_TIMEOUT = 90.0
# Wikimedia asks bulk readers to pace themselves; these pages are fetched at
# most a dozen at a time, once per pipeline run.
FETCH_PAUSE_SECONDS = 1.0

# Rating vocabulary -> unsigned strength. Every handicapper on these pages
# uses this ladder; "Tilt" is used by only some of them, which is precisely
# why the consensus is a mean over raters rather than any single scale.
STRENGTH = {"tossup": 0.0, "toss-up": 0.0, "tilt": 1.0, "lean": 2.0,
            "likely": 3.0, "safe": 4.0, "solid": 4.0}

# Cycles with a machine-readable ratings table. House ratings live on a
# dedicated page; Senate ratings live in the main Senate-elections article.
HOUSE_PAGE = "{cycle} United States House of Representatives election ratings"
SENATE_PAGE = "{cycle} United States Senate elections"
# Per-state House articles carry a Predictions table for EVERY district, not
# just the ones some rater calls competitive, so they are what gives the
# ~280 safe House seats real dated evidence instead of none. Multi-district
# states use the plural title, at-large states the singular one.
STATE_HOUSE_PAGES = (
    "{cycle} United States House of Representatives elections in {state}",
    "{cycle} United States House of Representatives election in {state}",
)
DEFAULT_CYCLES = (2016, 2018, 2020, 2022, 2024, 2026)
# 2016's Senate article predates the structured predictions table.
SENATE_CYCLES = (2018, 2020, 2022, 2024, 2026)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"])}

RATING_RE = re.compile(r"\{\{\s*USRaceRating\s*\|([^}]*)\}\}", re.IGNORECASE)
USHR_RE = re.compile(r"\{\{\s*ushr\s*\|\s*([A-Za-z]{2})\s*\|\s*([A-Za-z0-9]+)",
                     re.IGNORECASE)
PVI_RE = re.compile(r"\{\{\s*shading\s*PVI\s*\|\s*([RDrd])\s*\|\s*(\d+)",
                    re.IGNORECASE)
PVI_EVEN_RE = re.compile(r"\{\{\s*shading\s*PVI\s*\|\s*EVEN", re.IGNORECASE)
DATE_RE = re.compile(
    r"\{\{\s*small\s*\|\s*([A-Za-z]+)\.?\s*(\d{1,2}),?\s*(?:<br\s*/?>)?\s*(\d{4})",
    re.IGNORECASE)
RETIRING_RE = re.compile(r"\(\s*retiring\s*\)", re.IGNORECASE)
DISTRICT_SECTION_RE = re.compile(
    r"^=+\s*District\s+([0-9]+|at[- ]large)\s*=+\s*$", re.IGNORECASE | re.MULTILINE)
LONG_DATE_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})\b")
# Header cells that follow the ratings block and must not be read as raters.
OUTCOME_LABELS = {"winner", "result", "results", "elected", "outcome", "notes"}
LAST_RESULT_LABELS = ("last result", "previous result", "last election",
                      "last<br />election", "last margin")

# The national ratings pages head each column with an abbreviation ("DDHQ",
# "Cook", "Econ.") while the per-state pages spell the same organisation out
# ("Decision Desk HQ", "The Cook Political Report", "The Economist"). Since the
# consensus is an unweighted mean over raters, leaving both spellings in place
# counts those organisations twice and quietly overweights them. Names are
# canonicalized here, which also makes ``rater`` a stable part of a rating's
# identity across the two page types.
RATER_ALIASES = {
    "cook": "Cook Political Report",
    "cook political report": "Cook Political Report",
    "the cook political report": "Cook Political Report",
    "ie": "Inside Elections",
    "inside elections": "Inside Elections",
    "rothenberg": "Inside Elections",
    "stuart rothenberg": "Inside Elections",
    "sabato": "Sabato's Crystal Ball",
    "sabato's crystal ball": "Sabato's Crystal Ball",
    "ddhq": "Decision Desk HQ",
    "decision desk hq": "Decision Desk HQ",
    "econ": "The Economist", "econ.": "The Economist",
    "economist": "The Economist", "the economist": "The Economist",
    "st": "Split Ticket", "split ticket": "Split Ticket",
    "split ticket (website)": "Split Ticket",
    "silver": "Silver Bulletin", "silver bulletin": "Silver Bulletin",
    "fpo": "FiftyPlusOne", "fiftyplusone": "FiftyPlusOne",
    "wh": "Race to the WH", "race to the wh": "Race to the WH",
    "rcp": "RealClearPolitics", "realclearpolitics": "RealClearPolitics",
    "realclearpolling": "RealClearPolitics",
    "fox": "Fox News", "fox news": "Fox News",
    "538": "FiveThirtyEight", "fivethirtyeight": "FiveThirtyEight",
    "ed": "Elections Daily", "elections daily": "Elections Daily",
    "dk": "Daily Kos", "daily kos": "Daily Kos",
    "cbs": "CBS News", "cbs news": "CBS News",
    "nyt": "The New York Times", "the new york times": "The New York Times",
    "cnn": "CNN", "politico": "Politico", "votehub": "VoteHub",
}


def canonical_rater(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip(" .'\t")
    return RATER_ALIASES.get(cleaned.lower(), cleaned or "unattributed")


STATE_BY_NAME = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


def page_url(title: str) -> str:
    return f"{BASE}?title={title.replace(' ', '_')}&action=raw"


def _fetch(title: str) -> bytes:
    """Fetch page wikitext.

    Deliberately ``urllib`` rather than the ``httpx`` client every other
    adapter uses: Wikimedia's bot policy currently answers httpx's client
    signature with HTTP 403 ("Please respect our robot policy") on every
    endpoint — article HTML, ``action=raw``, ``api.php`` and
    ``api.wikimedia.org`` alike — while serving the identical request from
    the standard library. Verified both ways against these exact pages
    before this adapter shipped. The descriptive User-Agent below is what
    that policy asks automated clients to send.
    """
    request = urllib.request.Request(
        page_url(title), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        return response.read()


def _strip_markup(value: str) -> str:
    value = re.sub(r"<ref[^>]*/>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<ref.*?</ref>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"\{\{\s*efn.*?\}\}", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    return value


def _label(cell: str) -> str:
    """Human label of a header cell: the piped part of a wikilink, else text."""
    cleaned = _strip_markup(cell)
    # Drop any leading HTML attribute block ('class="unsortable" | Incumbent').
    attributes = re.match(r'\s*(?:[a-zA-Z-]+\s*=\s*"[^"]*"\s*)+\|(?!\|)', cleaned)
    if attributes:
        cleaned = cleaned[attributes.end():]
    link = (re.search(r"\[\[[^\]|]*\|([^\]]+)\]\]", cleaned)
            or re.search(r"\[\[([^\]|]+)\]\]", cleaned))
    if link:
        return link.group(1).strip()
    text = re.sub(r"\{\{.*?\}\}", "", cleaned, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip(" |'\t").strip()


def _cell_date(cell: str) -> str | None:
    match = DATE_RE.search(_strip_markup(cell))
    if not match:
        return None
    month = MONTHS.get(match.group(1)[:3].lower())
    if not month:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    except ValueError:
        return None


def rating_rows(text: str) -> list[str]:
    """Wikitable rows carrying at least one ``{{USRaceRating}}``.

    Rows are taken from the whole page rather than from an extracted table.
    Wikitext table nesting on these pages is not reliably balanced — a
    brace-matching table splitter silently truncated the 2022 House table at
    30 of its 143 rows and the 2026 Senate table at 24 of 35 — and the row
    filters below (a district template for the House, a state name in the row
    head for the Senate, plus the presence of a rating) are specific enough
    that page-wide splitting is both simpler and safer.
    """
    return [block for block in re.split(r"\n\|-", text) if RATING_RE.search(block)]


def rater_columns(text: str) -> list[tuple[str, str | None]]:
    """``(rater label, publication date)`` per rating column, in table order.

    The header is the last contiguous run of ``!`` cells before the first
    rated row. The rating columns are the cells after the "last result"
    column, up to any trailing outcome column. Some raters' cells carry an
    explanatory footnote instead of a date; those keep ``None`` here and
    inherit a conservative page-level date in :func:`_rows_from_page`.
    """
    first = RATING_RE.search(text)
    if not first:
        return []
    # Strip refs/footnotes across the whole head first: a <ref> block spans
    # physical lines, so without this the header's cells stop looking like a
    # contiguous run of "!" lines and only the final column is found.
    head = _strip_markup(text[:first.start()])
    # Data rows also open with a "!" cell (the district or state name), so the
    # header is not simply the last run of "!" lines before the first rating —
    # that finds the first data row. It is the LONGEST such run: header rows
    # carry a dozen-plus cells, a data row's leading cell is a run of one.
    runs: list[list[str]] = []
    current: list[str] = []
    for line in head.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!"):
            body = stripped[1:]
            if not re.match(r"\s*colspan", body, re.IGNORECASE):
                current.extend(re.split(r"!!", body))
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return []
    header = max(runs, key=len)
    labels = [_label(cell) for cell in header]
    start = 0
    for index, label in enumerate(labels):
        if any(key in label.lower() for key in LAST_RESULT_LABELS):
            start = index + 1
    columns = []
    for cell, label in zip(header[start:], labels[start:]):
        if label.lower() in OUTCOME_LABELS:
            break
        columns.append((canonical_rater(label) if label
                        else f"column-{len(columns) + 1}", _cell_date(cell)))
    return columns


def parse_rating(argument_blob: str) -> tuple[str, float] | None:
    """``{{USRaceRating|Lean|R|Flip}}`` -> ``("Lean R", -2.0)``. D is positive."""
    parts = [part.strip() for part in argument_blob.split("|") if part.strip()]
    if not parts:
        return None
    strength = STRENGTH.get(parts[0].lower())
    if strength is None:
        return None
    if strength == 0.0:
        return "Tossup", 0.0
    party = parts[1].upper() if len(parts) > 1 else ""
    if party == "D":
        return f"{parts[0].title()} D", strength
    if party == "R":
        return f"{parts[0].title()} R", -strength
    return None


def _house_seat(block: str) -> tuple[str, str, str] | None:
    match = USHR_RE.search(block)
    if not match:
        return None
    state = match.group(1).upper()
    if state not in STATES:
        return None
    raw = match.group(2).upper()
    number = 1 if raw == "AL" else int(raw) if raw.isdigit() else None
    if number is None:
        return None
    return state, f"{number:02d}", house_seat_key(state, number)


def _senate_seat(block: str) -> tuple[str, None, str] | None:
    """Senate rows head with a link to the state's race article."""
    head = block.split("\n|", 1)[0]
    special = "special" in head.lower()
    # Longest name first: a plain "Virginia" search matches inside "West
    # Virginia", which silently merged the two seats into one.
    for name, code in sorted(STATE_BY_NAME.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", head):
            return code, None, senate_seat_key(code, special=special)
    return None


def _pvi(block: str) -> float | None:
    match = PVI_RE.search(block)
    if match:
        return float(match.group(2)) * (1.0 if match.group(1).upper() == "D" else -1.0)
    return 0.0 if PVI_EVEN_RE.search(block) else None


def _last_result(block: str) -> float | None:
    """The prior two-party share cell, as a D-positive margin approximation."""
    match = re.search(r'data-sort-value="(-?\d+(?:\.\d+)?)"\s*\|\s*'
                      r'(\d{1,3}(?:\.\d+)?)\s*%\s*([DR])', block)
    if not match:
        return None
    share = float(match.group(2))
    signed = (share - (100.0 - share)) * (1.0 if match.group(3).upper() == "D" else -1.0)
    return round(signed, 2)


def _rows_from_page(text: str, cycle: int, chamber: str, url: str,
                    retrieved_at: str) -> tuple[list[dict], list[dict], dict]:
    columns = rater_columns(text)
    known_dates = [d for _, d in columns if d]
    # A rater whose header carries a footnote instead of a date inherits the
    # LATEST date on the page: conservative, because a vintage filter then
    # excludes it until every rater on that page had certainly published.
    fallback_date = max(known_dates) if known_dates else None
    blocks = rating_rows(text)
    seat_of = _house_seat if chamber == "house" else _senate_seat
    ratings: list[dict] = []
    contexts: list[dict] = []
    aligned = unaligned = 0
    for block in blocks:
        seat = seat_of(block)
        if seat is None:
            continue
        state, district, seat_key = seat
        parsed = [parse_rating(blob) for blob in RATING_RE.findall(block)]
        names: list[tuple[str, str | None]]
        if len(parsed) == len(columns):
            names = columns
            aligned += 1
        else:
            # Column drift on this row: keep the ratings (the consensus is an
            # unweighted mean, so it is unaffected) but do not assert an
            # attribution the table does not support.
            names = [(f"column-{i + 1}", fallback_date) for i in range(len(parsed))]
            unaligned += 1
        for (rater, published), item in zip(names, parsed):
            if item is None:
                continue
            rating_date = published or fallback_date
            if not rating_date:
                continue        # no defensible vintage: drop rather than guess
            label, score = item
            ratings.append({
                "cycle": cycle, "chamber": chamber, "state": state,
                "district": district, "seat_key": seat_key, "rater": rater,
                "rating": label, "score": score, "rating_date": rating_date,
                "available_at": rating_date, "retrieved_at": retrieved_at,
                "source": SOURCE, "source_url": url,
            })
        incumbent = None
        cells = [c for c in block.split("\n|") if "sortname" in c]
        if cells:
            name = re.search(r"\{\{\s*sortname\s*\|([^}]*)\}\}", cells[0])
            if name:
                parts = [p.strip() for p in name.group(1).split("|")][:2]
                incumbent = " ".join(p for p in parts if p) or None
        contexts.append({
            "cycle": cycle, "chamber": chamber, "seat_key": seat_key,
            "cook_pvi": _pvi(block), "incumbent": incumbent,
            "incumbent_retiring": bool(RETIRING_RE.search(block)),
            "last_result_margin": _last_result(block),
            "observed_at": fallback_date or retrieved_at[:10],
            "source": SOURCE, "source_url": url,
        })
    stats = {"rater_columns": len(columns), "rows": len(blocks),
             "attributed_rows": aligned, "unattributed_rows": unaligned}
    return ratings, contexts, stats


def _long_date(value: str) -> str | None:
    match = LONG_DATE_RE.search(_strip_markup(value))
    if not match:
        return None
    month = MONTHS.get(match.group(1)[:3].lower())
    if not month:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    except ValueError:
        return None


def _district_sections(text: str) -> list[tuple[str, str]]:
    """``[(district number, section text)]`` for a per-state House article.

    At-large states have no "District N" headings at all; their whole page is
    one district, which ``house_seat_key`` normalizes to 01.
    """
    matches = list(DISTRICT_SECTION_RE.finditer(text))
    if not matches:
        return [("1", text)]
    sections = []
    for index, match in enumerate(matches):
        raw = match.group(1)
        number = "1" if raw.lower().replace(" ", "-") == "at-large" else raw
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((number, text[match.end():end]))
    return sections


def parse_state_page(text: str, cycle: int, state: str, url: str,
                     retrieved_at: str) -> tuple[list[dict], dict]:
    """Parse a per-state House article's per-district Predictions tables.

    These tables are transposed relative to the national ratings page — one
    row per rater, carrying that rater's own "as of" date in its own cell —
    which makes them the more precisely dated of the two sources.
    """
    ratings: list[dict] = []
    districts = 0
    undated = 0
    for number, section in _district_sections(text):
        try:
            seat_key = house_seat_key(state, int(number))
        except ValueError:
            continue
        found = False
        for block in re.split(r"\n\|-", section):
            match = RATING_RE.search(block)
            if not match:
                continue
            item = parse_rating(match.group(1))
            if item is None:
                continue
            rating_date = _long_date(block[match.end():]) or _long_date(block)
            if not rating_date:
                undated += 1
                continue          # no defensible vintage: drop rather than guess
            # The rater is the row's first cell. Refs are stripped first
            # because a <ref> block spans lines and would otherwise split the
            # cell in half — which collapsed every rater to one name and, with
            # rater in the row's uniqueness key, silently dropped all but one
            # rating per date.
            cells = [c for c in _strip_markup(block).split("\n|") if c.strip()]
            rater = canonical_rater(_label(cells[0]) if cells else "")
            label, score = item
            ratings.append({
                "cycle": cycle, "chamber": "house", "state": state,
                "district": f"{int(number):02d}", "seat_key": seat_key,
                "rater": rater, "rating": label, "score": score,
                "rating_date": rating_date, "available_at": rating_date,
                "retrieved_at": retrieved_at, "source": SOURCE, "source_url": url,
            })
            found = True
        districts += int(found)
    return ratings, {"districts_with_ratings": districts,
                     "rating_rows": len(ratings), "undated_rows": undated}


def parse(text: str, cycle: int, chamber: str, url: str,
          retrieved_at: str) -> tuple[list[dict], list[dict], dict]:
    """Parse one ratings page. Returns ``(ratings, contexts, stats)``."""
    if not RATING_RE.search(text):
        return [], [], {"skipped": "no rating templates on page"}
    return _rows_from_page(text, cycle, chamber, url, retrieved_at)


def ingest(cycles: tuple[int, ...] = DEFAULT_CYCLES) -> dict:
    """Ingest ratings for every configured cycle.

    Historical cycles are what the model's ratings coefficient is *fitted*
    on, so they are refreshed alongside the current one; their pages are
    stable and the inserts are idempotent, so re-running is a no-op.
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    ratings: list[dict] = []
    contexts: list[dict] = []
    pages: dict[str, dict] = {}
    failures: list[str] = []
    for cycle in cycles:
        for chamber, template, allowed in (("house", HOUSE_PAGE, cycles),
                                           ("senate", SENATE_PAGE, SENATE_CYCLES)):
            if cycle not in allowed:
                continue
            title = template.format(cycle=cycle)
            url = page_url(title)
            try:
                time.sleep(FETCH_PAUSE_SECONDS)
                payload = _fetch(title)
            except Exception as exc:                      # network/page drift
                failures.append(f"{cycle} {chamber}: {exc}")
                continue
            text = payload.decode("utf-8", errors="replace")
            page_ratings, page_contexts, stats = parse(
                text, cycle, chamber, url, retrieved_at)
            source_id = store.record_source(
                SOURCE, url, LICENSE, available_at=retrieved_at,
                sha256=sha256(payload), record_count=len(page_ratings),
                note=f"{cycle} {chamber} expert race ratings")
            for row in page_ratings:
                row["source_id"] = source_id
            for row in page_contexts:
                row["source_id"] = source_id
            ratings.extend(page_ratings)
            contexts.extend(page_contexts)
            pages[f"{cycle}-{chamber}"] = stats
    # Per-state House articles, current cycle only. The national page lists
    # only the seats somebody calls competitive; these pages carry a dated
    # Predictions table for every district, which is what gives the remaining
    # ~280 House races real evidence rather than none.
    current = max(cycles)
    state_stats: dict[str, dict] = {}
    for state_name, code in sorted(STATE_BY_NAME.items()):
        page_ratings: list[dict] = []
        for template in STATE_HOUSE_PAGES:
            title = template.format(cycle=current, state=state_name)
            try:
                time.sleep(FETCH_PAUSE_SECONDS)
                payload = _fetch(title)
            except Exception:
                continue          # try the at-large title, then give up
            text = payload.decode("utf-8", errors="replace")
            if text.lstrip()[:9].upper().startswith("#REDIRECT"):
                continue
            url = page_url(title)
            page_ratings, stats = parse_state_page(text, current, code, url,
                                                   retrieved_at)
            source_id = store.record_source(
                SOURCE, url, LICENSE, available_at=retrieved_at,
                sha256=sha256(payload), record_count=len(page_ratings),
                note=f"{current} {code} per-district House ratings")
            for row in page_ratings:
                row["source_id"] = source_id
            state_stats[code] = stats
            break
        if page_ratings:
            ratings.extend(page_ratings)
        elif code not in state_stats:
            failures.append(f"{current} {code} state page: unreachable or unparsed")

    inserted = store.insert_rows("race_ratings", ratings) if ratings else 0
    context_rows = store.insert_rows("seat_context", contexts) if contexts else 0
    summary = {
        "source": SOURCE, "pages": pages,
        "state_pages": {"states": len(state_stats),
                        "districts_with_ratings": sum(
                            v["districts_with_ratings"] for v in state_stats.values()),
                        "undated_rows_dropped": sum(
                            v["undated_rows"] for v in state_stats.values())},
        "records_seen": len(ratings),
        "rating_rows": inserted, "seat_context_rows": context_rows,
        "seats": len({(r["cycle"], r["seat_key"]) for r in ratings}),
        "current_cycle_seats": len({r["seat_key"] for r in ratings
                                    if r["cycle"] == max(cycles)}),
    }
    if failures:
        summary["failures"] = failures
    if not ratings:
        summary["skipped"] = "no ratings parsed from any configured page"
    return summary
