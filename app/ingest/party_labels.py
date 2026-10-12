"""Which party's side is each general-election candidate on?

The MEDSL returns carry a party column, but it cannot be used as-is to turn a
race into a Democratic-vs-Republican margin:

* **Independents who caucus with a party.** Angus King (ME) and Bernie Sanders
  (VT) are labelled "INDEPENDENT" and so counted as neither party: King's 54%
  in 2018 became "R+54", because the only major-party candidates left were the
  Republican and a 10% Democrat. Both win as, and vote with, Democrats.
* **Fusion tickets.** A candidate appears once per ballot line (Dodd 1992:
  Democratic + "A Connecticut Party"; New York's Working Families and
  Conservative lines). Counting only the major-party line undercounts both
  sides by different amounts.
* **Missing or mis-simplified labels.** Wyoming 2020 and Alaska 2022 carry no
  party for anyone; Duckworth (IL 2022) and Van Hollen (MD 2022) are
  "DEMOCRATIC" but simplified to OTHER; Minnesota's pre-1995 Republicans are
  "INDEPENDENT-REPUBLICAN". Every one of those races silently fell out of the
  history (or entered it with a +/-100 margin).

``resolve`` returns "D", "R" or None for one candidate in one race; ``tally``
turns a race's rows into Democratic / Republican / other votes with
candidates (not ballot lines) as the unit.

The override table is deliberately short, each entry a documented fact, and
every entry is a candidate who is on the Democratic or Republican side in
substance. Third-party and ordinary independent candidates are left as
"other", exactly as before.
"""
from __future__ import annotations

import re
from collections import defaultdict

_SUFFIXES = {"JR", "SR", "II", "III", "IV"}
_NON_CANDIDATE = ("OVER VOTES", "OVERVOTES", "UNDER VOTES", "UNDERVOTES", "BLANK",
                  "SCATTERING", "WRITE-IN", "WRITE IN", "EXHAUSTED", "TOTAL VOTES",
                  "NONE OF THESE")


def surname(name: str) -> str:
    """Last name, upper-cased, suffix-free; handles "SURNAME, FIRST" and
    "FIRST LAST" and quoted nicknames."""
    text = str(name or "").upper()
    if "," in text and not re.search(r",\s*(JR|SR|II|III|IV)\b", text):
        text = text.split(",")[0]
    tokens = [t for t in re.sub(r"[^A-Z \-']", " ", text).split() if t not in _SUFFIXES]
    return tokens[-1] if tokens else ""


# (year, state, surname) -> side. Each entry is a documented fact.
ALIGNED_INDEPENDENTS: dict[tuple[int, str, str], str] = {}


# (year, state) -> human note shown beside a result, for races won or
# contested by a candidate who is on a party's side without being its nominee.
ALIGNED_NOTES: dict[tuple[int, str], str] = {}
# (year, state, chamber) of races whose real contest involved an independent on
# a party's side. Polls of such a race are not usable as Democrat-vs-Republican
# polls: the stored Maine 2012/2018 polls are Republican vs the weak Democratic
# NOMINEE (R+16 to R+31), while the race was won by King (D+30).
ALIGNED_RACES: set[tuple[int, str, str]] = set()


def _add(side: str, state: str, name: str, years, note: str | None = None,
         chamber: str = "senate") -> None:
    for year in years:
        ALIGNED_INDEPENDENTS[(year, state, name)] = side
        if note:
            ALIGNED_NOTES[(year, state)] = note.format(year=year)
            ALIGNED_RACES.add((year, state, chamber))


def aligned_race(year: int, state: str, chamber: str) -> bool:
    """True when the race's real contest involved a caucus-aligned independent,
    so its party-labelled head-to-head polls do not describe the ballot."""
    return (year, state, chamber) in ALIGNED_RACES


def aligned_note(year: int, state: str) -> str | None:
    """Explanation for a result whose winner/challenger is a caucus-aligned
    independent (e.g. "Angus King (I) -- independent, caucuses with Democrats")."""
    return ALIGNED_NOTES.get((year, state))


# Independents who win -- and caucus -- as Democrats.
_add("D", "VT", "SANDERS", range(1990, 2006, 2),   # U.S. House, Independent
     "Bernie Sanders (I) -- independent, caucuses with Democrats", chamber="house")
_add("D", "VT", "SANDERS", (2006, 2012, 2018, 2024),  # U.S. Senate, Independent
     "Bernie Sanders (I) -- independent, caucuses with Democrats")
_add("D", "ME", "KING", (2012, 2018, 2024),         # Independent, caucuses with Democrats
     "Angus King (I) -- independent, caucuses with Democrats")
_add("D", "CT", "LIEBERMAN", (2006,),               # "Connecticut for Lieberman"; caucused D
     "Joe Lieberman (I) -- independent, caucused with Democrats")
_add("D", "VA", "BYRD", (1976,),                    # Harry Byrd Jr., Independent, caucused D
     "Harry Byrd Jr. (I) -- independent, caucused with Democrats")
# Independents the Democratic Party backed after fielding no nominee of its
# own. Left as "other" they erase a genuinely contested race -- the ballot
# would read as an uncontested Republican win.
_add("D", "UT", "MCMULLIN", (2022,),                # Utah Democrats endorsed, no nominee
     "Evan McMullin (I) -- independent backed by Democrats, who fielded no nominee")
_add("D", "NE", "OSBORN", (2024,),                  # Nebraska Democrats endorsed, no nominee
     "Dan Osborn (I) -- independent backed by Democrats, who fielded no nominee")
_add("D", "KS", "ORMAN", (2014,),                   # Democratic nominee withdrew; party backed Orman
     "Greg Orman (I) -- independent; the Democratic nominee withdrew")
# Candidates whose ballot label is blank or wrong in the source file.
_add("R", "WY", "LUMMIS", (2020,))
_add("D", "WY", "DAVID", (2020,))                   # Merav Ben-David
_add("R", "AK", "MURKOWSKI", (2010, 2022))          # 2010 write-in; 2022 label blank
_add("R", "AK", "TSHIBAKA", (2022,))
_add("R", "AK", "KELLEY", (2022,))
_add("D", "AK", "CHESBRO", (2022,))
_add("D", "MS", "MUSGROVE", (2008,))                # 2008 special: "no party affiliation" jungle
_add("R", "MS", "WICKER", (2008,))
_add("R", "MO", "EMERSON", (1996,))                 # Republican running as an independent write-in
# House: Virgil Goode was elected as an Independent in 2000 and caucused with
# the Democrats until he became a Republican in 2002.
_add("D", "VA", "GOODE", (2000,),
     "Virgil Goode (I) -- independent, caucused with Democrats until 2002", chamber="house")


def classify_label(*labels: str) -> str | None:
    """D / R from a party label; None for third parties and independents."""
    for label in labels:
        text = str(label or "").upper()
        if not text:
            continue
        # "INDEPENDENT-REPUBLICAN" is Minnesota's Republican Party until 1995
        if "REPUBLICAN" in text and "LIBERTARIAN" not in text:
            return "R"
        if "DEMOCRAT" in text or "DFL" in text.split() or text.startswith("DFL"):
            return "D"
    return None


def resolve(year: int, state: str, candidate: str, *labels: str) -> str | None:
    override = ALIGNED_INDEPENDENTS.get((year, state, surname(candidate)))
    if override:
        return override
    return classify_label(*labels)


def is_non_candidate(name: str) -> bool:
    text = str(name or "").upper().strip()
    return not text or any(marker in text for marker in _NON_CANDIDATE)


def _candidate_key(name: str) -> str:
    return re.sub(r"[^A-Z ]", "", str(name or "").upper()).strip()


def tally(year: int, state: str, rows: list[dict]) -> dict:
    """Democratic / Republican / other votes for one race.

    ``rows`` are the race's general-election rows with ``candidate``,
    ``votes`` and the party columns under ``labels``. Votes are summed per
    CANDIDATE across ballot lines (fusion); a candidate whose lines straddle
    both parties goes to the party of the line that gave them more votes.
    """
    by_candidate: dict[str, dict] = defaultdict(lambda: {"votes": 0.0, "lines": []})
    for row in rows:
        if is_non_candidate(row["candidate"]):
            continue
        entry = by_candidate[_candidate_key(row["candidate"])]
        entry["name"] = row["candidate"]
        entry["votes"] += row["votes"]
        entry["lines"].append((row["votes"], resolve(year, state, row["candidate"], *row["labels"])))
    totals = {"D": 0.0, "R": 0.0, "other": 0.0}
    winners = []
    for entry in by_candidate.values():
        sides: dict[str, float] = defaultdict(float)
        for votes, side in entry["lines"]:
            if side:
                sides[side] += votes
        side = max(sides, key=sides.get) if sides else None
        totals[side or "other"] += entry["votes"]
        winners.append((entry["votes"], entry["name"], side))
    top = max(winners) if winners else (0.0, "", None)
    ranked = sorted(winners, reverse=True)
    total = totals["D"] + totals["R"] + totals["other"]
    return {**totals, "top_votes": top[0], "top_candidate": top[1], "top_side": top[2],
            "candidates": ranked,
            "comparable": is_two_party_contest(ranked, totals["D"] + totals["R"], total)}


MIN_TWO_PARTY_VOTE_SHARE = 0.70


def is_two_party_contest(ranked: list[tuple], major_votes: float, total_votes: float) -> bool:
    """Was this a Democrat-vs-Republican race a margin can describe?

    NOT when the top two finishers are on the same side (Alaska 2022 and 2010:
    Murkowski vs Tshibaka / Miller, both Republicans, with the Democrat at 11%
    and 24% -- "R+78" and "R+52" are not partisan margins; Lieberman vs Lamont
    in Connecticut 2006; Louisiana's same-party runoffs), nor when third
    parties and independents hold more than 30% (Alaska 2016: a Libertarian at
    29% and an independent at 13% left the Democrat at 11.6%; Florida 2010 with
    Crist at 30%). Across 50 years this flags 6 Senate and 18 House races."""
    if len(ranked) >= 2 and ranked[0][2] and ranked[0][2] == ranked[1][2]:
        return False
    return total_votes <= 0 or major_votes / total_votes >= MIN_TWO_PARTY_VOTE_SHARE
