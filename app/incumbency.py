"""Who is actually on the ballot: incumbent running vs. open seat.

A seat's history says how it has voted; whether the person who won it last
time is running again says how much of that history is personal. Open seats
drift toward the district's underlying partisanship; incumbents hold a
premium over it. This is one of the classic poll-free inputs, and it is
knowable before every election (nominees are set at the primaries).

Historical flags come from the bundled MEDSL certified returns: a general-
election nominee is an INCUMBENT when a same-party candidate with the same
surname won that office in the same state at the previous election(s) --
the House in the prior cycle (on any district number, so incumbents who
moved districts after a redraw still count), the Senate in any of the prior
three cycles (either seat). Matching on (state, party, surname) is simple on
purpose; appointed senators and special-election winners are the known
misses, and both are rare.

2026 flags come from ``data/vintage/incumbency_2026.csv`` (see
``load_current``), a snapshot of which sitting members are on the November
ballot. It is built from the unitedstates legislators list plus published
retirement / primary-loss lists, never from polls.
"""
from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from pathlib import Path

from .ingest.base import STATES, house_seat_key, senate_seat_key
from .ingest.medsl import BUNDLED_HOUSE_FILE, BUNDLED_SENATE_FILE, DEM_PARTIES, REP_PARTIES

CURRENT_FILE = Path(__file__).resolve().parents[1] / "data" / "vintage" / "incumbency_2026.csv"
_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


def surname(name: str) -> str:
    tokens = [t for t in re.sub(r"[^A-Z ]", " ", str(name).upper()).split()
              if t not in _SUFFIXES]
    return tokens[-1] if tokens else ""


def _nominees(path: Path, chamber: str) -> dict[tuple[int, str], dict[str, tuple[str, float]]]:
    """{(cycle, seat_key): {"D"|"R": (surname, votes)}} -- top vote-getter per party."""
    text = path.read_text(errors="replace")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    best: dict[tuple[int, str], dict[str, tuple[str, float]]] = defaultdict(dict)
    for raw in csv.DictReader(io.StringIO(text), delimiter=delimiter):
        row = {k.lower(): (v or "").strip().strip('"') for k, v in raw.items() if k}
        if row.get("stage", "GEN").upper() not in ("GEN", "GENERAL"):
            continue
        state = row.get("state_po", "")
        if state not in STATES:
            continue
        special = row.get("special", "FALSE").upper() in ("TRUE", "1")
        party_raw = (row.get("party_simplified") or row.get("party") or "").upper()
        party = "D" if party_raw in DEM_PARTIES else "R" if party_raw in REP_PARTIES else None
        if party is None:
            continue
        try:
            votes = float(row.get("candidatevotes") or 0)
        except ValueError:
            continue
        cycle = int(row["year"])
        seat = (house_seat_key(state, row.get("district") or 0) if chamber == "house"
                else senate_seat_key(state, special))
        current = best[(cycle, seat)].get(party)
        if current is None or votes > current[1]:
            best[(cycle, seat)][party] = (surname(row.get("candidate", "")), votes)
    return best


class IncumbencyLookup:
    """(dem_incumbent_running, rep_incumbent_running) per (cycle, seat)."""

    def __init__(self, house_path: Path = BUNDLED_HOUSE_FILE,
                 senate_path: Path = BUNDLED_SENATE_FILE,
                 current_path: Path = CURRENT_FILE):
        self._flags: dict[tuple[int, str], tuple[bool, bool]] = {}
        for chamber, path in (("house", house_path), ("senate", senate_path)):
            if path.exists():
                self._derive(_nominees(path, chamber), chamber)
        self._load_current(current_path)

    def _derive(self, nominees: dict, chamber: str) -> None:
        winners: dict[tuple[int, str], set[tuple[str, str]]] = defaultdict(set)
        for (cycle, seat), parties in nominees.items():
            if "D" in parties and "R" in parties:
                party = "D" if parties["D"][1] > parties["R"][1] else "R"
            elif parties:
                party = next(iter(parties))
            else:
                continue
            state = seat.split("-")[1]
            winners[(cycle, state)].add((party, parties[party][0]))
        lookback = (2,) if chamber == "house" else (2, 4, 6)
        for (cycle, seat), parties in nominees.items():
            state = seat.split("-")[1]
            prior = set().union(*(winners.get((cycle - k, state), set()) for k in lookback))
            dem = "D" in parties and ("D", parties["D"][0]) in prior
            rep = "R" in parties and ("R", parties["R"][0]) in prior
            self._flags[(cycle, seat)] = (dem, rep)

    def _load_current(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open() as handle:
            for row in csv.DictReader(handle):
                running = row["incumbent_running"].strip().lower() in ("1", "true", "yes")
                party = row["incumbent_party"].strip().upper()
                self._flags[(int(row["cycle"]), row["seat_key"])] = (
                    running and party == "D", running and party == "R")

    def flags(self, cycle: int, seat_key: str) -> tuple[bool, bool] | None:
        """None when the seat's nominees are unknown for that cycle."""
        return self._flags.get((cycle, seat_key))

    def value(self, cycle: int, seat_key: str) -> float | None:
        """+1 Democratic incumbent running, -1 Republican, 0 open seat,
        None unknown. Incumbent-vs-incumbent (post-redraw pairings) is 0."""
        flags = self.flags(cycle, seat_key)
        if flags is None:
            return None
        dem, rep = flags
        return float(dem) - float(rep)


def ballot_status(cycle: int, path: Path = CURRENT_FILE) -> dict[str, dict]:
    """{seat_key: {"running": bool, "status": str, "party": str}} for ``cycle``
    from the bundled ballot-status snapshot (empty when it is absent)."""
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["seat_key"]: {
                    "running": row["incumbent_running"].strip().lower() in ("1", "true", "yes"),
                    "status": row["status"], "party": row["incumbent_party"]}
                for row in csv.DictReader(handle) if int(row["cycle"]) == cycle}
