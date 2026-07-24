"""Vintage-safe feature construction.

Every feature for a race in cycle ``t`` uses only information available
before that election: results from cycles strictly before ``t``, polls with a
field date on or before the as-of date, and the president's party (fixed years
in advance). Missing inputs are flagged, never imputed with invented values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from . import redistricting

# The president's party during each November election. Facts of record.
PRESIDENT_PARTY = {
    1998: "D", 2000: "D", 2002: "R", 2004: "R", 2006: "R", 2008: "R",
    2010: "D", 2012: "D", 2014: "D", 2016: "D", 2018: "R", 2020: "R",
    2022: "D", 2024: "D", 2026: "R",
}

FEATURE_NAMES = [
    "intercept", "prior_margin", "has_prior", "prior_winner",
    "state_lean", "has_state_lean",
    "environment", "midterm_environment",
    "generic_ballot", "has_generic_ballot",
    "poll_average", "has_polls",
]

# National generic-congressional-ballot polls are stored under this seat key
# (chamber "national"); their time-decayed average is a shared environment
# input available to every race, polled or not.
GENERIC_BALLOT_SEAT = "us-generic"

# Result sources ranked by authority when several report the same seat-cycle.
SOURCE_PRIORITY = ["official-results-csv", "medsl-constituency-returns",
                   "fivethirtyeight-raw-polls", "synthetic-demo"]

POLL_HALF_LIFE_DAYS = 21.0
PRIOR_CLIP = 50.0


@dataclass
class FeatureRow:
    seat_key: str
    cycle: int
    chamber: str
    state: str
    district: str | None
    x: list[float]
    actual_margin: float | None = None
    poll_count: int = 0
    last_poll_date: str | None = None
    has_prior: bool = False
    detail: dict = field(default_factory=dict)


def environment_signs(cycle: int) -> tuple[float, float]:
    """(environment, midterm_environment): +1 favors the out-party being D."""
    president = PRESIDENT_PARTY.get(cycle)
    if president is None:
        raise ValueError(f"no president-party fact for cycle {cycle}")
    sign = 1.0 if president == "R" else -1.0
    return sign, sign if cycle % 4 == 2 else 0.0


class ResultLookup:
    """Best-source seat/cycle margins from ingested election results."""

    def __init__(self, results: list[dict]):
        self._by_seat: dict[tuple[int, str], dict] = {}
        for row in results:
            key = (row["cycle"], row["seat_key"])
            current = self._by_seat.get(key)
            if current is None or self._rank(row["source"]) < self._rank(current["source"]):
                self._by_seat[key] = row

    @staticmethod
    def _rank(source: str) -> int:
        try:
            return SOURCE_PRIORITY.index(source)
        except ValueError:
            return len(SOURCE_PRIORITY)

    def margin(self, cycle: int, seat_key: str) -> float | None:
        row = self._by_seat.get((cycle, seat_key))
        return None if row is None else row["dem_margin"]

    def prior(self, cycle: int, seat_key: str, chamber: str) -> tuple[float | None, int | None]:
        """Most recent same-seat result strictly before ``cycle``."""
        if seat_key.endswith("-special"):
            # A special fills a seat whose last regular election ran under
            # the base seat key, on that seat's own class schedule.
            seat_key = seat_key.removesuffix("-special")
            lookback = [cycle - 2, cycle - 4, cycle - 6]
        elif chamber == "house":
            lookback = [cycle - 2, cycle - 4]
        else:
            lookback = [cycle - 6, cycle - 8, cycle - 4]
        for prior_cycle in lookback:
            if prior_cycle >= cycle:
                continue
            margin = self.margin(prior_cycle, seat_key)
            if margin is not None:
                return margin, prior_cycle
        return None, None

    def cycles(self, chamber: str | None = None) -> list[int]:
        return sorted({c for (c, seat) in self._by_seat
                       if not chamber or seat.startswith(chamber)})

    def seats(self, cycle: int, chamber: str) -> list[dict]:
        return [row for (c, seat), row in self._by_seat.items()
                if c == cycle and seat.startswith(chamber)]


class StateLean:
    """Vintage-safe state partisan baseline: the mean two-party House-district
    margin across a state, from the most recent cycle strictly before the
    target. This is the single strongest fundamentals signal for a Senate
    race (a state that votes R+40 for its House delegation is not a
    competitive Senate seat), and — unlike the 6-year-stale prior Senate
    margin — it exists for *every* state every cycle, closing the gap that
    made no-prior seats like Idaho and Louisiana collapse to a meaningless
    environment-only default. Built from the same House results the model
    already ingests, so it needs no new data source."""

    # Individual district margins are clipped to +/-DISTRICT_CLIP before
    # averaging: uncontested districts (one major party doesn't run) show up
    # as ~+/-100 and, unclipped, wildly distort the state mean (Massachusetts
    # read D+84 vs a true ~D+25; deep-red states were similarly exaggerated).
    # Clipping at 40 caps those blowouts while preserving genuine safe-district
    # signal. Validated against 2024 presidential two-party margins across all
    # 35 states with 2026 Senate races: mean absolute error drops from 7.3
    # points (raw mean) to 3.7 (clipped), max error 21 -> 12.
    DISTRICT_CLIP = 40.0

    def __init__(self, results: "ResultLookup"):
        by_state_cycle: dict[tuple[str, int], list[float]] = {}
        for (cycle, seat), row in results._by_seat.items():
            if seat.startswith("house-"):
                clipped = max(-self.DISTRICT_CLIP, min(self.DISTRICT_CLIP, row["dem_margin"]))
                by_state_cycle.setdefault((row["state"], cycle), []).append(clipped)
        self._mean: dict[tuple[str, int], float] = {
            key: sum(vals) / len(vals) for key, vals in by_state_cycle.items()}
        self._cycles_by_state: dict[str, list[int]] = {}
        for (state, cycle) in self._mean:
            self._cycles_by_state.setdefault(state, []).append(cycle)
        for state in self._cycles_by_state:
            self._cycles_by_state[state].sort()

    def lean(self, state: str, before_cycle: int) -> tuple[float | None, int | None]:
        prior_cycles = [c for c in self._cycles_by_state.get(state, []) if c < before_cycle]
        if not prior_cycles:
            return None, None
        cycle = prior_cycles[-1]
        return self._mean[(state, cycle)], cycle


class PollLookup:
    """Time-decayed poll averages with an explicit as-of cutoff."""

    def __init__(self, polls: list[dict]):
        self._by_seat: dict[tuple[int, str], list[dict]] = {}
        for p in polls:
            self._by_seat.setdefault((p["cycle"], p["seat_key"]), []).append(p)

    def average(self, cycle: int, seat_key: str, as_of: str) -> tuple[float | None, int, str | None]:
        """(weighted margin, poll count, last field date), polls after as_of excluded."""
        cutoff = _to_date(as_of)
        usable = [p for p in self._by_seat.get((cycle, seat_key), [])
                  if _to_date(p["poll_date"]) <= cutoff]
        if not usable:
            return None, 0, None
        weight_sum = value_sum = 0.0
        for p in usable:
            age = (cutoff - _to_date(p["poll_date"])).days
            weight = 0.5 ** (age / POLL_HALF_LIFE_DAYS)
            if p.get("partisan"):
                weight *= 0.5
            weight_sum += weight
            value_sum += weight * p["dem_margin"]
        last = max(p["poll_date"] for p in usable)
        return (value_sum / weight_sum if weight_sum > 0 else None), len(usable), last


def _to_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def clip(value: float, bound: float = PRIOR_CLIP) -> float:
    return max(-bound, min(bound, value))


def build_row(seat_key: str, cycle: int, chamber: str, state: str,
              district: str | None, results: ResultLookup, poll_lookup: PollLookup,
              as_of: str, actual_margin: float | None = None,
              holder_party: str | None = None,
              state_lean: "StateLean | None" = None) -> FeatureRow:
    prior_margin, prior_cycle = results.prior(cycle, seat_key, chamber)
    poll_avg, poll_count, last_poll = poll_lookup.average(cycle, seat_key, as_of)
    gb_avg, gb_count, _ = poll_lookup.average(cycle, GENERIC_BALLOT_SEAT, as_of)
    environment, midterm_environment = environment_signs(cycle)
    lean_value, lean_cycle = (state_lean.lean(state, cycle) if state_lean else (None, None))
    # A district that redrew its U.S. House lines mid-decade (see
    # redistricting) has a *stale* prior: it describes different boundaries.
    # An earlier version of this code dropped the prior entirely for such
    # seats, falling back to the much coarser state_lean. Walk-forward-tested
    # against 2022 -- the one real historical cycle where nearly every House
    # district's map changed after the census -- that made things worse, not
    # better: accuracy on the affected seats fell to 48.5% (worse than a coin
    # flip) vs 90.1% when the stale prior was kept, because most of a
    # redrawn district's population and partisan character persists through
    # a redraw, and a same-state average is a far weaker signal than even a
    # stale district-specific result. So the prior is kept as the point
    # estimate; only the seat's uncertainty (sigma, in model.py) widens for
    # `redrawn` seats, reflecting the real but smaller risk that this
    # particular redraw moved the district more than history suggests.
    redrawn = (chamber == "house"
               and redistricting.prior_is_stale(state, prior_cycle, cycle))
    has_prior = prior_margin is not None
    has_polls = poll_avg is not None
    has_gb = gb_avg is not None
    has_lean = lean_value is not None
    if has_prior and prior_margin != 0:
        prior_winner = 1.0 if prior_margin > 0 else -1.0
    else:
        # No ingested prior result: the party currently holding the seat is
        # still a pre-election fact and carries the same signal.
        prior_winner = {"D": 1.0, "R": -1.0}.get(holder_party or "", 0.0)
    x = [
        1.0,
        clip(prior_margin) if has_prior else 0.0,
        1.0 if has_prior else 0.0,
        prior_winner,
        clip(lean_value) if has_lean else 0.0,
        1.0 if has_lean else 0.0,
        environment,
        midterm_environment,
        clip(gb_avg, 25.0) if has_gb else 0.0,
        1.0 if has_gb else 0.0,
        poll_avg if has_polls else 0.0,
        1.0 if has_polls else 0.0,
    ]
    return FeatureRow(
        seat_key=seat_key, cycle=cycle, chamber=chamber, state=state,
        district=district, x=x, actual_margin=actual_margin,
        poll_count=poll_count, last_poll_date=last_poll, has_prior=has_prior,
        detail={"prior_cycle": prior_cycle, "as_of": as_of,
                "state_lean_cycle": lean_cycle, "redrawn": redrawn})


def historical_rows(results: ResultLookup, poll_lookup: PollLookup, chamber: str,
                    cycles: list[int] | None = None,
                    election_dates: dict[int, str] | None = None,
                    state_lean: "StateLean | None" = None) -> list[FeatureRow]:
    """One row per seat with a known outcome; as-of is that cycle's election day."""
    if state_lean is None:
        state_lean = StateLean(results)
    rows: list[FeatureRow] = []
    for cycle in cycles or results.cycles(chamber):
        if cycle not in PRESIDENT_PARTY:
            continue
        as_of = (election_dates or {}).get(cycle) or f"{cycle}-11-08"
        for result in results.seats(cycle, chamber):
            rows.append(build_row(
                result["seat_key"], cycle, chamber, result["state"],
                result.get("district"), results, poll_lookup, as_of,
                actual_margin=result["dem_margin"], state_lean=state_lean))
    return rows
