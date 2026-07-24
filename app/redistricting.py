"""Redistricting events: when a state's U.S. House map changed.

A district's past result is only a valid baseline for a future race if the
boundaries are the same. The regular post-2020-census maps took effect in
2022, so a 2026 district's 2024 result is normally on current lines. But
several states redrew *mid-decade* for 2026, which makes their 2024 (and
earlier) district results stale — those results describe boundaries that no
longer exist.

This module records those events as data (state -> first cycle the NEW map
is in effect). ``features`` uses it two ways for any race whose most recent
available prior predates the newest map affecting its state:

  1. the stale district prior is dropped (the race falls back to the
     redistricting-immune ``state_lean`` plus incumbency), and
  2. the seat is flagged ``redrawn`` so the model widens its uncertainty
     (mandate hypothesis H-005: "recently redrawn districts require greater
     uncertainty").

``state_lean`` itself is a statewide aggregate and is unaffected by
within-state redistricting, so it remains valid for every state.

ACCURACY NOTE: this list asserts factual claims about specific 2026 maps and
must be kept current from an authoritative redistricting tracker (e.g. the
Brennan Center or Ballotpedia redistricting pages). It is intentionally
conservative — it contains only well-documented, high-confidence mid-decade
changes. Adding a state here only ever *reduces* the model's reliance on that
state's stale district priors in favour of its statewide lean; omitting a
state that did change leaves its districts on the pre-existing (possibly
stale) behaviour. Extend it as more maps are finalized/verified.
"""
from __future__ import annotations

# state -> earliest election cycle its CURRENT (2026) U.S. House map is in
# effect. A district prior from a cycle < this value is on superseded lines.
# Baseline for every other state is the post-2020 census round (2022).
POST_CENSUS_CYCLE = 2022

MIDDECADE_REMAP_CYCLE: dict[str, int] = {
    # Texas enacted a new mid-decade congressional map in a 2025 special
    # session (widely reported, GOP-drawn to add seats); 2024 results are on
    # the 2021 lines.
    "TX": 2026,
    # California voters passed Proposition 50 (Nov 2025) adopting a new
    # congressional map for 2026 in response; 2024 results are on the prior
    # citizen-commission lines.
    "CA": 2026,
    # Missouri's legislature passed a new GOP-drawn map (targeting the KC-area
    # 5th district); upheld as constitutional by the Missouri Supreme Court
    # (May 2026) and, as of this writing, in effect for 2026 despite an
    # ongoing referendum-suspension fight. 2024 results are on the prior lines.
    "MO": 2026,
    # North Carolina's legislature passed a new GOP-drawn map in Oct 2025; a
    # federal three-judge panel allowed it to be used for the 2026 midterms
    # (ruling Nov 26, 2025). 2024 results are on the 2023 lines.
    "NC": 2026,
    # Ohio's Redistricting Commission approved a new map on Oct 31, 2025 after
    # missing the legislature's statutory deadline (increasing the GOP seat
    # share); the state supreme court found no jurisdiction to block it before
    # 2026. 2024 results are on the prior lines.
    "OH": 2026,
    # Utah: a court-ordered remedial map (adopted Nov 2025) replaced the
    # legislature's map after years of litigation over Salt Lake County
    # splitting; the Utah Supreme Court dismissed the legislature's appeal
    # (Feb 2026), leaving it in place for 2026. 2024 results are on the old,
    # 4-safe-seat lines.
    "UT": 2026,
    # Louisiana's legislature redrew its map (signed by Gov. Landry) after the
    # US Supreme Court's Louisiana v. Callais decision (Apr 29, 2026) struck
    # down the prior map's second majority-Black district; the new map is
    # expected to be used in November 2026. 2024 results are on the
    # struck-down map.
    "LA": 2026,
    # Florida's legislature passed, and Gov. DeSantis signed (May 4, 2026), a
    # new GOP-favored map reworking 21 of 28 districts; the Florida Supreme
    # Court denied an injunction (Jun 10-11, 2026), leaving it in effect for
    # 2026 pending the underlying suit. 2024 results are on the prior lines.
    "FL": 2026,
}


def current_map_cycle(state: str) -> int:
    """First cycle the state's present U.S. House boundaries are in effect."""
    return MIDDECADE_REMAP_CYCLE.get(state, POST_CENSUS_CYCLE)


def prior_is_stale(state: str, prior_cycle: int, target_cycle: int) -> bool:
    """True if a district result from ``prior_cycle`` predates the map now in
    effect for ``target_cycle`` (i.e. it describes superseded boundaries)."""
    if prior_cycle is None:
        return False
    return prior_cycle < current_map_cycle(state) <= target_cycle
