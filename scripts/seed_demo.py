"""Seed a clearly-labelled synthetic demo dataset and run the REAL pipeline.

This exists so the full system (ingestion tables -> training -> backtests ->
snapshots -> API) can be exercised locally with zero network access. All rows
are tagged ``source='synthetic-demo'`` and every forecast produced carries a
``demo-...`` data version, which the API surfaces as DEMO MODE.
"""
from __future__ import annotations

import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import store  # noqa: E402
from app.forecast import HOUSE_APPORTIONMENT, SENATE_CLASS2, build_forecasts  # noqa: E402
from app.ingest.base import house_seat_key, senate_seat_key  # noqa: E402

SOURCE = "synthetic-demo"
CYCLES = [2014, 2016, 2018, 2020, 2022, 2024]


def seed() -> None:
    store.init_db()
    rng = Random(2026)
    results, polls, incumbents = [], [], []
    lean: dict[str, float] = {}
    for state, seats in HOUSE_APPORTIONMENT.items():
        for number in range(1, seats + 1):
            lean[house_seat_key(state, number)] = rng.gauss(0, 18)
    for state in SENATE_CLASS2:
        lean[senate_seat_key(state)] = rng.gauss(0, 12)
    for cycle in CYCLES:
        national = rng.gauss(0, 4)
        for seat_key, base in lean.items():
            chamber = seat_key.split("-")[0]
            if chamber == "senate" and (cycle - 2026) % 6 != 0:
                continue
            state = seat_key.split("-")[1]
            margin = base + national + rng.gauss(0, 5)
            district = seat_key.split("-")[2] if chamber == "house" else None
            results.append({
                "cycle": cycle, "chamber": chamber, "state": state,
                "district": district, "seat_key": seat_key,
                "dem_pct": 50 + margin / 2, "rep_pct": 50 - margin / 2,
                "dem_margin": margin, "winner_party": "D" if margin > 0 else "R",
                "source": SOURCE})
            if abs(base) < 10 and rng.random() < .5:
                polls.append({
                    "external_id": f"demo-{cycle}-{seat_key}", "cycle": cycle,
                    "chamber": chamber, "state": state, "district": district,
                    "seat_key": seat_key, "pollster": "Demo Synthetic Poll",
                    "methodology": "synthetic", "sample_size": 600, "partisan": None,
                    "poll_date": f"{cycle}-10-{rng.randint(10, 28)}",
                    "election_date": f"{cycle}-11-08",
                    "dem_pct": None, "rep_pct": None,
                    "dem_margin": margin + rng.gauss(0, 4), "source": SOURCE})
    for seat_key, base in lean.items():
        chamber = seat_key.split("-")[0]
        state = seat_key.split("-")[1]
        incumbents.append({
            "cycle": 2026, "chamber": chamber, "state": state,
            "district": seat_key.split("-")[2] if chamber == "house" else None,
            "seat_key": seat_key, "party": "D" if base > 0 else "R",
            "name": f"Demo Incumbent {seat_key}", "source": SOURCE})
    store.insert_rows("election_results", results)
    store.insert_rows("polls", polls)
    store.insert_rows("incumbents", incumbents)
    store.set_meta("senate_dem_seats_not_up", "34")
    store.record_source(SOURCE, None, "n/a (synthetic)", available_at=store.now(),
                        sha256=None, record_count=len(results) + len(polls),
                        note="deterministic synthetic demo records")
    store.seed_research_claims([
        {"id": "H-001", "claim": "Seat partisan history is a strong baseline.",
         "chamber": "house", "metric": "prior_margin", "mechanism": "Partisan alignment",
         "status": "Production", "validation": "Expanding-window backtests",
         "decision": "Included", "source": "Project research synthesis"},
        {"id": "S-001", "claim": "Senate races are more candidate-sensitive than House races.",
         "chamber": "senate", "metric": "residual_sigma", "mechanism": "Statewide personal brands",
         "status": "Production", "validation": "Chamber-specific residual pools",
         "decision": "Separate Senate fit", "source": "Project research synthesis"},
    ])
    summary = build_forecasts(prefix="demo")
    print("Seeded clearly labelled synthetic demo dataset.")
    print(f"forecast: {summary['data_version']}, races={summary['races']}")


if __name__ == "__main__":
    seed()
