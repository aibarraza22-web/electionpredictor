"""2026 forecast pipeline.

Builds the real race universe (the 435 post-2020-census House districts, the
33 class-2 Senate seats, and special elections detected from appointed-seat
term data), trains the model on all ingested history, freezes immutable
per-race snapshots, and stores chamber-control simulations.
"""
from __future__ import annotations

import json
from datetime import date

from . import store
from .backtest import run_backtests
from .features import PollLookup, ResultLookup, build_row
from .ingest.base import house_seat_key, senate_seat_key
from .model import MarginModel
from .simulation import simulate_control

CYCLE = 2026
MODEL_VERSION = "2026.2"

# Seats per state, 2020 census apportionment (sums to 435).
HOUSE_APPORTIONMENT = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WI": 8,
    "WV": 2, "WY": 1,
}

# Senate class 2: regularly scheduled in November 2026.
SENATE_CLASS2 = ["AL", "AK", "AR", "CO", "DE", "GA", "ID", "IL", "IA", "KS",
                 "KY", "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH",
                 "NJ", "NM", "NC", "OK", "OR", "RI", "SC", "SD", "TN", "TX",
                 "VA", "WV", "WY"]

RANKED_CHOICE_STATES = {"AK", "ME"}

# Research registry: every claim from the project mandate that the current
# system operationalizes, with its honest lifecycle status. Validation always
# points at stored, queryable evidence — never at prose.
RESEARCH_CLAIMS = [
    {"id": "H-001", "claim": "Seat partisan history (prior result) is a strong initial baseline.",
     "chamber": "both", "metric": "prior_margin",
     "mechanism": "Partisan alignment persists between cycles",
     "status": "Production",
     "validation": "Expanding-window: compare champion vs baseline-prior-result at /api/models/comparison",
     "decision": "Included in both chamber models", "source": "Project research mandate"},
    {"id": "H-002", "claim": "District polling deserves more weight closer to Election Day.",
     "chamber": "both", "metric": "poll_average (21-day half-life decay)",
     "mechanism": "Recent opinion measures current candidate standing",
     "status": "Production",
     "validation": "Horizon breakdown (0/30/90 days pre-election) stored in each champion backtest run config",
     "decision": "Time-decayed average in polled tier", "source": "Project research mandate"},
    {"id": "H-003", "claim": "Absence of polls is not evidence a race is tied.",
     "chamber": "both", "metric": "two-tier model routing",
     "mechanism": "Unpolled races fall back to fundamentals, with wider uncertainty",
     "status": "Production",
     "validation": "Separate fundamentals fit + unpolled subgroup metrics in run config",
     "decision": "Dedicated fundamentals tier", "source": "Project research mandate"},
    {"id": "H-004", "claim": "The president's party is penalized in midterms.",
     "chamber": "both", "metric": "midterm_environment",
     "mechanism": "Midterm referendum dynamics against the White House",
     "status": "Production",
     "validation": "Compare champion vs baseline-environment-only; midterm-cycle subgroup metrics",
     "decision": "Environment + midterm interaction features", "source": "Project research mandate"},
    {"id": "H-005", "claim": "Recently redrawn districts require greater uncertainty.",
     "chamber": "house", "metric": "has_prior / lookback restriction",
     "mechanism": "New boundaries break seat-history comparability",
     "status": "Production",
     "validation": "House prior lookback restricted to post-redistricting cycles; +5pt sigma when no prior",
     "decision": "Structural: lookback limits + variance inflation", "source": "Project research mandate"},
    {"id": "S-001", "claim": "Senate races are more candidate-sensitive than House races.",
     "chamber": "senate", "metric": "chamber-specific residual sigma",
     "mechanism": "Statewide personal brands decouple from partisanship",
     "status": "Validated",
     "validation": "Separate Senate fit; residual sigmas stored per chamber in model_versions.coefficients",
     "decision": "Chamber-specific models (mandate requirement 6)", "source": "Project research mandate"},
    {"id": "F-001", "claim": "Challenger fundraising may be more informative than total spending.",
     "chamber": "both", "metric": "FEC receipts/cash-on-hand",
     "mechanism": "Money proxies candidate quality and enthusiasm",
     "status": "Collecting data",
     "validation": "FEC adapter ingests live totals; NO historical vintage series yet, so no leakage-safe backtest is possible",
     "decision": "Displayed per race; excluded from the model until vintage-tested",
     "source": "Project research mandate"},
    {"id": "A-001", "claim": "Alaska/Maine ranked-choice races need transfer-round simulation.",
     "chamber": "senate", "metric": "election_system flag",
     "mechanism": "Multi-candidate elimination changes win conditions",
     "status": "Proposed",
     "validation": "Not yet modeled; races are flagged ranked_choice and carry standard uncertainty",
     "decision": "Open challenger-model slot; margins for AK/ME treated as two-party approximations",
     "source": "Project research mandate"},
    {"id": "P-001", "claim": "Polling errors are correlated within a cycle, not independent.",
     "chamber": "both", "metric": "shared national shock (3.5pt sigma)",
     "mechanism": "Common-mode polling and environment misses",
     "status": "Production",
     "validation": "Margin-space control simulation decomposes national vs idiosyncratic error",
     "decision": "Correlated simulation structure", "source": "Project research mandate"},
    {"id": "P-002", "claim": "REPORTED FAILURE: at election-eve cutoff the polls-only baseline "
                             "marginally beats the blended model on polled races.",
     "chamber": "both", "metric": "baseline-polls-only vs champion",
     "mechanism": "Election-eve polling already impounds most fundamentals information",
     "status": "Experimental",
     "validation": "See /api/models/comparison (identical walk-forward protocol); the blend "
                   "remains ahead of every fundamentals baseline and is required for the "
                   "~78% of 2026 races with no polling",
     "decision": "Champion retained for full-universe coverage; a higher-poll-weight "
                 "challenger is the top open experiment. Not silently tuned away, per "
                 "the no-post-hoc-fitting rule",
     "source": "This project's own backtests"},
]


def build_race_universe() -> list[dict]:
    """Upsert the 2026 race table from ingested incumbency data."""
    incumbents = store.all_incumbents(CYCLE)
    timestamp = store.now()
    rows: list[dict] = []
    for state, seats in HOUSE_APPORTIONMENT.items():
        for number in range(1, seats + 1):
            seat_key = house_seat_key(state, number)
            incumbent = incumbents.get(seat_key)
            rows.append({
                "id": f"{CYCLE}-{seat_key}", "cycle": CYCLE, "chamber": "house",
                "state": state, "district": f"{number:02d}", "seat_key": seat_key,
                "name": f"{state}-{number:02d}",
                "incumbent_party": incumbent["party"] if incumbent else None,
                "incumbent_name": incumbent["name"] if incumbent else None,
                "open_seat": incumbent is None,
                "special": False,
                "election_system": "ranked_choice" if state in RANKED_CHOICE_STATES else "plurality",
                "updated_at": timestamp,
            })
    senate_seats = [(state, False) for state in SENATE_CLASS2]
    # Specials come from ingested appointed-seat terms, not a hardcoded list.
    senate_seats += [(inc["state"], True) for key, inc in incumbents.items()
                     if key.startswith("senate-") and key.endswith("-special")]
    for state, special in senate_seats:
        seat_key = senate_seat_key(state, special)
        incumbent = incumbents.get(seat_key)
        label = f"{state} Senate" + (" (special)" if special else "")
        rows.append({
            "id": f"{CYCLE}-{seat_key}", "cycle": CYCLE, "chamber": "senate",
            "state": state, "district": None, "seat_key": seat_key, "name": label,
            "incumbent_party": incumbent["party"] if incumbent else None,
            "incumbent_name": incumbent["name"] if incumbent else None,
            "open_seat": incumbent is None,
            "special": special,
            "election_system": "ranked_choice" if state in RANKED_CHOICE_STATES else "plurality",
            "updated_at": timestamp,
        })
    store.upsert_races(rows)
    return rows


def data_version(counts: dict, prefix: str = "live") -> str:
    return f"{prefix}-{date.today().isoformat()}-r{counts['election_results']}-p{counts['polls']}"


def build_forecasts(as_of: str | None = None, prefix: str = "live",
                    with_backtests: bool = True) -> dict:
    """Train on ingested history, freeze snapshots, store control simulations."""
    as_of = as_of or date.today().isoformat()
    races = build_race_universe()
    results = ResultLookup(store.all_results())
    poll_lookup = PollLookup(store.all_polls())

    training: list = []
    for chamber in ("house", "senate"):
        from .features import historical_rows
        training.extend(historical_rows(results, poll_lookup, chamber,
                                        cycles=[c for c in results.cycles(chamber) if c < CYCLE]))
    trained_chambers = {row.chamber for row in training}
    if not {"house", "senate"} <= trained_chambers:
        raise RuntimeError(
            "cannot train: no ingested historical results for "
            f"{sorted({'house', 'senate'} - trained_chambers)}; run ingestion first")
    model = MarginModel().fit(training)

    version = data_version(store.counts(), prefix)
    snapshots = []
    feature_meta = {}
    for race in races:
        row = build_row(race["seat_key"], CYCLE, race["chamber"], race["state"],
                        race["district"], results, poll_lookup, as_of,
                        holder_party=race["incumbent_party"])
        payload = model.forecast_payload(row, race["id"])
        payload.update({"as_of": as_of, "model_version": MODEL_VERSION,
                        "data_version": version})
        snapshots.append(payload)
        feature_meta[race["id"]] = {"has_prior": row.has_prior, "poll_count": row.poll_count}
    inserted = store.insert_forecasts(snapshots)

    control = {}
    for chamber, base in (("house", 0), ("senate", int(store.get_meta("senate_dem_seats_not_up") or 0))):
        # Simulate from what is actually persisted, so snapshots and control
        # numbers can never disagree (snapshots are immutable: a same-day
        # rerun keeps the first frozen set).
        stored = store.latest_forecasts(chamber)
        control[chamber] = simulate_control(stored, chamber, base_dem_seats=base)
        store.save_control_snapshot(stored[0]["as_of"], chamber, MODEL_VERSION,
                                    stored[0]["data_version"], control[chamber])

    store.upsert_model_version({
        "id": MODEL_VERSION, "chamber": "both", "status": "champion",
        "created_at": store.now(),
        "description": "Chamber-specific ridge regression on vintage-safe "
                       "fundamentals + time-decayed polling averages",
        "coefficients": model.to_json()})
    store.seed_research_claims(RESEARCH_CLAIMS)
    backtests = run_backtests(MODEL_VERSION) if with_backtests else []
    store.set_meta("last_forecast_as_of", as_of)
    store.set_meta("last_data_version", version)
    coverage = {
        "races": len(races),
        "with_prior_result": sum(1 for m in feature_meta.values() if m["has_prior"]),
        "with_polls": sum(1 for m in feature_meta.values() if m["poll_count"] > 0),
    }
    store.set_meta("coverage", json.dumps(coverage))
    return {"as_of": as_of, "data_version": version, "races": len(races),
            "snapshots_inserted": inserted, "coverage": coverage,
            "control": {k: {"democratic_control_probability": v["democratic_control_probability"]}
                        for k, v in control.items()},
            "backtests": [r["id"] for r in backtests]}
