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
MODEL_VERSION = "2026.8"

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
     "chamber": "house", "metric": "variance inflation only, on top of a kept prior",
     "mechanism": "New boundaries add real risk that a district's past margin no longer "
                  "reflects its makeup, but most of a redrawn district's population and "
                  "partisan character persists through a redraw",
     "status": "Production",
     "validation": "app.redistricting records mid-decade remaps (TX, CA, MO, NC, OH, UT, "
                   "LA, FL for 2026). FIRST ATTEMPT (2026.6) dropped the stale district "
                   "prior entirely, falling back to state_lean; walk-forward tested "
                   "against 2022 -- the one real historical cycle where nearly every "
                   "House district's map changed post-census -- this made accuracy on "
                   "the affected seats WORSE (48.5%, worse than a coin flip) than the "
                   "unmodified prior (90.1%), and inflated the 2026 House median from "
                   "235 to 246 by systematically mispredicting redrawn deep-red seats "
                   "(e.g. Utah's 4 GOP-held seats) as competitive. CORRECTED (2026.7): "
                   "the prior is kept as the point estimate; only sigma widens (+4pt-sd) "
                   "for redrawn seats, which matched the full-revert walk-forward score "
                   "almost exactly (90.15% vs 90.15%) while still pricing in the genuine "
                   "extra boundary risk",
     "decision": "Structural: event-dated variance inflation, prior retained",
     "source": "Project research mandate"},
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
    {"id": "N-001", "claim": "REPORTED FAILURE: the raw generic-ballot average worsened held-out "
                             "accuracy in both chambers and was rejected from the champion.",
     "chamber": "both", "metric": "generic_ballot (time-decayed national average)",
     "mechanism": "GB polls carry cycle-varying partisan bias that ~10 training cycles "
                  "cannot separate from real environment shifts",
     "status": "Rejected for no predictive value",
     "validation": "challenger-generic-ballot vs champion at /api/models/comparison "
                   "(identical walk-forward protocol); 883 GB polls remain ingested",
     "decision": "Excluded from champion; auto-re-tested as a challenger every run so "
                 "promotion happens on evidence if live 2026 data changes the verdict. "
                 "A bias-corrected GB (house-effect adjusted) is the natural next experiment",
     "source": "This project's own backtests"},
    {"id": "S-002", "claim": "State-specific effects (partial-pooled per-state residual offsets) "
                             "are the disciplined form of 'niche state metrics'.",
     "chamber": "both", "metric": "shrunken per-state training-residual offsets (k=8)",
     "mechanism": "Persistent state-level polling/candidate error (e.g. Maine's history of "
                  "fundamentals misses) earns a data-sized correction, not a hand-picked story",
     "status": "Experimental",
     "validation": "challenger-state-effects vs champion at /api/models/comparison; "
                   "per-race disagreement visible at /api/races/{id}/models",
     "decision": "Runs as a challenger every cycle; promoted only on a robust "
                 "walk-forward win in both chambers",
     "source": "Project research mandate + user hypothesis"},
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
    {"id": "N-002", "claim": "FIXED BUG: the control simulation's shared national-shock size "
                             "was a hardcoded constant (3.5pts), understating real cycle-to-"
                             "cycle correlated error and producing false aggregate certainty.",
     "chamber": "house", "metric": "national_error_sigma (backtest.national_error_sigma)",
     "mechanism": "Individual-seat sigma was correctly wide (~26pts, core tier), but the "
                  "simulation treated most of it as independent per-seat noise; independent "
                  "noise across 435 seats washes out via the law of large numbers, turning "
                  "a modest average lean into near-certainty at the chamber level. The MEDSL "
                  "House backfill (raising seat-prior coverage from 105/470 to 468/470) made "
                  "this visible: House control jumped to 95.9% Democratic against an actual "
                  "current chamber of 218R/212D",
     "status": "Production",
     "validation": "national_error_sigma computed from the SD of out-of-sample walk-forward "
                   "cycle-level mean error (14 House cycles: -10.6 to +10.1pts observed) - "
                   "5.52pts, not 3.5. Verified the fix moves House control from an implausible "
                   "95.9% to 87.8% and the rating distribution from 378 Toss-ups (uninformative "
                   "core-tier default) to a realistic 215D/194R/26-toss-up split matching the "
                   "real chamber's near-even composition",
     "decision": "national_sigma is now computed per chamber from real backtest history and "
                 "wired through simulate_control(); no more hardcoded constant. Ruled out "
                 "alternative causes first: per-cycle-feature ridge shrinkage (0-100x sweep) "
                 "barely moved the aggregate number, and pooled calibration bins were "
                 "reasonable - the bug was specifically in how per-seat uncertainty was "
                 "decomposed into shared-vs-independent components for the simulation, not "
                 "in the margin coefficients themselves",
     "source": "This project's own backtests, investigated live in response to a user-observed "
              "implausible 2026 forecast"},
    {"id": "S-003", "claim": "State partisan lean (clipped mean of a state's House-district "
                             "margins) is a strong Senate fundamentals baseline and fills the "
                             "safe-seat gap the stale prior-Senate-margin leaves.",
     "chamber": "senate", "metric": "state_lean",
     "mechanism": "A statewide race tracks the state's overall partisan lean; the district "
                  "mean is a good proxy once uncontested-district blowouts are clipped",
     "status": "Production",
     "validation": "Validated against 2024 presidential two-party margins across all 35 "
                   "states with 2026 Senate races: mean abs error 3.7pts (raw district mean "
                   "was 7.3, distorted by uncontested seats - e.g. MA read D+84 vs true D+25). "
                   "Per-district clip at 40pts fixes it. Adding state_lean fixed Idaho and "
                   "Louisiana (no prior Senate result) collapsing from safe-R to D+3 toss-ups",
     "decision": "state_lean added to the core feature tier (available to every seat, every "
                 "cycle); Senate MAE improved 5.2->4.9",
     "source": "Project research mandate (state presidential lean) + user-flagged Senate issue"},
    {"id": "M-001", "claim": "The House and Senate should not share one champion spec.",
     "chamber": "both", "metric": "per-chamber champion selection by held-out log loss",
     "mechanism": "The Senate has ~14x fewer training races than the House, so it benefits "
                  "from stronger regularization",
     "status": "Production",
     "validation": "select_chamber_champions walk-forwards {base, ridge-strong, ridge-light, "
                   "state-effects} per chamber. House picks ridge-light (l2=2); Senate picks "
                   "ridge-strong (l2=8), improving Senate log loss 0.1536->0.1523. Scoreboard "
                   "stored in meta.chamber_champions",
     "decision": "Each chamber fits its own champion spec (mandate requirement 6)",
     "source": "Project research mandate + user request"},
    {"id": "N-003", "claim": "The 2026 topline is built up from individual seats on CURRENT "
                             "data; the national midterm swing on top is out-of-sample "
                             "validated, not an assumption that 2026 equals 2006.",
     "chamber": "house", "metric": "midterm_environment coefficient + pseudoreplication penalty",
     "mechanism": "Each seat is predicted from its own 2024 prior margin and state lean (current "
                  "maps/demographics); the president's party historically loses midterm seats",
     "status": "Production",
     "validation": "Decomposition: pure seat fundamentals give a House median of 216 (status "
                   "quo); the president's-party-midterm effect adds the rest. That effect was "
                   "confirmed to improve held-out prediction at BOTH the row level and the "
                   "cycle-level national-swing level (shrinking it to zero worsened cycle mean "
                   "error 4.6->5.1pts) - so it is earned, not assumed, and forcing the median "
                   "to the fundamentals-only 216 would override validated data with intuition. "
                   "BUT the coefficient was pseudo-replicated (6,088 House rows share ~14 "
                   "cycle values), inflating it; penalising cycle-level features by "
                   "rows-per-cycle corrects the effective sample size and moved the House "
                   "median 240->235 (matching the 2018 precedent of 235) at negligible "
                   "backtest cost",
     "decision": "Data-driven pseudoreplication penalty in MarginModel._penalties (replaces a "
                 "hardcoded multiplier). Seat features already use current-cycle data, so map/"
                 "demographic change IS captured per-seat; the remaining D-lean is the "
                 "validated midterm effect, expressed with wide intervals (House 80%: ~[210,260])",
     "source": "This project's backtests, investigated in response to a user methodology note"},
    {"id": "H-006", "claim": "The 2025-26 mid-decade redraws are net-Republican, so scoring "
                             "redrawn seats on their pre-redraw 2024 margins overstates "
                             "Democrats; encode each redraw's documented net seat change.",
     "chamber": "house", "metric": "redistricting.NET_DEM_SEAT_SHIFT + features.RedrawAdjust",
     "mechanism": "A partisan map cracks a state's most-marginal seats for the drawing party; "
                  "the retained old margin points the wrong way for exactly those seats",
     "status": "Production",
     "validation": "Documented net deltas (TX -5, FL -4, OH -2, MO/NC/LA -1, CA +5, UT +1; net "
                   "~-8 D) override the |delta| most-marginal seats per state to a lean of the "
                   "new party. Moves the House median 235->233 and P(D House) 0.83->0.79. The "
                   "topline effect is small BY DESIGN: individual unpolled House seats carry ~26pt "
                   "sigma this far out, so an 8-seat documented shift sits well inside the 80% "
                   "interval [~211,257] - which is also why the user's ~223 intuition is fully "
                   "consistent with the model (it is below the median, not outside the range). "
                   "The bigger, correct effect is on the redrawn seats' individual RATINGS.",
     "decision": "Ship as a sourced, per-seat structural input, NOT tuned to a topline. It cannot "
                 "be walk-forward validated (2026 has not happened); the ideal replacement is real "
                 "presidential-by-new-district partisanship, which the environment's network "
                 "policy currently blocks (Ballotpedia/Wikipedia return 403).",
     "source": "Documented enacted-map seat targets (NPR, NBC, state commissions), 2026"},
    {"id": "M-002", "claim": "REJECTED: recency-weighting the training cycles to shrink the "
                             "midterm swing (and pull the topline down) fails out of sample.",
     "chamber": "house", "metric": "walk-forward mean log loss vs exponential cycle half-life",
     "mechanism": "Down-weighting older cycles was hypothesised to reflect the smaller modern "
                  "midterm waves (2022 was only R+2.8) and lower the D-lean",
     "status": "Rejected",
     "validation": "Walk-forward 2006-2024, core tier: uniform weighting logloss 0.2707 beats "
                   "every half-life tested (12->0.2744, 8->0.2764, 6->0.2783, 4->0.2821). Older "
                   "cycles carry real signal; shrinking them only degrades accuracy. Also "
                   "confirmed the two R-president-midterm precedents (2006, 2018) had D swings of "
                   "+16 and +18 in median district margin - the model's regularized +5.75 is "
                   "already FAR below them, so the swing is conservative, not inflated.",
     "decision": "Keep uniform cycle weighting. Lowering the topline by recency-weighting or "
                 "shrinking a conservative swing would be fitting the answer, not the data.",
     "source": "This project's walk-forward backtests, in response to a user target-number note"},
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


def _store_alternative_model_snapshots(training, feature_rows, as_of, version):
    """Per-race predictions for every challenger and baseline, so users can
    switch between models on any race and see where they disagree. All are
    labelled by model_version; the champion alone drives ratings and control
    simulations."""
    from statistics import pstdev

    from .backtest import BASELINES, CHALLENGERS
    from .domain import normal_cdf, rating
    from .model import MIN_SIGMA, MarginModel, ridge_fit

    alternatives = []
    for name, model_kwargs in CHALLENGERS.items():
        challenger = MarginModel(**model_kwargs).fit(training)
        for race_id, row in feature_rows.items():
            p = challenger.predict(row)
            low80, high80 = p.interval(1.282)
            low95, high95 = p.interval(1.960)
            alternatives.append({
                "race_id": race_id, "as_of": as_of, "model_version": name,
                "data_version": version,
                "dem_probability": round(p.dem_probability, 4),
                "margin": round(p.mean, 2),
                "low80": round(low80, 2), "high80": round(high80, 2),
                "low95": round(low95, 2), "high95": round(high95, 2),
                "rating": rating(p.dem_probability), "quality": "-",
                "components": json.dumps({"_model": name})})
    for name, spec in BASELINES.items():
        idx = spec["features"]
        by_chamber: dict[str, tuple[list[float], float]] = {}
        for chamber in ("house", "senate"):
            rows = [r for r in training if r.chamber == chamber
                    and (not spec["polled_only"] or r.poll_count > 0)]
            if len(rows) < 30:
                continue
            xs = [[r.x[i] for i in idx] for r in rows]
            ys = [r.actual_margin for r in rows]
            weights = ridge_fit(xs, ys)
            residuals = [y - sum(w * v for w, v in zip(weights, x))
                         for x, y in zip(xs, ys)]
            by_chamber[chamber] = (weights, max(MIN_SIGMA, pstdev(residuals)))
        for race_id, row in feature_rows.items():
            if row.chamber not in by_chamber:
                continue
            if spec["polled_only"] and row.poll_count == 0:
                continue  # a polls-only model has nothing to say without polls
            weights, sigma = by_chamber[row.chamber]
            mean = sum(w * row.x[i] for w, i in zip(weights, idx))
            probability = min(.995, max(.005, normal_cdf(mean / sigma)))
            alternatives.append({
                "race_id": race_id, "as_of": as_of, "model_version": name,
                "data_version": version,
                "dem_probability": round(probability, 4),
                "margin": round(mean, 2),
                "low80": round(mean - 1.282 * sigma, 2), "high80": round(mean + 1.282 * sigma, 2),
                "low95": round(mean - 1.960 * sigma, 2), "high95": round(mean + 1.960 * sigma, 2),
                "rating": rating(probability), "quality": "-",
                "components": json.dumps({"_model": name})})
    store.insert_forecasts(alternatives)


def build_forecasts(as_of: str | None = None, prefix: str = "live",
                    with_backtests: bool = True) -> dict:
    """Train on ingested history, freeze snapshots, store control simulations."""
    as_of = as_of or date.today().isoformat()
    races = build_race_universe()
    results = ResultLookup(store.all_results())
    poll_lookup = PollLookup(store.all_polls())
    from .features import StateLean, RedrawAdjust
    state_lean = StateLean(results)
    redraw_adjust = RedrawAdjust(results)

    training: list = []
    for chamber in ("house", "senate"):
        from .features import historical_rows
        training.extend(historical_rows(results, poll_lookup, chamber,
                                        cycles=[c for c in results.cycles(chamber) if c < CYCLE],
                                        state_lean=state_lean))
    trained_chambers = {row.chamber for row in training}
    if not {"house", "senate"} <= trained_chambers:
        raise RuntimeError(
            "cannot train: no ingested historical results for "
            f"{sorted({'house', 'senate'} - trained_chambers)}; run ingestion first")
    # Each chamber picks its own champion spec on held-out log loss (the
    # mandate calls for different structures per chamber; this decides it on
    # evidence). Fit one model per chamber on that chamber's training rows.
    from .backtest import select_chamber_champions
    champions = select_chamber_champions(results, poll_lookup, state_lean)
    models: dict[str, MarginModel] = {}
    for chamber, choice in champions.items():
        chamber_rows = [r for r in training if r.chamber == chamber]
        models[chamber] = MarginModel(**choice["kwargs"]).fit(chamber_rows)

    version = data_version(store.counts(), prefix)
    snapshots = []
    feature_meta = {}
    feature_rows = {}
    for race in races:
        row = build_row(race["seat_key"], CYCLE, race["chamber"], race["state"],
                        race["district"], results, poll_lookup, as_of,
                        holder_party=race["incumbent_party"], state_lean=state_lean,
                        redraw_adjust=redraw_adjust)
        feature_rows[race["id"]] = row
        payload = models[race["chamber"]].forecast_payload(row, race["id"])
        payload.update({"as_of": as_of, "model_version": MODEL_VERSION,
                        "data_version": version})
        snapshots.append(payload)
        feature_meta[race["id"]] = {"has_prior": row.has_prior, "poll_count": row.poll_count}
    inserted = store.insert_forecasts(snapshots)
    _store_alternative_model_snapshots(training, feature_rows, as_of, version)

    champion_desc = "; ".join(f"{ch}: {choice['name']}" for ch, choice in champions.items())
    store.set_meta("chamber_champions", json.dumps(
        {ch: {"spec": choice["name"], "kwargs": choice["kwargs"],
              "log_loss_scoreboard": choice["scoreboard"]}
         for ch, choice in champions.items()}))
    store.upsert_model_version({
        "id": MODEL_VERSION, "chamber": "both", "status": "champion",
        "created_at": store.now(),
        "description": "Per-chamber ridge regression on vintage-safe "
                       f"fundamentals (state lean, seat history, environment) + "
                       f"time-decayed polling. Chamber champions -> {champion_desc}",
        "coefficients": json.dumps({ch: json.loads(m.to_json()) for ch, m in models.items()})})
    store.seed_research_claims(RESEARCH_CLAIMS)
    # Backtests run before the control simulation: they compute the
    # empirical national-shock size (see backtest.national_error_sigma) that
    # the simulation needs to avoid false aggregate certainty from averaging
    # away correlated seat-level errors.
    backtests = run_backtests(MODEL_VERSION) if with_backtests else []

    control = {}
    for chamber, base in (("house", 0), ("senate", int(store.get_meta("senate_dem_seats_not_up") or 0))):
        # Simulate from the champion's persisted snapshots, so snapshots and
        # control numbers can never disagree (snapshots are immutable: a
        # same-day rerun keeps the first frozen set).
        stored = store.latest_forecasts(chamber, model_version=MODEL_VERSION)
        nat_sigma = store.get_meta(f"national_sigma_{chamber}")
        kwargs = {"national_sigma": float(nat_sigma)} if nat_sigma else {}
        control[chamber] = simulate_control(stored, chamber, base_dem_seats=base, **kwargs)
        store.save_control_snapshot(stored[0]["as_of"], chamber, MODEL_VERSION,
                                    stored[0]["data_version"], control[chamber])

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
