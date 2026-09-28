"""2026 forecast pipeline.

Builds the real race universe (the 435 post-2020-census House districts, the
33 class-2 Senate seats, and special elections detected from appointed-seat
term data), fits the poll-free track-record model (``app.track_record``) on
all ingested history, freezes immutable per-race snapshots, and stores
chamber-control simulations.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

from . import gates, store
from .domain import rating
from .features import ResultLookup
from .ingest.base import house_seat_key, senate_seat_key
from .ratings import RatingLookup, is_unanimously_safe
from .simulation import simulate_control

CYCLE = 2026
# 2026.11: full 1976-2024 Senate history from the official single-source
# Dataverse file (was a 2004-2024 two-source reconstruction), champion
# selection scored on recent cycles only (CHAMPION_SCORING_SINCE=2010), and
# the poll-blend/toss-up-ceiling investigations (P-002, P-004) which found no
# further change to make. Net: Senate winner accuracy 0.872 -> 0.893 on a
# fixed 2010-2024 walk-forward. Bumped (over 2026.10) because forecast
# snapshots are immutable per (race_id, as_of, model_version): a same-day
# rerun under an unchanged version keeps the day's first frozen numbers, so a
# real prediction-affecting change must bump the version to surface.
# 2026.21: POLL-FREE. The published forecast is the track-record model
# (app.track_record, research claim T-004): no polls, no poll-derived expert
# ratings, no unvalidated campaign layer -- only inputs that also existed for
# every past election, so the whole system is replayed on 2010-2024 exactly as
# it runs on 2026.
MODEL_VERSION = "2026.21"
# Competitive races must have at least presidential partisanship on their
# current lines plus a known ballot status (grade C). Redrawn 2026 seats have,
# by construction, no same-map history, so A/B cannot be required of them.
POLL_FREE_REQUIRED_GRADES = {"A", "B", "C"}
# A 50-50 Senate is decided by the Vice President's vote. Vice President
# JD Vance (R) holds it through January 2029, so a tie is REPUBLICAN control.
# Earlier versions simulated the tie as Democratic control, overstating the
# Democrats' Senate chances by the full probability of a 50-50 outcome.
SENATE_TIE_BREAK_PARTY = "republican"
# Release-gate floor: the 2026 map has 153 House seats and all 35 Senate seats
# on the published ratings pages. Anything far below that means the ratings
# ingest failed and the forecast would silently fall back to history alone.
MIN_RATED_RACES = 150
ELECTION_DATE = "2026-11-03"

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
    {"id": "P-002", "claim": "RESOLVED: the polls-only baseline's tiny edge on polled races is "
                             "not robustly capturable; the blend already trusts polls near-fully.",
     "chamber": "both", "metric": "polled-race winner accuracy vs poll-coefficient / blend weight",
     "mechanism": "Election-eve polling already impounds most fundamentals information, so the "
                  "optimal poll weight is close to 1",
     "status": "Validated",
     "validation": "Followed up the open experiment with real historical polls (538 raw-polls, "
                   "1998-2022) ingested locally. On polled races (2010-2024) polls-only wins "
                   "narrowly (house 0.839 vs blend 0.818; senate 0.932 vs 0.928), but the blend's "
                   "full-tier poll coefficient is ALREADY 0.886 (house) / 0.967 (senate), and "
                   "sweeping its ridge penalty from 4.0 down to 0.01 does not move winner "
                   "accuracy - the model is at the poll-trust ceiling. A poll-count-weighted "
                   "blend and a crude fundamentals+poll average both did WORSE. The residual gap "
                   "is noise-level (~14 of 688 house races) and did not generalise across "
                   "formulations, so capturing it would be post-hoc fitting.",
     "decision": "Keep the blend (near-optimal poll weight, and required for the ~78% of 2026 "
                 "races with no polls). The competitive-race ceiling (~82% house / ~93% senate "
                 "polled) is set by irreducible ~3-4pt poll error, not by the blend.",
     "source": "This project's own backtests + user push to maximise winner accuracy"},
    {"id": "P-004", "claim": "The competitive-race (toss-up) accuracy ceiling is set by "
                             "irreducible outcome randomness and UNPREDICTABLE poll bias, not by "
                             "a shortage of data or model capacity.",
     "chamber": "both", "metric": "decomposition of wrong calls on polled races (2010-2024)",
     "mechanism": "A near-50/50 race is partly a coin flip; and whether a given cycle's polls are "
                  "accurate is not knowable until after the election",
     "status": "Validated",
     "validation": "Ingested real 538 polls and dissected every wrong call on polled races. Of "
                   "125 House misses: 43% were within 3pt (near coin-flips); of the 5pt+ misses, "
                   "33 had the POLLS also wrong (irreducible poll error) and only 19 had polls "
                   "right but fundamentals overriding. Critically, deferring fully to polls to "
                   "capture those 19 is NOT robust: polls-only beats the blend in 2010/2016/2018 "
                   "but LOSES in 2020/2022, when polls were systematically biased and the "
                   "fundamentals correctly anchored the call. You cannot know at prediction time "
                   "which regime you are in, so the blend is the robust optimum.",
     "decision": "No change: the poll/fundamentals blend is robust-optimal. Further gains need "
                 "signals this environment blocks (FEC finance API, precinct/demographic data) "
                 "and would still be bounded by the irreducible floor. Documented rather than "
                 "chased, per the no-post-hoc-fitting rule.",
     "source": "User push to raise toss-up accuracy; this project's backtests"},
    {"id": "P-005", "claim": "REJECTED: campaign-finance receipts disparity, tested for real with "
                             "complete historical FEC data for BOTH chambers, does NOT improve "
                             "competitive-race accuracy -- it makes House worse and is noise, not "
                             "signal, for Senate.",
     "chamber": "both", "metric": "walk-forward winner accuracy on polled races with real FEC "
                                  "finance data, 2012-2024 (all 7 cycles, both chambers)",
     "mechanism": "The environment's network policy opened mid-session (Harvard Dataverse and the "
                  "FEC API both became reachable), so this hypothesis -- previously blocked -- "
                  "could finally be tested with real data instead of reasoned about",
     "status": "Rejected",
     "validation": "Pulled the complete real historical FEC candidate-totals (api.open.fec.gov, "
                   "DEMO_KEY, House+Senate, 2012-2024, 17,387 candidate-cycle rows -- every cycle, "
                   "both chambers) and computed each race's receipts disparity. Tested three ways, "
                   "walk-forward, on the same polled-race subset: (1) directional agreement on the "
                   "model's own wrong calls was noisy, not robust once all cycles were in (House: "
                   "71/50/71/71/30/62% across 2012-2022 -- the 2020 cycle actually fell BELOW "
                   "chance); (2) blending finance into the prediction at every weight 0.05-0.50 "
                   "made accuracy WORSE for both chambers (House 0.8085->0.7319, Senate "
                   "0.9023->0.8563, both monotonically worse with more weight); (3) adding finance "
                   "as a genuine ridge-fit feature hurt House (0.8165->0.7984) but showed a small "
                   "apparent GAIN for Senate (0.9080->0.9253). Investigated that gain specifically: "
                   "it is +3 correct calls out of 174, concentrated entirely in 2 of 6 cycles "
                   "(2018, 2020), with 2014/2022 going the other way -- a small-sample artifact, "
                   "not a robust cycle-independent effect, and it contradicts the more conservative "
                   "blend-weight test on the identical data. Mechanistically: finance correlates "
                   "with actual outcome at only 0.22 vs polls' 0.76, and is not simply redundant "
                   "with polls (mutual correlation only 0.19) -- an independently WEAK, noisy "
                   "signal in both chambers.",
     "decision": "Do not add raw receipts disparity as a linear model input, for either chamber. Directional agreement "
                 "on hindsight misses is not sufficient evidence, and a positive result in only "
                 "one of two rigorous test methodologies -- especially one traceable to two "
                 "specific cycles out of six -- is not evidence either; tests (2) and one that "
                 "survives a mechanism check are what determine whether a feature earns its "
                 "place, and raw receipts disparity fails on both counts in both chambers. "
                 "The narrower decision remains in force under the provisional multi-signal "
                 "overlay in P-006.",
     "source": "User challenge to use 'insane amounts of data' to raise toss-up accuracy; tested "
               "for real once the network policy allowed it, rather than assumed blocked"},
    {"id": "P-006", "claim": "ACTIVE PROVISIONAL: campaign capacity, candidate-quality "
                             "asymmetry, and source-backed campaign events directly adjust the "
                             "published margin, with ordinary effects capped at three points and "
                             "exceptional events capped at six.",
     "chamber": "both", "metric": "live margin overlay with explicit per-race attribution",
     "mechanism": "Early money can build durable capacity, candidate advantages can separate "
                  "otherwise similar races, and major opponent liabilities can create larger "
                  "overperformance. Recent polls absorb information already visible to voters.",
     "status": "Provisional",
     "validation": "The combined Campaign Fundraising Impact and Campaign Overperformance "
                   "research bounds ordinary execution effects near 1-3 points and reserves "
                   "larger movement for exceptional candidate or opponent events. Tests enforce "
                   "stage weighting, credibility discounts, poll absorption, attribution, caps, "
                   "and added uncertainty. Complete historical as-of candidate and event vintages "
                   "are not yet available, so this is not presented as a fitted causal coefficient.",
     "decision": "Activate the bounded overlay in model 2026.18 at the user's direction. Keep "
                 "P-005's rejection of raw receipts disparity, publish every component, widen "
                 "uncertainty when the overlay is active, and replace point priors with fitted "
                 "coefficients once complete historical vintages permit it.",
     "source": "Campaign Fundraising Impact + Campaign Overperformance Analysis; explicit user directive"},
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
    {"id": "S-004", "claim": "ROOT CAUSE of bad Senate margins/tipping-point/ratings: the "
                             "Senate had no real training data. Bundle real MEDSL Senate returns.",
     "chamber": "senate", "metric": "election_results (senate) row count and provenance",
     "mechanism": "Harvard Dataverse egress is blocked from the build environment and the "
                  "Senate file is guestbook-gated, so the 'live fetch' silently produced zero "
                  "rows; the Senate model then fell back to synthetic/stale forecasts",
     "status": "Production",
     "validation": "Bundled data/vintage/medsl_us_senate_2004_2024.csv: real statewide returns, "
                   "2004-2020 (Dataverse MEDSL) + 2024 (MEDSL open GitHub repo), 344 seat-cycle "
                   "margins across 10 cycles, spot-checked vs known results (2018 TX D-2.6, 2014 "
                   "NC D-1.6, 2012 MA D+7.6). With real data the Senate forecast is driven by "
                   "actual history + state_lean instead of noise, and the tipping point resolves "
                   "to a genuine battleground (GA).",
     "decision": "Ship the bundled Senate snapshot as the default source (mirrors the House "
                 "bundle); DATA_SOURCES.md corrected (the old 'fetched live, no gating' note "
                 "was false for the deploy environment)",
     "source": "User-reported Senate margins/tipping-point/ratings issues"},
    {"id": "S-005", "claim": "The Senate's few July toss-ups are the honest state of a no-poll "
                             "fundamentals forecast, not an over-polarization bug.",
     "chamber": "senate", "metric": "walk-forward log loss with/without prior_winner",
     "mechanism": "prior_winner adds a flat incumbency signal; it looks like it over-polarizes "
                  "competitive seats (NC/GA, decided by ~1.8pts in 2020, read ~D-9)",
     "status": "Validated",
     "validation": "Tested the fix: dropping or shrinking prior_winner makes competitive races "
                   "look competitive (NC/GA -> ~toss-up, +7 toss-ups) BUT worsens walk-forward "
                   "log loss in BOTH chambers (house 0.262->0.275, senate 0.367->0.400) and "
                   "winner accuracy - so incumbency genuinely predicts and the confidence is "
                   "earned. Making competitive races 'look' like toss-ups by shrinking it would "
                   "be fitting intuition, not data. Real toss-ups will emerge once 2026 Senate "
                   "polls are ingested (polls pull competitive races toward their true closeness).",
     "decision": "Keep prior_winner as-is; do not distort means to manufacture toss-ups. The "
                 "fix for Senate realism was real DATA (S-004) plus polls, not model tuning.",
     "source": "User-reported 'no toss-ups' + investigation"},
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
    {"id": "S-006", "claim": "Senate winner accuracy improves from more training history plus "
                             "selecting the champion on RECENT cycles.",
     "chamber": "senate", "metric": "walk-forward winner accuracy on a fixed 2010-2024 test set",
     "mechanism": "More cycles stabilise the environment/incumbency coefficients; and the spec "
                  "that predicts the modern era best is not the one that predicts the 1980s-90s "
                  "best (heavy ticket-splitting, Southern realignment)",
     "status": "Production",
     "validation": "(a) Extending the bundled Senate file from 2004-2020 to the full 1976-2020 "
                   "MEDSL history (+2024) lifts winner accuracy 0.872->0.893 and log loss "
                   "0.367->0.322 on the IDENTICAL 2010-2024 walk-forward, for both specs. (b) "
                   "select_chamber_champions now scores candidates only on cycles >= 2010 (still "
                   "training each fit on all earlier history); scoring over all of 1982-2024 had "
                   "let the old cycles pick ridge-lighter (0.884 recent winner accuracy) over "
                   "state-effects (0.893). Net Senate winner accuracy: 0.872 -> 0.893.",
     "decision": "Bundle full 1976-2024 Senate returns; add CHAMPION_SCORING_SINCE=2010 to the "
                 "champion selection. Verified this does not change or harm the House champion.",
     "source": "User request to raise winner accuracy"},
    {"id": "H-007", "claim": "The House model is near the achievable floor for its feature set; "
                             "the residual seat-count error is irreducible national-wave "
                             "magnitude, not missing features.",
     "chamber": "house", "metric": "walk-forward log loss / per-cycle seat error across feature "
                                   "and regularization variants",
     "mechanism": "The dominant errors are wave-reversal cycles (2010 predicted +68 D seats too "
                  "many, 2020 +32) where a national swing hits every seat at once",
     "status": "Validated",
     "validation": "Deep exploration, all walk-forward: (a) a second prior (seat's result 4yr "
                   "back) and an explicit wave-detector barely move log loss (0.2707->0.2698) "
                   "and leave the 2010 miss at +67 - a seat's own history cannot forecast the "
                   "NATIONAL swing; (b) elasticity interactions (env x prior, env x lean) do "
                   "not help (<=0.0002) and env x prior HURTS (0.2736); (c) a full l2 sweep is "
                   "flat (0.2609 at l2=1 to 0.2650 at l2=16, winner accuracy 0.913 throughout). "
                   "The small gaps between all variants are the evidence: there is no easy win "
                   "left, and the ~16.7-seat topline MAE (T-001) is dominated by wave magnitude "
                   "that no available ex-ante feature predicts.",
     "decision": "Keep the linear feature set; widen the champion grid to a real l2 ladder + "
                 "state-effects at two shrinkage levels so the choice is empirical (House now "
                 "picks l2=1). Do not add features that don't earn their place out of sample.",
     "source": "User request to go deep on the House model"},
    {"id": "M-003", "claim": "REJECTED (House) / OPEN LEAD (Senate): incumbency-status features "
                             "and gradient boosting. Also a METHODOLOGY correction: a simplified "
                             "test harness manufactured illusory gains.",
     "chamber": "both", "metric": "walk-forward winner accuracy under the PRODUCTION tier routing",
     "mechanism": "Open seats lose the incumbent's personal vote (real effect: mean |actual - "
                  "prior| is 20.9pts open vs 17.2 incumbent-defended); and a nonlinear learner "
                  "can capture interactions ridge cannot",
     "status": "Rejected",
     "validation": "Derived real incumbent-running/open-seat status for 10,349 House and 744 "
                   "Senate seat-cycles from MEDSL candidate names (the sitting member's presence "
                   "on the ballot is a pre-election fact, so it is vintage-safe). In a SIMPLIFIED "
                   "harness (single pooled ridge over core+poll features) both the incumbency "
                   "features and a GradientBoostingRegressor looked like large wins -- incumbency "
                   "improved and was never worse in ANY cycle, and GBM lifted Senate 0.860->0.914. "
                   "Re-tested through the REAL model architecture (separate core/full tier fits, "
                   "full tier trained only on polled rows) BOTH collapsed: incumbency made things "
                   "slightly worse (House polled 0.8183->0.8154, Senate polled 0.9155->0.9014), "
                   "and GBM hurt the House (0.9313->0.9224). ROOT CAUSE: the harness's pooled fit "
                   "was a WEAKER baseline than production (its Senate ridge scored 0.860 where "
                   "production scores 0.914), so both changes were mostly compensating for the "
                   "harness, not beating the real model. The extra features also overfit the "
                   "small polled-only training set the full tier uses.",
     "decision": "Reverted both. Lesson recorded: candidate changes MUST be evaluated against the "
                 "production architecture, not a simplified stand-in -- a weaker baseline "
                 "manufactures gains that vanish on deployment. FOLLOW-UP (resolved): the "
                 "apparent Senate GBM edge was then put through the same per-cycle gate that "
                 "killed the finance and polls-only hypotheses, and FAILED it. Against production "
                 "ridge with no incumbency features, Senate GBM is identical overall (0.9137 vs "
                 "0.9137; improved 2 cycles, worse 3, tied 3) and its polled-race edge "
                 "(0.9155->0.9249) is a coin flip -- better in 2012/2014, worse in 2016/2022. "
                 "House GBM is plainly worse (0.9319->0.9232; worse in 6 of 8 cycles, -18 seats "
                 "in 2010). Part of the earlier Senate edge was interaction with the incumbency "
                 "features, which are themselves rejected. Gradient boosting is therefore NOT "
                 "adopted, and scikit-learn/numpy stay out of the dependency set entirely.",
     "source": "User push to significantly improve accuracy with an open network"},
    {"id": "T-001", "claim": "The median of the simulated seat distribution is the best single "
                             "topline number; the mode and 'average of the top few outcomes' "
                             "are not improvements.",
     "chamber": "house", "metric": "walk-forward MAE of each estimator vs certified seat count",
     "mechanism": "For a near-symmetric seat distribution the median, mean and mode nearly "
                  "coincide; picking the mode or a top-k average just adds noise",
     "status": "Validated",
     "validation": "backtest.topline_estimator_backtest, walk-forward 2008-2024: mean absolute "
                   "error median 16.7, mean 16.7, mode 18.7, mean-of-top-4 17.5. Median (tied "
                   "with mean) is best; the hypothesised mode / top-k averages are slightly "
                   "WORSE. The winning MAE of ~16.7 seats is the model's irreducible seat-count "
                   "error this far out, driven by wave-reversal cycles (2010, 2020) with no "
                   "consistent directional bias to correct - so the 2026 median (~232) carries a "
                   "genuine +/-16-seat error bar, and a topline in the low 220s is well within it.",
     "decision": "SUPERSEDED by T-002: on the current model the smoothed mode beats the median. "
                 "The original conclusion (keep the median) was correct for the model as it stood "
                 "at the time; re-testing after the Senate-data and champion-selection changes "
                 "reversed it, which is why estimator choice is re-run rather than assumed.",
     "source": "This project's walk-forward backtests, in response to a user topline-statistic idea"},
    {"id": "P-007", "claim": "FIXED: win probabilities were derived from MARGIN-SIZE uncertainty, "
                             "which made genuinely safe seats read as competitive and erased "
                             "toss-ups.",
     "chamber": "both", "metric": "Brier / log loss / winner accuracy, walk-forward 2010-2024",
     "mechanism": "The margin intervals are well calibrated (~80% coverage at the 80% level), but "
                  "most of the model's margin error is on the MAGNITUDE of blowouts, which never "
                  "threatens the winner. Feeding that same sigma into a normal CDF therefore "
                  "overstated flip risk for safe seats and understated separation elsewhere",
     "status": "Production",
     "validation": "Symptom: Alabama at R+21 read as a 19% Democratic flip chance ('Lean "
                   "Republican'), every Senate race landed in Lean/Likely, and no race was a "
                   "Toss-up. Fix: blend the normal CDF with a logistic (Platt) calibration fitted "
                   "on each chamber's own training outcomes. The blend weight was swept "
                   "0.25-0.75 walk-forward rather than picked: 0.25 improves Brier, log loss AND "
                   "winner accuracy in BOTH chambers (House 0.0617->0.0591 Brier, 0.2288->0.2199 "
                   "log loss; Senate 0.0658->0.0650, 0.2298->0.2267, winner accuracy "
                   "0.9137->0.9173) and is the only weight better in EVERY House cycle (8 better, "
                   "0 worse). Heavier weights score slightly better in aggregate but lose "
                   "individual cycles, and full Platt (weight 1.0) was overconfident, sending a "
                   "R+39 seat to 0.000. Effect on the 2026 Senate: Georgia becomes a genuine "
                   "Toss-up and Alabama moves Lean -> Likely Republican.",
     "decision": "model.CALIBRATION_WEIGHT = 0.25, fitted per chamber in MarginModel.fit and "
                 "persisted through to_json/from_json. Margin point estimates and intervals are "
                 "unchanged -- this only affects the margin -> probability mapping. Regression "
                 "test asserts safe seats sharpen without the favoured side ever flipping.",
     "source": "User report that Senate margins/ratings were way off"},
    {"id": "D-001", "claim": "FIXED BUG: no 2026 race had ANY polling -- every per-race poll was "
                             "being silently discarded, capping every race at data grade C.",
     "chamber": "both", "metric": "coverage.with_polls; per-race quality grade",
     "mechanism": "VoteHub labels congressional polls 'us-senator' / 'us-representative', but the "
                  "adapter matched on 'senate' / 'house' and queried invented poll_type values, "
                  "so only the generic ballot survived -- and that attaches to no seat",
     "status": "Production",
     "validation": "Live coverage read with_polls: 0 across all 470 races despite 533 stored poll "
                   "records. Diagnosed against the live API: of 5,378 polls, 295 are us-senator "
                   "and 59 us-representative, and ALL were dropped. Two further obstacles were "
                   "real: VoteHub candidate polls carry no party labels (only names and "
                   "percentages), and Senate polls carry no state (it is in the subject line). "
                   "Fixed by resolving party from the FEC candidate registry by surname within "
                   "the seat, requiring exactly one Democrat and one Republican -- which also "
                   "excludes primaries (60 of them) rather than misreading a D-vs-D matchup as a "
                   "general election. Verified end to end with a complete index: 211 of 295 "
                   "Senate polls attach across 16 seats (MI 39, ME 31, TX 26, NC 22, NH 16, ...) "
                   "plus House seats via seat_name. Unresolvable polls are skipped, never "
                   "guessed.",
     "decision": "votehub adapter rewritten: single unfiltered fetch, seat attribution for both "
                 "chambers, FEC-backed party resolution with a time budget so a throttled "
                 "registry walk degrades to fewer attached polls rather than a hung pipeline. "
                 "quality_grade also now receives REAL finance coverage instead of a hardcoded "
                 "False. Effect on grades: a seat with 4+ polls reaches A, 2-3 polls B, and 1 "
                 "poll plus finance B. Races that genuinely have no polling stay at C -- that is "
                 "the honest reading, and the fix is more polls, not a looser scale.",
     "source": "User: more races should reach data grades B and A"},
    {"id": "D-002", "claim": "Polls feeding the forecast must be independent, adequately "
                             "sampled and reasonably fresh -- and must attach to the seat "
                             "actually on the ballot.",
     "chamber": "both", "metric": "poll quality gate + 2026 seat resolution",
     "mechanism": "A party-sponsored or year-old poll inflates a race's data grade without "
                  "adding trustworthy signal; and a poll mapped to the wrong seat_key is lost "
                  "entirely",
     "status": "Production",
     "validation": "SECOND SEAT BUG: Ohio's and Florida's 2026 Senate contests are SPECIAL "
                   "elections (senate-OH-special), but polls of '2026 Ohio' were mapped to "
                   "senate-OH -- a race that does not exist -- so 26 real polls were discarded "
                   "even after D-001. Seat resolution now reads the actual 2026 race universe "
                   "and skips genuinely ambiguous states (a regular AND a special seat up at "
                   "once) rather than guessing. QUALITY GATE applied to the 354 congressional "
                   "polls: 122 partisan/party-sponsored, 14 stale (>365 days) and 8 "
                   "undersized (<300) are excluded, leaving 210 independent polls. Campaign "
                   "internals and all-adult samples are excluded on the same basis. Result: 14 "
                   "Senate seats carry polling, 10 of them at grade A -- including Ohio "
                   "(10 polls) and Florida (9), which previously had none.",
     "decision": "votehub gains _quality_reject (internal / partisan / population / sample "
                 "size / staleness) and _senate_seat_for_state; quality_grade now also receives "
                 "the REAL age of the most recent poll, so a race polled only long ago is "
                 "marked down instead of scoring as though the data were current.",
     "source": "User: include Ohio's polling, get more A/B grades, and only high-quality, "
               "unbiased, recent polls"},
    {"id": "T-003", "claim": "FIXED INCONSISTENCY: the headline seat count disagreed with the "
                             "race list, and the simulation had drifted away from the published "
                             "per-race probabilities.",
     "chamber": "both", "metric": "topline MAE + headline-vs-race-list agreement",
     "mechanism": "Summing 435 fractional probabilities is not the same as counting how many "
                  "races are over 50%; and once win probabilities became calibrated (P-006) the "
                  "simulation, which sampled raw margin intervals, no longer reproduced them",
     "status": "Production",
     "validation": "Reported by the user: clicking through the individual House races gave 211D/"
                   "224R while the headline said 226D. Measured live: 214 races favored D but a "
                   "228.6 probability sum. Two fixes, both evidence-led. (1) The simulation now "
                   "derives each race's effective sigma from its PUBLISHED probability, so the "
                   "simulated mean equals the sum of the per-race probabilities by construction "
                   "and can never drift from the ratings again. (2) The headline is now chosen "
                   "per chamber on walk-forward MAE (2010-2024): the House uses the count of "
                   "favored races (MAE 11.75 vs 15.12 for the simulated peak, better in 5 of 8 "
                   "cycles) which ALSO makes the headline reproduce the race list exactly; the "
                   "Senate keeps the simulated peak (MAE 1.62 vs 2.38), where counting favorites "
                   "was never better in any single cycle -- 35 races is too few for the count to "
                   "be stable.",
     "decision": "simulate_control reports favored_democratic_seats, most_likely_democratic_seats "
                 "and a chamber-specific headline_democratic_seats with headline_basis; the "
                 "dashboard leads with the headline and shows the alternatives beneath it. A "
                 "regression test asserts the House headline equals the favored-race count and "
                 "that the simulated mean tracks the published probability sum.",
     "source": "User: 'if you look at the individual races you get 224R/211D even though the "
               "topline is 226D'"},
    {"id": "T-002", "claim": "The headline seat count should be the SMOOTHED MODE (most likely "
                             "outcome), not the median -- but the raw mode is too noisy to use.",
     "chamber": "both", "metric": "walk-forward MAE of each estimator vs certified seat count",
     "mechanism": "The simulated seat distribution is mildly skewed, so its peak and its median "
                  "differ; the peak is the single most probable outcome",
     "status": "Production",
     "validation": "Re-ran backtest.topline_estimator_backtest on the current model, walk-forward "
                   "2010-2024, adding the modal and smoothed-modal estimators plus a "
                   "'seat count where P(control) crosses 50%' variant the user proposed. Result: "
                   "smoothed mode (peak of a +/-3-seat window) is best in BOTH chambers -- House "
                   "MAE 13.6 vs 14.3 for the median, Senate 1.50 vs 1.75, and in the Senate it is "
                   "never worse than the median in any individual cycle (2 better, 6 tied). The "
                   "RAW mode is not usable: it scored 12.75 and 14.12 on two runs of the same "
                   "configuration, moving several seats on simulation noise alone, which is "
                   "exactly why the reported figure is smoothed. The 50%-crossing variant tied "
                   "the median (it IS the median by construction) and mean-of-top-4 was worst.",
     "decision": "simulation.simulate_control now reports most_likely_democratic_seats as the "
                 "smoothed peak (raw argmax kept alongside as modal_democratic_seats_raw for "
                 "transparency), and the dashboard leads with it while still showing the median "
                 "and the 80/95% intervals. A regression test asserts the smoothed value is "
                 "stable across independent simulation seeds.",
     "source": "User question: is the median really the best topline for the House?"},
    {"id": "SIM-001", "claim": "FIXED BUG: the simulation's tipping-point seat was wrong.",
     "chamber": "both", "metric": "pivotal-seat identification in simulate_control",
     "mechanism": "It recorded whichever race came LAST in list order among a simulation's "
                  "Democratic wins - an artifact of iteration order, not the pivotal seat",
     "status": "Production",
     "validation": "Now each simulation ranks all seats by realized margin and takes the one at "
                   "the majority-making rank (accounting for safe not-up seats via "
                   "base_dem_seats). Verified: the House pivot is a seat forecast at ~0 margin "
                   "(WI-03, +0.5), and a synthetic 34-safe-D Senate correctly returns the 17th "
                   "most-Democratic contested seat. Especially visible for the Senate's short "
                   "race list, where the old bug was most wrong.",
     "decision": "Per-simulation pivotal-seat tally in simulate_control", "source": "User bug report"},
]

RESEARCH_CLAIMS.extend([
    {"id": "C-001",
     "claim": "Campaign execution may explain modest overperformance, while large departures from an even structural baseline require stronger evidence.",
     "chamber": "both", "metric": "structural baseline versus final margin",
     "mechanism": "Candidate quality, opponent weakness, money, messaging, and events can move a close race but are endogenous and incompletely observed.",
     "status": "Active provisional overlay",
     "validation": "Per-race structural, polling, and campaign layers are frozen separately; ordinary campaign effects are bounded at three points and exceptional source-backed events at six.",
     "decision": "Apply and expose the bounded decomposition, add uncertainty, and publish decisive-win bands.",
     "source": "Campaign Overperformance Analysis; project backtesting discipline"},
    {"id": "C-002",
     "claim": "Early money must be measured by campaign stage and relative to both comparable candidates and the actual opponent.",
     "chamber": "both", "metric": "FEC reporting vintages and opponent-relative finance context",
     "mechanism": "Early receipts can signal viability, affect candidate exit, and buy capacity, but totals also respond to expected competitiveness.",
     "status": "Active provisional overlay",
     "validation": "Append-only FEC vintages now permit future lagged velocity, cash, burn, and stage tests; simple receipt disparity remains rejected under P-005.",
     "decision": "Use richer finance capacity with stage, credibility, and poll-absorption discounts while retaining P-005's rejection of raw receipts disparity.",
     "source": "Campaign Fundraising Impact; FEC OpenFEC; Case and coauthors; Thomsen"},
    {"id": "C-003",
     "claim": "Candidate quality and campaign shocks must be source-backed and known as of the forecast timestamp.",
     "chamber": "both", "metric": "candidate observation and event-ledger coverage",
     "mechanism": "Withdrawals, replacements, experience, legal events, and institutional support can change a race independently of district fundamentals.",
     "status": "Active provisional overlay",
     "validation": "CSV adapters require observed_at, available_at, source URL, reliability, and explicit model eligibility.",
     "decision": "Score enumerated observations and explicitly eligible events within hard caps; prohibit opaque LLM-generated adjustments.",
     "source": "Campaign Overperformance Analysis; project research mandate"},
])

RESEARCH_CLAIMS.extend([
    {"id": "R-001",
     "claim": "ACCEPTED: published expert race ratings, blended in as a fitted "
              "overlay rather than as a model feature, improve held-out accuracy "
              "on the seats they cover. House (490 held-out rated seats): Brier "
              "0.1642 -> 0.1325, log loss 0.5251 -> 0.4345, winner accuracy 0.7755 "
              "-> 0.8449, margin MAE 9.69 -> 6.00. Senate (66): Brier 0.0871 -> "
              "0.0763, log loss 0.3594 -> 0.3388, winner accuracy 0.8788 -> 0.8939, "
              "but margin MAE 11.34 -> 11.99 -- the Senate buys better win "
              "probabilities at the cost of slightly worse point margins on a "
              "sample of 66. These figures are recomputed by the overlay's own fit "
              "on every run and served at /api/data-health, never hand-entered.",
     "chamber": "both",
     "metric": "walk-forward Brier / log loss / winner accuracy / margin MAE on "
               "seats with published ratings",
     "mechanism": "Handicappers observe candidate recruitment, retirements, "
                  "district-level private polling, ad reservations and primary "
                  "outcomes months before any of it reaches a public poll. For the "
                  "~90% of 2026 races with no polling at all, that is the only "
                  "current-cycle seat-level information that exists.",
     "status": "Champion component",
     "validation": "Expanding-window walk-forward over 2016-2024, the same protocol "
                   "that selects the chamber champions. The rating->margin slope is "
                   "fitted in within-cycle deviation form on strictly earlier cycles "
                   "(3.8-4.6 pts per rating step for the House, 5.9-7.0 for the "
                   "Senate; stable in every held-out cycle) and the national level "
                   "comes from the model, never from outcomes. Blend weights are "
                   "chosen by held-out log loss per chamber and separately for "
                   "polled and unpolled seats: a polled race keeps more of the "
                   "model, which is what the data asks for (House MAE bottoms near "
                   "w=0.6 on polled seats but keeps falling to w=1 on unpolled "
                   "ones). The weight is capped at 0.75 -- historical pages carry "
                   "FINAL pre-election ratings while the live feed is ~2 months "
                   "out, and archived late-August revisions of the 2020/2022/2024 "
                   "pages show the slope holds (4.18/4.21/3.84 vs 3.70/4.06/3.86) "
                   "but residual spread roughly doubles.",
     "decision": "Apply the overlay to the population its slope was fitted on: "
                 "every Senate seat (those pages list the full map) and every House "
                 "seat at least one rater declines to call safe. Unrated seats, and "
                 "House seats every rater calls safe, keep the model's own "
                 "prediction. Rated-seat sigma is re-estimated from the blended "
                 "walk-forward residuals, because the model's pooled sigma (fitted "
                 "over uncontested blowouts too) over-covered competitive races "
                 "badly -- walk-forward coverage80 ~0.93 against a nominal 0.80.",
     "source": "User: the toplines have not changed; get every competitive race to "
               "data grade A/B on real data and make it move the predictions"},
    {"id": "R-002",
     "claim": "REJECTED: expert consensus added to the ridge as a "
              "(rating_consensus, has_rating) feature pair.",
     "chamber": "both",
     "metric": "walk-forward Brier / winner accuracy on rated seats",
     "mechanism": "``has_rating`` is a SELECTION indicator -- a seat appears on the "
                  "ratings page because someone already judged it competitive -- so "
                  "a single global coefficient turns that selection into a biased "
                  "constant shift whose sign depends on the D/R mix of whichever "
                  "cycles happen to be in training.",
     "status": "Rejected",
     "validation": "Identical walk-forward protocol. House rated seats got WORSE: "
                   "Brier 0.185 -> 0.202, winner accuracy 0.761 -> 0.712, margin MAE "
                   "13.82 -> 14.40. The Senate degraded too (Brier 0.088 -> 0.093). "
                   "Rescaling the feature changed nothing, confirming the problem is "
                   "structural rather than a ridge-penalty artifact.",
     "decision": "Keep the ratings out of FEATURE_NAMES. The same data helps a great "
                 "deal through the R-001 overlay, which never asks the ratings for "
                 "the national level -- only for the spread between seats.",
     "source": "First implementation attempt of R-001, kept because the negative "
               "result is what justifies the overlay's shape"},
    {"id": "R-004",
     "claim": "ACCEPTED: on a seat whose district was redrawn after its most "
              "recent result, the expert consensus should fully replace the "
              "model's margin -- fitted blend weight 1.00 for the redrawn "
              "stratum versus 0.75 elsewhere, held-out log loss 0.3908 vs "
              "0.4017 at 0.75.",
     "chamber": "house",
     "metric": "walk-forward log loss on rated seats whose district prior is stale",
     "mechanism": "A redrawn seat's prior margin describes boundaries that no "
                  "longer exist, so the model's single strongest feature is "
                  "known-wrong for exactly that seat, while the handicappers "
                  "are looking at the new map.",
     "status": "Champion component",
     "validation": "The 2022 cycle is the natural experiment: post-2020-census "
                   "maps took effect that year, so every 2022 House seat's 2020 "
                   "prior is stale while 2018/2020/2024 priors are not. That "
                   "gives 139 held-out redrawn rated seats, fitted under the same "
                   "walk-forward protocol as every other weight. The redrawn "
                   "stratum also has the TIGHTEST fitted residual sigma of the "
                   "three (5.11 vs 6.97 polled and 10.61 unpolled). Overall "
                   "House rated-seat metrics improved with it: Brier 0.1643 -> "
                   "0.1303, winner accuracy 0.7735 -> 0.8531, MAE 9.69 -> 5.78. "
                   "The population is the corrected one: redistricting history "
                   "is a SEQUENCE of map changes, so a state that redrew both "
                   "for 2022 and again for 2026 is stale at both transitions. "
                   "Reading only the latest map withheld 48 of the 143 rated "
                   "2022 seats -- every one in a state that later remapped -- "
                   "which fitted the stratum on a geographically selected "
                   "subset and then applied it to exactly the excluded states.",
     "decision": "Fit and apply a third `redrawn` stratum with a 1.00 ceiling, "
                 "and lift the unanimously-safe exclusion for those seats -- it "
                 "was stranding the worst cases (CA-40 published D+22.8 while "
                 "all raters said Safe Republican, purely because its consensus "
                 "landed on exactly -4.0). Redrawn seats are fitted and applied "
                 "on the same population, and the fitted consensus range "
                 "(+/-3.89) makes applying at +/-4.0 a negligible extrapolation; "
                 "redrawn seats rated -3.89 historically finished at -19 to -24 "
                 "points.",
     "source": "User: the prior margin is not adjusted for redistricting -- TN-09 "
               "reads lean-Dem on a prior that redistricting has superseded"},
    {"id": "R-003",
     "claim": "A model version that does not change the published competitive-race "
              "numbers must not be publishable.",
     "chamber": "both", "metric": "release gates on the payloads about to be frozen",
     "mechanism": "Model 2026.18 wired a campaign layer into the margin equation "
                  "whose candidate-profile and campaign-event feeds were never "
                  "configured in production (CANDIDATE_PROFILES_URL and "
                  "CAMPAIGN_EVENTS_URL both empty; the run reported "
                  "with_campaign_events: 0), so it shipped as a finance-only "
                  "adjustment and moved the toplines by almost nothing: House "
                  "Democratic control 0.6432 -> 0.6441 and Senate 0.5474 -> 0.5616 "
                  "between the last 2026.17 run and the first 2026.18 run. Nothing "
                  "caught it because nothing was checking.",
     "status": "Enforced",
     "validation": "app.gates runs before any snapshot is inserted: every "
                   "competitive race must carry data grade A or B; a new model "
                   "version must move at least 75% of comparable competitive races; "
                   "and the ratings feed must have delivered current-cycle coverage. "
                   "A failure refuses the publish and leaves the previous forecast "
                   "standing.",
     "decision": "Gate the pipeline, and report the gate results in the run summary "
                 "and at /api/data-health.",
     "source": "User: the topline numbers still haven't changed at all"},
])

RESEARCH_CLAIMS.extend([
    {"id": "T-004",
     "claim": "A forecast with NO polls -- built only from certified results, presidential "
              "partisanship (Cook PVI), incumbency and the national midterm pattern, and "
              "judged by its own track record on past elections -- calls individual races "
              "about as well as the polls + expert-ratings model it replaces.",
     "chamber": "both",
     "metric": "winner accuracy (all / toss-ups decided by <10), log loss, seat-total error, "
               "walk-forward replay 2012-2024",
     "mechanism": "Five poll-free systems (seat history; presidential lean; presidential lean "
                  "+ incumbency; last result + state lean; everything in one regression) are "
                  "each run walk-forward, so every one has a genuine out-of-sample record on "
                  "every past race. Tested two ways of using that record. (1) WHICH SYSTEM: "
                  "choosing a different system per seat from its last three elections never "
                  "beat one system chosen on the chamber's hundreds of past close races "
                  "(House 94.4% vs 94.6% of races, toss-ups 73.4% vs 74.0%; Senate 88.6% vs "
                  "91.6%, toss-ups 68.0% vs 72.0%; per-state choice was worse still) -- three "
                  "elections cannot tell skill from luck. The chamber-wide choice is stable: "
                  "it picked the same system in every cycle from 2014 on. (2) HOW SURE: a "
                  "seat's and state's own past misses DO predict their future misses. Each "
                  "race's uncertainty is a shared national term plus a local term shrunk "
                  "from its own and its state's residuals: log loss improved in 11 of 12 "
                  "held-out chamber-cycles vs one uniform uncertainty (House 0.179 -> 0.171, "
                  "Senate 0.268 -> 0.260).",
     "status": "Production",
     "validation": "Whole procedure replayed per held-out cycle using only earlier cycles "
                   "(system choice, per-race fallback, uncertainty, calibration). vs the "
                   "polls + ratings model on the same races, 2014-2024: House 94.8% vs 94.5% "
                   "of races, toss-ups 74.0% vs 74.5%, mean seat-total miss 14.0 vs 11.8 "
                   "(polls mainly help size a wave: 2018 miss -37 vs -26); Senate 91.1% vs "
                   "92.1%, toss-ups 70.7% vs 72.4%, seat-total miss 2.3 vs 2.7. Published "
                   "systems: House = presidential lean + incumbency, Senate = average of all "
                   "five. Slope-only probability calibration is adopted per chamber only "
                   "when it beats the raw odds on mean log loss AND in most held-out cycles "
                   "(both pass: House 4 of 6, Senate 4 of 5); a free intercept was rejected "
                   "because it would tilt every 2026 toss-up toward the Democrats on four "
                   "cycles' average miss. The headline rule (T-003) was re-tested: the "
                   "Senate keeps the simulated total (seat MAE 1.68 vs 2.33, 5 of 6 cycles); "
                   "the House keeps the count of favored races (the probability total's "
                   "lower MAE came from 2018 alone and was worse in 4 of 6 cycles).",
     "decision": "Model 2026.21 publishes the track-record model (app.track_record). Polls, "
                 "poll-derived expert ratings and the never-backtested campaign layer are no "
                 "longer forecast inputs; handicapper consensus is kept as a published "
                 "diagnostic of where the two disagree. Every race page shows the system's "
                 "record in that seat or state and what every system says.",
     "source": "User: I don't trust the polls; I want a data-driven system that works for "
               "previous years -- based on these data points this system correctly predicted "
               "the past elections here -- applied to 2026, no polls"},
    {"id": "D-003",
     "claim": "FIXED DATA BUGS found building the poll-free model: uncontested races and "
              "mid-decade redistricting silently corrupted seat history, and the PVI "
              "systems were trained on rows that had no PVI.",
     "chamber": "both", "metric": "coefficients + held-out log loss",
     "mechanism": "(a) A race with no major-party opponent is stored as a +/-100 margin; it "
                  "was used as a seat's 'last result' and as a training target. (b) Only the "
                  "2022 census and 2026 redraws were recorded, so e.g. PA-13 entered 2018 "
                  "with a D+100 prior from lines a court had replaced with an R+22 district "
                  "(historical redraws now listed in redistricting.HISTORICAL_MIDDECADE_REMAPS). "
                  "(c) Fitting the PVI systems on 1990-2006 rows that have no PVI let "
                  "incumbency stand in for partisanship: its coefficient inflated to 22.6 "
                  "margin points and PA-07, NY-17 and CO-08 read as R+19 to R+21.",
     "status": "Production",
     "validation": "Uncontested rows excluded from targets and priors; same-map priors use "
                   "the full redistricting history; each system trains only on rows carrying "
                   "its own inputs (incumbency -> 11.7 points, within published estimates). "
                   "Held-out log loss 2014-24: House 0.197 -> 0.169, Senate 0.233 -> 0.224. "
                   "Up-weighting ex-ante competitive races was tested and not adopted (better "
                   "in 3 of 6 House cycles, worse in the Senate).",
     "decision": "track_record.trains_on / usable; redistricting.map_changed; "
                 "track_record.UNCONTESTED.",
     "source": "Found while validating T-004"},
    {"id": "S-007",
     "claim": "FIXED BUG: the control simulation counted a 50-50 Senate as Democratic "
              "control.",
     "chamber": "senate", "metric": "Senate control probability",
     "mechanism": "A tie is broken by the Vice President; JD Vance (R) holds that vote "
                  "through January 2029, so 50-50 is Republican control. The default "
                  "tie_break_party='democratic' overstated Democratic chances by the whole "
                  "probability of a tie.",
     "status": "Production",
     "validation": "On the 2026.21 forecast the fix moved Democratic Senate control from "
                   "54% to 43% before calibration: about one simulation in nine was an "
                   "exact 50-50 tie.",
     "decision": "forecast.SENATE_TIE_BREAK_PARTY = 'republican', used by the published "
                 "simulation and the scenario API.",
     "source": "Found while validating T-004"},
])

# Claims whose mechanism the poll-free model (T-004) no longer publishes. Kept
# in the registry -- the evidence stands -- but no longer driving the forecast.
_SUPERSEDED_BY_T004 = {
    "H-002": "polls are not a forecast input",
    "D-001": "polls are not a forecast input",
    "D-002": "polls are not a forecast input",
    "P-006": "the campaign layer was never validated on past elections",
    "C-001": "the campaign layer was never validated on past elections",
    "C-002": "the campaign layer was never validated on past elections",
    "C-003": "the campaign layer was never validated on past elections",
    "R-001": "expert ratings are largely poll-derived; kept as a diagnostic only",
    "R-004": "redrawn seats now use presidential partisanship on their NEW lines",
    "H-006": "redrawn seats now use presidential partisanship on their NEW lines",
}
for _claim in RESEARCH_CLAIMS:
    if _claim["id"] in _SUPERSEDED_BY_T004:
        _claim["status"] = f"Superseded by T-004 (2026.21): {_SUPERSEDED_BY_T004[_claim['id']]}"

RESEARCH_EVIDENCE = [
    {"id": "E-R001-RATINGS", "claim_id": "R-001",
     "citation": "Wikipedia, United States House/Senate election ratings (2016-2026), "
                 "aggregating Cook Political Report, Inside Elections, Sabato's "
                 "Crystal Ball, DDHQ, The Economist, Split Ticket, Silver Bulletin, "
                 "RealClearPolitics, Fox News and others",
     "source_url": "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_election_ratings",
     "data_period": "2016, 2018, 2020, 2022, 2024 (fitting) and 2026 (live)",
     "interpretation": "A published, dated, multi-rater consensus is real "
                       "pre-election information about seats that carry no polling.",
     "expected_mechanism": "Handicappers price candidate quality, retirements, "
                           "recruitment, private polling and ad spending well before "
                           "public polls exist.",
     "proposed_feature": "Per-seat consensus on the shared Safe/Likely/Lean/Tilt/"
                         "Tossup ladder, applied through a fitted overlay.",
     "leakage_risk": "Reading a rating published after the forecast's as-of date. "
                     "Blocked twice: the store filters on rating_date and "
                     "RatingLookup filters again against the row's as_of.",
     "validation_test": "Expanding-window walk-forward 2016-2024, scored on rated "
                        "seats, against the identical model without the overlay.",
     "result": "Accepted: House rated-seat log loss 0.525 -> 0.437, winner accuracy "
               "0.775 -> 0.841, margin MAE 9.69 -> 6.17.",
     "decision": "Champion component (R-001)."},
    {"id": "E-C002-CASE", "claim_id": "C-002",
     "citation": "Conceptualizing and Measuring Early Campaign Fundraising in Congressional Elections",
     "source_url": "https://www.cambridge.org/core/journals/political-science-research-and-methods/article/conceptualizing-and-measuring-early-campaign-fundraising-in-congressional-elections/48B7870A4EC0EC7B3AE060FBC42873C0",
     "data_period": "U.S. congressional elections; see article",
     "interpretation": "Predictive measurement roadmap, not a universal causal multiplier.",
     "expected_mechanism": "Candidate-centered and election-centered early money capture different information.",
     "proposed_feature": "Stage-normalized receipts and opponent-relative ratios.",
     "leakage_risk": "Using final-cycle totals in an early forecast.",
     "validation_test": "As-of FEC-vintage ablation by forecast horizon.",
     "result": "Data infrastructure implemented; production effect not yet validated.",
     "decision": "Context only."},
    {"id": "E-C002-THOMSEN", "claim_id": "C-002",
     "citation": "Early Money and Strategic Candidate Exit",
     "source_url": "https://www.cambridge.org/core/journals/british-journal-of-political-science/article/early-money-and-strategic-candidate-exit/3EFFBCA74202AE908F9978B12F630683",
     "data_period": "U.S. congressional primaries; see article",
     "interpretation": "Early fundraising is associated with viability and experienced-candidate exit.",
     "expected_mechanism": "Early money changes campaign trajectories before voters choose.",
     "proposed_feature": "Early-stage velocity, candidate status, and withdrawal events.",
     "leakage_risk": "Backfilling withdrawal knowledge before its public timestamp.",
     "validation_test": "Prequential horizon test with available_at cutoffs.",
     "result": "Event/vintage storage implemented; effect unvalidated.",
     "decision": "Context only."},
    {"id": "E-C001-DYNAMIC", "claim_id": "C-001",
     "citation": "Electoral Campaigns as Dynamic Contests",
     "source_url": "https://academic.oup.com/jeea/article/22/6/2782/7595784",
     "data_period": "Dynamic theoretical and empirical campaign setting; see article",
     "interpretation": "Campaign resources and popularity evolve through time.",
     "expected_mechanism": "The timing and relative allocation of resources matter, with diminishing returns.",
     "proposed_feature": "Forecast-horizon states and finance velocity.",
     "leakage_risk": "Using later resource allocation at earlier horizons.",
     "validation_test": "Fixed-population walk-forward horizon comparison.",
     "result": "Time-indexed storage and adaptive refresh implemented.",
     "decision": "No production margin coefficient yet."},
    {"id": "E-C002-FEC", "claim_id": "C-002",
     "citation": "OpenFEC API and electronic filing documentation",
     "source_url": "https://api.open.fec.gov/developers/",
     "data_period": "Current and historical federal filings",
     "interpretation": "Primary administrative data source.",
     "expected_mechanism": "Coverage and retrieval timestamps enable vintage-safe finance features.",
     "proposed_feature": "Immutable content-addressed FEC snapshots.",
     "leakage_risk": "Treating amended or final totals as known earlier.",
     "validation_test": "Snapshot cutoff and amendment tests.",
     "result": "Implemented and tested.",
     "decision": "Production data infrastructure."},
    {"id": "E-P005-INTERNAL", "claim_id": "P-005",
     "citation": "Election Predictor vintage-safe finance ablation",
     "source_url": "https://github.com/aibarraza22-web/electionpredictor/blob/main/RESEARCH_MANIFEST.md",
     "data_period": "2012–2024, both chambers",
     "interpretation": "Predictive test within this project, not a causal estimate.",
     "expected_mechanism": "Receipts may proxy quality but also react to competitiveness.",
     "proposed_feature": "Simple receipts disparity.",
     "leakage_risk": "Final totals and endogenous response to race strength.",
     "validation_test": "Identical expanding-window blend and feature tests.",
     "result": "Worsened held-out forecasts in both chambers.",
     "decision": "Rejected from champion."},
]


def _poll_age_days(last_poll_date: str | None, as_of: str) -> int | None:
    """Days between the most recent poll and the forecast date, so the data
    grade reflects how FRESH the polling is, not merely how much there is."""
    if not last_poll_date:
        return None
    try:
        latest = date.fromisoformat(str(last_poll_date)[:10])
        current = date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return None
    return (current - latest).days


def build_race_universe() -> list[dict]:
    """Upsert the 2026 race table from ingested incumbency data.

    ``open_seat`` means the sitting member is NOT on the November ballot
    (retiring, lost renomination, redistricted away), from the bundled 2026
    ballot-status snapshot; without a status row it falls back to "no
    sitting member"."""
    from .incumbency import ballot_status
    incumbents = store.all_incumbents(CYCLE)
    statuses = ballot_status(CYCLE)

    def is_open(seat_key: str, incumbent: dict | None) -> bool:
        status = statuses.get(seat_key)
        return (not status["running"]) if status else incumbent is None

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
                "open_seat": is_open(seat_key, incumbent),
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
            "open_seat": is_open(seat_key, incumbent),
            "special": special,
            "election_system": "ranked_choice" if state in RANKED_CHOICE_STATES else "plurality",
            "updated_at": timestamp,
        })
    store.upsert_races(rows)
    return rows


def data_version(fingerprint: str, prefix: str = "live") -> str:
    return f"{prefix}-{fingerprint}"


SYSTEM_BOARD_PREFIX = "challenger-"


def _scored_dicts(rows: list[tuple]) -> list[dict]:
    """(predicted, sigma, actual, cycle, probability) -> backtest.metrics rows."""
    out = []
    for predicted, sigma, actual, cycle, win_probability in rows:
        out.append({
            "cycle": cycle, "probability": win_probability,
            "predicted_margin": predicted, "actual_margin": actual,
            "dem_won": 1 if actual > 0 else 0,
            "low80": predicted - 1.282 * sigma, "high80": predicted + 1.282 * sigma,
            "low95": predicted - 1.960 * sigma, "high95": predicted + 1.960 * sigma,
            "low50": predicted - 0.674 * sigma, "high50": predicted + 0.674 * sigma})
    return out


def store_track_record_backtests(models: dict, model_version: str = MODEL_VERSION) -> list[dict]:
    """Persist the replayed walk-forward backtest of the poll-free model and of
    every candidate system, under the identical protocol, and the comparison
    table the dashboard reads."""
    from uuid import uuid4

    from .backtest import metrics as backtest_metrics
    from .track_record import CANDIDATES, label

    runs, comparison = [], {}
    for chamber, model in models.items():
        store.set_meta(f"national_sigma_{chamber}",
                       str(round(model.backtest["national_error_sigma_pts"], 3)))
        by_candidate: dict[str, list] = defaultdict(list)
        for candidate, cycle, _seat, predicted, sigma, actual, win_p in model.backtest_rows:
            by_candidate[candidate].append((predicted, sigma, actual, cycle, win_p))
        for candidate in ["published"] + CANDIDATES:
            scored = _scored_dicts(by_candidate.get(candidate, []))
            if not scored:
                continue
            summary = backtest_metrics(scored)
            cycles = sorted({row["cycle"] for row in scored})
            version = (model_version if candidate == "published"
                       else f"{SYSTEM_BOARD_PREFIX}{candidate}")
            by_cycle = {str(c): backtest_metrics([r for r in scored if r["cycle"] == c])
                        for c in cycles}
            run = {
                "id": f"bt-{chamber}-{uuid4().hex[:10]}", "run_at": store.now(),
                "model_version": version, "chamber": chamber,
                "cycles": json.dumps(cycles), "n_races": summary["n_races"],
                "brier": summary["brier"], "log_loss": summary["log_loss"],
                "winner_accuracy": summary["winner_accuracy"],
                "margin_mae": summary["margin_mae"], "margin_rmse": summary["margin_rmse"],
                "coverage80": summary["coverage80"], "coverage95": summary["coverage95"],
                "calibration": json.dumps(summary["calibration"]),
                "by_cycle": json.dumps(by_cycle),
                "config": json.dumps({
                    "design": "walk-forward replay of the whole poll-free procedure: "
                              "every system trained only on earlier cycles, the published "
                              "system re-chosen and every race's uncertainty re-estimated "
                              "from earlier cycles only",
                    "system": "published (track-record choice)" if candidate == "published"
                              else label(candidate),
                    "chosen_system_by_cycle": model.backtest["chosen_system_by_cycle"],
                    "track_record_metrics": (model.backtest["summary"] if candidate == "published"
                                             else model.backtest["per_system"].get(candidate)),
                    "subgroups": {"by_cycle_track_record": model.backtest["by_cycle"]}
                                 if candidate == "published" else {},
                    "national_error_sigma_pts": model.backtest["national_error_sigma_pts"],
                    "polls_used": False}),
            }
            store.save_backtest_run(run)
            if candidate == "published":
                runs.append(run)
            comparison.setdefault(chamber, {})[version] = {
                "brier": summary["brier"], "log_loss": summary["log_loss"],
                "winner_accuracy": summary["winner_accuracy"],
                "margin_mae": summary["margin_mae"], "n_races": summary["n_races"]}
    store.set_meta("model_comparison", json.dumps(
        {"run_at": store.now(), "champion": model_version, "chambers": comparison,
         "note": "Poll-free. Identical walk-forward replay for every row; "
                 f"{SYSTEM_BOARD_PREFIX}* rows are the individual systems the "
                 "published forecast chooses between by track record."}))
    return runs


def run_track_record_backtests() -> list[dict]:
    """Fit the poll-free model on stored history and persist its backtests."""
    from .track_record import Inputs, TrackRecordModel
    inputs = Inputs(ResultLookup(store.all_results()))
    models = {ch: TrackRecordModel(ch).fit(inputs, CYCLE) for ch in ("house", "senate")}
    return store_track_record_backtests(models)


def _expert_consensus(as_of: str) -> dict[str, dict]:
    """Published handicapper consensus per seat -- DIAGNOSTIC ONLY. It is no
    longer a forecast input (it is largely built on polls); it is kept so the
    places where the poll-free model and the handicappers disagree are
    published rather than hidden."""
    try:
        lookup = RatingLookup(store.all_race_ratings(as_of=as_of))
    except Exception:  # pragma: no cover - a missing table must not block a forecast
        return {}
    out = {}
    for seat_key in lookup.seats(CYCLE):
        summary = lookup.consensus(CYCLE, seat_key, as_of)
        if summary:
            out[seat_key] = summary
    return out


def build_forecasts(as_of: str | None = None, prefix: str = "live",
                    with_backtests: bool = True, force: bool = False,
                    enforce_gates: bool = True,
                    min_rated_races: int = MIN_RATED_RACES) -> dict:
    """Train the poll-free track-record model on ingested history, freeze
    snapshots, store backtests and control simulations.

    ``min_rated_races`` is accepted for compatibility and ignored: expert
    ratings are no longer a forecast input (research claim T-004)."""
    from .campaign import victory_bands
    from .track_record import (CANDIDATES, SYSTEMS, Inputs, TrackRecordModel,
                               label, probability)

    as_of = as_of or store.now()
    fingerprint = store.data_fingerprint()
    if (prefix == "live" and not force
            and (store.get_meta("last_data_version") or "").startswith("live-")
            and store.get_meta("last_input_fingerprint") == fingerprint
            and store.get_meta("last_model_version") == MODEL_VERSION):
        return {"as_of": store.get_meta("last_forecast_as_of"),
                "data_version": store.get_meta("last_data_version"),
                "skipped": "no input or model change"}
    races = build_race_universe()
    results = ResultLookup(store.all_results())
    inputs = Inputs(results)
    missing = [ch for ch in ("house", "senate") if not inputs.history(ch)]
    if missing:
        raise RuntimeError(f"cannot train: no ingested historical results for {missing}; "
                           "run ingestion first")
    models = {ch: TrackRecordModel(ch).fit(inputs, CYCLE) for ch in ("house", "senate")}

    version = data_version(fingerprint, prefix)
    previous_by_race = {item["race_id"]: item for item in
                        store.latest_forecasts(model_version=MODEL_VERSION)}
    outgoing_version = store.latest_champion_version()
    outgoing_by_race = ({item["race_id"]: item for item in store.latest_forecasts()}
                        if outgoing_version and outgoing_version != MODEL_VERSION
                        else {})
    consensus = _expert_consensus(as_of)

    snapshots, board, feature_meta = [], [], {}
    for race in races:
        chamber = race["chamber"]
        model = models[chamber]
        row = inputs.row(chamber, CYCLE, race["seat_key"], race["state"], race["district"],
                         holder_party=race["incumbent_party"])
        out = model.predict(row)
        mean, sigma, p = out["mean"], out["sigma"], out["probability"]
        system = out["system"]
        every = {}
        for candidate in CANDIDATES:
            c_sigma = model.uncertainty[candidate].local(CYCLE, row.seat_key, row.state)["sigma"]
            c_mean = out["systems"][candidate]
            # Same odds calibration as the published number, so the board's
            # rows are comparable with it (and identical for the published one).
            c_p = probability(c_mean, c_sigma, model.calibration)
            every[candidate] = {"label": label(candidate), "margin": round(c_mean, 2),
                                "dem_probability": round(c_p, 4)}
            board.append({
                "race_id": race["id"], "as_of": as_of,
                "model_version": f"{SYSTEM_BOARD_PREFIX}{candidate}",
                "data_version": version,
                "dem_probability": round(c_p, 4),
                "margin": round(c_mean, 2),
                "low80": round(c_mean - 1.282 * c_sigma, 2),
                "high80": round(c_mean + 1.282 * c_sigma, 2),
                "low95": round(c_mean - 1.960 * c_sigma, 2),
                "high95": round(c_mean + 1.960 * c_sigma, 2),
                "rating": rating(c_p), "quality": "-",
                "components": json.dumps({"_model": candidate})})
        previous = previous_by_race.get(race["id"])
        change = None
        if previous:
            change = {"margin_points": round(mean - float(previous.get("margin") or 0.0), 2),
                      "dem_probability_points": round(
                          100 * (p - float(previous.get("dem_probability") or 0.0)), 2)}
        fallback = system != model.system
        expert = consensus.get(race["seat_key"])
        components = {group: round(value, 3) for group, value in out["contributions"].items()
                      if abs(value) >= 0.0005}
        components["_model"] = f"track-record:{system}"
        components["_analysis"] = {
            "method": "poll-free track record",
            "system": system, "system_label": label(system),
            "system_about": (SYSTEMS[system]["about"] if system in SYSTEMS
                             else "the plain average of all five systems"),
            "why_this_system": (
                f"{label(system)} is used because {label(model.system)} -- the "
                f"system with the best {chamber} close-race record -- needs an "
                "input this race does not have" if fallback else
                f"it has the best record on past {chamber} close races "
                "(out-of-sample, 2010 onward); re-chosen on every run"),
            "chamber_ranking": [{"system": c, **model.scoreboard[c]} for c in model.ranking],
            "inputs": {
                "presidential_lean_pvi": row.pvi, "state_house_lean": (
                    round(row.lean, 2) if row.lean is not None else None),
                "same_map_prior_margin": (round(row.prior, 2) if row.prior is not None else None),
                "same_map_prior_cycle": row.prior_cycle,
                "incumbent_running": (None if row.inc is None else
                                      {1.0: "D", -1.0: "R"}.get(row.inc, "open")),
                "redrawn_for_2026": row.redrawn},
            "every_system": every,
            "track_record_here": out["seat_record"],
            "track_record_here_summary": {
                "elections": len(out["seat_record"]),
                "called_correctly": sum(1 for e in out["seat_record"] if e["called_correctly"])},
            "uncertainty": out["uncertainty"],
            "victory_bands": victory_bands(mean, sigma, p),
            "change_since_previous": change,
            "expert_consensus_for_reference": (
                {"consensus": expert["consensus"], "n_raters": expert["n_raters"],
                 "newest_rating_date": expert["newest_rating_date"],
                 "used": False} if expert else None),
        }
        payload = {
            "race_id": race["id"], "dem_probability": round(p, 4), "margin": round(mean, 2),
            "low80": round(mean - 1.282 * sigma, 2), "high80": round(mean + 1.282 * sigma, 2),
            "low95": round(mean - 1.960 * sigma, 2), "high95": round(mean + 1.960 * sigma, 2),
            "rating": rating(p), "quality": out["grade"],
            "components": json.dumps(components),
            "as_of": as_of, "model_version": MODEL_VERSION, "data_version": version}
        snapshots.append(payload)
        feature_meta[race["id"]] = {
            "chamber": chamber, "pvi": row.pvi is not None,
            "prior": row.prior is not None, "ballot": row.inc is not None,
            "open": row.inc == 0, "redrawn": row.redrawn, "system": system,
            "margin": round(mean, 2), "rating": payload["rating"],
            "consensus": expert["consensus"] if expert else None,
            "consensus_safe": is_unanimously_safe(expert)}

    coverage = {
        "races": len(races),
        "with_pvi": sum(1 for m in feature_meta.values() if m["pvi"]),
        "with_same_map_result": sum(1 for m in feature_meta.values() if m["prior"]),
        "with_ballot_status": sum(1 for m in feature_meta.values() if m["ballot"]),
        "open_seats": sum(1 for m in feature_meta.values() if m["open"]),
        "redrawn_seats": sum(1 for m in feature_meta.values() if m["redrawn"]),
        "with_polls": 0, "polls_used": False,
        "competitive_races": len(gates.competitive_races(snapshots)),
    }
    coverage["competitive_races_grade_a_or_b"] = sum(
        1 for p in gates.competitive_races(snapshots) if p.get("quality") in gates.REQUIRED_GRADES)
    gate_results = []
    if enforce_gates:
        gate_results = [
            gates.check_poll_free_inputs(coverage),
            gates.check_competitive_data_grade(snapshots, required=POLL_FREE_REQUIRED_GRADES),
            gates.check_model_moved(snapshots, outgoing_by_race, MODEL_VERSION, outgoing_version),
        ]
    inserted = store.insert_forecasts(snapshots)
    store.insert_forecasts(board)

    track_record_meta = {
        chamber: {"published_system": model.system, "label": label(model.system),
                  "ranking": [{"system": c, **model.scoreboard[c]} for c in model.ranking],
                  "backtest": {k: model.backtest[k] for k in
                               ("cycles", "chosen_system_by_cycle", "summary", "by_cycle",
                                "national_error_sigma_pts", "calibration", "per_system")}}
        for chamber, model in models.items()}
    store.set_meta("track_record", json.dumps(track_record_meta))
    store.upsert_model_version({
        "id": MODEL_VERSION, "chamber": "both", "status": "champion",
        "created_at": store.now(),
        "description": "Poll-free track-record model: five systems built only from "
                       "certified results, presidential partisanship (Cook PVI), "
                       "incumbency and the national midterm pattern, each run "
                       "walk-forward; each chamber publishes the system with the best "
                       "out-of-sample close-race record, and every race's uncertainty "
                       "combines a national term with its own seat/state track record. "
                       + "; ".join(f"{ch}: {label(m.system)}" for ch, m in models.items()),
        "coefficients": json.dumps({
            ch: {name: dict(zip(system.names, [round(w, 4) for w in system.weights]))
                 for name, system in m.systems.items()} for ch, m in models.items()})})
    store.seed_research_claims(RESEARCH_CLAIMS)
    store.seed_research_evidence(RESEARCH_EVIDENCE)
    if with_backtests:
        backtests = store_track_record_backtests(models)
    else:
        backtests = []
        for chamber, model in models.items():
            store.set_meta(f"national_sigma_{chamber}",
                           str(round(model.backtest["national_error_sigma_pts"], 3)))

    control = {}
    for chamber, base in (("house", 0), ("senate", int(store.get_meta("senate_dem_seats_not_up") or 0))):
        stored = store.latest_forecasts(chamber, model_version=MODEL_VERSION)
        nat_sigma = store.get_meta(f"national_sigma_{chamber}")
        kwargs = {"national_sigma": float(nat_sigma)} if nat_sigma else {}
        # Headline rule (T-003) re-tested on the poll-free model's replay,
        # 2014-2024: the Senate's simulated total still beats counting
        # favorites (seat MAE 1.68 vs 2.33, better in 5 of 6 cycles); in the
        # House the probability total's lower MAE (12.6 vs 14.0) comes from
        # the 2018 wave alone -- it is worse in 4 of 6 cycles -- so the House
        # keeps the count of favored races, which also equals the race list.
        control[chamber] = simulate_control(stored, chamber, base_dem_seats=base,
                                            tie_break_party=SENATE_TIE_BREAK_PARTY, **kwargs)
        store.save_control_snapshot(stored[0]["as_of"], chamber, MODEL_VERSION,
                                    stored[0]["data_version"], control[chamber])

    store.set_meta("last_forecast_as_of", as_of)
    store.set_meta("last_data_version", version)
    store.set_meta("last_input_fingerprint", fingerprint)
    store.set_meta("last_model_version", MODEL_VERSION)
    # Where the poll-free model and the handicappers point at different
    # parties. Published, never silently reconciled: the ratings are not an
    # input any more, so these are genuine disagreements to argue with.
    sign_conflicts = sorted(
        ({"race_id": race_id, "model_margin": meta["margin"], "consensus": meta["consensus"]}
         for race_id, meta in feature_meta.items()
         if meta["consensus"] is not None and abs(meta["margin"]) >= 5.0
         and abs(meta["consensus"]) >= 1.0 and (meta["margin"] > 0) != (meta["consensus"] > 0)),
        key=lambda item: -abs(item["model_margin"]))
    consensus_safe_conflicts = sorted(
        race_id for race_id, meta in feature_meta.items()
        if meta.get("consensus_safe") and meta["rating"] in gates.COMPETITIVE_RATINGS)
    coverage["with_expert_ratings_for_reference"] = sum(
        1 for m in feature_meta.values() if m["consensus"] is not None)
    coverage["model_vs_consensus_sign_conflicts"] = len(sign_conflicts)
    coverage["competitive_but_consensus_safe"] = len(consensus_safe_conflicts)
    store.set_meta("coverage", json.dumps(coverage))
    store.set_meta("expert_rating_overlay", json.dumps(None))
    store.set_meta("competitive_but_consensus_safe", json.dumps(consensus_safe_conflicts))
    store.set_meta("model_vs_consensus_sign_conflicts", json.dumps(sign_conflicts[:25]))
    store.set_meta("release_gates", json.dumps(
        {"run_at": store.now(), "model_version": MODEL_VERSION,
         "enforced": enforce_gates, "results": gate_results}))
    return {"as_of": as_of, "data_version": version, "races": len(races),
            "snapshots_inserted": inserted, "coverage": coverage,
            "track_record": {ch: {"published_system": m.system,
                                  "backtest_summary": m.backtest["summary"]}
                             for ch, m in models.items()},
            "model_vs_consensus_sign_conflicts": sign_conflicts[:25],
            "release_gates": gate_results,
            "control": {k: {"democratic_control_probability": v["democratic_control_probability"],
                            "headline_democratic_seats": v.get("headline_democratic_seats")}
                        for k, v in control.items()},
            "backtests": [r["id"] for r in backtests]}
