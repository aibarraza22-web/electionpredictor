# Model card

**Model version:** 2026.21 — poll-free track-record model
(`app/track_record.py`): five ridge systems over certified results, Cook PVI on
current lines, incumbency and the national midterm pattern; each chamber
publishes the system with the best out-of-sample close-race record, with
per-race uncertainty sized by the seat's and state's own past misses. No polls,
no expert ratings, no campaign layer.

**Use:** research, transparent forecast workflow development, and public
forecast presentation with the provenance caveats below surfaced by
`/api/data-health`. **Not for:** campaign decisions or certainty claims.

**Training data:** certified House/Senate outcomes and polls ingested by the
configured adapters (see DATA_SOURCES.md), 1998 onward; exact cycles and row
counts are stored with each fit in `model_versions.coefficients`.

**Validation:** expanding-window backtests run weekly, on manual full runs,
and after controlled methodology changes. Frequent data-only refreshes reuse
the last validated model. Metrics live in `/api/backtests`, never in prose.

**Validation (2026.21):** the whole procedure is replayed per held-out cycle
2012–2024 using only earlier cycles. Against the previous polls + ratings
model on the same races (2014–2024): House 94.8% vs 94.5% of races called,
toss-ups 74.0% vs 74.5%, but a larger seat-total miss in wave years (mean 14.0
vs 11.8 seats; 2018: −37 vs −26); Senate 91.1% vs 92.1%, toss-ups 70.7% vs
72.4%, seat-total miss 2.3 vs 2.7. Without polls the size of a national wave
is the main thing the model cannot see; the simulation's national-shock term
(±6.8 House / ±6.1 Senate margin points, from past cycle-level misses) carries
that uncertainty.

**Release gates:** a run refuses to publish unless every race has
presidential partisanship on its current lines and ≥95% have a known ballot
status, every competitive race carries data grade A–C (C = partisanship +
ballot status, the most a redrawn seat can have), and a new model version
moves at least 75% of comparable competitive races. Results are stored and
served at `/api/data-health`.

**Known weaknesses:**

* (2026.21) Without polls the model knows only the *average* midterm swing,
  not this year's. It rates 12 Republican-held swing seats (e.g. PA-07,
  NY-17, CO-08, IA-01) Republican where handicappers lean Democratic; these
  disagreements are published at `/api/data-health`
  (`model_vs_consensus_sign_conflicts`) rather than reconciled.
* (2026.21) Incumbency is one average effect. An incumbent running in a
  heavily redrawn district (AL-02, FL-09) likely keeps less of it than the
  historical average credits.

* Only 43 of 470 2026 races carry any polling. Those races now lean on the
  expert-ratings overlay instead of seat history alone, which is a large
  measured improvement (see claim R-001) but is still a secondary signal:
  the overlay's weight is capped at 0.75 and its interval is widened 1.45x
  for the gap between the final-vintage ratings it is fitted on and the
  months-out ratings it is applied to.
* The overlay does not act on **settled** House seats every rater calls safe —
  that is outside the population its slope was fitted on. Where the model
  calls such a seat competitive anyway, the disagreement is published
  (`competitive_but_consensus_safe`) rather than reconciled. Redrawn seats are
  the exception and do get the overlay, because there the stale prior is the
  thing that is wrong.
* Mid-decade redistricting is tracked in `app/redistricting.py` as a hand-
  maintained, sourced list of states with new 2026 maps. **A state missing
  from that list keeps a prior margin describing boundaries that no longer
  exist** — Tennessee and Alabama were both missing until model 2026.20, and
  TN-09 published as a toss-up off a D+48 prior for a district that has since
  been split. The run's model-versus-consensus sign-conflict report exists to
  surface that failure mode; the list still has to be maintained by hand as
  further maps are enacted or struck down.
* Expert ratings are other forecasters' judgements, not primary observation.
  Using them imports their errors, and their correlation with each other
  means the consensus is narrower evidence than the rater count suggests.
* Where no seat prior is ingested, intervals widen and quality grades drop.
* Incumbency = current seat holder; announced retirements are not marked
  open without a candidate-status source.
* FEC totals now retain immutable reporting vintages and expose stage,
  velocity, cash, burn, and opponent-relative context. A prior vintage-safe
  test found that simple receipts disparity worsened both chambers, so the
  model does not use raw receipts as a linear feature. Model 2026.18 instead
  uses a bounded, stage-aware capacity overlay with explicit credibility and
  poll-absorption discounts. It is provisional, fully attributed, and widens
  uncertainty when active.
* Candidate-quality observations and campaign events require a timestamp and
  source URL. Comparable candidate observations and explicitly model-eligible
  events can affect the provisional overlay within hard caps.
* Redistricting breaks seat-history comparability (lookback is restricted to
  post-redistricting cycles for the House).

Update the model version only after completed outcomes or a controlled,
validated methodology change.
