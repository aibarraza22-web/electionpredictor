# Model card

**Model version:** 2026.20 — chamber-specific ridge regressions
(fundamentals + polling tiers) trained on ingested primary-source history,
blended with a walk-forward-fitted overlay on published expert race ratings,
then a bounded campaign-development adjustment. Each layer is attributed
separately in every frozen snapshot.

**Use:** research, transparent forecast workflow development, and public
forecast presentation with the provenance caveats below surfaced by
`/api/data-health`. **Not for:** campaign decisions or certainty claims.

**Training data:** certified House/Senate outcomes and polls ingested by the
configured adapters (see DATA_SOURCES.md), 1998 onward; exact cycles and row
counts are stored with each fit in `model_versions.coefficients`.

**Validation:** expanding-window backtests run weekly, on manual full runs,
and after controlled methodology changes. Frequent data-only refreshes reuse
the last validated model. Metrics live in `/api/backtests`, never in prose.

**Release gates:** a run refuses to publish unless every competitive race
carries data grade A or B, a new model version moves at least 75% of
comparable competitive races, and the expert-ratings feed delivered
current-cycle coverage. Results are stored and served at
`/api/data-health`.

**Known weaknesses:**

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
