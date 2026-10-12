# Model card

**Model version:** 2026.23 — track-record model (`app/track_record.py`):
five ridge systems over certified results, Cook PVI on current lines,
incumbency and the national midterm pattern; each chamber publishes the system
with the best out-of-sample close-race record, per-race uncertainty is sized by
the seat's and state's own past misses, **polls are a layer used only on races
that have them**, and House incumbents carry part of their personal vote
forward. No expert ratings, no campaign layer.

**Use:** research, transparent forecast workflow development, and public
forecast presentation with the provenance caveats below surfaced by
`/api/data-health`. **Not for:** campaign decisions or certainty claims.

**Training data:** certified House/Senate outcomes and polls ingested by the
configured adapters (see DATA_SOURCES.md), 1998 onward; exact cycles and row
counts are stored with each fit in `model_versions.coefficients`.

**Validation:** expanding-window backtests run weekly, on manual full runs,
and after controlled methodology changes. Frequent data-only refreshes reuse
the last validated model. Metrics live in `/api/backtests`, never in prose.

**Margin audit (2026.23):** across the 2012–24 replay the margins show no tilt
toward either party (actual minus predicted: House +0.4 in Democratic-favored
seats and −0.8 in Republican-favored; Senate +0.3 / +1.8); the large cycle-level
misses (House 2012 +5.2, 2020 −4.3) are the national environment. Margins in
lopsided House districts were biased and are fixed by a lean curve term (error
7.32 → 6.78 points). Conflicting evidence now widens uncertainty (House log
loss 0.1787 → 0.1669, Senate 0.2022 → 0.1942; Maine 2020 95% → 81%). Polls
overstate Democrats by ~5 points in Republican-leaning states; a correction
for that improved Senate winner accuracy but worsened log loss and was not
adopted.

**Validation (2026.22):** the whole procedure is replayed per held-out cycle
2012–2024 using only earlier cycles. All races: House 93.9% called, toss-ups
(decided by <10 points) 75.1%, average error 7.3 points; Senate 90.9%,
toss-ups 71.4%, error 6.5. On races that had polls (2012–2022), polls take
House winner accuracy 80.1% → 83.9% and error 9.3 → 5.6 points; in the Senate
they cut error 6.8 → 6.3 and log loss 0.282 → 0.205 but do not improve who
wins (91.1% → 90.5%). Compared with the earlier 2026.21 on the same corrected
data: House toss-ups 72.3% → 75.1%, log loss 0.207 → 0.183; Senate log loss
0.241 → 0.202 with winner accuracy 90.5% → 90.9%. Races that were not
Democrat-vs-Republican contests (Alaska 2010/2016/2022, Lieberman vs Lamont,
Louisiana's same-party runoffs) are excluded from learning and grading. Without
polls the size of a national wave is what the model cannot see; the
simulation's national-shock term (±6.8 House / ±6.1 Senate margin points, from
past cycle-level misses) carries that uncertainty. Per-race weights fitted to
each result were NOT used: they would make every backtest look perfect and say
nothing about 2026. A Maine-specific adjustment was tested in its general form
(a senator's own persistent over/under-performance) and has no carry-over
(slope 0.05 across 131 senator pairs), so none was added; Maine's one 2020 miss
is a polling miss.

**Release gates:** a run refuses to publish unless every race has
presidential partisanship on its current lines and ≥95% have a known ballot
status, every competitive race carries data grade A–C (C = partisanship +
ballot status, the most a redrawn seat can have), and a new model version
moves at least 75% of comparable competitive races. Results are stored and
served at `/api/data-health`.

**Known weaknesses:**

* (2026.22) On races without polls the model knows only the *average* midterm
  swing, not this year's. It rates 12 Republican-held swing seats (e.g. PA-07,
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
