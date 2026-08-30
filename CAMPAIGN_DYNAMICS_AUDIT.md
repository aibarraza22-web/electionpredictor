# Campaign dynamics upgrade audit

## Baseline retained

The existing system already had the right scientific core: vintage-safe
features, expanding-window evaluation, chamber-specific ridge models,
immutable snapshots, baseline/challenger comparisons, data provenance,
poll-quality gates, and correlated control simulations. Replacing that core
without evidence would make the forecast less defensible.

## Material gaps found

1. `finance` was unique by candidate but ingestion used conflict-ignore, so a
   candidate could remain frozen at the first report ingested.
2. Finance had no historical reporting vintages, making early-money,
   acceleration, cash, burn, and amendment tests impossible.
3. Forecast snapshots were date-granular, so multiple meaningful intraday
   changes could not coexist under one model version.
4. The daily workflow could not become more responsive near Election Day and
   rebuilt even when source contents did not change.
5. Source provenance recorded successes but not skipped or failed attempts.
6. Candidate-quality and campaign-event observations had no auditable schema.
7. The dashboard did not distinguish structural standing, polling movement,
   campaign evidence, or decisive-win probabilities.

## Implemented decisions

* Preserve the existing champion. The project's 2012–2024 test of simple
  receipts disparity worsened both chambers, so finance still applies zero
  production points.
* Add content-addressed, append-only FEC vintages and correct the latest view.
* Compute stage, velocity, burn, cash, donor-type, and opponent-relative
  finance context without presenting it as causal.
* Add source-backed candidate observations and campaign events with strict
  `available_at` cutoffs. No protected traits or opaque LLM margin scores.
* Freeze structural, polling, and campaign layers separately and publish
  calibrated narrow/4-point/8-point win bands.
* Wake the workflow hourly but adapt actual refreshes from daily to every two
  hours near Election Day. Skip publication when the input fingerprint and
  model version are unchanged.
* Track every source attempt and expose failures and staleness in data health.
* Extend validation with nominal 50% coverage, expected calibration error,
  calibration slope/intercept, and competitive-race margin MAE.

## Verification

The complete test suite passes: 48 tests. A fresh synthetic end-to-end run
created forecasts for all 468 scheduled House and class-2 Senate races. A
separate run using the bundled official MEDSL history produced these
expanding-window diagnostics:

| Chamber | Races | Brier | Log loss | Winner accuracy | Margin MAE | 80% coverage | 95% coverage | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| House | 4,783 | 0.0711 | 0.2593 | 91.74% | 18.53 | 84.2% | 94.8% | 0.0333 |
| Senate | 380 | 0.1112 | 0.3711 | 85.53% | 15.85 | 90.3% | 94.7% | 0.0497 |

That isolated run intentionally contained MEDSL results only, with no live
polls, FEC data, incumbency feed, or candidate/event feed. It verifies the
pipeline and leakage-safe metrics but is not a claim about the fully
configured production deployment.

## Features deliberately not promoted

* Simple receipts disparity: rejected by the existing vintage-safe ablation.
* Candidate-quality score: not promoted because complete historical as-of
  observations are not yet available.
* Scandal or endorsement margin adjustments: not promoted because manually
  assigned point values would be subjective and leak hindsight.
* Advertising and name-recognition terms: pending a stable, licensed,
  historical source.
* Gradient boosting or a Bayesian replacement: not introduced without an
  identical cycle-holdout comparison proving a material gain.

## Next evidence threshold

Accumulate multiple FEC reporting vintages and candidate/event observations,
then build horizon-specific finance and candidate-quality challengers. Promote
only if they improve recent-cycle held-out log loss and Brier score without
degrading calibration, competitive-race error, or stability across cycles.
