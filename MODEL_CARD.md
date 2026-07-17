# Model card

**Model version:** 2026.2 — chamber-specific ridge regressions
(fundamentals + polling tiers) trained on ingested primary-source history.

**Use:** research, transparent forecast workflow development, and public
forecast presentation with the provenance caveats below surfaced by
`/api/data-health`. **Not for:** campaign decisions or certainty claims.

**Training data:** certified House/Senate outcomes and polls ingested by the
configured adapters (see DATA_SOURCES.md), 1998 onward; exact cycles and row
counts are stored with each fit in `model_versions.coefficients`.

**Validation:** expanding-window backtests re-run on every pipeline
execution; metrics live in `/api/backtests`, never in documentation.

**Known weaknesses:**

* Unpolled races rely on seat history and national environment only; where
  no seat prior is ingested, intervals widen and quality grades drop.
* Incumbency = current seat holder; announced retirements are not marked
  open without a candidate-status source.
* Campaign finance is displayed but deliberately not a model input until it
  passes vintage-correct backtesting.
* Redistricting breaks seat-history comparability (lookback is restricted to
  post-redistricting cycles for the House).

Update the model version only after completed outcomes or a controlled,
validated methodology change.
