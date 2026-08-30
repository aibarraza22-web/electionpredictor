# Model card

**Model version:** 2026.17 — chamber-specific ridge regressions
(fundamentals + polling tiers) trained on ingested primary-source history.

**Use:** research, transparent forecast workflow development, and public
forecast presentation with the provenance caveats below surfaced by
`/api/data-health`. **Not for:** campaign decisions or certainty claims.

**Training data:** certified House/Senate outcomes and polls ingested by the
configured adapters (see DATA_SOURCES.md), 1998 onward; exact cycles and row
counts are stored with each fit in `model_versions.coefficients`.

**Validation:** expanding-window backtests run weekly, on manual full runs,
and after controlled methodology changes. Frequent data-only refreshes reuse
the last validated model. Metrics live in `/api/backtests`, never in prose.

**Known weaknesses:**

* Unpolled races rely on seat history and national environment only; where
  no seat prior is ingested, intervals widen and quality grades drop.
* Incumbency = current seat holder; announced retirements are not marked
  open without a candidate-status source.
* FEC totals now retain immutable reporting vintages and expose stage,
  velocity, cash, burn, and opponent-relative context. A prior vintage-safe
  test found that simple receipts disparity worsened both chambers, so the
  production campaign adjustment remains exactly zero until a richer
  challenger wins held-out validation.
* Candidate-quality observations and campaign events require a timestamp and
  source URL. They are auditable context, not subjective scores.
* Redistricting breaks seat-history comparability (lookback is restricted to
  post-redistricting cycles for the House).

Update the model version only after completed outcomes or a controlled,
validated methodology change.
