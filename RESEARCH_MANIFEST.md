# Research manifest

The database registry records claim ID, claim, source, chamber, metric,
mechanism, lifecycle status, validation result, and acceptance decision.
The companion machine-readable evidence registry at
`/api/research-evidence` records citations, direct links, data periods,
interpretation, leakage risks, validation tests, results, and decisions.
Claims C-001 through C-003 merge campaign-overperformance and early-money
research into measurable layers: structural baseline, polling contribution,
FEC reporting vintages, candidate observations, and a campaign-event ledger.

Claims R-001 through R-003 cover published expert race ratings. R-001 is the
accepted result — a walk-forward-fitted overlay on the rating consensus, the
largest measured accuracy gain in the model's history on the seats it covers.
R-002 is the rejected first attempt at the same data (the consensus as a plain
model feature, which made those seats worse) and is kept because the negative
result is what determined the overlay's shape. R-003 records the release gates
that now refuse to publish a model version whose competitive-race numbers do
not actually move. R-004 extends the overlay with a `redrawn` stratum: where a
seat's district prior is stale, the held-out fit says the expert consensus
should replace the model's margin outright.

The working hypotheses are that excellent execution can sometimes explain a
modest one-to-three-point departure from fundamentals, while a much larger
departure from a genuinely even baseline normally requires identifiable
candidate-quality asymmetry, opponent weakness, an issue/environment shift,
or evidence that the baseline was wrong. These are hypotheses, not fixed
coefficients. No campaign feature enters the champion until repeatable
prequential results improve log loss, Brier score, margin error, calibration,
or uncertainty coverage without leakage. The expert-ratings overlay cleared
that bar and is applied; the same data failed it as a model feature and is
not.

Known open work, recorded rather than quietly skipped: the overlay's slope is
fitted on the competitive population the historical national ratings pages
list, so it is not applied to House seats every rater calls safe. Per-state
articles now supply full-coverage ratings for the current cycle, and once
several cycles of them have accumulated the slope can be refitted across the
whole range and that exclusion revisited. Until then, races where the model
calls a unanimously-safe seat competitive are published as an explicit
disagreement (`competitive_but_consensus_safe`), not reconciled.
