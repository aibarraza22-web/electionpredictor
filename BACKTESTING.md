# Backtesting

Design: **expanding-window prequential**. For each held-out cycle the model
is re-fit on strictly earlier cycles only; predictions are frozen with an
election-day poll cutoff and scored against certified outcomes. Runtime
assertions verify that no future cycle enters training and no poll after the
as-of date enters features (`app/backtest.py::walk_forward`).

Reported per run and per cycle: Brier score, log loss, winner accuracy,
margin MAE/RMSE, 50/80/95% interval coverage, 10-bin calibration, expected
calibration error, calibration slope/intercept, and competitive-race margin
MAE. Extended metrics are stored inside each run's versioned config so the
existing production schema remains backward compatible.

The **expert-ratings overlay** is fitted and scored by the same protocol
(`app/ratings.py::RatingOverlay.fit`): for each held-out cycle the
rating→margin slope comes from strictly earlier cycles, the level comes from
a model fitted on strictly earlier cycles, and the blend weight is chosen by
held-out log loss per chamber and per polled/unpolled stratum. The full
weight scoreboard and a model-only-vs-with-overlay comparison on the rated
seats are recomputed every run and served at `/api/data-health`
(`expert_rating_overlay.held_out_metrics`). Vintage safety has two
independent guards: `store.all_race_ratings` filters on `rating_date` and
`RatingLookup.consensus` filters again against the row's own as-of date, so
a rating published after the date being predicted from cannot be read.

Every pipeline run also evaluates **baseline models under the identical
protocol** — prior-result-only, incumbency-only, environment-only, uniform
swing, and polls-only — and stores a champion-vs-baseline table at
`/api/models/comparison`. When a baseline beats the champion on some slice,
that result is recorded in the research registry as a reported failure, not
tuned away. Each champion run's config also stores **subgroup metrics**
(polled/unpolled, midterm vs presidential cycles, competitive vs safe,
Dem-held vs Rep-held) and **forecast-horizon metrics** (poll cutoffs 0/30/90
days before the election).

Run with `python scripts/backtest.py`. Full validation runs weekly, on manual
workflow runs, and after methodology changes; frequent data-only refreshes use
`python scripts/forecast.py --skip-backtests`. Results are persisted to `backtest_runs` and
served at `/api/backtests` — **the application never reports performance
numbers that a stored run did not compute**, and this document deliberately
quotes none: query the API of a deployment for its own validated metrics,
which reflect exactly the data that deployment ingested.

Caveats stored with each run's config: cycles whose only results source is
the polled-race dataset over-represent competitive districts, so House
metrics on that subset are harder than the full-universe equivalent.
