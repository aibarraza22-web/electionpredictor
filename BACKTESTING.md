# Backtesting

Design: **expanding-window prequential**. For each held-out cycle the model
is re-fit on strictly earlier cycles only; predictions are frozen with an
election-day poll cutoff and scored against certified outcomes. Runtime
assertions verify that no future cycle enters training and no poll after the
as-of date enters features (`app/backtest.py::walk_forward`).

Reported per run and per cycle: Brier score, log loss, winner accuracy,
margin MAE/RMSE, 80/95% interval coverage, and 10-bin calibration.

Run with `python scripts/backtest.py` (also re-run automatically by every
forecast pipeline execution). Results are persisted to `backtest_runs` and
served at `/api/backtests` — **the application never reports performance
numbers that a stored run did not compute**, and this document deliberately
quotes none: query the API of a deployment for its own validated metrics,
which reflect exactly the data that deployment ingested.

Caveats stored with each run's config: cycles whose only results source is
the polled-race dataset over-represent competitive districts, so House
metrics on that subset are harder than the full-universe equivalent.
