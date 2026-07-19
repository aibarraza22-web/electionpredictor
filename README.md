# Congressional Forecast Lab

A production-grade congressional forecasting platform for the 2026 midterms:
real primary-source ingestion, PostgreSQL persistence, a trained
chamber-specific model with vintage-safe features, validated expanding-window
backtests, immutable forecast snapshots, correlated control simulations, and
an API/dashboard whose **mode is derived from actual data provenance** — the
system reports `live` only when real ingested sources produced the forecasts,
`demo` for the clearly-labelled synthetic dataset, and `unconfigured` when
nothing has been ingested.

## Quick start (live data)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# optional: export DATEBASE_URL=postgresql://... (defaults to local SQLite)
python scripts/ingest.py           # real polls, results, incumbency (+MEDSL/FEC where reachable)
python scripts/forecast.py         # train, backtest, freeze snapshots, simulate control
uvicorn app.main:app --reload      # http://localhost:8000 and /docs

pytest -q
```

Or `docker compose up --build` (bundles PostgreSQL 16). A zero-network
synthetic demo — clearly labelled as such end-to-end — is available with
`python scripts/seed_demo.py`.

## Data sources (see DATA_SOURCES.md)

| Adapter | Source | What it provides |
|---|---|---|
| `fte_polls` | FiveThirtyEight raw-polls dataset (CC-BY-4.0) | 1998–2022 House/Senate polls + certified outcomes of polled races |
| `legislators` | @unitedstates congress-legislators (CC0) | Current incumbency; 2026 Senate classes incl. specials |
| `medsl` | MIT Election Data + Science Lab, Harvard Dataverse (CC0) | Full district-level House 1976–2022 / Senate 1976–2020 returns |
| `fec` | Federal Election Commission API | Live 2026 candidate finance totals (needs `FEC_API_KEY`) |
| `polls_feed` | Any CSV in the 538 raw-polls schema | Live 2026 polling (`POLLS_FEED_URL`) |
| `scripts/import_csv.py` | State certified results | Official results not yet in an aggregate release |

Every raw record carries source, URL, license, `retrieved_at`,
`available_at`, and a payload hash; ingestion is idempotent.

## Scientific safeguards

* Forecast snapshots are inserted with a unique race/date/model key and never updated.
* Features are vintage-safe: results enter only for later cycles, polls are cut off at the as-of date, and walk-forward backtests assert both properties at runtime.
* All published performance numbers come from stored backtest runs (`/api/backtests`); nothing is hand-entered.
* Missing inputs are flagged and widen uncertainty; they are never imputed with invented values, and coverage is reported by `/api/data-health`.

See [METHODOLOGY.md](METHODOLOGY.md), [MODEL_CARD.md](MODEL_CARD.md),
[DATA_SOURCES.md](DATA_SOURCES.md), [BACKTESTING.md](BACKTESTING.md), and
[DEPLOYMENT.md](DEPLOYMENT.md).
