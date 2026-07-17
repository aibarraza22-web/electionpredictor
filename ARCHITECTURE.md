# Architecture

```
GitHub Actions (daily)                     Serving (Vercel / uvicorn / Docker)
────────────────────────                   ────────────────────────────────────
scripts/ingest.py                          app/main.py   FastAPI + dashboard
  app/ingest/*  ── raw_sources,              reads snapshots, races, polls,
                   election_results,         backtests, provenance
                   polls, incumbents,      app/index.py  Vercel entrypoint
                   finance
scripts/forecast.py
  app/features.py  vintage-safe rows   ──►  PostgreSQL (DATABASE_URL)
  app/model.py     ridge fits               SQLite fallback for local dev
  app/backtest.py  walk-forward runs
  app/forecast.py  race universe,
                   snapshots, control sims
```

* `app/db.py` — SQLAlchemy Core schema + engine (PostgreSQL or SQLite from
  the same code); `app/store.py` — repository functions; raw sources are
  append-only and hashed.
* `app/features.py` — as-of feature construction with source precedence.
* `app/model.py` — pure-Python ridge fits (no numeric dependencies), stored
  as versioned coefficient data.
* `app/backtest.py` — expanding-window validation persisted to
  `backtest_runs`.
* `app/forecast.py` — real 2026 race universe (2020-census apportionment,
  Senate class 2 + ingested specials), immutable snapshots, stored control
  simulations.
* `app/simulation.py` — seeded correlated margin-space simulation.
* The API is typed by FastAPI/OpenAPI; heavy computation happens in the
  pipeline, requests only read (scenarios run a small labelled simulation).
