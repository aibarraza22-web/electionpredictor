# Architecture

```
GitHub Actions (daily)                     Serving (Vercel / uvicorn / Docker)
────────────────────────                   ────────────────────────────────────
scripts/ingest.py                          app/main.py   FastAPI + dashboard
  app/ingest/*  ── raw_sources,              reads snapshots, races, polls,
                   election_results,         backtests, provenance
                   polls, incumbents,      app/index.py  Vercel entrypoint
                   finance, race_ratings
scripts/forecast.py
  app/features.py  vintage-safe rows   ──►  PostgreSQL (DATEBASE_URL)
  app/model.py     ridge fits
  app/ratings.py   expert consensus +        SQLite fallback for local dev
                   fitted overlay
  app/campaign.py  finance vintages,
                   candidates, events
  app/backtest.py  walk-forward runs
  app/gates.py     refuse-to-publish checks
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
* `app/ratings.py` — expert-rating consensus per seat and the walk-forward-
  fitted `RatingOverlay` that blends it into the published margin. This is
  the only place a rating becomes a number the forecast acts on.
* `app/gates.py` — release gates evaluated against the payloads about to be
  frozen; a failure refuses the publish and leaves the previous forecast
  standing.
* `app/campaign.py` — stage/opponent-relative finance context, auditable
  candidate/event context, structural/poll decomposition, and calibrated
  narrow/4-point/8-point victory bands, applied on top of the ratings
  overlay within hard caps.
* The API is typed by FastAPI/OpenAPI; heavy computation happens in the
  pipeline, requests only read (scenarios run a small labelled simulation).
