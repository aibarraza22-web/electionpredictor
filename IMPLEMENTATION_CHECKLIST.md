# Implementation checklist

- [x] PostgreSQL-backed storage layer (SQLAlchemy Core) with SQLite dev fallback and provenance-tracked, idempotent ingestion.
- [x] Real primary-source adapters: 538 raw-polls (polls + certified outcomes), congress-legislators (incumbency/specials), MEDSL Dataverse (full district returns), FEC finance, live polls feed, official-results CSV.
- [x] Vintage-safe feature construction and chamber-specific trained model (fundamentals + polling tiers) with residual-based uncertainty.
- [x] Expanding-window backtests with runtime leakage assertions; metrics persisted and served, never hand-entered.
- [x] Real 2026 race universe (435 districts, class-2 + special Senate seats), immutable snapshots, stored correlated control simulations.
- [x] Honest mode reporting (live/demo/unconfigured) driven by data provenance; env-token admin API with audit logs.
- [x] Scheduled ingestion+forecast pipeline, CI (SQLite + PostgreSQL), tests, Docker Compose with PostgreSQL, Vercel build fix (project metadata + committed uv.lock).

## Remaining to reach full parity with the research mandate

- [ ] 2024 certified results via MEDSL 2024 release or official-results CSVs (until then, 2022 is the latest full-coverage cycle).
- [ ] Candidate-status source for retirements/open seats and challenger quality.
- [ ] Live 2026 district/state polling feed (`POLLS_FEED_URL`) once a licensed aggregation is chosen.
- [ ] Finance and generic-ballot model terms — only after vintage-correct backtests show improvement.
- [ ] Ranked-choice transfer simulation for AK/ME and Georgia runoff module.
