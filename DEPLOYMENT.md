# Deployment

## Production topology

* **Serving:** Vercel (FastAPI entrypoint `app/index.py`) or any
  uvicorn/container host (`docker compose up --build` bundles PostgreSQL).
* **Persistence:** managed PostgreSQL via `DATEBASE_URL` (Neon, Vercel
  Postgres, RDS...). `postgres://` shorthand is accepted. Without it the app
  falls back to local SQLite — on Vercel that filesystem is ephemeral and
  `/api/data-health` flags it as non-durable.
* **Pipeline:** `.github/workflows/forecast.yml` wakes hourly, then gates to
  daily when more than 180 days out, every six hours at 61–180 days, every
  three hours at 15–60 days, and every two hours during the final 14 days.
  It ingests, fingerprints inputs, and freezes a new snapshot only when data
  or the model changed. Full backtests run weekly and on manual runs.
  The serving layer only reads; heavy work never happens in a request.

## Vercel setup

1. Import the repo in Vercel with these project settings (Settings →
   Build & Development):
   * **Framework Preset: Other** — the committed `vercel.json` rewrites all
     traffic to the `api/index.py` serverless function, which
     `@vercel/python` builds from the root `requirements.txt`. This path
     does not depend on the beta FastAPI preset. (The repo also carries
     valid `[project]` metadata and `uv.lock`, so the FastAPI preset works
     too if selected.)
   * Build command / output directory: leave empty.
2. Set environment variables: `DATEBASE_URL` (required for durability;
   yes, that's spelled "DATE" not "DATA" — this is intentional, matching
   the app's expected variable name, see app/db.py),
   `ADMIN_TOKEN` (optional; admin API stays disabled without it). Redeploy
   after changing env vars — they apply only to new deployments.
3. In GitHub, add repository secret `DATABASE_URL` (correctly spelled —
   the workflow remaps it to `DATEBASE_URL` for the app; same value as
   above) plus
   optionally `FEC_API_KEY`, and repository variables `POLLS_FEED_URL`,
   `CANDIDATE_PROFILES_URL`, and `CAMPAIGN_EVENTS_URL`; then
   run the "Scheduled forecast pipeline" workflow once manually.
4. Visit `/api/data-health` — it must report `mode: live`,
   `durable_storage: true`, and list the ingested sources.

## Operational rules

* `/api/data-health` is the health endpoint; alert on `mode` != live, stale
  sources, failed latest ingestion attempts, stale `last_forecast_as_of`, or
  empty `backtest_runs`.
* Source credentials live only in a secret manager (GitHub secrets / Vercel
  env), never in the repo.
* Ingestion validates availability timestamps and preserves raw payload
  hashes; forecasts write a content fingerprint. Frequent retraining happens
  through the pipeline, while full historical validation runs separately to
  control cost.
