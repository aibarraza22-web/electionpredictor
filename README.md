# Congressional Forecast Lab

A runnable, **clearly labelled demo-mode** congressional forecasting platform. It serves all 435 synthetic House records and scheduled 2026 Senate records, an interactive national dashboard, race APIs, model/research registries, immutable forecast snapshots, simulation, and protected audited admin actions. It deliberately does **not** present synthetic inputs as live political data.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000 and http://localhost:8000/docs
pytest -q
```

Or: `docker compose up --build`. Seed/reset demo data with `python scripts/seed_demo.py`.

## Scientific safeguards

* Forecast snapshots are inserted with a unique race/date/model key and never updated.
* Historical rows require an `available_cycle` at or before their forecast cycle; walk-forward predictions freeze before that cycle joins training.
* Demo endpoints return no invented polls, finance, or validation performance. Replaceable ingest adapters must carry source and availability timestamps.

See [Methodology](METHODOLOGY.md), [model card](MODEL_CARD.md), [data sources](DATA_SOURCES.md), and [deployment](DEPLOYMENT.md).
