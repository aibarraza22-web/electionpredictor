"""Congressional Forecast Lab API.

Serves forecast snapshots, race data, backtest results, and data-provenance
endpoints from the database. The reported ``mode`` is derived from what has
actually been ingested and validated — the API never labels output "live"
unless real sources produced it, and never reports model performance that a
stored backtest run did not compute.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import db, store
from .simulation import simulate_control


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    yield


app = FastAPI(
    title="Congressional Forecast Lab", version="2026.2", lifespan=lifespan,
    description="Leakage-aware congressional forecasting API. Data provenance "
                "and validation status are reported by /api/data-health.")


def current_mode() -> str:
    version = store.get_meta("last_data_version") or ""
    if version.startswith("live"):
        return "live"
    if version.startswith("demo"):
        return "demo"
    return "unconfigured"


def health_payload() -> dict:
    mode = current_mode()
    counts = store.counts()
    warnings = []
    if mode == "demo":
        warnings.append("Synthetic demo inputs; do not interpret as live forecasts.")
    if mode == "unconfigured":
        warnings.append("No data ingested yet. Run scripts/ingest.py then "
                        "scripts/forecast.py (or the scheduled pipeline).")
    if not db.is_durable():
        warnings.append("SQLite on serverless storage is ephemeral; set "
                        "DATABASE_URL to a managed PostgreSQL instance.")
    if mode == "live" and not counts["backtest_runs"]:
        warnings.append("No validated backtest run stored; treat forecasts as unvalidated.")
    coverage = store.get_meta("coverage")
    return {
        "mode": mode,
        "database_backend": db.backend(),
        "durable_storage": db.is_durable(),
        "counts": counts,
        "sources": store.sources_summary(),
        "coverage": json.loads(coverage) if coverage else None,
        "last_forecast_as_of": store.get_meta("last_forecast_as_of"),
        "data_version": store.get_meta("last_data_version"),
        "warnings": warnings,
    }


DASHBOARD = """<!doctype html><title>Congressional Forecast Lab</title>
<style>body{font:16px system-ui;max-width:1100px;margin:auto;background:#09111f;color:#e7eefb;padding:2rem}
.banner{padding:1rem;border-radius:8px;margin-bottom:1rem}.live{background:#0f3d2e}.demo{background:#8b3b12}.unconfigured{background:#3d3d0f}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.card{background:#15233a;padding:1rem;border-radius:8px}
a{color:#7fcbff}small{color:#9db2d0}</style>
<h1>Congressional Forecast Lab</h1><div id=banner class=banner></div><div id=cards class=grid></div>
<p><a href=/docs>OpenAPI</a> · <a href=/api/races?chamber=house>House races</a> · <a href=/api/backtests>Backtests</a> ·
<a href=/api/research>Research registry</a> · <a href=/api/models>Models</a> · <a href=/api/data-health>Data health</a></p>
<script>
(async()=>{
 const h=await (await fetch('/api/data-health')).json();
 const banner=document.getElementById('banner'),cards=document.getElementById('cards');
 banner.className='banner '+h.mode;
 if(h.mode==='unconfigured'){
  banner.innerHTML='<b>NOT CONFIGURED.</b> No data has been ingested; no forecasts exist yet. '+
   'Run the ingestion + forecast pipeline (see DEPLOYMENT.md), then reload.';
  cards.innerHTML='<div class=card><h2>Setup</h2>1. Provision PostgreSQL, set DATABASE_URL<br>2. python scripts/ingest.py<br>3. python scripts/forecast.py</div>';
  return;
 }
 banner.innerHTML= h.mode==='live'
  ? '<b>LIVE FORECAST.</b> Built from ingested primary sources as of '+h.last_forecast_as_of+
    ' (data version '+h.data_version+'). '+(h.warnings.length?('<br><small>'+h.warnings.join(' ')+'</small>'):'')
  : '<b>DEMO MODE — not live forecasting data.</b> '+h.warnings.join(' ');
 const c=await (await fetch('/api/forecast/control')).json();
 const pct=x=>(x*100).toFixed(1)+'%';
 cards.innerHTML=
  '<div class=card><h2>House</h2><b>'+pct(c.house.democratic_control_probability)+'</b> Democratic control<br><small>median '+c.house.median_democratic_seats+' seats</small></div>'+
  '<div class=card><h2>Senate</h2><b>'+pct(c.senate.democratic_control_probability)+'</b> Democratic control<br><small>median '+c.senate.median_democratic_seats+' seats</small></div>'+
  '<div class=card><h2>Data</h2>'+h.counts.election_results+' results · '+h.counts.polls+' polls<br><small>'+
    h.sources.map(s=>s.source).join(', ')+'</small></div>';
})();
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD


@app.get("/api/data-health")
def data_health():
    return health_payload()


@app.get("/api/forecast/control")
def forecast_control():
    house = store.latest_control_snapshot("house")
    senate = store.latest_control_snapshot("senate")
    if not house or not senate:
        raise HTTPException(404, "No control simulation stored; run the forecast pipeline.")
    return {"mode": current_mode(), "as_of": house["as_of"],
            "data_version": house["data_version"],
            "house": house["payload"], "senate": senate["payload"]}


@app.get("/api/forecast/{chamber}")
def forecast_chamber(chamber: str):
    if chamber not in ("house", "senate"):
        raise HTTPException(404)
    snapshots = store.latest_forecasts(chamber)
    if not snapshots:
        raise HTTPException(404, "No forecasts stored; run the forecast pipeline.")
    return {"mode": current_mode(),
            "model_version": snapshots[0]["model_version"],
            "data_version": snapshots[0]["data_version"],
            "as_of": snapshots[0]["as_of"],
            "forecasts": snapshots}


@app.get("/api/races")
def list_races(chamber: str | None = None, state: str | None = None):
    return store.list_races(chamber, state)


@app.get("/api/races/{race_id}")
def get_race(race_id: str):
    race = store.get_race(race_id)
    if not race:
        raise HTTPException(404)
    return {**race, "forecast": store.latest_forecast(race_id), "mode": current_mode()}


@app.get("/api/races/{race_id}/history")
def race_history(race_id: str):
    return store.forecast_history(race_id)


@app.get("/api/races/{race_id}/components")
def race_components(race_id: str):
    snapshot = store.latest_forecast(race_id)
    if not snapshot:
        raise HTTPException(404)
    return {"race_id": race_id, "components": json.loads(snapshot["components"]),
            "as_of": snapshot["as_of"]}


@app.get("/api/races/{race_id}/polls")
def race_polls(race_id: str):
    race = store.get_race(race_id)
    if not race:
        raise HTTPException(404)
    rows = store.polls_for_seat(race["seat_key"], race["cycle"])
    return {"race_id": race_id, "polls": rows,
            "note": None if rows else "No ingested polls for this race yet; "
                                      "nothing is fabricated in their place."}


@app.get("/api/races/{race_id}/finance")
def race_finance(race_id: str):
    race = store.get_race(race_id)
    if not race:
        raise HTTPException(404)
    rows = store.finance_for_seat(race["seat_key"], race["cycle"])
    return {"race_id": race_id, "finance": rows,
            "note": None if rows else "No FEC filings ingested for this race yet "
                                      "(configure FEC_API_KEY and run ingestion)."}


@app.get("/api/models")
def models():
    return store.list_model_versions()


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    row = next((m for m in store.list_model_versions() if m["id"] == model_id), None)
    if not row:
        raise HTTPException(404)
    if row.get("coefficients"):
        row["coefficients"] = json.loads(row["coefficients"])
    row["formula_documentation"] = "METHODOLOGY.md"
    return row


@app.get("/api/models/{model_id}/learning-history")
def learning_history(model_id: str):
    return {"model_id": model_id,
            "process": ["freeze held-out forecast", "score election",
                        "append completed cycle", "re-estimate before next cycle"],
            "documentation": "SEQUENTIAL_LEARNING.md"}


@app.get("/api/research")
def research():
    return store.list_research_claims()


@app.get("/api/research/{claim_id}")
def research_claim(claim_id: str):
    claim = store.get_research_claim(claim_id)
    if not claim:
        raise HTTPException(404)
    return claim


@app.get("/api/metrics")
def metrics_catalog():
    return {"metrics": ["Brier score", "log loss", "winner accuracy", "margin MAE",
                        "margin RMSE", "calibration", "80/95 interval coverage"],
            "computed_by": "expanding-window backtest runs; see /api/backtests"}


@app.get("/api/backtests")
def backtests():
    runs = store.list_backtest_runs()
    for run in runs:
        for key in ("cycles", "calibration", "config"):
            if run.get(key):
                run[key] = json.loads(run[key])
        run.pop("by_cycle", None)  # served on the detail endpoint
    return {"framework": "expanding-window prequential", "runs": runs,
            "note": None if runs else "No backtest run stored yet; forecasts "
                                      "must not be presented as validated."}


@app.get("/api/backtests/{run_id}")
def backtest_detail(run_id: str):
    run = store.get_backtest_run(run_id)
    if not run:
        raise HTTPException(404)
    for key in ("cycles", "calibration", "config", "by_cycle"):
        if run.get(key):
            run[key] = json.loads(run[key])
    return run


class Scenario(BaseModel):
    national_environment: float = Field(0, ge=-15, le=15,
                                        description="pct-point shift applied to every race margin")


@app.post("/api/scenarios")
def scenario(s: Scenario):
    out = {"label": "Scenario — not the official forecast",
           "national_environment": s.national_environment, "mode": current_mode()}
    for chamber, base_key in (("house", None), ("senate", "senate_dem_seats_not_up")):
        snapshots = store.latest_forecasts(chamber)
        if not snapshots:
            raise HTTPException(404, "No forecasts stored; run the forecast pipeline.")
        shifted = [{**f, "margin": f["margin"] + s.national_environment} for f in snapshots]
        base = int(store.get_meta(base_key) or 0) if base_key else 0
        out[chamber] = simulate_control(shifted, chamber, simulations=2000, base_dem_seats=base)
    return out


ADMIN_ACTIONS = ("refresh", "ingest", "forecast", "backtest")


@app.post("/api/admin/{action}")
def admin(action: str, reason: str = "operator request",
          authorization: str | None = Header(None)):
    if action not in ADMIN_ACTIONS:
        raise HTTPException(404)
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        raise HTTPException(503, "ADMIN_TOKEN is not configured; admin API disabled.")
    if authorization != f"Bearer {token}":
        raise HTTPException(401, "Admin bearer token required")
    store.audit("admin", action, reason)
    summary: dict = {"accepted": True, "audit_logged": True, "action": action}
    if action in ("refresh", "ingest"):
        from .ingest import ADAPTERS
        summary["ingest"] = {}
        for name in ("fte_polls", "legislators", "fec"):
            try:
                summary["ingest"][name] = ADAPTERS[name]()
            except Exception as exc:  # keep going; report per-source status
                summary["ingest"][name] = {"error": str(exc)}
    if action in ("refresh", "forecast"):
        from .forecast import build_forecasts
        summary["forecast"] = build_forecasts()
    if action == "backtest":
        from .backtest import run_backtests
        from .forecast import MODEL_VERSION
        summary["backtests"] = [r["id"] for r in run_backtests(MODEL_VERSION)]
    return summary
