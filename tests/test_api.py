from random import Random

import pytest


def _seed_synthetic(cycles=(2016, 2018, 2020, 2022, 2024), n_house=60):
    """Compact synthetic dataset, honestly labelled, exercising the real pipeline."""
    from app import store
    from app.forecast import SENATE_CLASS2, build_forecasts
    from app.ingest.base import house_seat_key, senate_seat_key

    rng = Random(11)
    results, incumbents = [], []
    seats = [house_seat_key("CA", i + 1) for i in range(n_house)] + \
            [senate_seat_key(state) for state in SENATE_CLASS2]
    lean = {seat: rng.gauss(0, 15) for seat in seats}
    for cycle in cycles:
        for seat, base in lean.items():
            chamber = seat.split("-")[0]
            if chamber == "senate" and (cycle - 2026) % 6 != 0:
                continue
            margin = base + rng.gauss(0, 5)
            results.append({
                "cycle": cycle, "chamber": chamber, "state": seat.split("-")[1],
                "district": seat.split("-")[2] if chamber == "house" else None,
                "seat_key": seat, "dem_margin": margin,
                "winner_party": "D" if margin > 0 else "R", "source": "synthetic-demo"})
    for seat, base in lean.items():
        chamber = seat.split("-")[0]
        incumbents.append({
            "cycle": 2026, "chamber": chamber, "state": seat.split("-")[1],
            "district": seat.split("-")[2] if chamber == "house" else None,
            "seat_key": seat, "party": "D" if base > 0 else "R",
            "name": "Demo Incumbent", "source": "synthetic-demo"})
    store.insert_rows("election_results", results)
    store.insert_rows("incumbents", incumbents)
    store.set_meta("senate_dem_seats_not_up", "34")
    store.record_source("synthetic-demo", None, "n/a", store.now(), None, len(results))
    return build_forecasts(prefix="demo")


@pytest.fixture()
def client(temp_db, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    from fastapi.testclient import TestClient
    from app.main import app
    summary = _seed_synthetic()
    assert summary["races"] == 468  # 435 House + 33 class-2 (no specials seeded)
    with TestClient(app) as test_client:
        yield test_client


def test_race_universe_and_forecasts(client):
    house = client.get("/api/races?chamber=house").json()
    senate = client.get("/api/races?chamber=senate").json()
    assert len(house) == 435
    assert len(senate) == 33
    forecasts = client.get("/api/forecast/house").json()
    assert forecasts["mode"] == "demo"
    assert len(forecasts["forecasts"]) == 435
    control = client.get("/api/forecast/control").json()
    assert 0 <= control["house"]["democratic_control_probability"] <= 1
    assert control["senate"]["tie_break_assumption"] == "democratic"


def test_race_detail_history_components(client):
    race = client.get("/api/races/2026-house-CA-01").json()
    assert race["forecast"]["rating"]
    assert client.get("/api/races/2026-house-CA-01/history").json()
    components = client.get("/api/races/2026-house-CA-01/components").json()
    assert "components" in components
    polls = client.get("/api/races/2026-house-CA-01/polls").json()
    assert polls["polls"] == [] and "fabricated" in polls["note"]


def test_backtests_are_real_runs(client):
    payload = client.get("/api/backtests").json()
    assert payload["runs"], "pipeline must persist backtest runs"
    champion = next(r for r in payload["runs"]
                    if not str(r["model_version"]).startswith(("baseline", "challenger")))
    assert champion["brier"] is not None and champion["n_races"] > 0
    assert "subgroups" in champion["config"]
    detail = client.get(f"/api/backtests/{champion['id']}").json()
    assert detail["by_cycle"]
    baselines = [r for r in payload["runs"]
                 if str(r["model_version"]).startswith("baseline")]
    assert baselines, "baseline comparisons must be stored"
    comparison = client.get("/api/models/comparison").json()
    assert "baseline-prior-result" in comparison["chambers"]["house"]


def test_data_health_reports_demo_mode(client):
    health = client.get("/api/data-health").json()
    assert health["mode"] == "demo"
    assert any("demo" in w.lower() for w in health["warnings"])
    # champion snapshots for all 468 races, plus challenger/baseline
    # alternates for the per-race model board
    assert health["counts"]["forecasts"] >= 468


def test_admin_requires_token(client, monkeypatch):
    assert client.post("/api/admin/backtest").status_code == 401
    assert client.post("/api/admin/backtest",
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.post("/api/admin/backtest",
                           headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200 and response.json()["audit_logged"]
    monkeypatch.delenv("ADMIN_TOKEN")
    assert client.post("/api/admin/backtest").status_code == 503


def test_scenario_shifts_control(client):
    neutral = client.post("/api/scenarios", json={"national_environment": 0}).json()
    blue = client.post("/api/scenarios", json={"national_environment": 8}).json()
    assert blue["house"]["democratic_control_probability"] >= \
        neutral["house"]["democratic_control_probability"]
