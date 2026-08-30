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
    campaign = client.get("/api/races/2026-house-CA-01/campaign").json()
    assert campaign["analysis"]["campaign_adjustment"] == 0.0
    assert "dem_by_at_least_8" in campaign["analysis"]["victory_bands"]
    polls = client.get("/api/races/2026-house-CA-01/polls").json()
    assert polls["polls"] == [] and "fabricated" in polls["note"]


def test_finance_overlay_changes_published_margin_and_probability(client):
    from app import store
    from app.forecast import build_forecasts

    before = client.get("/api/races/2026-house-CA-01").json()["forecast"]
    common = {
        "cycle": 2026, "seat_key": "house-CA-01", "committee_id": None,
        "individual_contributions": 0, "other_committee_contributions": 0,
        "candidate_contributions": 0, "debts_owed": 0,
        "coverage_start": "2025-01-01", "coverage_end": "2026-06-30",
        "retrieved_at": "2026-07-15T00:00:00+00:00", "source": "test",
    }
    store.insert_rows("finance_snapshots", [
        {**common, "candidate_id": "D1", "candidate": "Dem", "party": "D",
         "receipts": 1_000_000, "disbursements": 300_000, "cash_on_hand": 700_000,
         "payload_hash": "D1-strong"},
        {**common, "candidate_id": "R1", "candidate": "Rep", "party": "R",
         "receipts": 100_000, "disbursements": 80_000, "cash_on_hand": 20_000,
         "payload_hash": "R1-weak"},
    ])
    build_forecasts(as_of="2026-08-31T00:00:00+00:00", prefix="demo-finance",
                    with_backtests=False, force=True)
    after = client.get("/api/races/2026-house-CA-01").json()["forecast"]
    analysis = client.get("/api/races/2026-house-CA-01/campaign").json()["analysis"]
    assert analysis["campaign_adjustment"] > 0
    assert after["margin"] != before["margin"]
    assert after["dem_probability"] != before["dem_probability"]
    assert analysis["final_margin"] == after["margin"]


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
    assert health["coverage"]["with_candidate_profiles"] == 0
    assert health["coverage"]["with_campaign_events"] == 0


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


def test_latest_champion_version_resolves_numerically_and_skips_baselines(temp_db):
    """The API resolves the champion version from stored data, so pipeline
    output surfaces without a redeploy. Ordering must be numeric: plain string
    comparison would rank '2026.9' above '2026.11'."""
    from app import store
    rows = []
    for version, as_of in (("2026.9", "2026-07-01"), ("2026.11", "2026-07-02"),
                           ("baseline-prior-result", "2026-07-03"),
                           ("challenger-state-effects", "2026-07-03")):
        rows.append({
            "race_id": "2026-house-OH-01", "as_of": as_of, "model_version": version,
            "data_version": "test", "dem_probability": 0.5, "margin": 0.0,
            "low80": -1.0, "high80": 1.0, "low95": -2.0, "high95": 2.0,
            "rating": "Toss-up", "quality": "C", "components": "{}"})
    store.insert_forecasts(rows)
    assert store.latest_champion_version() == "2026.11"
