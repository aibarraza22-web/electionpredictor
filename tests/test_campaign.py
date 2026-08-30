from datetime import datetime, timezone

from app import store
from app.campaign import finance_context, victory_bands
from scripts.should_refresh import refresh_decision


def _finance(candidate, candidate_id, party, receipts, cash, end, retrieved, spending=0):
    return {
        "cycle": 2026, "seat_key": "house-AZ-01", "candidate": candidate,
        "candidate_id": candidate_id, "committee_id": None, "party": party,
        "receipts": receipts, "disbursements": spending, "cash_on_hand": cash,
        "individual_contributions": receipts * .7,
        "other_committee_contributions": 0, "candidate_contributions": 0,
        "debts_owed": 0, "coverage_start": "2025-01-01", "coverage_end": end,
        "retrieved_at": retrieved, "payload_hash": f"{candidate_id}-{end}-{receipts}",
        "source": "test",
    }


def test_finance_context_is_stage_relative_but_neutral_to_margin():
    rows = [
        _finance("Dem", "D1", "D", 100_000, 70_000, "2025-12-31", "2026-01-10T00:00:00+00:00"),
        _finance("Dem", "D1", "D", 300_000, 210_000, "2026-03-31", "2026-04-10T00:00:00+00:00"),
        _finance("Rep", "R1", "R", 150_000, 90_000, "2026-03-31", "2026-04-10T00:00:00+00:00"),
    ]
    context = finance_context(rows, "2026-04-15", "2026-11-03")
    assert context["production_margin_adjustment"] == 0.0
    assert context["comparison"]["dem_to_rep_receipts_ratio"] == 2.0
    assert context["candidates"][0]["latest_period_receipts_per_day"] is not None


def test_victory_bands_reconcile_to_calibrated_party_probabilities():
    bands = victory_bands(2.0, 6.0, .61)
    assert abs(bands["dem_narrow_0_to_4"] + bands["dem_by_at_least_4"] - .61) < .001
    assert abs(bands["rep_narrow_0_to_4"] + bands["rep_by_at_least_4"] - .39) < .001
    assert bands["dem_by_at_least_8"] <= bands["dem_by_at_least_4"]


def test_finance_snapshots_are_content_vintages(temp_db):
    first = _finance("Dem", "D1", "D", 100, 50, "2026-03-31",
                     "2026-04-10T00:00:00+00:00")
    assert store.insert_rows("finance_snapshots", [first]) == 1
    assert store.insert_rows("finance_snapshots", [first]) == 0
    amended = {**first, "receipts": 120, "payload_hash": "amended"}
    assert store.insert_rows("finance_snapshots", [amended]) == 1
    assert len(store.finance_history_for_seat("house-AZ-01", 2026)) == 2


def test_adaptive_refresh_cadence():
    def utc(month, day, hour):
        return datetime(2026, month, day, hour, tzinfo=timezone.utc)

    assert refresh_decision(utc(1, 1, 9))["run"]
    assert not refresh_decision(utc(1, 1, 10))["run"]
    assert refresh_decision(utc(6, 1, 9))["cadence"] == "every-6-hours"
    assert refresh_decision(utc(9, 15, 12))["cadence"] == "every-3-hours"
    assert refresh_decision(utc(10, 25, 12))["cadence"] == "every-2-hours"
    assert refresh_decision(utc(1, 1, 10), manual=True)["run"]
