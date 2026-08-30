from datetime import datetime, timezone

from app import store
from app.campaign import (campaign_adjustment, candidate_context, event_context,
                          finance_context, victory_bands)
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


def test_finance_context_is_stage_relative_and_changes_margin():
    rows = [
        _finance("Dem", "D1", "D", 100_000, 70_000, "2025-12-31", "2026-01-10T00:00:00+00:00"),
        _finance("Dem", "D1", "D", 300_000, 210_000, "2026-03-31", "2026-04-10T00:00:00+00:00"),
        _finance("Rep", "R1", "R", 150_000, 90_000, "2026-03-31", "2026-04-10T00:00:00+00:00"),
    ]
    context = finance_context(rows, "2026-04-15", "2026-11-03")
    assert context["production_margin_adjustment"] > 0.0
    assert context["comparison"]["dem_to_rep_receipts_ratio"] == 2.0
    assert 0 < context["comparison"]["signal_credibility"] <= 1
    assert context["candidates"][0]["latest_period_receipts_per_day"] is not None


def test_campaign_overlay_is_bounded_and_poll_absorbed():
    finance = {"production_margin_adjustment": 1.5}
    candidates = {"production_margin_adjustment": 1.0}
    events = {"pre_poll_adjustment": 1.0, "post_poll_adjustment": 0.0,
              "exceptional": False}
    unpolled = campaign_adjustment(1.0, finance, candidates, events)
    polled = campaign_adjustment(1.0, finance, candidates, events,
                                 poll_count=4, poll_age_days=7)
    assert unpolled["margin_adjustment"] == 3.0
    assert 0 < polled["margin_adjustment"] < unpolled["margin_adjustment"]
    assert unpolled["added_sigma"] > 0


def test_candidate_and_post_poll_exceptional_event_are_scored():
    profiles = [
        {"candidate_id": "D1", "candidate": "Dem", "party": "D",
         "profile_type": "elected_official", "value": 1,
         "observed_at": "2026-01-01", "available_at": "2026-01-01",
         "source_url": "https://example.com/d"},
        {"candidate_id": "R1", "candidate": "Rep", "party": "R",
         "profile_type": "first_time_candidate", "value": 1,
         "observed_at": "2026-01-01", "available_at": "2026-01-01",
         "source_url": "https://example.com/r"},
    ]
    candidates = candidate_context(profiles)
    assert candidates["production_margin_adjustment"] == .75
    rows = [{
        "external_id": "scandal-1", "candidate_id": "R1", "event_type": "indictment",
        "event_date": "2026-08-20", "available_at": "2026-08-20",
        "reliability": "primary", "model_eligible": True,
        "details": '{"severity":"major"}', "source_url": "https://example.com/event",
    }]
    events = event_context(rows, {"R1": "R"}, as_of="2026-08-30",
                           last_poll_date="2026-08-15")
    assert events["exceptional"]
    assert events["post_poll_adjustment"] > 0
    result = campaign_adjustment(0.0, {"production_margin_adjustment": 0.0},
                                 candidates, events, poll_count=4, poll_age_days=15)
    assert result["margin_adjustment"] > candidates["production_margin_adjustment"] * .35
    assert result["ordinary_or_exceptional_cap"] == 6.0


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
