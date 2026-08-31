"""Release gates: a forecast that only claims to be new must not publish."""
import pytest

from app import gates
from app.gates import GateFailure


def _payload(race_id, rating="Toss-up", quality="B", margin=0.0, probability=0.5):
    return {"race_id": race_id, "rating": rating, "quality": quality,
            "margin": margin, "dem_probability": probability}


def test_competitive_races_are_the_ones_the_model_calls_close():
    payloads = [_payload("a", "Toss-up"), _payload("b", "Lean Democratic"),
                _payload("c", "Lean Republican"), _payload("d", "Likely Republican"),
                _payload("e", "Safe Democratic")]
    assert {p["race_id"] for p in gates.competitive_races(payloads)} == {"a", "b", "c"}


def test_competitive_data_grade_gate_passes_on_a_or_b():
    result = gates.check_competitive_data_grade(
        [_payload("a", quality="A"), _payload("b", quality="B"),
         _payload("safe", "Safe Democratic", quality="D")])
    assert result["passed"] is True
    assert result["competitive_races"] == 2


def test_competitive_data_grade_gate_names_the_failing_races():
    with pytest.raises(GateFailure) as failure:
        gates.check_competitive_data_grade(
            [_payload("2026-house-CA-03", quality="C"), _payload("ok", quality="A")])
    message = str(failure.value)
    assert "2026-house-CA-03 (C)" in message
    assert "1 of 2" in message


def test_model_moved_gate_fails_a_version_that_changes_nothing():
    payloads = [_payload(f"r{i}", margin=1.0, probability=0.51) for i in range(30)]
    previous = {p["race_id"]: {"margin": 1.0, "dem_probability": 0.51}
                for p in payloads}
    with pytest.raises(GateFailure) as failure:
        gates.check_model_moved(payloads, previous, "2026.19", "2026.18")
    assert "changed only 0 of 30" in str(failure.value)


def test_model_moved_gate_passes_when_the_numbers_actually_move():
    payloads = [_payload(f"r{i}", margin=4.0, probability=0.60) for i in range(30)]
    previous = {p["race_id"]: {"margin": 1.0, "dem_probability": 0.51}
                for p in payloads}
    result = gates.check_model_moved(payloads, previous, "2026.19", "2026.18")
    assert result["passed"] is True
    assert result["moved_fraction"] == 1.0


def test_model_moved_gate_needs_a_real_share_not_one_moved_race():
    # 5 of 30 moved is exactly the 2026.18 failure mode: a change that reaches
    # a handful of races and leaves the toplines alone.
    payloads = [_payload(f"r{i}", margin=4.0 if i < 5 else 1.0,
                         probability=0.60 if i < 5 else 0.51) for i in range(30)]
    previous = {p["race_id"]: {"margin": 1.0, "dem_probability": 0.51}
                for p in payloads}
    with pytest.raises(GateFailure):
        gates.check_model_moved(payloads, previous, "2026.19", "2026.18")


def test_model_moved_gate_reports_when_it_cannot_evaluate():
    payloads = [_payload(f"r{i}") for i in range(30)]
    # No previous version at all (a fresh database).
    assert gates.check_model_moved(payloads, {}, "2026.19", None)["evaluated"] is False
    # Same version: a routine data refresh is allowed to leave races unchanged.
    previous = {p["race_id"]: {"margin": 0.0, "dem_probability": 0.5} for p in payloads}
    assert gates.check_model_moved(
        payloads, previous, "2026.19", "2026.19")["evaluated"] is False
    # Too few comparable races for the fraction to mean anything.
    few = [_payload("a"), _payload("b")]
    assert gates.check_model_moved(
        few, {"a": {"margin": 0.0, "dem_probability": 0.5}},
        "2026.19", "2026.18")["evaluated"] is False


def test_current_cycle_ratings_gate_catches_a_silent_feed_failure():
    assert gates.check_current_cycle_ratings(
        {"with_expert_ratings": 470}, minimum=150)["passed"] is True
    with pytest.raises(GateFailure) as failure:
        gates.check_current_cycle_ratings({"with_expert_ratings": 0}, minimum=150)
    assert "did not deliver" in str(failure.value)
