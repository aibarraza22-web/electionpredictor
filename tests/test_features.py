from app.features import (PollLookup, RedrawAdjust, ResultLookup, StateLean,
                          build_row, environment_signs)


def _result(cycle, seat_key, margin, chamber="senate", source="fivethirtyeight-raw-polls"):
    return {"cycle": cycle, "chamber": chamber, "state": seat_key.split("-")[1],
            "district": None, "seat_key": seat_key, "dem_margin": margin,
            "source": source}


def test_environment_signs():
    assert environment_signs(2026) == (1.0, 1.0)    # R president, midterm
    assert environment_signs(2024) == (-1.0, 0.0)   # D president, presidential year
    assert environment_signs(2022) == (-1.0, -1.0)  # D president, midterm


def test_poll_as_of_cutoff_is_enforced():
    polls = PollLookup([
        {"cycle": 2026, "seat_key": "senate-NC", "poll_date": "2026-06-01",
         "dem_margin": 2.0, "partisan": None},
        {"cycle": 2026, "seat_key": "senate-NC", "poll_date": "2026-10-01",
         "dem_margin": 30.0, "partisan": None},
    ])
    average, count, last = polls.average(2026, "senate-NC", "2026-07-17")
    assert count == 1 and last == "2026-06-01" and abs(average - 2.0) < 1e-9
    average_late, count_late, _ = polls.average(2026, "senate-NC", "2026-10-02")
    assert count_late == 2 and average_late > 2.0


def test_senate_prior_uses_same_seat_six_years_back():
    results = ResultLookup([_result(2020, "senate-NC", -1.8),
                            _result(2022, "senate-NC", -3.2)])
    margin, cycle = results.prior(2026, "senate-NC", "senate")
    assert (margin, cycle) == (-1.8, 2020)


def test_special_prior_falls_back_to_base_seat():
    results = ResultLookup([_result(2022, "senate-OH", -6.1)])
    margin, cycle = results.prior(2026, "senate-OH-special", "senate")
    assert (margin, cycle) == (-6.1, 2022)


def test_source_priority_prefers_official_records():
    results = ResultLookup([
        _result(2020, "senate-NC", -5.0, source="fivethirtyeight-raw-polls"),
        _result(2020, "senate-NC", -1.7, source="medsl-constituency-returns"),
    ])
    assert results.margin(2020, "senate-NC") == -1.7


def test_holder_party_fallback_when_no_prior():
    results = ResultLookup([])
    polls = PollLookup([])
    row = build_row("house-CA-30", 2026, "house", "CA", "30", results, polls,
                    "2026-07-17", holder_party="D")
    assert row.has_prior is False
    assert row.x[3] == 1.0  # prior_winner picks up the seat holder's party
    row_r = build_row("house-CA-30", 2026, "house", "CA", "30", results, polls,
                      "2026-07-17", holder_party="R")
    assert row_r.x[3] == -1.0


def test_redrawn_district_keeps_stale_prior_but_flags_redrawn():
    # A CA district (mid-decade remap for 2026): its 2024 result is on lines
    # that no longer exist, but walk-forward testing against the 2022
    # post-census cycle showed dropping such priors *hurts* accuracy (48.5%
    # vs 90.1%) -- most of a district's population persists through a
    # redraw, so the stale prior stays the point-estimate input. The seat is
    # still flagged `redrawn` so model.py widens its uncertainty.
    results = ResultLookup([
        _result(2024, "house-CA-30", 12.0, chamber="house"),
        _result(2024, "house-CA-31", 8.0, chamber="house"),
    ])
    polls = PollLookup([])
    lean = StateLean(results)
    row = build_row("house-CA-30", 2026, "house", "CA", "30", results, polls,
                    "2026-07-17", holder_party="D", state_lean=lean)
    assert row.has_prior is True             # stale 2024 prior kept
    assert row.detail["redrawn"] is True      # but flagged for wider sigma
    assert row.x[1] == 12.0                   # prior_margin retained as-is
    assert row.x[3] == 1.0                    # prior_winner from the real prior
    assert row.x[5] == 1.0                    # has_state_lean still set
    assert row.x[4] != 0.0                    # statewide lean also available


def test_non_redrawn_state_keeps_prior():
    # Pennsylvania is not in the mid-decade remap list: a 2024 House prior
    # is on the current map and must be retained.
    results = ResultLookup([_result(2024, "house-PA-01", 4.0, chamber="house")])
    polls = PollLookup([])
    row = build_row("house-PA-01", 2026, "house", "PA", "01", results, polls,
                    "2026-07-17", holder_party="D")
    assert row.has_prior is True


def test_redraw_adjust_flips_documented_seat_count_toward_new_party():
    # A GOP redraw documented to cost Democrats 1 seat: the single most-marginal
    # D district is overridden to lean-R; the safe D district and the R district
    # are left alone.
    results = ResultLookup([
        _result(2024, "house-ZZ-01", 3.0, chamber="house"),   # marginal D -> should flip
        _result(2024, "house-ZZ-02", 40.0, chamber="house"),  # safe D -> untouched
        _result(2024, "house-ZZ-03", -20.0, chamber="house"), # R -> untouched
    ])
    adj = RedrawAdjust(results, shifts={"ZZ": -1})
    assert adj.prior_override("house-ZZ-01") == -RedrawAdjust.CRACK_MARGIN
    assert adj.prior_override("house-ZZ-02") is None
    assert adj.prior_override("house-ZZ-03") is None


def test_redraw_adjust_positive_delta_flips_marginal_r_seat_to_dem():
    results = ResultLookup([
        _result(2024, "house-ZZ-01", -2.0, chamber="house"),   # marginal R -> flip to D
        _result(2024, "house-ZZ-02", -30.0, chamber="house"),  # safe R -> untouched
    ])
    adj = RedrawAdjust(results, shifts={"ZZ": 1})
    assert adj.prior_override("house-ZZ-01") == RedrawAdjust.CRACK_MARGIN
    assert adj.prior_override("house-ZZ-02") is None


def test_redraw_adjust_only_applies_to_redrawn_target_seats():
    # The override reaches build_row only when the seat is redrawn (a mid-decade
    # remap state, target cycle >= the new map). A 2024 prior in a non-remap
    # state is never overridden even if RedrawAdjust names it.
    results = ResultLookup([_result(2024, "house-CA-30", 5.0, chamber="house")])
    polls = PollLookup([])
    adj = RedrawAdjust(results, shifts={"CA": -1})
    # CA IS a remap state -> the override applies
    ca = build_row("house-CA-30", 2026, "house", "CA", "30", results, polls,
                   "2026-07-17", holder_party="D", redraw_adjust=adj)
    assert ca.detail["redrawn"] is True
    assert ca.x[1] == -RedrawAdjust.CRACK_MARGIN   # marginal D overridden to lean-R
    # a training-vintage row (target 2024) is not redrawn -> no override
    hist = build_row("house-CA-30", 2024, "house", "CA", "30", results, polls,
                     "2024-11-05", holder_party="R", redraw_adjust=adj)
    assert hist.detail["redrawn"] is False
