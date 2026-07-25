from app.simulation import simulate_control


def _races(margins):
    """One forecast dict per margin, with a symmetric 80% interval."""
    return [{"race_id": f"seat-{i:03d}", "margin": m, "high80": m + 7, "low80": m - 7}
            for i, m in enumerate(margins)]


def test_tipping_point_is_the_majority_making_seat():
    # A full 435-seat House laid out in one-point steps around the 218 line, so
    # the majority-making seat (rank 218) sits at a known, near-zero margin.
    # 435 seats from D+217 down to R+217 in steps of 1; rank 218 (0-based 217)
    # has margin 217 - 217 = 0.
    margins = [217 - i for i in range(435)]
    sim = simulate_control(_races(margins), "house", simulations=4000,
                           national_sigma=1.0)
    pivot_margin = next(m for i, m in enumerate(margins)
                        if f"seat-{i:03d}" == sim["tipping_point"])
    assert abs(pivot_margin) <= 3  # the pivot is the seat on the majority line


def test_tipping_point_independent_of_input_order():
    # The old bug recorded whichever race was LAST in list order among a
    # simulation's Democratic wins, so shuffling the input changed the answer.
    # The pivotal seat must depend only on the margins, not their order.
    margins = [40 - 2 * i for i in range(220)]  # 220 seats, spanning the 218 line
    forward = _races(margins)
    reverse = list(reversed(forward))
    a = simulate_control(forward, "house", simulations=6000, seed=1, national_sigma=3.0)
    b = simulate_control(reverse, "house", simulations=6000, seed=1, national_sigma=3.0)
    assert a["tipping_point"] == b["tipping_point"]


def test_senate_pivot_accounts_for_safe_not_up_seats():
    # 35 contested seats, 34 safe Democratic seats not up, majority = 51.
    # The pivot is the (51 - 34) = 17th most-Democratic contested seat, which
    # here is the one with margin +30 - 16*2 = -2.
    margins = [30 - 2 * i for i in range(35)]
    sim = simulate_control(_races(margins), "senate", simulations=4000,
                           base_dem_seats=34, national_sigma=1.0)
    pivot_margin = next(m for i, m in enumerate(margins)
                        if f"seat-{i:03d}" == sim["tipping_point"])
    assert pivot_margin == -2


def test_control_probability_and_seat_bounds():
    margins = [40 - 2 * i for i in range(220)]
    sim = simulate_control(_races(margins), "house", simulations=5000)
    assert 0.0 <= sim["democratic_control_probability"] <= 1.0
    lo, hi = sim["interval_80"]
    assert lo <= sim["median_democratic_seats"] <= hi


def test_smoothed_mode_is_stable_and_reported():
    """The headline 'most likely' seat count is the smoothed peak, not the raw
    argmax: the raw mode moves several seats between runs on simulation noise
    alone, so it must not be the number shown to users."""
    margins = [40 - 2 * i for i in range(220)]
    a = simulate_control(_races(margins), "house", simulations=6000, seed=1)
    b = simulate_control(_races(margins), "house", simulations=6000, seed=99)
    assert "most_likely_democratic_seats" in a and "modal_democratic_seats_raw" in a
    # smoothed peak is stable across independent simulation seeds
    assert abs(a["most_likely_democratic_seats"] - b["most_likely_democratic_seats"]) <= 2
    lo, hi = a["interval_80"]
    assert lo <= a["most_likely_democratic_seats"] <= hi


def test_headline_matches_race_list_for_house_and_simulation_stays_consistent():
    """Two guarantees users can check by hand:
    1. the House headline equals the number of races the party is favoured in,
       so counting the race list reproduces the topline exactly; and
    2. the simulated mean equals the sum of the published per-race
       probabilities, so the distribution cannot drift away from the ratings.
    """
    margins = [30 - 1.5 * i for i in range(60)]
    races = []
    for i, m in enumerate(margins):
        races.append({"race_id": f"r{i}", "margin": m, "high80": m + 18, "low80": m - 18,
                      "dem_probability": max(0.006, min(0.994, 0.5 + m / 90))})
    house = simulate_control(races, "house", simulations=8000)
    favored = sum(1 for r in races if r["dem_probability"] > 0.5)
    assert house["favored_democratic_seats"] == favored
    assert house["headline_democratic_seats"] == favored  # House: count of favourites
    # simulated mean tracks the sum of published probabilities
    assert abs(house["expected_democratic_seats"] - sum(r["dem_probability"] for r in races)) < 1.5
    # Senate uses the simulated peak instead (too few races for a stable count)
    senate = simulate_control(races, "senate", simulations=8000, base_dem_seats=34)
    assert senate["headline_democratic_seats"] == senate["most_likely_democratic_seats"]
    assert senate["favored_democratic_seats"] == 34 + favored
