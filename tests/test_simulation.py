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
