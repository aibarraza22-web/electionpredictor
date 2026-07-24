"""Chamber-control simulation.

Runs in margin space: each race's forecast margin carries its own sigma
(recovered from the stored 80% interval), decomposed into a shared national
shock plus idiosyncratic noise, so race errors stay correlated the way
polling/fundamentals misses actually are.

The national-shock size drives how much per-seat uncertainty stays
*correlated* rather than averaging out across hundreds of seats via the law
of large numbers — get it wrong and a modest, honest per-seat lean can turn
into false aggregate certainty. It must come from actual data
(``app.backtest.national_error_sigma``: the standard deviation of
out-of-sample cycle-level mean prediction error), not a guessed constant.
``FALLBACK_NATIONAL_SIGMA`` exists only for callers with no backtest history
available (e.g. unit tests).
"""
from __future__ import annotations

from collections import Counter
from math import sqrt
from random import Random
from statistics import mean, median

FALLBACK_NATIONAL_SIGMA = 5.5  # pct points; see module docstring
Z80 = 1.282


def simulate_control(forecasts: list[dict], chamber: str, simulations: int = 25000,
                     seed: int = 2026, base_dem_seats: int = 0,
                     tie_break_party: str = "democratic",
                     national_sigma: float = FALLBACK_NATIONAL_SIGMA) -> dict:
    rng = Random(seed)
    threshold = 218 if chamber == "house" else 51
    races = []
    for f in forecasts:
        sigma = max((f["high80"] - f["low80"]) / (2 * Z80), 1.0)
        idio = sqrt(max(sigma ** 2 - national_sigma ** 2, 1.0))
        races.append((f["race_id"], f["margin"], idio))
    counts = []
    decisive: Counter = Counter()
    # The tipping-point seat is the one at the majority-making rank: order every
    # seat from most- to least-Democratic and the seat sitting at the control
    # threshold is the pivot — whoever wins it wins the chamber, given every
    # safer seat breaks their way. The safe not-up Democratic seats
    # (base_dem_seats) occupy the top ranks, so among the *contested* seats,
    # sorted most-Democratic first, the pivot is at this 0-based index. (The old
    # code instead recorded whichever race came last in list order among a
    # simulation's Democratic wins — an artifact of iteration order, not a
    # pivotal seat; it was especially wrong for the Senate's short race list.)
    pivot_index = threshold - base_dem_seats - 1
    for _ in range(simulations):
        national = rng.gauss(0, national_sigma)
        seats = base_dem_seats
        realized = []
        for race_id, margin, idio in races:
            m = margin + national + rng.gauss(0, idio)
            realized.append((m, race_id))
            if m > 0:
                seats += 1
        counts.append(seats)
        if 0 <= pivot_index < len(realized):
            realized.sort(reverse=True)  # by realized margin, most-D first
            decisive[realized[pivot_index][1]] += 1
    distribution = Counter(counts)
    sorted_counts = sorted(counts)
    dem_control = sum(
        seats >= threshold or (chamber == "senate" and seats == 50 and tie_break_party == "democratic")
        for seats in counts) / simulations

    def quantile(q: float) -> int:
        return sorted_counts[round((len(sorted_counts) - 1) * q)]

    return {
        "chamber": chamber, "simulations": simulations,
        "democratic_control_probability": round(dem_control, 4),
        "republican_control_probability": round(1 - dem_control, 4),
        "expected_democratic_seats": round(mean(counts), 2),
        "median_democratic_seats": median(counts),
        "most_likely_democratic_seats": distribution.most_common(1)[0][0],
        "interval_80": [quantile(.1), quantile(.9)],
        "interval_95": [quantile(.025), quantile(.975)],
        "distribution": {str(k): v for k, v in sorted(distribution.items())},
        "tipping_point": decisive.most_common(1)[0][0] if decisive else None,
        "tie_break_assumption": tie_break_party,
        "national_error_sigma_pts": round(national_sigma, 2),
    }
