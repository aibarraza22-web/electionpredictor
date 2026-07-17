"""Chamber-control simulation.

Runs in margin space: each race's forecast margin carries its own sigma
(recovered from the stored 80% interval), decomposed into a shared national
shock plus idiosyncratic noise, so race errors stay correlated the way
polling/fundamentals misses actually are.
"""
from __future__ import annotations

from collections import Counter
from math import sqrt
from random import Random
from statistics import mean, median

NATIONAL_SIGMA = 3.5  # pct points of shared national margin error
Z80 = 1.282


def simulate_control(forecasts: list[dict], chamber: str, simulations: int = 25000,
                     seed: int = 2026, base_dem_seats: int = 0,
                     tie_break_party: str = "democratic") -> dict:
    rng = Random(seed)
    threshold = 218 if chamber == "house" else 51
    races = []
    for f in forecasts:
        sigma = max((f["high80"] - f["low80"]) / (2 * Z80), 1.0)
        idio = sqrt(max(sigma ** 2 - NATIONAL_SIGMA ** 2, 1.0))
        races.append((f["race_id"], f["margin"], idio))
    counts = []
    decisive: Counter = Counter()
    for _ in range(simulations):
        national = rng.gauss(0, NATIONAL_SIGMA)
        seats = base_dem_seats
        last_winner = None
        for race_id, margin, idio in races:
            if margin + national + rng.gauss(0, idio) > 0:
                seats += 1
                last_winner = race_id
        counts.append(seats)
        if last_winner:
            decisive[last_winner] += 1
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
        "national_error_sigma_pts": NATIONAL_SIGMA,
    }
