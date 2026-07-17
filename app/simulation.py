from __future__ import annotations
from collections import Counter
from random import Random
from statistics import mean, median
from .domain import Forecast

def simulate_control(forecasts: list[Forecast], chamber: str, simulations: int = 50000, seed: int = 2026, base_dem_seats: int = 0, tie_break_party: str = "democratic") -> dict:
    """Shared national shock preserves correlated race errors without claiming independence."""
    rng = Random(seed); counts=[]; decisive=Counter()
    threshold = 218 if chamber == "house" else 51
    for _ in range(simulations):
        national = rng.gauss(0, .035); seats = base_dem_seats
        winners=[]
        for f in forecasts:
            probability = f.dem_probability if hasattr(f, 'dem_probability') else f['dem_probability']
            race_id = f.race_id if hasattr(f, 'race_id') else f['race_id']
            p = min(.999, max(.001, probability + national + rng.gauss(0, .025)))
            if rng.random() < p: seats += 1; winners.append(race_id)
        counts.append(seats)
        if winners: decisive[winners[-1]] += 1
    distribution=Counter(counts); sorted_counts=sorted(counts)
    controls=sum(x >= threshold or (chamber == "senate" and x == 50 and tie_break_party == "democratic") for x in counts)/simulations
    def quantile(q): return sorted_counts[round((len(sorted_counts)-1)*q)]
    return {"chamber": chamber, "simulations": simulations, "democratic_control_probability": controls, "republican_control_probability": 1-controls,
      "expected_democratic_seats": mean(counts), "median_democratic_seats": median(counts), "most_likely_democratic_seats": distribution.most_common(1)[0][0],
      "interval_80": [quantile(.1),quantile(.9)], "interval_95": [quantile(.025),quantile(.975)], "distribution": dict(sorted(distribution.items())),
      "tipping_point": decisive.most_common(1)[0][0] if decisive else None, "tie_break_assumption": tie_break_party}
