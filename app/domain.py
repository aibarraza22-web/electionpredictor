from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from math import erf, sqrt

RATINGS = ((.97, "Safe Democratic"), (.85, "Likely Democratic"), (.60, "Lean Democratic"), (.40, "Toss-up"), (.15, "Lean Republican"), (.03, "Likely Republican"), (0, "Safe Republican"))

def normal_cdf(value: float) -> float:
    return .5 * (1 + erf(value / sqrt(2)))

def rating(probability_dem: float) -> str:
    return next(label for threshold, label in RATINGS if probability_dem >= threshold)

def quality_grade(poll_count: int, poll_age_days: int | None, candidate_known: bool, finance_fresh: bool, boundary_certain: bool = True) -> str:
    score = min(poll_count, 4) + int(candidate_known) + int(finance_fresh) + int(boundary_certain)
    if poll_age_days is not None and poll_age_days > 60: score -= 1
    return "A" if score >= 6 else "B" if score >= 4 else "C" if score >= 2 else "D" if score >= 1 else "F"

@dataclass(frozen=True)
class RaceFeatures:
    race_id: str; chamber: str; state: str; baseline: float; prior_margin: float = 0
    national_environment: float = 0; incumbent_dem: bool = False; open_seat: bool = False
    candidate_edge: float = 0; finance_edge: float = 0; expert_edge: float = 0
    poll_margin: float | None = None; poll_count: int = 0; days_to_election: int = 120
    archetype: str = "general"; election_system: str = "plurality"

@dataclass(frozen=True)
class Forecast:
    race_id: str; dem_probability: float; expected_margin: float; interval80: tuple[float, float]; interval95: tuple[float, float]
    rating: str; data_quality: str; components: dict[str, float]

class EnsembleModel:
    """Transparent chamber-specific regularized ensemble; coefficients are versioned data."""
    house_weights = {"baseline": .62, "national": .72, "prior": .16, "incumbency": 1.2, "open": -.4, "candidate": .55, "finance": .32, "expert": .55}
    senate_weights = {"baseline": .56, "national": .32, "prior": .24, "incumbency": 2.4, "open": -.7, "candidate": 1.05, "finance": .48, "expert": .42}
    def forecast(self, f: RaceFeatures) -> Forecast:
        w = self.house_weights if f.chamber == "house" else self.senate_weights
        c = {"partisan baseline": w["baseline"] * f.baseline, "national environment": w["national"] * f.national_environment,
             "previous result": w["prior"] * f.prior_margin, "incumbency": w["incumbency"] * int(f.incumbent_dem),
             "open seat": w["open"] * int(f.open_seat), "candidate strength": w["candidate"] * f.candidate_edge,
             "fundraising": w["finance"] * f.finance_edge, "expert consensus": w["expert"] * f.expert_edge}
        if f.poll_margin is not None:
            poll_weight = min(.72, .12 + (180 - min(180, f.days_to_election)) / 300 + .08 * min(f.poll_count, 4))
            pre_poll = sum(c.values()); c["polling"] = poll_weight * (f.poll_margin - pre_poll)
        mean = sum(c.values())
        sigma = (7.0 if f.chamber == "house" else 8.5) + (2.5 if f.poll_count == 0 else 0) + (2 if f.open_seat else 0) + (2.5 if f.archetype == "newly_redrawn" else 0)
        p = min(.995, max(.005, normal_cdf(mean / sigma)))
        return Forecast(f.race_id, p, mean, (mean-1.282*sigma, mean+1.282*sigma), (mean-1.96*sigma, mean+1.96*sigma), rating(p), quality_grade(f.poll_count, None, True, True), c)
