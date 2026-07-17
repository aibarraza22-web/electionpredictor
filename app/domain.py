"""Shared forecasting domain utilities."""
from __future__ import annotations

from math import erf, sqrt

RATINGS = ((.97, "Safe Democratic"), (.85, "Likely Democratic"), (.60, "Lean Democratic"),
           (.40, "Toss-up"), (.15, "Lean Republican"), (.03, "Likely Republican"),
           (0, "Safe Republican"))


def normal_cdf(value: float) -> float:
    return .5 * (1 + erf(value / sqrt(2)))


def rating(probability_dem: float) -> str:
    return next(label for threshold, label in RATINGS if probability_dem >= threshold)


def quality_grade(poll_count: int, poll_age_days: int | None, candidate_known: bool,
                  finance_fresh: bool, boundary_certain: bool = True) -> str:
    score = min(poll_count, 4) + int(candidate_known) + int(finance_fresh) + int(boundary_certain)
    if poll_age_days is not None and poll_age_days > 60:
        score -= 1
    return "A" if score >= 6 else "B" if score >= 4 else "C" if score >= 2 else "D" if score >= 1 else "F"
