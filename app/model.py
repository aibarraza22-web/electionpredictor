"""Trained margin model.

Two chamber-specific ridge regressions over the vintage-safe features in
``app.features``:

* a **full** fit (fundamentals + polling averages) applied to races that
  have polls, and
* a **fundamentals** fit (poll columns excluded) applied to races without
  polls — fitting it on all historical races keeps the seat-history and
  environment weights honest instead of letting the polled-race intercept
  extrapolate to unpolled seats.

Everything is fit purely in Python (the feature dimension is 8, so a dense
solve is trivial and the serverless bundle stays free of numeric
dependencies). Predictive uncertainty comes from each fit's own training
residuals and is converted to win probabilities with a normal CDF.
Coefficients are stored as versioned data, never hand-tuned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from statistics import pstdev

from .domain import normal_cdf, quality_grade, rating
from .features import FEATURE_NAMES, FeatureRow

DEFAULT_L2 = 4.0
MIN_SIGMA = 4.0     # pct points; guards degenerate residual pools
N_POLL_FEATURES = 2  # poll_average, has_polls sit at the end of the vector

FUNDAMENTALS_NAMES = FEATURE_NAMES[:-N_POLL_FEATURES]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular normal equations; check features")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = (a[row][n] - sum(a[row][k] * solution[k] for k in range(row + 1, n))) / a[row][row]
    return solution


def ridge_fit(xs: list[list[float]], ys: list[float], l2: float = DEFAULT_L2) -> list[float]:
    n_features = len(xs[0])
    xtx = [[sum(x[i] * x[j] for x in xs) for j in range(n_features)] for i in range(n_features)]
    for i in range(1, n_features):  # do not penalize the intercept
        xtx[i][i] += l2
    xty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(n_features)]
    return _solve(xtx, xty)


@dataclass(frozen=True)
class Prediction:
    mean: float
    sigma: float
    model: str  # "full" or "fundamentals"

    @property
    def dem_probability(self) -> float:
        return min(0.995, max(0.005, normal_cdf(self.mean / self.sigma)))

    def interval(self, z: float) -> tuple[float, float]:
        return (self.mean - z * self.sigma, self.mean + z * self.sigma)


def _residual_sigma(residuals: list[float]) -> float:
    return max(MIN_SIGMA, pstdev(residuals)) if len(residuals) > 1 else MIN_SIGMA


class MarginModel:
    """Per-chamber full and fundamentals-only ridge fits with residual sigmas."""

    def __init__(self, l2: float = DEFAULT_L2):
        self.l2 = l2
        self.weights: dict[str, dict[str, list[float]]] = {}
        self.sigma: dict[str, dict[str, float]] = {}
        self.training_meta: dict[str, dict] = {}

    def fit(self, rows: list[FeatureRow]) -> "MarginModel":
        by_chamber: dict[str, list[FeatureRow]] = {}
        for row in rows:
            if row.actual_margin is not None:
                by_chamber.setdefault(row.chamber, []).append(row)
        for chamber, chamber_rows in by_chamber.items():
            ys = [r.actual_margin for r in chamber_rows]
            fund_xs = [r.x[:-N_POLL_FEATURES] for r in chamber_rows]
            fund_weights = ridge_fit(fund_xs, ys, self.l2)
            fund_residuals = [
                y - sum(w * v for w, v in zip(fund_weights, x))
                for x, y in zip(fund_xs, ys)]

            polled = [r for r in chamber_rows if r.poll_count]
            if len(polled) >= 3 * len(FEATURE_NAMES):
                full_xs = [r.x for r in polled]
                full_ys = [r.actual_margin for r in polled]
                full_weights = ridge_fit(full_xs, full_ys, self.l2)
                full_residuals = [
                    y - sum(w * v for w, v in zip(full_weights, x))
                    for x, y in zip(full_xs, full_ys)]
            else:  # not enough polled history: everything routes to fundamentals
                full_weights, full_residuals = None, fund_residuals

            self.weights[chamber] = {"fundamentals": fund_weights, "full": full_weights}
            self.sigma[chamber] = {"fundamentals": _residual_sigma(fund_residuals),
                                   "full": _residual_sigma(full_residuals)}
            self.training_meta[chamber] = {
                "n": len(chamber_rows), "n_polled": len(polled),
                "cycles": sorted({r.cycle for r in chamber_rows}), "l2": self.l2,
            }
        return self

    def predict(self, row: FeatureRow) -> Prediction:
        chamber = self.weights[row.chamber]
        use_full = row.poll_count > 0 and chamber["full"] is not None
        if use_full:
            weights, x, kind = chamber["full"], row.x, "full"
        else:
            weights, x, kind = chamber["fundamentals"], row.x[:-N_POLL_FEATURES], "fundamentals"
        mean = sum(w * v for w, v in zip(weights, x))
        sigma = self.sigma[row.chamber][kind]
        if not row.has_prior:
            sigma = sqrt(sigma ** 2 + 25.0)  # no seat history: add 5pt-sd term
        return Prediction(mean, sigma, kind)

    def forecast_payload(self, row: FeatureRow, race_id: str) -> dict:
        prediction = self.predict(row)
        if prediction.model == "full":
            names, weights, x = FEATURE_NAMES, self.weights[row.chamber]["full"], row.x
        else:
            names, weights, x = (FUNDAMENTALS_NAMES,
                                 self.weights[row.chamber]["fundamentals"],
                                 row.x[:-N_POLL_FEATURES])
        components = {name: round(w * v, 3)
                      for name, w, v in zip(names, weights, x) if v != 0}
        components["_model"] = prediction.model
        probability = prediction.dem_probability
        low80, high80 = prediction.interval(1.282)
        low95, high95 = prediction.interval(1.960)
        return {
            "race_id": race_id,
            "dem_probability": round(probability, 4),
            "margin": round(prediction.mean, 2),
            "low80": round(low80, 2), "high80": round(high80, 2),
            "low95": round(low95, 2), "high95": round(high95, 2),
            "rating": rating(probability),
            "quality": quality_grade(
                poll_count=row.poll_count, poll_age_days=None,
                candidate_known=True, finance_fresh=False,
                boundary_certain=row.has_prior),
            "components": json.dumps(components),
        }

    def to_json(self) -> str:
        return json.dumps({
            "feature_names": FEATURE_NAMES, "l2": self.l2,
            "weights": self.weights, "sigma": self.sigma,
            "training": self.training_meta})

    @classmethod
    def from_json(cls, payload: str) -> "MarginModel":
        data = json.loads(payload)
        model = cls(l2=data.get("l2", DEFAULT_L2))
        model.weights = data["weights"]
        model.sigma = data["sigma"]
        model.training_meta = data.get("training", {})
        return model
