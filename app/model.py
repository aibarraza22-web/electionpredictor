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
# Cycle-level features (environment/midterm_environment/generic_ballot) are
# pseudo-replicated across a cycle's district rows; their ridge penalty is
# scaled by the training set's rows-per-cycle so their effective sample size
# is the number of cycles, not rows. See MarginModel._penalties. (This
# replaced an earlier hardcoded multiplier with the data-driven factor.)
MIN_SIGMA = 4.0     # pct points; guards degenerate residual pools
N_POLL_FEATURES = 2   # poll_average, has_polls sit at the end of the vector
N_NATIONAL_FEATURES = 2  # generic_ballot, has_generic_ballot sit just before them
N_TAIL = N_POLL_FEATURES + N_NATIONAL_FEATURES

FUNDAMENTALS_NAMES = FEATURE_NAMES[:-N_POLL_FEATURES]
CORE_NAMES = FEATURE_NAMES[:-N_TAIL]


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


def ridge_fit(xs: list[list[float]], ys: list[float],
              l2: float | list[float] = DEFAULT_L2) -> list[float]:
    """``l2`` may be a scalar (applied to every non-intercept feature) or a
    per-feature list — needed because some features are pseudo-replicated:
    ``environment``/``midterm_environment`` take the same value for every
    row in a cycle, so a district-level row count vastly overstates how
    much independent information backs their coefficient (e.g. 6,000+ House
    rows but only ~7 distinct midterm cycles). Naive row-level ridge treats
    that as 6,000 independent observations and can inflate the coefficient
    far beyond what the small number of true cycle-level data points
    supports; such features get a much larger penalty (see
    the rows-per-cycle factor in MarginModel._penalties)."""
    n_features = len(xs[0])
    penalties = l2 if isinstance(l2, list) else [l2] * n_features
    xtx = [[sum(x[i] * x[j] for x in xs) for j in range(n_features)] for i in range(n_features)]
    for i in range(1, n_features):  # do not penalize the intercept
        xtx[i][i] += penalties[i]
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
    """Per-chamber full and fundamentals-only ridge fits with residual sigmas.

    ``state_effects=True`` adds partial-pooled per-state offsets: the mean
    training residual for each (chamber, state), shrunk toward zero by
    n/(n+k) so a state's "personal pattern" only moves the forecast in
    proportion to how much evidence supports it. This is the disciplined
    form of state-specific modeling — a Maine offset estimated from Maine's
    past surprises, not a hand-picked story — and it runs as a challenger:
    it becomes champion only if it wins the walk-forward comparison.
    """

    def __init__(self, l2: float = DEFAULT_L2, state_effects: bool = False,
                 state_shrinkage: float = 8.0, use_generic_ballot: bool = False):
        # use_generic_ballot defaults False: the raw GB average DEGRADED
        # held-out Brier in both chambers (see research claim N-001), so it
        # is excluded from the champion and re-tested as a challenger on
        # every run.
        self.l2 = l2
        self.state_effects = state_effects
        self.state_shrinkage = state_shrinkage
        self.use_generic_ballot = use_generic_ballot
        self.weights: dict[str, dict[str, list[float]]] = {}
        self.sigma: dict[str, dict[str, float]] = {}
        self.state_offset: dict[str, dict[str, float]] = {}
        self.training_meta: dict[str, dict] = {}

    def _indices(self) -> dict[str, list[int]]:
        base = list(range(len(CORE_NAMES)))
        gb = [len(CORE_NAMES), len(CORE_NAMES) + 1] if self.use_generic_ballot else []
        polls = [len(FEATURE_NAMES) - 2, len(FEATURE_NAMES) - 1]
        return {"full": base + gb + polls, "fundamentals": base + gb, "core": base}

    def _penalties(self, idx: list[int], cycle_multiplier: float) -> list[float]:
        """Per-feature ridge penalty for a tier's index set. Cycle-level
        features (environment, midterm_environment, generic_ballot) take one
        value per cycle, so thousands of district rows that share that value
        are not independent evidence for the coefficient — they are
        pseudo-replicates of ~14 real cycle-level observations. Their penalty
        is scaled by ``cycle_multiplier`` (the training set's rows-per-cycle,
        computed in ``fit``) so the effective sample size for these features
        is the number of cycles, not the number of rows. Without this the
        national midterm coefficient inflates (a ~+8.7 uniform 2026 swing from
        an effective n of two R-president midterms); with it the coefficient
        reflects the real cycle-level evidence."""
        cycle_level = {FEATURE_NAMES.index("environment"),
                       FEATURE_NAMES.index("midterm_environment"),
                       FEATURE_NAMES.index("generic_ballot")}
        return [self.l2 * cycle_multiplier if i in cycle_level else self.l2
                for i in idx]

    def fit(self, rows: list[FeatureRow]) -> "MarginModel":
        by_chamber: dict[str, list[FeatureRow]] = {}
        for row in rows:
            if row.actual_margin is not None:
                by_chamber.setdefault(row.chamber, []).append(row)
        indices = self._indices()
        for chamber, chamber_rows in by_chamber.items():
            # Pseudoreplication correction factor: rows per distinct cycle.
            n_cycles = len({r.cycle for r in chamber_rows}) or 1
            cycle_mult = max(1.0, len(chamber_rows) / n_cycles)
            pen = {tier: self._penalties(indices[tier], cycle_mult)
                   for tier in ("fundamentals", "core", "full")}
            ys = [r.actual_margin for r in chamber_rows]
            fund_xs = [[r.x[i] for i in indices["fundamentals"]] for r in chamber_rows]
            fund_weights = ridge_fit(fund_xs, ys, pen["fundamentals"])
            fund_residuals = [
                y - sum(w * v for w, v in zip(fund_weights, x))
                for x, y in zip(fund_xs, ys)]
            # Core tier: no race polls AND no generic ballot. Used when the
            # current cycle has no national polling ingested yet, so the model
            # never extrapolates through a feature absent at prediction time.
            core_xs = [[r.x[i] for i in indices["core"]] for r in chamber_rows]
            core_weights = ridge_fit(core_xs, ys, pen["core"])
            core_residuals = [
                y - sum(w * v for w, v in zip(core_weights, x))
                for x, y in zip(core_xs, ys)]

            polled = [r for r in chamber_rows if r.poll_count]
            if len(polled) >= 3 * len(FEATURE_NAMES):
                full_xs = [[r.x[i] for i in indices["full"]] for r in polled]
                full_ys = [r.actual_margin for r in polled]
                full_weights = ridge_fit(full_xs, full_ys, pen["full"])
                full_residuals = [
                    y - sum(w * v for w, v in zip(full_weights, x))
                    for x, y in zip(full_xs, full_ys)]
            else:  # not enough polled history: everything routes to fundamentals
                full_weights, full_residuals = None, fund_residuals

            self.weights[chamber] = {"fundamentals": fund_weights, "full": full_weights,
                                     "core": core_weights}
            self.sigma[chamber] = {"fundamentals": _residual_sigma(fund_residuals),
                                   "full": _residual_sigma(full_residuals),
                                   "core": _residual_sigma(core_residuals)}
            self.training_meta[chamber] = {
                "n": len(chamber_rows), "n_polled": len(polled),
                "cycles": sorted({r.cycle for r in chamber_rows}), "l2": self.l2,
            }
            if self.state_effects:
                by_state: dict[str, list[float]] = {}
                for row in chamber_rows:
                    residual = row.actual_margin - self._base_mean(row)
                    by_state.setdefault(row.state, []).append(residual)
                self.state_offset[chamber] = {
                    state: (sum(vals) / len(vals)) * (len(vals) / (len(vals) + self.state_shrinkage))
                    for state, vals in by_state.items()}
        return self

    def _tier(self, row: FeatureRow) -> str:
        chamber = self.weights[row.chamber]
        if row.poll_count > 0 and chamber["full"] is not None:
            return "full"
        has_gb = row.x[len(CORE_NAMES) + 1] > 0
        return "fundamentals" if (has_gb and self.use_generic_ballot) else "core"

    def _base_mean(self, row: FeatureRow) -> float:
        tier = self._tier(row)
        weights = self.weights[row.chamber][tier]
        x = [row.x[i] for i in self._indices()[tier]]
        return sum(w * v for w, v in zip(weights, x))

    def predict(self, row: FeatureRow) -> Prediction:
        kind = self._tier(row)
        mean = self._base_mean(row)
        if self.state_effects:
            mean += self.state_offset.get(row.chamber, {}).get(row.state, 0.0)
        sigma = self.sigma[row.chamber][kind]
        if not row.has_prior:
            sigma = sqrt(sigma ** 2 + 25.0)  # no seat history: add 5pt-sd term
        if row.detail.get("redrawn"):
            # Mid-decade redraw: the seat lost its district prior above (so it
            # already carries the no-history term), but even the statewide lean
            # leaves the new district's specific composition genuinely unknown.
            # Add a further 4pt-sd term so redrawn seats read as the toss-ups
            # they are, not as false safe seats (mandate hypothesis H-005).
            sigma = sqrt(sigma ** 2 + 16.0)
        return Prediction(mean, sigma, kind)

    def forecast_payload(self, row: FeatureRow, race_id: str) -> dict:
        prediction = self.predict(row)
        idx = self._indices()[prediction.model]
        names = [FEATURE_NAMES[i] for i in idx]
        weights = self.weights[row.chamber][prediction.model]
        x = [row.x[i] for i in idx]
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
            "state_effects": self.state_effects,
            "use_generic_ballot": self.use_generic_ballot,
            "state_offset": self.state_offset,
            "weights": self.weights, "sigma": self.sigma,
            "training": self.training_meta})

    @classmethod
    def from_json(cls, payload: str) -> "MarginModel":
        data = json.loads(payload)
        model = cls(l2=data.get("l2", DEFAULT_L2),
                    state_effects=data.get("state_effects", False),
                    use_generic_ballot=data.get("use_generic_ballot", False))
        model.state_offset = data.get("state_offset", {})
        model.weights = data["weights"]
        model.sigma = data["sigma"]
        model.training_meta = data.get("training", {})
        return model
