"""Expert-rating consensus: from published ratings to a model input.

``app.ingest.race_ratings`` records *what each handicapper published and
when*.  This module turns those observations into the single number the
model consumes, and nothing else does that translation.

The consensus is the unweighted mean of the raters' signed strengths on the
shared ladder (Safe/Solid 4, Likely 3, Lean 2, Tilt 1, Tossup 0; Democratic
positive).  Unweighted is deliberate: the raters are highly correlated and
their per-cycle relative accuracy is not estimable from the few cycles of
published history, so weighting them would be fitting noise.  For each seat
only each rater's most recent rating on or before the as-of date is used, so
the consensus tracks rating *changes* rather than averaging a rater against
their own earlier opinion.

Only the *seats a rater published* are represented. Absence is never scored:
an unrated seat is left entirely to the fitted model, exactly as before.

:class:`RatingOverlay` at the bottom of this module is what actually moves a
published margin, and it is the second design tried — feeding the consensus
to the ridge as a feature was tested and rejected. See its docstring.
"""
from __future__ import annotations

from datetime import date, datetime
from math import log
from statistics import fmean, pstdev

# Consensus is expressed on the raters' own scale (-4 Safe R .. +4 Safe D).
# RatingOverlay's fitted slope is what converts a step on this ladder into
# margin points (see research claim R-001).
MAX_CONSENSUS = 4.0
# Below this many raters the mean is one or two opinions, not a consensus;
# it is still used, but the data grade does not treat it as strong evidence.
STRONG_RATER_COUNT = 5
# A rating this old is stale enough that it predates events the model should
# be reacting to.
FRESH_RATING_DAYS = 60
USABLE_RATING_DAYS = 180


def _day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class RatingLookup:
    """Per-seat expert consensus with an explicit as-of cutoff.

    Rows are pre-filtered by the caller (``store.all_race_ratings(as_of=...)``)
    and filtered again here, so a walk-forward backtest can never read a
    rating published after the date it is predicting from.
    """

    def __init__(self, rows: list[dict]):
        self._by_seat: dict[tuple[int, str], list[dict]] = {}
        for row in rows:
            self._by_seat.setdefault((row["cycle"], row["seat_key"]), []).append(row)

    def seats(self, cycle: int) -> set[str]:
        return {seat for c, seat in self._by_seat if c == cycle}

    def observations(self, cycle: int, seat_key: str,
                     as_of: str | None = None) -> list[dict]:
        rows = self._by_seat.get((cycle, seat_key), [])
        cutoff = _day(as_of)
        if cutoff is not None:
            rows = [r for r in rows if (_day(r.get("rating_date")) or cutoff) <= cutoff]
        return rows

    def latest_by_rater(self, cycle: int, seat_key: str,
                        as_of: str | None = None) -> list[dict]:
        """Each rater's most recent published rating as of the cutoff."""
        newest: dict[str, dict] = {}
        for row in self.observations(cycle, seat_key, as_of):
            rater = str(row.get("rater"))
            current = newest.get(rater)
            if current is None or str(row.get("rating_date")) >= str(current.get("rating_date")):
                newest[rater] = row
        return sorted(newest.values(), key=lambda r: str(r.get("rater")))

    def consensus(self, cycle: int, seat_key: str,
                  as_of: str | None = None) -> dict | None:
        """``None`` when no rater has published this seat as of the cutoff."""
        latest = self.latest_by_rater(cycle, seat_key, as_of)
        if not latest:
            return None
        scores = [float(row["score"]) for row in latest]
        dates = sorted(str(row["rating_date"]) for row in latest)
        cutoff = _day(as_of) or date.today()
        newest = _day(dates[-1])
        return {
            "consensus": round(fmean(scores), 4),
            "n_raters": len(scores),
            # Disagreement among raters is real forecast uncertainty; the
            # forecast widens the interval by it rather than hiding it.
            "disagreement": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "newest_rating_date": dates[-1],
            "oldest_rating_date": dates[0],
            "age_days": max(0, (cutoff - newest).days) if newest else None,
            "raters": [{"rater": row["rater"], "rating": row["rating"],
                        "score": float(row["score"]),
                        "rating_date": row["rating_date"],
                        "source_url": row.get("source_url")}
                       for row in latest],
        }


def rating_evidence_points(summary: dict | None) -> int:
    """Data-grade credit for expert coverage: 0, 1, or 2 points.

    Two points require a real consensus (several independent raters) that is
    also current. One point covers thin or ageing coverage. This is the same
    shape as the poll-count credit: it counts *evidence actually present*,
    never relabels its absence.
    """
    if not summary:
        return 0
    age = summary.get("age_days")
    if age is None:
        return 0
    if summary["n_raters"] >= STRONG_RATER_COUNT and age <= FRESH_RATING_DAYS:
        return 2
    if age <= USABLE_RATING_DAYS:
        return 1
    return 0


# --- production overlay -------------------------------------------------

# The fitted blend weight is capped rather than taken at its held-out optimum
# (which is 1.0 for the House: log loss keeps falling to w=1). Two reasons,
# both measured rather than assumed:
#
#  * Every historical ratings page carries each rater's FINAL pre-election
#    snapshot, while the 2026 ratings the forecast actually consumes are
#    ~2 months out. Re-parsing the archived late-August revisions of the 2020,
#    2022 and 2024 pages shows the slope survives the vintage gap (4.18/4.21/
#    3.84 vs 3.70/4.06/3.86 final) but the residual spread does not: it roughly
#    doubles (~9.0-9.5 pts vs ~5.2). A weight fitted on final ratings therefore
#    overstates how much an August rating deserves.
#  * A secondary signal replacing the fitted model outright removes the
#    model's own polling and seat history from competitive races entirely.
#
# The measured cost of the cap on House rated seats is small (walk-forward log
# loss 0.4374 at w=0.70 vs 0.4280 at w=1.00).
MAX_OVERLAY_WEIGHT = 0.75
WEIGHT_GRID = tuple(round(0.05 * i, 2) for i in range(0, 16))
# Below this many held-out training rows a stratum's weight is noise; it
# inherits the chamber-wide weight instead.
MIN_STRATUM_ROWS = 60
# The fitted rated-seat sigma comes from residuals of blended predictions made
# against each cycle's FINAL published ratings, while the live feed runs
# ~2 months ahead of the election. Re-parsing the archived late-August
# revisions of the 2020/2022/2024 House ratings pages puts that vintage gap at
# roughly 1.6x/1.8x/1.0x on residual spread; the mean, 1.45, is applied so the
# published interval is not tighter than the vintage supports. It can never
# widen a seat past the model's own pre-overlay sigma, which is the status quo.
VINTAGE_SIGMA_INFLATION = 1.45
MIN_SLOPE_ROWS = 50
MIN_CYCLE_RATED_SEATS = 20


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < MIN_SLOPE_ROWS:
        return None
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def is_unanimously_safe(summary: dict | None) -> bool:
    """True when every rater calls the seat Safe/Solid for the same party."""
    if not summary or not summary.get("raters"):
        return False
    scores = [float(r["score"]) for r in summary["raters"]]
    return all(abs(score) >= MAX_CONSENSUS for score in scores) and (
        min(scores) > 0 or max(scores) < 0)


def overlay_consensus(summary: dict | None, chamber: str) -> float | None:
    """The consensus the overlay may act on, or ``None`` to leave the seat alone.

    The rule is one thing: apply the overlay only to the population its slope
    was fitted on. That population differs by chamber, because the source
    pages do.

    * The **Senate** ratings tables list every seat up that cycle, safe ones
      included, so the fitted slope already spans the full range and every
      Senate seat is in scope.
    * The **House** ratings page lists only seats at least one rater declines
      to call safe. Per-state articles now also supply ratings for the ~280
      House seats every rater calls safe — real evidence, which is why they
      count toward the data grade and are displayed — but feeding them to a
      slope fitted on competitive seats would be extrapolation: 3.9 points per
      rating step maps a unanimous Safe rating to single-digit margin points
      when such seats actually win by twenty or more.

    Where the model calls a unanimously-safe House seat competitive, the
    forecast reports that disagreement rather than quietly splitting the
    difference (see ``competitive_but_consensus_safe`` in the run summary).
    """
    if not summary:
        return None
    if chamber == "house" and is_unanimously_safe(summary):
        return None
    return summary["consensus"]


def consensus_of(row) -> float | None:
    summary = row.detail.get("ratings") if row.detail else None
    return overlay_consensus(summary, row.chamber)


def _stratum(row) -> str:
    return "polled" if row.poll_count > 0 else "unpolled"


class RatingOverlay:
    """Blends the fitted model with an expert-consensus reading of each seat.

    Adding the consensus to the model's feature vector was tested and
    REJECTED — it made rated seats worse, because ``has_rating`` is a
    selection indicator (a seat is rated because someone thinks it is
    competitive) and one global coefficient turns that selection into a
    biased constant. This overlay avoids that by never asking the ratings for
    the national level, only for the *spread between seats*:

        implied(seat) = level + slope * (consensus(seat) - mean consensus)

    ``level`` is the model's own mean prediction across this cycle's rated
    seats, so the national environment still comes entirely from the fitted
    model and outcomes are never consulted. ``slope`` is fitted in
    within-cycle deviation form on strictly earlier cycles, which removes each
    cycle's environment from the fit as well; it lands at 3.8-4.6 margin
    points per rating step for the House and 5.9-7.0 for the Senate, stable
    across every held-out cycle.

    The blend weight is fitted per chamber and separately for polled and
    unpolled seats, by held-out log loss under the same walk-forward protocol
    the champion is chosen with — a race with real polling keeps more of the
    model, which is what the data asks for (walk-forward House MAE bottoms at
    w~0.6 on polled seats but keeps falling to w=1 on unpolled ones).
    """

    def __init__(self, chamber: str):
        self.chamber = chamber
        self.slope: float | None = None
        self.weights: dict[str, float] = {}
        self.sigma: dict[str, float] = {}
        self.fit_meta: dict = {}

    # -- fitting ---------------------------------------------------------

    @staticmethod
    def _slope(rows) -> float | None:
        """Pooled within-cycle deviation slope over rows with outcomes."""
        xs: list[float] = []
        ys: list[float] = []
        by_cycle: dict[int, list] = {}
        for row in rows:
            if row.actual_margin is None or consensus_of(row) is None:
                continue
            by_cycle.setdefault(row.cycle, []).append(row)
        for rated in by_cycle.values():
            if len(rated) < MIN_CYCLE_RATED_SEATS:
                continue
            mean_c = fmean([consensus_of(r) for r in rated])
            mean_m = fmean([r.actual_margin for r in rated])
            xs += [consensus_of(r) - mean_c for r in rated]
            ys += [r.actual_margin - mean_m for r in rated]
        return _ols_slope(xs, ys)

    def fit(self, rows, min_training_cycles: int = 3) -> "RatingOverlay":
        """Walk-forward fit of slope, blend weights, and rated-seat sigma."""
        from .model import MarginModel, MIN_SIGMA, Prediction

        rows = [r for r in rows if r.chamber == self.chamber
                and r.actual_margin is not None]
        cycles = sorted({r.cycle for r in rows})
        # (stratum, weight) -> accumulated held-out log loss / residuals
        samples: list[dict] = []
        for test_cycle in cycles:
            training = [r for r in rows if r.cycle < test_cycle]
            if len({r.cycle for r in training}) < min_training_cycles:
                continue
            test = [r for r in rows if r.cycle == test_cycle
                    and consensus_of(r) is not None]
            if len(test) < MIN_CYCLE_RATED_SEATS:
                continue
            slope = self._slope(training)
            if slope is None:
                continue
            model = MarginModel().fit(training)
            predictions = {r.seat_key: model.predict(r) for r in test}
            level = fmean([p.mean for p in predictions.values()])
            mean_c = fmean([consensus_of(r) for r in test])
            for row in test:
                base = predictions[row.seat_key]
                samples.append({
                    "stratum": _stratum(row), "base": base,
                    "implied": level + slope * (consensus_of(row) - mean_c),
                    "actual": row.actual_margin,
                    "won": 1 if row.actual_margin > 0 else 0})
        self.slope = self._slope(rows)
        if not samples or self.slope is None:
            self.fit_meta = {"fitted": False,
                             "reason": "not enough rated history to fit an overlay"}
            return self

        def log_loss(subset: list[dict], weight: float) -> float:
            total = 0.0
            for item in subset:
                base = item["base"]
                mean = (1 - weight) * base.mean + weight * item["implied"]
                blended = Prediction(mean, base.sigma, base.model,
                                     calibration=base.calibration)
                p = min(1 - 1e-6, max(1e-6, blended.dem_probability))
                total -= (item["won"] * log(p) + (1 - item["won"]) * log(1 - p))
            return total / len(subset)

        def best_weight(subset: list[dict]) -> tuple[float, dict[str, float]]:
            board = {w: round(log_loss(subset, w), 5) for w in WEIGHT_GRID
                     if w <= MAX_OVERLAY_WEIGHT}
            return min(board, key=board.get), board

        overall_weight, overall_board = best_weight(samples)
        boards = {"chamber": overall_board}
        for stratum in ("polled", "unpolled"):
            subset = [s for s in samples if s["stratum"] == stratum]
            if len(subset) >= MIN_STRATUM_ROWS:
                weight, board = best_weight(subset)
                self.weights[stratum] = weight
                boards[stratum] = board
            else:
                self.weights[stratum] = overall_weight
        # Rated-seat sigma from the blended walk-forward residuals. The
        # model's own sigma is pooled over every seat including uncontested
        # blowouts, so it over-covers competitive races badly (walk-forward
        # coverage80 ~0.93-0.98 against a nominal 0.80). Measuring it on the
        # population the overlay actually applies to makes the published
        # interval mean what it says.
        for stratum in ("polled", "unpolled"):
            weight = self.weights[stratum]
            residuals = [
                ((1 - weight) * s["base"].mean + weight * s["implied"]) - s["actual"]
                for s in samples if s["stratum"] == stratum]
            if len(residuals) > 5:
                self.sigma[stratum] = max(MIN_SIGMA, pstdev(residuals))
        # Held-out scoreboard for the overlay itself: the same samples, scored
        # with the model alone and with the chosen weights. This is the number
        # the claim in RESEARCH_CLAIMS R-001 is read from, so it is computed
        # here rather than quoted from a notebook.
        from .backtest import metrics

        def scored(weight_of) -> list[dict]:
            rows = []
            for item in samples:
                base = item["base"]
                weight = weight_of(item["stratum"])
                mean = (1 - weight) * base.mean + weight * item["implied"]
                blended = Prediction(mean, base.sigma, base.model,
                                     calibration=base.calibration)
                rows.append({
                    "probability": blended.dem_probability,
                    "predicted_margin": mean, "actual_margin": item["actual"],
                    "dem_won": item["won"],
                    "low80": blended.interval(1.282)[0],
                    "high80": blended.interval(1.282)[1],
                    "low95": blended.interval(1.960)[0],
                    "high95": blended.interval(1.960)[1],
                    "cycle": 0})
            return rows

        keys = ("n_races", "brier", "log_loss", "winner_accuracy", "margin_mae")
        model_only = metrics(scored(lambda _: 0.0))
        with_overlay = metrics(scored(lambda stratum: self.weights.get(stratum, 0.0)))
        self.fit_meta = {
            "fitted": True, "slope_margin_points_per_rating_step": round(self.slope, 3),
            "held_out_metrics": {
                "population": "seats with published ratings, walk-forward",
                "model_only": {k: model_only.get(k) for k in keys},
                "with_overlay": {k: with_overlay.get(k) for k in keys}},
            "weights": dict(self.weights),
            "rated_seat_sigma": {k: round(v, 3) for k, v in self.sigma.items()},
            "held_out_rows": len(samples),
            "weight_cap": MAX_OVERLAY_WEIGHT,
            "log_loss_by_weight": boards,
        }
        return self

    # -- application -----------------------------------------------------

    def cycle_context(self, predictions: list[float], consensus: list[float]) -> dict | None:
        """Level and mean consensus for one cycle's rated seats.

        Both come from information available before the election: the level is
        the model's own mean prediction, the mean consensus is the raters'.
        """
        if not self.is_fitted or not predictions or len(predictions) != len(consensus):
            return None
        return {"level": fmean(predictions), "mean_consensus": fmean(consensus),
                "n_rated": len(predictions)}

    @property
    def is_fitted(self) -> bool:
        return bool(self.fit_meta.get("fitted"))

    def apply(self, prediction, consensus: float | None, context: dict | None,
              polled: bool, summary: dict | None = None) -> tuple[object, dict]:
        """Return ``(blended prediction, explanation)``.

        With no rating, no fit, or no cycle context the model's own prediction
        is returned untouched — the overlay never invents a signal.

        ``summary`` is the seat's full rating summary and is used only to
        explain *why* an unused rating was unused: a seat every rater calls
        safe has plenty of published ratings, and reporting "no published
        rating" next to a table of them is simply wrong.
        """
        from .model import Prediction

        if consensus is None or context is None or not self.is_fitted:
            if not self.is_fitted:
                reason = "no fitted overlay for this chamber"
            elif context is None:
                reason = "no rated seats in this cycle to anchor the overlay"
            elif is_unanimously_safe(summary):
                reason = ("every rater calls this seat safe, which is outside "
                          "the competitive population the overlay's slope was "
                          "fitted on")
            elif summary:
                reason = "published ratings are not usable for this seat"
            else:
                reason = "no published rating for this seat"
            return prediction, {"applied": False, "reason": reason}
        stratum = "polled" if polled else "unpolled"
        weight = self.weights.get(stratum, 0.0)
        if weight <= 0.0:
            # The held-out fit says this stratum is better off with the model
            # alone. Then the overlay must leave the seat completely alone —
            # including its sigma, which is fitted on the blended prediction
            # and means nothing when no blending happens.
            return prediction, {"applied": False, "stratum": stratum,
                                "blend_weight": 0.0,
                                "reason": "fitted blend weight is zero for this "
                                          "chamber and stratum"}
        implied = context["level"] + self.slope * (consensus - context["mean_consensus"])
        mean = (1 - weight) * prediction.mean + weight * implied
        fitted_sigma = self.sigma.get(stratum)
        sigma = (min(prediction.sigma, fitted_sigma * VINTAGE_SIGMA_INFLATION)
                 if fitted_sigma else prediction.sigma)
        blended = Prediction(mean, sigma, prediction.model,
                             calibration=prediction.calibration,
                             calibration_weight=prediction.calibration_weight)
        return blended, {
            "applied": True, "stratum": stratum, "blend_weight": weight,
            "consensus": round(consensus, 3),
            "cycle_mean_consensus": round(context["mean_consensus"], 3),
            "slope_margin_points_per_rating_step": round(self.slope, 3),
            "model_level_for_rated_seats": round(context["level"], 3),
            "ratings_implied_margin": round(implied, 2),
            "model_margin": round(prediction.mean, 2),
            "blended_margin": round(mean, 2),
            "margin_shift": round(mean - prediction.mean, 2),
            "model_sigma": round(prediction.sigma, 3),
            "rated_seat_sigma": round(sigma, 3),
            "fitted_rated_seat_sigma": round(fitted_sigma, 3) if fitted_sigma else None,
            "vintage_sigma_inflation": VINTAGE_SIGMA_INFLATION,
        }

    def to_json(self) -> dict:
        return {"chamber": self.chamber, **self.fit_meta}
