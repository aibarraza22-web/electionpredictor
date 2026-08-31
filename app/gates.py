"""Release gates: refuse to publish a forecast that only *claims* to be new.

Model 2026.18 shipped a campaign-adjustment layer that was wired into the
margin equation and then, in production, moved almost nothing: two of the
three feeds it needed (candidate profiles and campaign events) were never
configured, so it ran finance-only and the published toplines shifted by
0.1 point in the House (Democratic control 0.6432 -> 0.6441) and 1.4 in the
Senate (0.5474 -> 0.5616). Nothing in the pipeline noticed, because nothing
in the pipeline was checking.

These gates make that failure loud instead of silent. They run against the
payloads that are *about* to be frozen, before any snapshot is inserted, so a
failure leaves the previous forecast standing rather than publishing a
version that cannot back its own claims.
"""
from __future__ import annotations

# A race the model itself calls competitive must be backed by real seat-level
# evidence — polls, published expert ratings, and current finance — not by
# history and the national environment alone.
COMPETITIVE_RATINGS = {"Toss-up", "Lean Democratic", "Lean Republican"}
REQUIRED_GRADES = {"A", "B"}
# When the model version changes, its predictions must differ from the
# outgoing champion's on a real share of comparable competitive races.
MOVEMENT_MARGIN_POINTS = 0.5
MOVEMENT_PROBABILITY_POINTS = 2.0
MIN_MOVED_FRACTION = 0.75
# Below this many comparable races the fraction is noise, so the gate reports
# "not evaluated" rather than passing or failing on two or three seats.
MIN_COMPARABLE_RACES = 20


class GateFailure(RuntimeError):
    """A publishable-forecast invariant was violated."""


def competitive_races(payloads: list[dict]) -> list[dict]:
    return [p for p in payloads if p.get("rating") in COMPETITIVE_RATINGS]


def check_competitive_data_grade(payloads: list[dict]) -> dict:
    """Every competitive race must carry data grade A or B."""
    competitive = competitive_races(payloads)
    failing = sorted(
        ({"race_id": p["race_id"], "rating": p["rating"], "quality": p.get("quality")}
         for p in competitive if p.get("quality") not in REQUIRED_GRADES),
        key=lambda item: item["race_id"])
    result = {"gate": "competitive_data_grade", "competitive_races": len(competitive),
              "failing_races": len(failing), "examples": failing[:15],
              "passed": not failing}
    if failing:
        names = ", ".join(f"{f['race_id']} ({f['quality']})" for f in failing[:10])
        raise GateFailure(
            f"{len(failing)} of {len(competitive)} competitive races are below data "
            f"grade B: {names}"
            f"{' ...' if len(failing) > 10 else ''}. A competitive race without "
            "seat-level evidence is a guess with a confidence interval attached; "
            "ingest polling/ratings coverage for these seats or stop calling them "
            "competitive.")
    return result


def check_model_moved(payloads: list[dict], previous: dict[str, dict],
                      model_version: str, previous_version: str | None) -> dict:
    """A new model version must actually change competitive-race forecasts.

    Compared on competitive races only: a new campaign or ratings layer is
    not expected to move a Safe seat, and including 400 safe races would let a
    version that changes nothing that matters still pass on rounding noise.
    """
    if not previous or previous_version in (None, model_version):
        return {"gate": "model_moved", "passed": True,
                "evaluated": False,
                "reason": "no previous model version to compare against"}
    comparable = [p for p in competitive_races(payloads) if p["race_id"] in previous]
    if len(comparable) < MIN_COMPARABLE_RACES:
        return {"gate": "model_moved", "passed": True, "evaluated": False,
                "reason": f"only {len(comparable)} comparable competitive races "
                          f"(need {MIN_COMPARABLE_RACES})"}
    moved = []
    for payload in comparable:
        before = previous[payload["race_id"]]
        margin_delta = abs(float(payload["margin"]) - float(before.get("margin") or 0.0))
        probability_delta = abs(
            100.0 * (float(payload["dem_probability"])
                     - float(before.get("dem_probability") or 0.0)))
        if (margin_delta >= MOVEMENT_MARGIN_POINTS
                or probability_delta >= MOVEMENT_PROBABILITY_POINTS):
            moved.append(payload["race_id"])
    fraction = len(moved) / len(comparable)
    result = {"gate": "model_moved", "evaluated": True,
              "previous_version": previous_version, "model_version": model_version,
              "comparable_competitive_races": len(comparable),
              "moved_races": len(moved), "moved_fraction": round(fraction, 4),
              "threshold": MIN_MOVED_FRACTION,
              "passed": fraction >= MIN_MOVED_FRACTION}
    if not result["passed"]:
        raise GateFailure(
            f"model {model_version} changed only {len(moved)} of {len(comparable)} "
            f"competitive races ({fraction:.0%}) versus {previous_version}; a new "
            f"version must move at least {MIN_MOVED_FRACTION:.0%}. Either its new "
            "inputs are missing in this database or the change does not reach the "
            "published numbers — publishing it would repeat the 2026.18 failure of "
            "announcing a change that production never showed.")
    return result


def check_current_cycle_ratings(coverage: dict, minimum: int) -> dict:
    """The expert-ratings feed must have actually delivered this cycle.

    A silent 403 or a page rename upstream would drop the model's only
    seat-level signal for the ~90% of races that carry no polling, and the
    pipeline would otherwise carry on and publish an unchanged forecast.
    """
    rated = int(coverage.get("with_expert_ratings") or 0)
    result = {"gate": "current_cycle_ratings", "rated_races": rated,
              "minimum": minimum, "passed": rated >= minimum}
    if not result["passed"]:
        raise GateFailure(
            f"only {rated} races carry expert ratings (minimum {minimum}); the "
            "race-ratings ingest did not deliver. Check the wikipedia-race-ratings "
            "adapter's reported failures before publishing.")
    return result
