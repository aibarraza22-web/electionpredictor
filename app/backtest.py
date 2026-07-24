"""Leakage-safe expanding-window (prequential) backtesting on ingested data.

For each held-out cycle the model is re-fit on strictly earlier cycles only,
predictions are frozen against election-day poll cutoffs, and then scored
against certified outcomes. Metrics are persisted to ``backtest_runs`` and
served verbatim by the API — the application never reports performance that
was not produced by one of these runs.
"""
from __future__ import annotations

import json
from math import log, sqrt
from statistics import pstdev
from uuid import uuid4

from . import store
from .domain import normal_cdf
from .features import FEATURE_NAMES, FeatureRow, PollLookup, ResultLookup, historical_rows
from .model import MIN_SIGMA, MarginModel, ridge_fit

MIN_TRAINING_CYCLES = 3

# Baseline models: the SAME walk-forward protocol restricted to a feature
# subset (indices into FEATURE_NAMES). The production model must beat these
# on held-out Brier/log loss to justify its complexity; results are stored,
# never asserted. "polled_only" restricts scoring to races with polls so the
# comparison is on equal information.
# Feature indices reference app.features.FEATURE_NAMES:
# 0 intercept, 1 prior_margin, 2 has_prior, 3 prior_winner, 4 state_lean,
# 5 has_state_lean, 6 environment, 7 midterm_environment, 8 generic_ballot,
# 9 has_generic_ballot, 10 poll_average, 11 has_polls.
BASELINES: dict[str, dict] = {
    "baseline-prior-result": {"features": [0, 1, 2], "polled_only": False},
    "baseline-incumbency-only": {"features": [0, 3], "polled_only": False},
    "baseline-state-lean": {"features": [0, 4, 5], "polled_only": False},
    "baseline-environment-only": {"features": [0, 6, 7], "polled_only": False},
    "baseline-uniform-swing": {"features": [0, 1, 2, 6, 7], "polled_only": False},
    "baseline-generic-ballot": {"features": [0, 1, 2, 8, 9], "polled_only": False},
    "baseline-polls-only": {"features": [0, 10, 11], "polled_only": True},
}

# Challenger models: full-feature variants competing with the champion under
# the identical protocol. Promotion requires winning, and the decision is
# recorded either way.
CHALLENGERS: dict[str, dict] = {
    "challenger-state-effects": {"state_effects": True},
    # Re-tests the generic-ballot feature every run: it degraded held-out
    # accuracy on first evaluation (claim N-001) and stays out of the
    # champion until this challenger wins.
    "challenger-generic-ballot": {"use_generic_ballot": True},
}

# Champion candidates: model specs that compete to be each chamber's
# production model. The winner is chosen PER CHAMBER by held-out log loss
# (the Senate, with far fewer training races, tends to prefer stronger
# regularization than the House) — the mandate calls for different model
# structures per chamber, and this makes that choice on evidence, not
# assertion. The generic-ballot spec is excluded here (documented as a
# rejected feature, claim N-001) but still tracked as a challenger.
CHAMPION_CANDIDATES: dict[str, dict] = {
    "base": {},
    "ridge-strong": {"l2": 8.0},
    "ridge-light": {"l2": 2.0},
    "state-effects": {"state_effects": True},
}


def metrics(scored: list[dict]) -> dict:
    """Aggregate forecast-quality metrics over frozen predictions."""
    n = len(scored)
    if not n:
        return {"n_races": 0}
    brier = sum((s["probability"] - s["dem_won"]) ** 2 for s in scored) / n
    eps = 1e-6
    log_loss = -sum(
        s["dem_won"] * log(max(s["probability"], eps))
        + (1 - s["dem_won"]) * log(max(1 - s["probability"], eps))
        for s in scored) / n
    accuracy = sum((s["probability"] > .5) == bool(s["dem_won"]) for s in scored) / n
    errors = [s["predicted_margin"] - s["actual_margin"] for s in scored]
    mae = sum(abs(e) for e in errors) / n
    rmse = sqrt(sum(e * e for e in errors) / n)
    coverage80 = sum(s["low80"] <= s["actual_margin"] <= s["high80"] for s in scored) / n
    coverage95 = sum(s["low95"] <= s["actual_margin"] <= s["high95"] for s in scored) / n
    bins = []
    for lower in [i / 10 for i in range(10)]:
        members = [s for s in scored if lower <= s["probability"] < lower + .1]
        if members:
            bins.append({
                "bin": f"{lower:.1f}-{lower + .1:.1f}", "n": len(members),
                "forecast": round(sum(m["probability"] for m in members) / len(members), 3),
                "observed": round(sum(m["dem_won"] for m in members) / len(members), 3)})
    return {"n_races": n, "brier": round(brier, 4), "log_loss": round(log_loss, 4),
            "winner_accuracy": round(accuracy, 4), "margin_mae": round(mae, 2),
            "margin_rmse": round(rmse, 2), "coverage80": round(coverage80, 3),
            "coverage95": round(coverage95, 3), "calibration": bins}


def national_error_sigma(scored: list[dict], min_cycles: int = 5) -> float | None:
    """Standard deviation of out-of-sample cycle-level mean prediction error
    (predicted - actual, averaged within each held-out cycle).

    Individual-seat sigma alone cannot tell a control simulation how much
    error is genuinely *shared* across seats in a given cycle (a national
    swing) versus independent per-seat noise that washes out over hundreds
    of seats. This directly measures the shared component from real
    historical misses, cycle by cycle - not a guessed constant. Returns
    None with too few evaluated cycles for the estimate to mean anything.
    """
    by_cycle: dict[int, list[float]] = {}
    for s in scored:
        by_cycle.setdefault(s["cycle"], []).append(s["predicted_margin"] - s["actual_margin"])
    if len(by_cycle) < min_cycles:
        return None
    cycle_means = [sum(errs) / len(errs) for errs in by_cycle.values()]
    return pstdev(cycle_means)


def walk_forward(rows: list[FeatureRow], chamber: str,
                 min_training_cycles: int = MIN_TRAINING_CYCLES,
                 model_kwargs: dict | None = None) -> tuple[list[dict], list[int]]:
    """Frozen out-of-sample predictions for every eligible cycle."""
    rows = [r for r in rows if r.chamber == chamber and r.actual_margin is not None]
    cycles = sorted({r.cycle for r in rows})
    scored: list[dict] = []
    evaluated: list[int] = []
    for test_cycle in cycles:
        training = [r for r in rows if r.cycle < test_cycle]
        training_cycles = {r.cycle for r in training}
        if len(training_cycles) < min_training_cycles:
            continue
        test = [r for r in rows if r.cycle == test_cycle]
        assert all(r.cycle < test_cycle for r in training), "future cycle leaked into training"
        assert all(not r.last_poll_date or r.last_poll_date <= r.detail["as_of"]
                   for r in test), "poll after as-of cutoff leaked into features"
        model = MarginModel(**(model_kwargs or {})).fit(training)
        for row in test:
            prediction = model.predict(row)
            low80, high80 = prediction.interval(1.282)
            low95, high95 = prediction.interval(1.960)
            scored.append({
                "cycle": test_cycle, "seat_key": row.seat_key,
                "probability": prediction.dem_probability,
                "predicted_margin": prediction.mean,
                "actual_margin": row.actual_margin,
                "dem_won": 1 if row.actual_margin > 0 else 0,
                "low80": low80, "high80": high80,
                "low95": low95, "high95": high95,
                "polled": row.poll_count > 0,
                "training_cycles": sorted(training_cycles),
            })
        evaluated.append(test_cycle)
    return scored, evaluated


def walk_forward_baseline(rows: list[FeatureRow], chamber: str, feature_idx: list[int],
                          polled_only: bool = False,
                          min_training_cycles: int = MIN_TRAINING_CYCLES) -> list[dict]:
    """The identical expanding-window protocol on a restricted feature set."""
    rows = [r for r in rows if r.chamber == chamber and r.actual_margin is not None]
    if polled_only:
        rows = [r for r in rows if r.poll_count > 0]
    cycles = sorted({r.cycle for r in rows})
    scored: list[dict] = []
    for test_cycle in cycles:
        training = [r for r in rows if r.cycle < test_cycle]
        if len({r.cycle for r in training}) < min_training_cycles:
            continue
        xs = [[r.x[i] for i in feature_idx] for r in training]
        ys = [r.actual_margin for r in training]
        weights = ridge_fit(xs, ys)
        residuals = [y - sum(w * v for w, v in zip(weights, x)) for x, y in zip(xs, ys)]
        sigma = max(MIN_SIGMA, pstdev(residuals)) if len(residuals) > 1 else MIN_SIGMA
        for row in (r for r in rows if r.cycle == test_cycle):
            mean = sum(w * row.x[i] for w, i in zip(weights, feature_idx))
            probability = min(.995, max(.005, normal_cdf(mean / sigma)))
            scored.append({
                "cycle": test_cycle, "seat_key": row.seat_key,
                "probability": probability, "predicted_margin": mean,
                "actual_margin": row.actual_margin,
                "dem_won": 1 if row.actual_margin > 0 else 0,
                "low80": mean - 1.282 * sigma, "high80": mean + 1.282 * sigma,
                "low95": mean - 1.960 * sigma, "high95": mean + 1.960 * sigma,
                "polled": row.poll_count > 0,
            })
    return scored


def subgroup_metrics(scored: list[dict], rows_by_key: dict[tuple, FeatureRow]) -> dict:
    """Performance sliced the way the research mandate asks for."""
    def sel(predicate):
        return metrics([s for s in scored if predicate(s)])

    def row_of(s):
        return rows_by_key.get((s["cycle"], s["seat_key"]))

    return {
        "polled": sel(lambda s: s["polled"]),
        "unpolled": sel(lambda s: not s["polled"]),
        "midterm_cycles": sel(lambda s: s["cycle"] % 4 == 2),
        "presidential_cycles": sel(lambda s: s["cycle"] % 4 == 0),
        "competitive_actual_lt10": sel(lambda s: abs(s["actual_margin"]) < 10),
        "safe_actual_ge10": sel(lambda s: abs(s["actual_margin"]) >= 10),
        "dem_held_seats": sel(lambda s: (r := row_of(s)) is not None and r.x[3] > 0),
        "rep_held_seats": sel(lambda s: (r := row_of(s)) is not None and r.x[3] < 0),
    }


def topline_estimator_backtest(results: ResultLookup, poll_lookup: PollLookup,
                               state_lean=None, cycles: list[int] | None = None,
                               simulations: int = 12000) -> dict:
    """Which summary of the simulated seat distribution is the best point
    estimate of the actual House seat count? Walk-forward: for each held-out
    cycle, fit on earlier cycles, simulate that cycle's seats, and compare
    candidate toplines (median, mean, mode, mean-of-top-4-modes) to the
    certified result. Answers, on data, the question of whether the median is
    the right headline number or whether some other statistic 'works every
    time' -- and quantifies the model's irreducible seat-count error."""
    from collections import Counter
    from statistics import mean, median
    from .features import build_row, StateLean
    from .simulation import simulate_control
    if state_lean is None:
        state_lean = StateLean(results)
    cycles = cycles or [c for c in results.cycles("house") if c >= 2008]
    errs: dict[str, list[float]] = {k: [] for k in
                                    ("median", "mean", "mode", "mean_top4")}
    per_cycle = []
    for t in cycles:
        train_cycles = [c for c in results.cycles("house") if c < t]
        if len(train_cycles) < MIN_TRAINING_CYCLES:
            continue
        train = historical_rows(results, poll_lookup, "house",
                                cycles=train_cycles, state_lean=state_lean)
        model = MarginModel(l2=2.0).fit(train)
        pays = []
        for r in results.seats(t, "house"):
            row = build_row(r["seat_key"], t, "house", r["state"], r.get("district"),
                            results, poll_lookup, f"{t}-11-08", state_lean=state_lean)
            pays.append(model.forecast_payload(row, r["seat_key"]))
        sim = simulate_control(pays, "house", simulations=simulations)
        counts: list[int] = []
        for k, v in sim["distribution"].items():
            counts += [int(k)] * v
        actual = sum(1 for r in results.seats(t, "house") if r["dem_margin"] > 0)
        top4 = [s for s, _ in Counter(counts).most_common(4)]
        estimates = {
            "median": median(counts), "mean": mean(counts),
            "mode": Counter(counts).most_common(1)[0][0],
            "mean_top4": sum(top4) / len(top4)}
        for name, val in estimates.items():
            errs[name].append(abs(val - actual))
        per_cycle.append({"cycle": t, "actual": actual,
                          **{k: round(v, 1) for k, v in estimates.items()}})
    mae = {k: round(sum(v) / len(v), 2) for k, v in errs.items() if v}
    best = min(mae, key=mae.get) if mae else None
    return {"mae_by_estimator": mae, "best_estimator": best,
            "per_cycle": per_cycle,
            "note": "MAE in seats vs certified House result; lower is better. The "
                    "model's own seat-count error floor is the winning MAE."}


def horizon_metrics(results: ResultLookup, poll_lookup: PollLookup,
                    chamber: str) -> dict:
    """Production-model accuracy at earlier poll cutoffs.

    The evaluation population is held FIXED — races that had polls by
    election eve — and the model is re-scored on that same set at each
    earlier cutoff (where a race may not yet have polls and falls back to
    the fundamentals tier, exactly as it would have in real time). A
    shifting population would otherwise shrink to tiny, unrepresentative
    samples at long horizons.
    """
    eve_rows = historical_rows(results, poll_lookup, chamber)
    eligible = {(r.cycle, r.seat_key) for r in eve_rows if r.poll_count > 0}
    out = {}
    for days_before, cutoffs in (
            (0, None),
            (30, {c: f"{c}-10-09" for c in range(1998, 2026, 2)}),
            (90, {c: f"{c}-08-10" for c in range(1998, 2026, 2)})):
        rows = eve_rows if cutoffs is None else historical_rows(
            results, poll_lookup, chamber, election_dates=cutoffs)
        scored, _ = walk_forward(rows, chamber)
        subset = [s for s in scored if (s["cycle"], s["seat_key"]) in eligible]
        out[str(days_before)] = metrics(subset)
    out["population"] = "races polled by election eve; earlier cutoffs may route them to the fundamentals tier"
    return out


def select_chamber_champions(results: ResultLookup, poll_lookup: PollLookup,
                             state_lean=None) -> dict[str, dict]:
    """Choose each chamber's champion spec by held-out log loss, evaluating
    CHAMPION_CANDIDATES under the identical walk-forward protocol. Returns
    ``{chamber: {"name", "kwargs", "scoreboard"}}``."""
    champions: dict[str, dict] = {}
    for chamber in ("house", "senate"):
        rows = historical_rows(results, poll_lookup, chamber, state_lean=state_lean)
        scoreboard = {}
        for name, kwargs in CHAMPION_CANDIDATES.items():
            scored, _ = walk_forward(rows, chamber, model_kwargs=kwargs)
            if scored:
                scoreboard[name] = metrics(scored)["log_loss"]
        if not scoreboard:
            champions[chamber] = {"name": "base", "kwargs": {}, "scoreboard": {}}
            continue
        best = min(scoreboard, key=scoreboard.get)
        champions[chamber] = {"name": best, "kwargs": CHAMPION_CANDIDATES[best],
                              "scoreboard": scoreboard}
    return champions


def run_backtests(model_version: str) -> list[dict]:
    """Persist expanding-window backtests: production model with subgroup and
    horizon breakdowns, plus every baseline under the identical protocol."""
    results = ResultLookup(store.all_results())
    poll_lookup = PollLookup(store.all_polls())
    runs = []
    comparison: dict[str, dict[str, dict]] = {}
    for chamber in ("house", "senate"):
        rows = historical_rows(results, poll_lookup, chamber)
        rows_by_key = {(r.cycle, r.seat_key): r for r in rows}
        scored, evaluated = walk_forward(rows, chamber)
        if not scored:
            continue
        summary = metrics(scored)
        by_cycle = {
            str(cycle): metrics([s for s in scored if s["cycle"] == cycle])
            for cycle in evaluated}
        nat_sigma = national_error_sigma(scored)
        if nat_sigma is not None:
            store.set_meta(f"national_sigma_{chamber}", str(round(nat_sigma, 3)))
        run = {
            "id": f"bt-{chamber}-{uuid4().hex[:10]}",
            "run_at": store.now(), "model_version": model_version,
            "chamber": chamber, "cycles": json.dumps(evaluated),
            "n_races": summary["n_races"], "brier": summary["brier"],
            "log_loss": summary["log_loss"],
            "winner_accuracy": summary["winner_accuracy"],
            "margin_mae": summary["margin_mae"], "margin_rmse": summary["margin_rmse"],
            "coverage80": summary["coverage80"], "coverage95": summary["coverage95"],
            "calibration": json.dumps(summary["calibration"]),
            "by_cycle": json.dumps(by_cycle),
            "config": json.dumps({
                "design": "expanding-window prequential",
                "poll_cutoff": "election day", "min_training_cycles": MIN_TRAINING_CYCLES,
                "subgroups": subgroup_metrics(scored, rows_by_key),
                "horizons_days_before_election": horizon_metrics(results, poll_lookup, chamber),
                "national_error_sigma_pts": nat_sigma,
                "note": "coverage reflects ingested sources; polled-race-only cycles "
                        "over-represent competitive districts. national_error_sigma_pts "
                        "is the SD of out-of-sample cycle-level mean prediction error - "
                        "used by the control simulation as the shared national-shock size, "
                        "so aggregate control probability reflects real historical "
                        "cycle-to-cycle swings rather than averaging away seat uncertainty"}),
        }
        store.save_backtest_run(run)
        runs.append(run)
        comparison.setdefault(chamber, {})[model_version] = {
            "brier": summary["brier"], "log_loss": summary["log_loss"],
            "winner_accuracy": summary["winner_accuracy"],
            "margin_mae": summary["margin_mae"], "n_races": summary["n_races"]}

        for name, model_kwargs in CHALLENGERS.items():
            challenger_scored, _ = walk_forward(rows, chamber, model_kwargs=model_kwargs)
            if challenger_scored:
                challenger_summary = metrics(challenger_scored)
                store.save_backtest_run({
                    "id": f"bt-{chamber}-{uuid4().hex[:10]}",
                    "run_at": store.now(), "model_version": name,
                    "chamber": chamber,
                    "cycles": json.dumps(sorted({s['cycle'] for s in challenger_scored})),
                    "n_races": challenger_summary["n_races"],
                    "brier": challenger_summary["brier"],
                    "log_loss": challenger_summary["log_loss"],
                    "winner_accuracy": challenger_summary["winner_accuracy"],
                    "margin_mae": challenger_summary["margin_mae"],
                    "margin_rmse": challenger_summary["margin_rmse"],
                    "coverage80": challenger_summary["coverage80"],
                    "coverage95": challenger_summary["coverage95"],
                    "calibration": json.dumps(challenger_summary["calibration"]),
                    "by_cycle": None,
                    "config": json.dumps({
                        "design": "expanding-window prequential (challenger)",
                        "model_kwargs": model_kwargs}),
                })
                comparison[chamber][name] = {
                    "brier": challenger_summary["brier"],
                    "log_loss": challenger_summary["log_loss"],
                    "winner_accuracy": challenger_summary["winner_accuracy"],
                    "margin_mae": challenger_summary["margin_mae"],
                    "n_races": challenger_summary["n_races"]}

        for name, spec in BASELINES.items():
            baseline_scored = walk_forward_baseline(
                rows, chamber, spec["features"], spec["polled_only"])
            if not baseline_scored:
                continue
            baseline_summary = metrics(baseline_scored)
            baseline_run = {
                "id": f"bt-{chamber}-{uuid4().hex[:10]}",
                "run_at": store.now(), "model_version": name,
                "chamber": chamber,
                "cycles": json.dumps(sorted({s["cycle"] for s in baseline_scored})),
                "n_races": baseline_summary["n_races"],
                "brier": baseline_summary["brier"],
                "log_loss": baseline_summary["log_loss"],
                "winner_accuracy": baseline_summary["winner_accuracy"],
                "margin_mae": baseline_summary["margin_mae"],
                "margin_rmse": baseline_summary["margin_rmse"],
                "coverage80": baseline_summary["coverage80"],
                "coverage95": baseline_summary["coverage95"],
                "calibration": json.dumps(baseline_summary["calibration"]),
                "by_cycle": None,
                "config": json.dumps({
                    "design": "expanding-window prequential (baseline)",
                    "features": [FEATURE_NAMES[i] for i in spec["features"]],
                    "polled_only": spec["polled_only"]}),
            }
            store.save_backtest_run(baseline_run)
            comparison[chamber][name] = {
                "brier": baseline_summary["brier"],
                "log_loss": baseline_summary["log_loss"],
                "winner_accuracy": baseline_summary["winner_accuracy"],
                "margin_mae": baseline_summary["margin_mae"],
                "n_races": baseline_summary["n_races"]}
    if comparison:
        store.set_meta("model_comparison", json.dumps(
            {"run_at": store.now(), "champion": model_version,
             "chambers": comparison,
             "note": "identical expanding-window protocol; baseline-polls-only "
                     "is scored on polled races only"}))
    return runs
