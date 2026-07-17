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
from uuid import uuid4

from . import store
from .features import FeatureRow, PollLookup, ResultLookup, historical_rows
from .model import MarginModel

MIN_TRAINING_CYCLES = 3


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


def walk_forward(rows: list[FeatureRow], chamber: str,
                 min_training_cycles: int = MIN_TRAINING_CYCLES) -> tuple[list[dict], list[int]]:
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
        model = MarginModel().fit(training)
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


def run_backtests(model_version: str) -> list[dict]:
    """Run and persist expanding-window backtests for both chambers."""
    results = ResultLookup(store.all_results())
    poll_lookup = PollLookup(store.all_polls())
    runs = []
    for chamber in ("house", "senate"):
        rows = historical_rows(results, poll_lookup, chamber)
        scored, evaluated = walk_forward(rows, chamber)
        if not scored:
            continue
        summary = metrics(scored)
        by_cycle = {
            str(cycle): metrics([s for s in scored if s["cycle"] == cycle])
            for cycle in evaluated}
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
                "note": "coverage reflects ingested sources; polled-race-only cycles "
                        "over-represent competitive districts"}),
        }
        store.save_backtest_run(run)
        runs.append(run)
    return runs
