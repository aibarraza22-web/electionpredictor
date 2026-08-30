"""Campaign-development context kept separate from the production margin.

The project's own vintage-safe test found that a simple receipts disparity
made forecasts worse (research claim P-005).  This module fixes the data
foundation needed for better tests: reporting vintages, stage, velocity,
cash, burn, opponent-relative comparisons, candidate observations, and an
event ledger.  Until a challenger wins held-out validation, these signals are
shown transparently but contribute exactly zero points to the champion.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
from math import erf, log, sqrt


def _day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _number(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 3) if b > 0 else None


def _latest_candidate_rows(rows: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row.get("candidate_id") or row.get("candidate"))].append(row)
    latest = []
    for candidate_rows in by_candidate.values():
        candidate_rows.sort(key=lambda r: (r.get("coverage_end") or "",
                                           r.get("retrieved_at") or "", r.get("id") or 0))
        latest.append(candidate_rows[-1])
    return latest, by_candidate


def finance_context(rows: list[dict], as_of: str, election_date: str) -> dict:
    """Opponent-relative and stage-aware finance description.

    The resulting ``signal`` is descriptive, not a vote-margin adjustment.
    It is bounded and exposed only to help identify where campaign capacity
    differs from the structural model while richer historical vintages build.
    """
    cutoff = _day(as_of) or date.today()
    election = _day(election_date) or date(cutoff.year, 11, 3)
    latest, histories = _latest_candidate_rows(rows)
    candidates = []
    for row in latest:
        candidate_id = str(row.get("candidate_id") or row.get("candidate"))
        receipts = _number(row.get("receipts"))
        spending = _number(row.get("disbursements"))
        cash = _number(row.get("cash_on_hand"))
        history = histories[candidate_id]
        velocity = None
        if len(history) > 1:
            previous = next((x for x in reversed(history[:-1])
                             if x.get("coverage_end") != row.get("coverage_end")), None)
            days = ((_day(row.get("coverage_end")) or cutoff)
                    - (_day(previous.get("coverage_end")) if previous else cutoff)).days
            if previous and days > 0:
                velocity = max(0.0, receipts - _number(previous.get("receipts"))) / days
        start = _day(row.get("coverage_start"))
        elapsed = max(1, (( _day(row.get("coverage_end")) or cutoff) - start).days) if start else None
        candidates.append({
            "candidate_id": candidate_id,
            "candidate": row.get("candidate"), "party": row.get("party"),
            "coverage_end": row.get("coverage_end"),
            "receipts": round(receipts, 2), "disbursements": round(spending, 2),
            "cash_on_hand": round(cash, 2),
            "debts_owed": round(_number(row.get("debts_owed")), 2),
            "burn_rate": round(spending / receipts, 3) if receipts > 0 else None,
            "receipts_per_campaign_day": round(receipts / elapsed, 2) if elapsed else None,
            "latest_period_receipts_per_day": round(velocity, 2) if velocity is not None else None,
            "individual_share": round(
                _number(row.get("individual_contributions")) / receipts, 3) if receipts > 0 else None,
            "snapshot_count": len(history),
        })
    candidates.sort(key=lambda x: x["receipts"], reverse=True)

    def leader(party: str) -> dict | None:
        return next((c for c in candidates if c.get("party") == party), None)

    dem, rep = leader("D"), leader("R")
    comparison = None
    if dem and rep:
        dem_v = dem.get("latest_period_receipts_per_day") or 0.0
        rep_v = rep.get("latest_period_receipts_per_day") or 0.0
        # Bounded descriptive score.  It is deliberately NOT multiplied by a
        # margin coefficient anywhere in production.
        log_receipts = log((dem["receipts"] + 1) / (rep["receipts"] + 1))
        log_cash = log((dem["cash_on_hand"] + 1) / (rep["cash_on_hand"] + 1))
        log_velocity = log((dem_v + 1) / (rep_v + 1))
        signal = max(-1.0, min(1.0, (0.45 * log_receipts + 0.35 * log_cash
                                    + 0.20 * log_velocity) / 3.0))
        comparison = {
            "dem_candidate": dem["candidate"], "rep_candidate": rep["candidate"],
            "dem_to_rep_receipts_ratio": _ratio(dem["receipts"], rep["receipts"]),
            "dem_to_rep_cash_ratio": _ratio(dem["cash_on_hand"], rep["cash_on_hand"]),
            "dem_minus_rep_receipts": round(dem["receipts"] - rep["receipts"], 2),
            "dem_minus_rep_cash": round(dem["cash_on_hand"] - rep["cash_on_hand"], 2),
            "descriptive_campaign_signal": round(signal, 3),
        }
    days_out = (election - cutoff).days
    if days_out > 180:
        stage = "early"
    elif days_out > 60:
        stage = "middle"
    elif days_out > 14:
        stage = "late"
    else:
        stage = "closing"
    return {
        "status": "context-only-unvalidated",
        "production_margin_adjustment": 0.0,
        "reason": "Simple finance adjustments failed vintage-safe held-out tests; retain as context until a staged challenger wins.",
        "campaign_stage": stage, "days_until_election": days_out,
        "comparison": comparison, "candidates": candidates,
    }


def candidate_context(rows: list[dict]) -> dict:
    by_candidate: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("candidate_id"))
        entry = by_candidate.setdefault(key, {
            "candidate_id": key, "candidate": row.get("candidate"),
            "party": row.get("party"), "observations": []})
        entry["observations"].append({
            "type": row.get("profile_type"), "value": row.get("value"),
            "observed_at": row.get("observed_at"),
            "available_at": row.get("available_at"),
            "source_url": row.get("source_url"),
        })
    return {"status": "auditable-observations-not-scored",
            "candidates": list(by_candidate.values())}


def event_context(rows: list[dict]) -> dict:
    return {
        "status": "source-backed-ledger",
        "events": [{
            "external_id": r.get("external_id"), "candidate_id": r.get("candidate_id"),
            "event_type": r.get("event_type"), "event_date": r.get("event_date"),
            "available_at": r.get("available_at"), "reliability": r.get("reliability"),
            "model_eligible": bool(r.get("model_eligible")),
            "details": _json_value(r.get("details")), "source_url": r.get("source_url"),
        } for r in rows],
    }


def _json_value(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _normal_cdf(z: float) -> float:
    # Same stable approximation family used by app.domain, kept local to
    # avoid changing the production probability path.
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def victory_bands(mean: float, sigma: float, dem_probability: float) -> dict:
    """Party win-size probabilities reconciled to calibrated win totals."""
    sigma = max(float(sigma), 1e-6)
    raw_dem = max(1e-9, 1.0 - _normal_cdf((0.0 - mean) / sigma))
    raw_rep = max(1e-9, 1.0 - raw_dem)
    dem_ge4 = 1.0 - _normal_cdf((4.0 - mean) / sigma)
    dem_ge8 = 1.0 - _normal_cdf((8.0 - mean) / sigma)
    rep_ge4 = _normal_cdf((-4.0 - mean) / sigma)
    rep_ge8 = _normal_cdf((-8.0 - mean) / sigma)

    def scaled(value: float, raw_total: float, calibrated_total: float) -> float:
        return max(0.0, min(calibrated_total, value / raw_total * calibrated_total))

    d4, d8 = scaled(dem_ge4, raw_dem, dem_probability), scaled(dem_ge8, raw_dem, dem_probability)
    rp = 1.0 - dem_probability
    r4, r8 = scaled(rep_ge4, raw_rep, rp), scaled(rep_ge8, raw_rep, rp)
    return {
        "dem_narrow_0_to_4": round(max(0.0, dem_probability - d4), 4),
        "dem_by_at_least_4": round(d4, 4),
        "dem_by_at_least_8": round(d8, 4),
        "rep_narrow_0_to_4": round(max(0.0, rp - r4), 4),
        "rep_by_at_least_4": round(r4, 4),
        "rep_by_at_least_8": round(r8, 4),
    }


def forecast_analysis(mean: float, sigma: float, dem_probability: float,
                      components: dict, finance: dict, candidates: dict,
                      events: dict, previous: dict | None = None) -> dict:
    polling = sum(float(components.get(k) or 0.0)
                  for k in ("poll_average", "has_polls"))
    change = None
    if previous:
        change = {
            "margin_points": round(mean - float(previous.get("margin") or 0.0), 2),
            "dem_probability_points": round(
                100 * (dem_probability - float(previous.get("dem_probability") or 0.0)), 2),
        }
        change["material"] = (abs(change["margin_points"]) >= 0.5 or
                              abs(change["dem_probability_points"]) >= 2.0)
    return {
        "structural_baseline_margin": round(mean - polling, 2),
        "polling_adjustment": round(polling, 2),
        "campaign_adjustment": 0.0,
        "campaign_adjustment_status": "withheld-pending-out-of-sample-validation",
        "final_margin": round(mean, 2),
        "victory_bands": victory_bands(mean, sigma, dem_probability),
        "finance": finance, "candidate_quality": candidates,
        "events": events, "change_since_previous": change,
    }
