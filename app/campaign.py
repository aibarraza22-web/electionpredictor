"""Research-informed campaign-development adjustment.

The historical test in claim P-005 rejected a naive receipts-only feature.
This module therefore does not treat money as votes.  It uses a bounded,
stage-aware capacity overlay that discounts thin, stale, or poll-absorbed
signals and separately scores auditable candidate observations and eligible
campaign events.  Ordinary campaign effects are capped at three margin
points.  Only source-backed exceptional events can expand the cap to six.

The overlay is deliberately labelled provisional because complete historical
as-of vintages do not yet exist for every input.  Its uncertainty is added to
the forecast rather than hidden.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
from math import erf, exp, log, log1p, sqrt, tanh


FINANCE_STAGE_POINTS = {
    "early": 1.75,
    "middle": 1.35,
    "late": 0.90,
    "closing": 0.45,
}

PROFILE_POINTS = {
    "incumbent": 0.35,
    "challenger": 0.0,
    "open_seat_candidate": 0.0,
    "former_officeholder": 0.55,
    "elected_official": 0.50,
    "veteran": 0.15,
    "business_executive": 0.10,
    "first_time_candidate": -0.25,
    "previous_overperformance": 0.65,
    "major_party_support": 0.45,
    "competitive_primary": 0.10,
    "uncontested_primary": 0.15,
}

EVENT_POINTS = {
    "announcement": 0.0,
    "withdrawal": -2.00,
    "replacement": -0.50,
    "primary_result": 0.20,
    "endorsement": 0.15,
    "party_intervention": 0.45,
    "independent_expenditure_surge": 0.40,
    "scandal": -1.50,
    "indictment": -2.75,
    "legal_event": -0.80,
    "health_event": -0.75,
    "ballot_access": -2.25,
    "election_system": 0.0,
    "redistricting": 0.0,
}

EXCEPTIONAL_EVENTS = {"withdrawal", "replacement", "scandal", "indictment",
                      "legal_event", "health_event", "ballot_access"}
PERSISTENT_EVENTS = {"withdrawal", "replacement", "primary_result", "ballot_access"}
RELIABILITY_WEIGHT = {"primary": 1.0, "high": .9, "medium": .6, "low": .35}
SEVERITY_WEIGHT = {"minor": .5, "moderate": 1.0, "major": 1.5,
                   "catastrophic": 2.0}


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


def finance_context(rows: list[dict], as_of: str, election_date: str,
                    chamber: str | None = None) -> dict:
    """Build a stage-aware, opponent-relative campaign-capacity adjustment."""
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
            "net_cash": round(max(0.0, cash - _number(row.get("debts_owed"))), 2),
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
    days_out = (election - cutoff).days
    if days_out > 180:
        stage = "early"
    elif days_out > 60:
        stage = "middle"
    elif days_out > 14:
        stage = "late"
    else:
        stage = "closing"

    comparison = None
    adjustment = 0.0
    if dem and rep:
        dem_v = (dem.get("latest_period_receipts_per_day")
                 or dem.get("receipts_per_campaign_day") or 0.0)
        rep_v = (rep.get("latest_period_receipts_per_day")
                 or rep.get("receipts_per_campaign_day") or 0.0)
        log_receipts = log((dem["receipts"] + 1) / (rep["receipts"] + 1))
        log_cash = log((dem["net_cash"] + 1) / (rep["net_cash"] + 1))
        log_velocity = log((dem_v + 1) / (rep_v + 1))
        individual_delta = ((dem.get("individual_share") or 0.0)
                            - (rep.get("individual_share") or 0.0))
        burn_delta = max(-1.0, min(1.0, (rep.get("burn_rate") or 0.0)
                                        - (dem.get("burn_rate") or 0.0)))
        raw_signal = (0.35 * log_receipts + 0.30 * log_cash
                      + 0.20 * log_velocity + 0.10 * individual_delta * 2.0
                      + 0.05 * burn_delta)
        signal = tanh(raw_signal / 1.25)

        threshold = 250_000.0 if chamber == "senate" else 50_000.0
        minimum_receipts = min(dem["receipts"], rep["receipts"])
        scale_credibility = min(1.0, log1p(minimum_receipts) / log1p(threshold))
        snapshot_credibility = min(1.0, .55 + .15 * min(
            dem.get("snapshot_count") or 0, rep.get("snapshot_count") or 0))
        latest_end = min(_day(dem.get("coverage_end")) or cutoff,
                         _day(rep.get("coverage_end")) or cutoff)
        age_days = max(0, (cutoff - latest_end).days)
        freshness = max(.35, exp(-age_days / 150.0))
        credibility = scale_credibility * snapshot_credibility * freshness
        adjustment = FINANCE_STAGE_POINTS[stage] * signal * credibility
        comparison = {
            "dem_candidate": dem["candidate"], "rep_candidate": rep["candidate"],
            "dem_to_rep_receipts_ratio": _ratio(dem["receipts"], rep["receipts"]),
            "dem_to_rep_cash_ratio": _ratio(dem["cash_on_hand"], rep["cash_on_hand"]),
            "dem_minus_rep_receipts": round(dem["receipts"] - rep["receipts"], 2),
            "dem_minus_rep_cash": round(dem["cash_on_hand"] - rep["cash_on_hand"], 2),
            "descriptive_campaign_signal": round(signal, 3),
            "signal_credibility": round(credibility, 3),
            "stage_point_ceiling": FINANCE_STAGE_POINTS[stage],
            "finance_margin_adjustment_before_poll_absorption": round(adjustment, 3),
        }
    return {
        "status": "active-provisional" if comparison else "insufficient-opponent-data",
        "production_margin_adjustment": round(adjustment, 3),
        "reason": "Bounded capacity signal using stage, receipts, net cash, velocity, donor mix, burn, freshness, and reporting depth.",
        "campaign_stage": stage, "days_until_election": days_out,
        "comparison": comparison, "candidates": candidates,
    }


def candidate_context(rows: list[dict],
                      party_by_candidate: dict[str, str] | None = None) -> dict:
    party_by_candidate = party_by_candidate or {}
    by_candidate: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("candidate_id"))
        entry = by_candidate.setdefault(key, {
            "candidate_id": key, "candidate": row.get("candidate"),
            "party": row.get("party") or party_by_candidate.get(key),
            "observations": [], "quality_score": 0.0})
        raw_value = row.get("value")
        value = 1.0 if raw_value is None else max(0.0, min(2.0, _number(raw_value)))
        points = PROFILE_POINTS.get(str(row.get("profile_type")), 0.0) * value
        entry["quality_score"] += points
        entry["observations"].append({
            "type": row.get("profile_type"), "value": row.get("value"),
            "margin_points": round(points, 3),
            "observed_at": row.get("observed_at"),
            "available_at": row.get("available_at"),
            "source_url": row.get("source_url"),
        })
    candidates = list(by_candidate.values())
    for item in candidates:
        item["quality_score"] = round(max(-1.5, min(1.5, item["quality_score"])), 3)
    party_scores = {
        party: max((c["quality_score"] for c in candidates if c.get("party") == party),
                   default=None)
        for party in ("D", "R")
    }
    complete = party_scores["D"] is not None and party_scores["R"] is not None
    adjustment = ((party_scores["D"] or 0.0) - (party_scores["R"] or 0.0)) if complete else 0.0
    adjustment = max(-2.25, min(2.25, adjustment))
    return {
        "status": "active-provisional" if complete else "incomplete-party-comparison",
        "production_margin_adjustment": round(adjustment, 3),
        "party_scores": party_scores,
        "candidates": candidates,
    }


def event_context(rows: list[dict], party_by_candidate: dict[str, str] | None = None,
                  as_of: str | None = None, last_poll_date: str | None = None) -> dict:
    party_by_candidate = party_by_candidate or {}
    cutoff = _day(as_of) or date.today()
    poll_day = _day(last_poll_date)
    events = []
    pre_poll = post_poll = 0.0
    exceptional = False
    for row in rows:
        details = _json_value(row.get("details"))
        details = details if isinstance(details, dict) else {}
        kind = str(row.get("event_type") or "")
        event_day = _day(row.get("event_date"))
        candidate_id = str(row.get("candidate_id") or "")
        party = (details.get("party") or details.get("beneficiary_party")
                 or party_by_candidate.get(candidate_id))
        eligible = bool(row.get("model_eligible"))
        contribution = 0.0
        if eligible and event_day and event_day <= cutoff:
            if details.get("dem_margin_points") is not None:
                contribution = max(-5.0, min(5.0, _number(details["dem_margin_points"])))
            elif party in {"D", "R"}:
                candidate_points = (_number(details.get("margin_points"))
                                    if details.get("margin_points") is not None
                                    else EVENT_POINTS.get(kind, 0.0))
                contribution = candidate_points if party == "D" else -candidate_points
            severity = SEVERITY_WEIGHT.get(str(details.get("severity") or "moderate").lower(), 1.0)
            reliability = RELIABILITY_WEIGHT.get(str(row.get("reliability") or "low").lower(), .35)
            age = max(0, (cutoff - event_day).days)
            decay = 1.0 if kind in PERSISTENT_EVENTS else exp(-age / 90.0)
            contribution *= severity * reliability * decay
        after_poll = bool(event_day and (poll_day is None or event_day > poll_day))
        if after_poll:
            post_poll += contribution
        else:
            pre_poll += contribution
        if eligible and kind in EXCEPTIONAL_EVENTS and abs(contribution) >= .5:
            exceptional = True
        events.append({
            "external_id": row.get("external_id"), "candidate_id": row.get("candidate_id"),
            "party": party, "event_type": kind, "event_date": row.get("event_date"),
            "available_at": row.get("available_at"), "reliability": row.get("reliability"),
            "model_eligible": eligible, "after_last_poll": after_poll,
            "margin_adjustment": round(contribution, 3),
            "details": details, "source_url": row.get("source_url"),
        })
    total = max(-5.0, min(5.0, pre_poll + post_poll))
    return {
        "status": "active-provisional" if any(e["margin_adjustment"] for e in events)
                  else "no-eligible-scored-events",
        "production_margin_adjustment": round(total, 3),
        "pre_poll_adjustment": round(pre_poll, 3),
        "post_poll_adjustment": round(post_poll, 3),
        "exceptional": exceptional,
        "events": events,
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


def campaign_adjustment(base_mean: float, finance: dict, candidates: dict,
                        events: dict, poll_count: int = 0,
                        poll_age_days: int | None = None) -> dict:
    """Combine campaign signals without double-counting information in polls.

    Recent, repeated polls absorb most pre-existing campaign information.
    Events after the last poll are not absorbed.  Ordinary effects are capped
    at three points; exceptional source-backed events permit a six-point cap.
    """
    if poll_count >= 4 and poll_age_days is not None and poll_age_days <= 21:
        poll_absorption = .35
    elif poll_count >= 2 and poll_age_days is not None and poll_age_days <= 45:
        poll_absorption = .55
    elif poll_count:
        poll_absorption = .70
    else:
        poll_absorption = 1.0

    # Campaign differences are most outcome-relevant in competitive races.
    # Preserve a smaller effect in safe seats rather than pretending campaign
    # capacity becomes literally irrelevant outside the battleground set.
    competitiveness = max(.25, min(1.0, 1.0 - max(0.0, abs(base_mean) - 8.0) / 32.0))
    event_relevance = max(.50, competitiveness)

    finance_points = (_number(finance.get("production_margin_adjustment"))
                      * poll_absorption * competitiveness)
    candidate_points = (_number(candidates.get("production_margin_adjustment"))
                        * poll_absorption * competitiveness)
    event_pre = (_number(events.get("pre_poll_adjustment"))
                 * poll_absorption * event_relevance)
    event_post = _number(events.get("post_poll_adjustment")) * event_relevance
    raw = finance_points + candidate_points + event_pre + event_post
    cap = 6.0 if events.get("exceptional") else 3.0
    adjustment = max(-cap, min(cap, raw))
    active_inputs = [
        name for name, value in (("finance", finance_points),
                                 ("candidate_quality", candidate_points),
                                 ("events_before_last_poll", event_pre),
                                 ("events_after_last_poll", event_post))
        if abs(value) >= .001
    ]
    # The overlay is not backed by complete historical as-of vintages. Add a
    # small independent uncertainty term whenever it changes the mean.
    uncertainty = min(2.5, .75 + .25 * abs(adjustment)) if active_inputs else 0.0
    return {
        "status": "active-research-informed-provisional" if active_inputs
                  else "no-supported-campaign-signal",
        "margin_adjustment": round(adjustment, 3),
        "raw_margin_adjustment": round(raw, 3),
        "ordinary_or_exceptional_cap": cap,
        "added_sigma": round(uncertainty, 3),
        "poll_absorption_multiplier": poll_absorption,
        "competitive_race_multiplier": round(competitiveness, 3),
        "active_inputs": active_inputs,
        "components": {
            "finance": round(finance_points, 3),
            "candidate_quality": round(candidate_points, 3),
            "events_before_last_poll": round(event_pre, 3),
            "events_after_last_poll": round(event_post, 3),
        },
    }


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


def forecast_analysis(base_mean: float, final_mean: float, sigma: float,
                      dem_probability: float,
                      components: dict, finance: dict, candidates: dict,
                      events: dict, campaign: dict,
                      previous: dict | None = None,
                      expert_ratings: dict | None = None,
                      expert_overlay: dict | None = None) -> dict:
    polling = sum(float(components.get(k) or 0.0)
                  for k in ("poll_average", "has_polls"))
    change = None
    if previous:
        change = {
            "margin_points": round(final_mean - float(previous.get("margin") or 0.0), 2),
            "dem_probability_points": round(
                100 * (dem_probability - float(previous.get("dem_probability") or 0.0)), 2),
        }
        change["material"] = (abs(change["margin_points"]) >= 0.5 or
                              abs(change["dem_probability_points"]) >= 2.0)
    overlay_shift = float((expert_overlay or {}).get("margin_shift") or 0.0)
    return {
        "structural_baseline_margin": round(base_mean - polling, 2),
        "polling_adjustment": round(polling, 2),
        "model_margin": round(base_mean, 2),
        "expert_rating_adjustment": round(overlay_shift, 2),
        "expert_ratings": expert_ratings,
        "expert_rating_overlay": expert_overlay,
        "pre_campaign_margin": round(base_mean + overlay_shift, 2),
        "campaign_adjustment": campaign["margin_adjustment"],
        "campaign_adjustment_status": campaign["status"],
        "campaign_adjustment_detail": campaign,
        "final_margin": round(final_mean, 2),
        "victory_bands": victory_bands(final_mean, sigma, dem_probability),
        "finance": finance, "candidate_quality": candidates,
        "events": events, "change_since_previous": change,
    }
