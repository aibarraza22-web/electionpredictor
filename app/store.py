"""Repository functions over the database layer."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, func, insert, select, update

from . import db


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    db.init_db()


# --- meta -------------------------------------------------------------

def set_meta(key: str, value: str) -> None:
    with db.get_engine().begin() as c:
        c.execute(delete(db.meta).where(db.meta.c.key == key))
        c.execute(insert(db.meta).values(key=key, value=value))


def get_meta(key: str) -> str | None:
    with db.get_engine().connect() as c:
        row = c.execute(select(db.meta.c.value).where(db.meta.c.key == key)).fetchone()
    return row[0] if row else None


# --- provenance -------------------------------------------------------

def record_source(source: str, url: str | None, license_: str | None,
                  available_at: str, sha256: str | None, record_count: int,
                  note: str | None = None) -> int:
    with db.get_engine().begin() as c:
        result = c.execute(insert(db.raw_sources).values(
            source=source, url=url, license=license_, retrieved_at=now(),
            available_at=available_at, sha256=sha256,
            record_count=record_count, note=note))
        return result.inserted_primary_key[0]


def sources_summary() -> list[dict]:
    t = db.raw_sources
    with db.get_engine().connect() as c:
        rows = c.execute(
            select(t.c.source, func.count(), func.max(t.c.retrieved_at), func.sum(t.c.record_count))
            .group_by(t.c.source)).fetchall()
    return [{"source": s, "ingest_runs": n, "last_retrieved_at": r, "records": int(rc or 0)}
            for s, n, r, rc in rows]


# --- bulk ingest helpers ----------------------------------------------

def insert_rows(table_name: str, rows: Sequence[dict]) -> int:
    """Insert rows, ignoring duplicates (idempotent re-ingestion).

    Counts via RETURNING rather than cursor.rowcount: some drivers report -1
    (unknown) for batched ON CONFLICT DO NOTHING inserts, which would
    otherwise make successful ingestion look like it inserted nothing.
    """
    if not rows:
        return 0
    table = db.metadata.tables[table_name]
    pk_columns = list(table.primary_key.columns)
    inserted = 0
    with db.get_engine().begin() as c:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start:chunk_start + 500]
            statement = db.insert_ignore(table).returning(*pk_columns)
            result = c.execute(statement, chunk)
            inserted += len(result.fetchall())
    return inserted


# --- reads used by features/backtests ---------------------------------

def all_results(chamber: str | None = None) -> list[dict]:
    t = db.election_results
    q = select(t)
    if chamber:
        q = q.where(t.c.chamber == chamber)
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(q)]


def all_polls(chamber: str | None = None) -> list[dict]:
    t = db.polls
    q = select(t)
    if chamber:
        q = q.where(t.c.chamber == chamber)
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(q)]


def polls_for_seat(seat_key: str, cycle: int) -> list[dict]:
    t = db.polls
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)
                         .order_by(t.c.poll_date)).fetchall()
    return [dict(r._mapping) for r in rows]


def all_incumbents(cycle: int) -> dict[str, dict]:
    t = db.incumbents
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).where(t.c.cycle == cycle)).fetchall()
    return {r.seat_key: dict(r._mapping) for r in rows}


def all_finance(cycle: int) -> list[dict]:
    """Every finance row for a cycle (used for data-grade coverage)."""
    t = db.finance
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(select(t).where(t.c.cycle == cycle))]


def finance_for_seat(seat_key: str, cycle: int) -> list[dict]:
    t = db.finance
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)).fetchall()
    return [dict(r._mapping) for r in rows]


def upsert_finance_latest(rows: Sequence[dict]) -> int:
    """Maintain the legacy one-row-per-candidate finance view.

    Earlier code used INSERT IGNORE, which silently froze a candidate at the
    first report ever ingested.  Updates are now keyed by cycle/seat/name;
    immutable reporting vintages live in ``finance_snapshots`` below.
    """
    if not rows:
        return 0
    t = db.finance
    changed = 0
    allowed = {c.name for c in t.columns if c.name != "id"}
    with db.get_engine().begin() as c:
        for raw in rows:
            row = {k: v for k, v in raw.items() if k in allowed}
            key = (t.c.cycle == row["cycle"], t.c.seat_key == row["seat_key"],
                   t.c.candidate == row.get("candidate"))
            existing = c.execute(select(t.c.id).where(*key)).fetchone()
            if existing:
                c.execute(update(t).where(t.c.id == existing.id).values(**row))
            else:
                c.execute(insert(t).values(**row))
            changed += 1
    return changed


def finance_history_for_seat(seat_key: str, cycle: int,
                             as_of: str | None = None) -> list[dict]:
    t = db.finance_snapshots
    q = select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.coverage_end <= as_of[:10], t.c.retrieved_at <= as_of)
    q = q.order_by(t.c.coverage_end, t.c.retrieved_at, t.c.id)
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(q)]


def all_finance_snapshots(cycle: int, as_of: str | None = None) -> list[dict]:
    t = db.finance_snapshots
    q = select(t).where(t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.coverage_end <= as_of[:10], t.c.retrieved_at <= as_of)
    with db.get_engine().connect() as c:
        rows = c.execute(q.order_by(t.c.seat_key, t.c.coverage_end,
                                    t.c.retrieved_at, t.c.id)).fetchall()
    return [dict(r._mapping) for r in rows]


def profiles_for_seat(seat_key: str, cycle: int,
                      as_of: str | None = None) -> list[dict]:
    t = db.candidate_profiles
    q = select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.available_at <= as_of)
    with db.get_engine().connect() as c:
        rows = c.execute(q.order_by(t.c.available_at, t.c.id)).fetchall()
    return [dict(r._mapping) for r in rows]


def all_candidate_profiles(cycle: int, as_of: str | None = None) -> list[dict]:
    t = db.candidate_profiles
    q = select(t).where(t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.available_at <= as_of)
    with db.get_engine().connect() as c:
        rows = c.execute(q.order_by(t.c.seat_key, t.c.available_at, t.c.id)).fetchall()
    return [dict(r._mapping) for r in rows]


def events_for_seat(seat_key: str, cycle: int,
                    as_of: str | None = None) -> list[dict]:
    t = db.campaign_events
    q = select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.available_at <= as_of)
    with db.get_engine().connect() as c:
        rows = c.execute(q.order_by(t.c.available_at, t.c.id)).fetchall()
    return [dict(r._mapping) for r in rows]


def all_campaign_events(cycle: int, as_of: str | None = None) -> list[dict]:
    t = db.campaign_events
    q = select(t).where(t.c.cycle == cycle)
    if as_of:
        q = q.where(t.c.available_at <= as_of)
    with db.get_engine().connect() as c:
        rows = c.execute(q.order_by(t.c.seat_key, t.c.available_at, t.c.id)).fetchall()
    return [dict(r._mapping) for r in rows]


def start_ingestion_run(source: str, details: dict | None = None) -> int:
    with db.get_engine().begin() as c:
        result = c.execute(insert(db.ingestion_runs).values(
            source=source, started_at=now(), status="running",
            details=json.dumps(details or {})))
        return result.inserted_primary_key[0]


def finish_ingestion_run(run_id: int, status: str, records_seen: int = 0,
                         records_inserted: int = 0, error: str | None = None,
                         details: dict | None = None) -> None:
    t = db.ingestion_runs
    with db.get_engine().begin() as c:
        c.execute(update(t).where(t.c.id == run_id).values(
            completed_at=now(), status=status, records_seen=records_seen,
            records_inserted=records_inserted, error=error,
            details=json.dumps(details or {})))


def ingestion_health() -> list[dict]:
    """Latest attempted run per source, including failures and skips."""
    t = db.ingestion_runs
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).order_by(t.c.started_at.desc(), t.c.id.desc())).fetchall()
    latest: dict[str, dict] = {}
    for row in rows:
        item = dict(row._mapping)
        latest.setdefault(item["source"], item)
    return list(latest.values())


def data_fingerprint() -> str:
    """Stable digest of data contents used to decide whether a run changed."""
    payload: dict[str, list[dict]] = {}
    input_tables = (db.election_results, db.polls, db.incumbents,
                    db.finance_snapshots, db.candidate_profiles, db.campaign_events)
    with db.get_engine().connect() as c:
        for table in input_tables:
            items = []
            for result in c.execute(select(table)):
                row = dict(result._mapping)
                for volatile in ("id", "source_id", "retrieved_at"):
                    row.pop(volatile, None)
                items.append(row)
            payload[table.name] = sorted(items, key=lambda x: json.dumps(
                x, sort_keys=True, default=str))
    payload["meta"] = {
        "senate_dem_seats_not_up": get_meta("senate_dem_seats_not_up"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# --- races ------------------------------------------------------------

def upsert_races(rows: Sequence[dict]) -> None:
    t = db.races
    with db.get_engine().begin() as c:
        for row in rows:
            existing = c.execute(select(t.c.id).where(t.c.id == row["id"])).fetchone()
            if existing:
                c.execute(update(t).where(t.c.id == row["id"]).values(**row))
            else:
                c.execute(insert(t).values(**row))


def list_races(chamber: str | None = None, state: str | None = None) -> list[dict]:
    t = db.races
    q = select(t).order_by(t.c.id)
    if chamber:
        q = q.where(t.c.chamber == chamber)
    if state:
        q = q.where(t.c.state == state)
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(q)]


def get_race(race_id: str) -> dict | None:
    t = db.races
    with db.get_engine().connect() as c:
        row = c.execute(select(t).where(t.c.id == race_id)).fetchone()
    return dict(row._mapping) if row else None


# --- forecasts (immutable snapshots) ----------------------------------

def insert_forecasts(rows: Sequence[dict]) -> int:
    return insert_rows("forecasts", rows)


def latest_champion_version(fallback: str | None = None) -> str | None:
    """Newest champion model version actually present in ``forecasts``.

    The API must not pin reads to a compile-time constant: the pipeline
    (GitHub Actions, running the latest default branch) writes snapshots under
    whatever MODEL_VERSION its checkout has, while the deployed serverless app
    may still be running an older build. When those disagree the app queries a
    version the pipeline has moved past and silently serves stale numbers --
    the site appears frozen even though fresh forecasts exist. Resolving the
    version from the data instead makes new pipeline output surface without
    waiting on a redeploy.

    Baseline/challenger rows are excluded; among champion rows the most recent
    ``as_of`` wins, with a numeric (not lexicographic -- 2026.9 < 2026.11)
    version comparison as the tie-break.
    """
    f = db.forecasts
    with db.get_engine().connect() as c:
        rows = c.execute(
            select(f.c.model_version, func.max(f.c.as_of).label("as_of"))
            .group_by(f.c.model_version)).fetchall()
    champions = [r for r in rows
                 if not str(r.model_version).startswith(("baseline", "challenger"))]
    if not champions:
        return fallback

    def sort_key(row):
        parts = []
        for piece in str(row.model_version).split("."):
            parts.append(int(piece) if piece.isdigit() else 0)
        return (row.as_of or "", parts)

    return max(champions, key=sort_key).model_version


def latest_forecasts(chamber: str | None = None,
                     model_version: str | None = None) -> list[dict]:
    """All snapshots from the most recent as_of date (optionally one model)."""
    f, r = db.forecasts, db.races
    latest_q = select(func.max(f.c.as_of))
    if model_version:
        latest_q = latest_q.where(f.c.model_version == model_version)
    latest_as_of = latest_q.scalar_subquery()
    q = (select(f).join(r, r.c.id == f.c.race_id).where(f.c.as_of == latest_as_of))
    if model_version:
        q = q.where(f.c.model_version == model_version)
    if chamber:
        q = q.where(r.c.chamber == chamber)
    with db.get_engine().connect() as c:
        return [dict(x._mapping) for x in c.execute(q)]


def models_for_race(race_id: str) -> list[dict]:
    """Every model's latest snapshot for one race (champion + alternatives)."""
    f = db.forecasts
    latest_as_of = (select(func.max(f.c.as_of))
                    .where(f.c.race_id == race_id).scalar_subquery())
    with db.get_engine().connect() as c:
        rows = c.execute(select(f).where(f.c.race_id == race_id,
                                         f.c.as_of == latest_as_of)
                         .order_by(f.c.model_version)).fetchall()
    return [dict(r._mapping) for r in rows]


def forecast_history(race_id: str) -> list[dict]:
    f = db.forecasts
    with db.get_engine().connect() as c:
        rows = c.execute(select(f).where(f.c.race_id == race_id).order_by(f.c.as_of)).fetchall()
    return [dict(r._mapping) for r in rows]


def latest_forecast(race_id: str, model_version: str | None = None) -> dict | None:
    f = db.forecasts
    q = select(f).where(f.c.race_id == race_id)
    if model_version:
        q = q.where(f.c.model_version == model_version)
    with db.get_engine().connect() as c:
        row = c.execute(q.order_by(f.c.as_of.desc(), f.c.id.desc()).limit(1)).fetchone()
    return dict(row._mapping) if row else None


# --- control snapshots --------------------------------------------------

def save_control_snapshot(as_of: str, chamber: str, model_version: str,
                          data_version: str, payload: dict) -> None:
    insert_rows("control_snapshots", [{
        "as_of": as_of, "chamber": chamber, "model_version": model_version,
        "data_version": data_version, "payload": json.dumps(payload)}])


def latest_control_snapshot(chamber: str) -> dict | None:
    t = db.control_snapshots
    with db.get_engine().connect() as c:
        row = c.execute(select(t).where(t.c.chamber == chamber)
                        .order_by(t.c.as_of.desc(), t.c.id.desc()).limit(1)).fetchone()
    if not row:
        return None
    out = dict(row._mapping)
    out["payload"] = json.loads(out["payload"])
    return out


# --- backtests ----------------------------------------------------------

def save_backtest_run(row: dict) -> None:
    insert_rows("backtest_runs", [row])


def list_backtest_runs() -> list[dict]:
    t = db.backtest_runs
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).order_by(t.c.run_at.desc())).fetchall()
    return [dict(r._mapping) for r in rows]


def get_backtest_run(run_id: str) -> dict | None:
    t = db.backtest_runs
    with db.get_engine().connect() as c:
        row = c.execute(select(t).where(t.c.id == run_id)).fetchone()
    return dict(row._mapping) if row else None


# --- models / research / audit ------------------------------------------

def upsert_model_version(row: dict) -> None:
    t = db.model_versions
    with db.get_engine().begin() as c:
        if c.execute(select(t.c.id).where(t.c.id == row["id"])).fetchone():
            c.execute(update(t).where(t.c.id == row["id"]).values(**row))
        else:
            c.execute(insert(t).values(**row))


def list_model_versions() -> list[dict]:
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(select(db.model_versions))]


def seed_research_claims(rows: Iterable[dict]) -> None:
    insert_rows("research_claims", list(rows))


def seed_research_evidence(rows: Iterable[dict]) -> None:
    insert_rows("research_evidence", list(rows))


def list_research_claims() -> list[dict]:
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(select(db.research_claims))]


def list_research_evidence() -> list[dict]:
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(select(db.research_evidence))]


def get_research_claim(claim_id: str) -> dict | None:
    t = db.research_claims
    with db.get_engine().connect() as c:
        row = c.execute(select(t).where(t.c.id == claim_id)).fetchone()
    return dict(row._mapping) if row else None


def audit(actor: str, action: str, reason: str,
          previous_value: Any = None, new_value: Any = None) -> None:
    with db.get_engine().begin() as c:
        c.execute(insert(db.audit_logs).values(
            actor=actor, action=action, reason=reason,
            previous_value=json.dumps(previous_value) if previous_value is not None else None,
            new_value=json.dumps(new_value) if new_value is not None else None,
            created_at=now()))


def counts() -> dict[str, int]:
    out = {}
    with db.get_engine().connect() as c:
        for name in ("election_results", "polls", "incumbents", "finance",
                     "finance_snapshots", "candidate_profiles", "campaign_events",
                     "ingestion_runs", "races", "forecasts", "backtest_runs",
                     "research_evidence"):
            out[name] = c.execute(select(func.count()).select_from(db.metadata.tables[name])).scalar_one()
    return out
