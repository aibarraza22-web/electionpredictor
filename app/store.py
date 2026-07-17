"""Repository functions over the database layer."""
from __future__ import annotations

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
    """Insert rows, ignoring duplicates (idempotent re-ingestion)."""
    if not rows:
        return 0
    table = db.metadata.tables[table_name]
    inserted = 0
    with db.get_engine().begin() as c:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start:chunk_start + 500]
            result = c.execute(db.insert_ignore(table), chunk)
            inserted += result.rowcount if result.rowcount and result.rowcount > 0 else 0
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


def finance_for_seat(seat_key: str, cycle: int) -> list[dict]:
    t = db.finance
    with db.get_engine().connect() as c:
        rows = c.execute(select(t).where(t.c.seat_key == seat_key, t.c.cycle == cycle)).fetchall()
    return [dict(r._mapping) for r in rows]


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


def latest_forecasts(chamber: str | None = None) -> list[dict]:
    """All snapshots from the most recent as_of date."""
    f, r = db.forecasts, db.races
    latest_as_of = select(func.max(f.c.as_of)).scalar_subquery()
    q = (select(f).join(r, r.c.id == f.c.race_id).where(f.c.as_of == latest_as_of))
    if chamber:
        q = q.where(r.c.chamber == chamber)
    with db.get_engine().connect() as c:
        return [dict(x._mapping) for x in c.execute(q)]


def forecast_history(race_id: str) -> list[dict]:
    f = db.forecasts
    with db.get_engine().connect() as c:
        rows = c.execute(select(f).where(f.c.race_id == race_id).order_by(f.c.as_of)).fetchall()
    return [dict(r._mapping) for r in rows]


def latest_forecast(race_id: str) -> dict | None:
    f = db.forecasts
    with db.get_engine().connect() as c:
        row = c.execute(select(f).where(f.c.race_id == race_id)
                        .order_by(f.c.as_of.desc(), f.c.id.desc()).limit(1)).fetchone()
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


def list_research_claims() -> list[dict]:
    with db.get_engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(select(db.research_claims))]


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
                     "races", "forecasts", "backtest_runs"):
            out[name] = c.execute(select(func.count()).select_from(db.metadata.tables[name])).scalar_one()
    return out
