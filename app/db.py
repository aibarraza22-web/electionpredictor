"""Database layer.

Production persistence is PostgreSQL, selected with ``DATEBASE_URL``
(e.g. Neon, Vercel Postgres, RDS). Without it the app falls back to a local
SQLite file so development and tests need no external service. On Vercel the
filesystem is ephemeral, so a SQLite fallback there is explicitly flagged as
non-durable by ``/api/data-health``.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Float, Integer, MetaData, Table, Text, UniqueConstraint,
    create_engine, event,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

meta = Table(
    "meta", metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text),
)

raw_sources = Table(
    "raw_sources", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("url", Text),
    Column("license", Text),
    Column("retrieved_at", Text, nullable=False),
    Column("available_at", Text, nullable=False),
    Column("sha256", Text),
    Column("record_count", Integer),
    Column("note", Text),
)

election_results = Table(
    "election_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle", Integer, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("district", Text),
    Column("seat_key", Text, nullable=False),
    Column("dem_pct", Float),
    Column("rep_pct", Float),
    Column("dem_margin", Float, nullable=False),  # two-party margin, pct points
    Column("winner_party", Text),
    Column("source", Text, nullable=False),
    Column("source_id", Integer),
    UniqueConstraint("cycle", "seat_key", "source", name="uq_result_cycle_seat_source"),
)

polls = Table(
    "polls", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("external_id", Text),
    Column("cycle", Integer, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("district", Text),
    Column("seat_key", Text, nullable=False),
    Column("pollster", Text),
    Column("methodology", Text),
    Column("sample_size", Float),
    Column("partisan", Text),
    Column("poll_date", Text, nullable=False),
    Column("election_date", Text),
    Column("dem_pct", Float),
    Column("rep_pct", Float),
    Column("dem_margin", Float, nullable=False),
    Column("source", Text, nullable=False),
    Column("source_id", Integer),
    UniqueConstraint("source", "external_id", "seat_key", name="uq_poll_source_external"),
)

incumbents = Table(
    "incumbents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle", Integer, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("district", Text),
    Column("seat_key", Text, nullable=False),
    Column("party", Text),
    Column("name", Text),
    Column("source", Text, nullable=False),
    UniqueConstraint("cycle", "seat_key", name="uq_incumbent_cycle_seat"),
)

finance = Table(
    "finance", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle", Integer, nullable=False),
    Column("seat_key", Text, nullable=False),
    Column("candidate", Text),
    Column("party", Text),
    Column("receipts", Float),
    Column("disbursements", Float),
    Column("cash_on_hand", Float),
    Column("as_of", Text),
    Column("source", Text, nullable=False),
    UniqueConstraint("cycle", "seat_key", "candidate", name="uq_finance_candidate"),
)

# Append-only FEC vintages.  ``finance`` is retained as the compact latest
# view used by older deployments; it cannot represent amendments or the
# evolution of a campaign because its uniqueness key contains no reporting
# date.  This table is the research-safe source for stage/velocity features.
finance_snapshots = Table(
    "finance_snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle", Integer, nullable=False),
    Column("seat_key", Text, nullable=False),
    Column("candidate_id", Text, nullable=False),
    Column("committee_id", Text),
    Column("candidate", Text),
    Column("party", Text),
    Column("receipts", Float),
    Column("disbursements", Float),
    Column("cash_on_hand", Float),
    Column("individual_contributions", Float),
    Column("other_committee_contributions", Float),
    Column("candidate_contributions", Float),
    Column("debts_owed", Float),
    Column("coverage_start", Text),
    Column("coverage_end", Text, nullable=False),
    Column("retrieved_at", Text, nullable=False),
    Column("payload_hash", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("source_id", Integer),
    UniqueConstraint("cycle", "candidate_id", "coverage_end", "payload_hash",
                     name="uq_finance_snapshot_vintage"),
)

candidate_profiles = Table(
    "candidate_profiles", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle", Integer, nullable=False),
    Column("seat_key", Text, nullable=False),
    Column("candidate_id", Text, nullable=False),
    Column("candidate", Text),
    Column("party", Text),
    Column("profile_type", Text, nullable=False),
    Column("value", Float),
    Column("observed_at", Text, nullable=False),
    Column("available_at", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("source_url", Text),
    Column("source_id", Integer),
    UniqueConstraint("cycle", "candidate_id", "profile_type", "available_at", "source",
                     name="uq_candidate_profile_observation"),
)

campaign_events = Table(
    "campaign_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("external_id", Text, nullable=False),
    Column("cycle", Integer, nullable=False),
    Column("seat_key", Text, nullable=False),
    Column("candidate_id", Text),
    Column("event_type", Text, nullable=False),
    Column("event_date", Text, nullable=False),
    Column("available_at", Text, nullable=False),
    Column("reliability", Text, nullable=False),
    Column("model_eligible", Boolean, default=False),
    Column("details", Text),
    Column("source", Text, nullable=False),
    Column("source_url", Text),
    Column("source_id", Integer),
    UniqueConstraint("source", "external_id", name="uq_campaign_event_source"),
)

ingestion_runs = Table(
    "ingestion_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("started_at", Text, nullable=False),
    Column("completed_at", Text),
    Column("status", Text, nullable=False),
    Column("records_seen", Integer),
    Column("records_inserted", Integer),
    Column("error", Text),
    Column("details", Text),
)

races = Table(
    "races", metadata,
    Column("id", Text, primary_key=True),
    Column("cycle", Integer, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("district", Text),
    Column("seat_key", Text, nullable=False),
    Column("name", Text),
    Column("incumbent_party", Text),
    Column("incumbent_name", Text),
    Column("open_seat", Boolean, default=False),
    Column("special", Boolean, default=False),
    Column("election_system", Text, default="plurality"),
    Column("data_version", Text),
    Column("updated_at", Text),
)

forecasts = Table(
    "forecasts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", Text, nullable=False),
    Column("as_of", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("data_version", Text, nullable=False),
    Column("dem_probability", Float, nullable=False),
    Column("margin", Float, nullable=False),
    Column("low80", Float), Column("high80", Float),
    Column("low95", Float), Column("high95", Float),
    Column("rating", Text),
    Column("quality", Text),
    Column("components", Text),  # JSON
    Column("immutable", Boolean, default=True),
    UniqueConstraint("race_id", "as_of", "model_version", name="uq_forecast_snapshot"),
)

control_snapshots = Table(
    "control_snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("as_of", Text, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("data_version", Text),
    Column("payload", Text, nullable=False),  # JSON
    UniqueConstraint("as_of", "chamber", "model_version", name="uq_control_snapshot"),
)

backtest_runs = Table(
    "backtest_runs", metadata,
    Column("id", Text, primary_key=True),
    Column("run_at", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("chamber", Text, nullable=False),
    Column("cycles", Text),        # JSON list of evaluated cycles
    Column("n_races", Integer),
    Column("brier", Float),
    Column("log_loss", Float),
    Column("winner_accuracy", Float),
    Column("margin_mae", Float),
    Column("margin_rmse", Float),
    Column("coverage80", Float),
    Column("coverage95", Float),
    Column("calibration", Text),   # JSON bins
    Column("by_cycle", Text),      # JSON per-cycle metrics
    Column("config", Text),        # JSON
)

model_versions = Table(
    "model_versions", metadata,
    Column("id", Text, primary_key=True),
    Column("chamber", Text),
    Column("status", Text),
    Column("created_at", Text),
    Column("description", Text),
    Column("coefficients", Text),  # JSON, refit artifacts
)

research_claims = Table(
    "research_claims", metadata,
    Column("id", Text, primary_key=True),
    Column("claim", Text),
    Column("chamber", Text),
    Column("metric", Text),
    Column("mechanism", Text),
    Column("status", Text),
    Column("validation", Text),
    Column("decision", Text),
    Column("source", Text),
)

research_evidence = Table(
    "research_evidence", metadata,
    Column("id", Text, primary_key=True),
    Column("claim_id", Text, nullable=False),
    Column("citation", Text, nullable=False),
    Column("source_url", Text, nullable=False),
    Column("data_period", Text),
    Column("interpretation", Text),
    Column("expected_mechanism", Text),
    Column("proposed_feature", Text),
    Column("leakage_risk", Text),
    Column("validation_test", Text),
    Column("result", Text),
    Column("decision", Text),
)

audit_logs = Table(
    "audit_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("actor", Text),
    Column("action", Text),
    Column("reason", Text),
    Column("previous_value", Text),
    Column("new_value", Text),
    Column("created_at", Text),
)

_engine: Engine | None = None


def database_url() -> str:
    url = os.getenv("DATEBASE_URL", "").strip()
    if url:
        # Accept the common postgres:// shorthand used by managed providers.
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url
    if os.getenv("VERCEL"):
        return "sqlite:////tmp/forecast_lab.sqlite"
    path = Path(os.getenv("SQLITE_PATH", Path(__file__).parents[1] / "data" / "forecast_lab.sqlite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def is_durable() -> bool:
    """True when persistence survives serverless instance recycling."""
    return backend() == "postgresql" or not os.getenv("VERCEL")


def backend() -> str:
    return get_engine().dialect.name


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = database_url()
        kwargs = {"pool_pre_ping": True} if url.startswith("postgresql") else {}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
    return _engine


def reset_engine() -> None:
    """Dispose the cached engine (used by tests switching DATEBASE_URL)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    metadata.create_all(get_engine())


def insert_ignore(table: Table):
    """Dialect-appropriate INSERT .. ON CONFLICT DO NOTHING statement."""
    if backend() == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        return pg_insert(table).on_conflict_do_nothing()
    from sqlalchemy.dialects.sqlite import insert as sq_insert
    return sq_insert(table).on_conflict_do_nothing()
