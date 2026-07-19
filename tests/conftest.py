import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import db  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Isolated SQLite database per test."""
    monkeypatch.delenv("DATEBASE_URL", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.sqlite"))
    db.reset_engine()
    db.init_db()
    yield
    db.reset_engine()
