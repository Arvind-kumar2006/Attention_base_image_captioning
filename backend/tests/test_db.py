"""Database portability checks.

The models must create cleanly on PostgreSQL as well as SQLite, so the DDL is
compiled against both dialects here without needing a server.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

import backend.app.db.models  # noqa: F401  (registers the mappers)
from backend.app.db.base import Base, _engine_kwargs
from backend.app.db.models import Caption, EvaluationRun, Image

TABLES = (Image, Caption, EvaluationRun)


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.__name__)
def test_create_table_compiles_for_sqlite(table):
    statement = str(CreateTable(table.__table__).compile(dialect=sqlite.dialect()))
    assert table.__tablename__ in statement


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.__name__)
def test_create_table_compiles_for_postgresql(table):
    statement = str(CreateTable(table.__table__).compile(dialect=postgresql.dialect()))
    lowered = statement.lower()
    assert table.__tablename__ in lowered

    has_json = any(
        type(column.type).__name__ == "JSON" for column in table.__table__.columns
    )
    if has_json:
        # PostgreSQL renders JSONB for the JSON type; either spelling is fine.
        assert "json" in lowered
    # portable types only: no SQLite-only spellings
    assert "datetime" not in lowered  # rendered as timestamp with time zone


def test_no_sqlite_only_types_in_models():
    """UUID and JSONB would work on postgres but break SQLite, so they are avoided."""
    for table in TABLES:
        for column in table.__table__.columns:
            type_name = type(column.type).__name__.lower()
            assert "uuid" not in type_name, (table.__name__, column.name)
            assert type_name != "jsonb", (table.__name__, column.name)
            assert type_name != "array", (table.__name__, column.name)


def test_indexes_compile_for_postgresql():
    for table in TABLES:
        for index in table.__table__.indexes:
            statement = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
            assert index.name in statement


def test_sqlite_engine_allows_cross_thread_use():
    """FastAPI runs sync endpoints in a threadpool, so check_same_thread is off."""
    kwargs = _engine_kwargs("sqlite:///./app.db")
    assert kwargs["connect_args"]["check_same_thread"] is False


def test_postgres_engine_gets_pre_ping():
    """A stale connection should be replaced rather than raising mid-request."""
    kwargs = _engine_kwargs("postgresql://user:pw@localhost:5432/lumina")
    assert kwargs["pool_pre_ping"] is True
    assert "connect_args" not in kwargs


def test_temporary_engine_creates_all_tables():
    """A real create_all against SQLite proves the mappers are consistent."""
    engine = create_engine("sqlite:///:memory:", **_engine_kwargs("sqlite:///:memory:"))
    Base.metadata.create_all(bind=engine)
    names = set(Base.metadata.tables)
    assert {"images", "captions", "evaluation_runs"} <= names
    Base.metadata.drop_all(bind=engine)
