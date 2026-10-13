"""SQLAlchemy engine, session factory and declarative base."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.config import settings


class Base(DeclarativeBase):
    """Declarative base for every model in this package."""


def _engine_kwargs(url: str) -> dict[str, Any]:
    """Engine options that differ between SQLite and server databases."""
    if url.startswith("sqlite"):
        # FastAPI serves requests from a threadpool, so the connection is used
        # from a different thread than the one that created it. SQLite's default
        # same-thread check rejects that.
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine: Engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
    **_engine_kwargs(settings.database_url),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Always closed, including when the handler raises.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
