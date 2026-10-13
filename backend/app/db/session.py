"""Database bootstrap: create tables on startup."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base import Base, engine

logger = logging.getLogger(__name__)


def create_all(target: Engine | None = None) -> None:
    """Create any missing tables.

    Uses ``CREATE TABLE IF NOT EXISTS`` semantics, so it is safe to call on
    every startup against either SQLite or PostgreSQL.
    """
    target = target or engine
    import backend.app.db.models  # noqa: F401  (registers the mappers)

    Base.metadata.create_all(bind=target)
    logger.info("database ready: %s", _describe(target))


def _describe(target: Engine) -> str:
    """The database dialect and host, without any password in the log."""
    url = target.url
    if url.get_backend_name() == "sqlite":
        return "sqlite"
    return f"{url.get_backend_name()}://{url.host or ''}/{url.database or ''}"


def ping() -> bool:
    """True when a trivial query succeeds, i.e. the database is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as error:
        logger.warning("database ping failed: %r", error)
        return False
