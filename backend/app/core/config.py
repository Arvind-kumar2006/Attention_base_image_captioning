"""Application settings, read from the environment.

No secrets live in code. Every value has a working default except the paths
that must exist for the service to do anything useful.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Image types the API accepts. Anything else is rejected before decoding.
ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png")

#: Uploads above this size are refused with 413.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

#: Decoded images above this many pixels are refused; decompression bombs are the
#: usual way a tiny file turns into gigabytes of memory.
MAX_IMAGE_PIXELS = 40_000_000

#: Attention maps are stored flattened as num_words x 49, rounded to 4 places.
ATTENTION_DECIMALS = 4


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_path(name: str, default: str) -> Path:
    return Path(_env(name, default)).expanduser()


@dataclass
class Settings:
    """Runtime configuration, instantiated once at import time."""

    #: Where the trained checkpoint lives. Set CHECKPOINT_PATH in the environment.
    checkpoint_path: Path = field(
        default_factory=lambda: _env_path(
            "CHECKPOINT_PATH", "ml/weights/best.pt"
        )
    )
    device: str = field(default_factory=lambda: _env("DEVICE", "cpu"))

    #: Comma-separated allowed origins, or "*" for any.
    frontend_origin: str = field(
        default_factory=lambda: _env("FRONTEND_ORIGIN", "http://localhost:5173")
    )

    #: SQLite by default; any SQLAlchemy URL works, including PostgreSQL.
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite:///./app.db")
    )
    database_echo: bool = field(
        default_factory=lambda: _env("DATABASE_ECHO", "").lower() in ("1", "true", "yes")
    )

    upload_dir: Path = field(
        default_factory=lambda: _env_path("UPLOAD_DIR", "backend/uploads")
    )

    max_upload_bytes: int = field(
        default_factory=lambda: int(_env("MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES))
    )
    allowed_content_types: tuple[str, ...] = field(default_factory=lambda: ALLOWED_CONTENT_TYPES)

    @property
    def cors_origins(self) -> list[str]:
        """Origins allowed by CORS."""
        if self.frontend_origin.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.frontend_origin.split(",")
            if origin.strip()
        ]

    @property
    def is_sqlite(self) -> bool:
        """True when the database is SQLite, which needs different setup."""
        return self.database_url.startswith("sqlite")


def get_settings() -> Settings:
    """Settings for the current environment. Callers that need to re-read the
    environment (tests) get a fresh instance rather than a cached one."""
    return Settings()


settings = get_settings()
