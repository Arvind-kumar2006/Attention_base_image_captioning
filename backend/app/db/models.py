"""Database models: images, captions and evaluation runs.

Works on SQLite and PostgreSQL. Portable column types are used throughout —
``String`` instead of ``UUID``, ``JSON`` instead of ``JSONB`` — so the same
models create cleanly on either backend.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def new_uuid() -> str:
    """A fresh hex UUID4 as a string, portable across both databases."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class Image(Base):
    """One uploaded image on disk."""

    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    #: Path relative to UPLOAD_DIR, so moving the directory does not break rows.
    image_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    captions: Mapped[list["Caption"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class Caption(Base):
    """One generated caption with its attention map."""

    __tablename__ = "captions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    image_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    caption_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: The caption split into words.
    words: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    #: Flattened attention, num_words x 49, rounded to 4 decimals.
    attention: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    generation_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    image: Mapped[Image] = relationship(back_populates="captions")

    __table_args__ = (Index("ix_captions_created_at_id", "created_at", "id"),)


class EvaluationRun(Base):
    """A row of ``ml/evaluate.py`` results, so /stats can surface them."""

    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    checkpoint: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    split: Mapped[str] = mapped_column(String(32), nullable=False, default="test")
    tokenizer: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    num_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: BLEU-1..4, METEOR and CIDEr.
    scores: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
