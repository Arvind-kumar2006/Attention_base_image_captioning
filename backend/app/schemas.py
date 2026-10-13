"""Pydantic request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """GET /health."""

    status: str
    model_loaded: bool
    database: Optional[bool] = None
    version: Optional[str] = None


class CaptionCreateResponse(BaseModel):
    """POST /api/v1/captions."""

    id: str
    caption: str
    words: list[str]
    #: ``(num_words, 7, 7)`` floats; every row sums to 1.
    attention: list[list[list[float]]]
    time_ms: float
    image_url: str


class CaptionItem(BaseModel):
    """One row of the caption history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    image_id: str
    caption: str
    words: list[str]
    attention: list[list[list[float]]]
    time_ms: float
    created_at: datetime
    image_url: Optional[str] = None


class CaptionPage(BaseModel):
    """A page of captions, newest first."""

    total: int
    limit: int
    offset: int
    items: list[CaptionItem]


class EvaluationMetrics(BaseModel):
    """The most recent ml/evaluate.py run, if one is recorded."""

    id: str
    checkpoint: str = ""
    split: str = "test"
    tokenizer: str = ""
    num_images: int = 0
    scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime


class StatsResponse(BaseModel):
    """GET /api/v1/stats."""

    total_captions: int
    avg_generation_time_ms: float
    evaluation: Optional[EvaluationMetrics] = None


class ErrorResponse(BaseModel):
    """Every failure in this API uses this shape."""

    detail: str
