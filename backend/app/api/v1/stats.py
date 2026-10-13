"""Aggregate statistics endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...db.base import get_db
from ...db.models import Caption, EvaluationRun
from ...schemas import EvaluationMetrics, StatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stats"])


def _as_aware(value):
    """Attach UTC to a naive timestamp from SQLite."""
    from datetime import timezone

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.get("/stats", response_model=StatsResponse)
def get_stats(session: Session = Depends(get_db)) -> StatsResponse:
    """Total captions, mean generation time, and the newest evaluation.

    ``evaluation`` is null until a run of ``ml/evaluate.py`` has been recorded
    in the ``evaluation_runs`` table.
    """
    total = session.execute(select(func.count(Caption.id))).scalar_one()
    average = session.execute(select(func.avg(Caption.generation_time_ms))).scalar_one()

    latest = (
        session.execute(
            select(EvaluationRun)
            .order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    evaluation = None
    if latest is not None:
        evaluation = EvaluationMetrics(
            id=latest.id,
            checkpoint=latest.checkpoint,
            split=latest.split,
            tokenizer=latest.tokenizer,
            num_images=latest.num_images,
            scores={key: float(value) for key, value in (latest.scores or {}).items()},
            created_at=_as_aware(latest.created_at),
        )

    return StatsResponse(
        total_captions=int(total or 0),
        avg_generation_time_ms=round(float(average), 2) if average is not None else 0.0,
        evaluation=evaluation,
    )
