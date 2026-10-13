"""Caption endpoints: generate, list history, fetch one.

Every handler here is a plain ``def``, not ``async def``. That is deliberate:
FastAPI runs sync endpoints in a threadpool, so the blocking torch inference
does not stall the event loop. The only awaitable in the request path is
reading the upload, which is genuine I/O.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ...db.base import get_db
from ...db.models import Caption
from ...db.models import Image as ImageModel
from ...schemas import CaptionCreateResponse, CaptionItem, CaptionPage
from ...services.caption_service import (
    CaptionOutput,
    CaptionService,
    ModelNotLoadedError,
    reshape_attention,
)
from ...services.image_utils import decode_image, read_upload, save_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/captions", tags=["captions"])

IMAGE_URL_TEMPLATE = "/api/v1/images/{image_id}"


def get_caption_service(request: Request) -> CaptionService:
    """FastAPI dependency returning the loaded model.

    Raises:
        HTTPException: 503 when the model could not be loaded at startup.
    """
    service = getattr(request.app.state, "caption_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the caption model is not loaded yet",
        )
    return service


def _require_image_row(session: Session, image_id: str) -> ImageModel:
    """Fetch the image row or raise 404."""
    row = session.get(ImageModel, image_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no image with id {image_id!r}")
    return row


def _require_caption_row(session: Session, caption_id: str) -> Caption:
    """Fetch the caption row or raise 404."""
    row = session.get(Caption, caption_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no caption with id {caption_id!r}")
    return row


@router.post(
    "",
    response_model=CaptionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "not an image, or corrupt"},
        413: {"description": "file over the size limit"},
        503: {"description": "model not loaded"},
    },
)
def create_caption(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    service: CaptionService = Depends(get_caption_service),
) -> CaptionCreateResponse:
    """Upload an image and store the caption generated for it."""
    # Reading the upload is the one genuinely async step. The handler itself is a
    # plain def, so this runs in the threadpool via anyio.
    import anyio

    data = anyio.from_thread.run(read_upload, file)

    try:
        pil_image = decode_image(data)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400, detail=f"could not read the image: {error}"
        ) from error

    stored = save_image(
        pil_image,
        data,
        original_filename=file.filename or "upload",
        content_type=(file.content_type or "image/jpeg").lower(),
    )

    try:
        output: CaptionOutput = service.generate(pil_image)
    except ModelNotLoadedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("inference failed")
        raise HTTPException(
            status_code=500, detail=f"caption generation failed: {error}"
        ) from error

    image_row = ImageModel(
        id=stored.id,
        image_path=stored.relative_path,
        filename=stored.filename,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
    )
    session.add(image_row)

    caption_row = Caption(
        image_id=stored.id,
        caption_text=output.caption,
        words=output.words,
        attention=output.flat_attention,
        generation_time_ms=output.time_ms,
    )
    session.add(caption_row)
    session.commit()
    session.refresh(caption_row)

    logger.info(
        "caption %s for image %s in %.1f ms",
        caption_row.id,
        stored.id,
        output.time_ms,
    )

    return CaptionCreateResponse(
        id=caption_row.id,
        caption=output.caption,
        words=output.words,
        attention=output.attention,
        time_ms=output.time_ms,
        image_url=IMAGE_URL_TEMPLATE.format(image_id=stored.id),
    )


@router.get("", response_model=CaptionPage)
def list_captions(
    session: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, description="case-insensitive substring match"),
) -> CaptionPage:
    """Caption history, newest first, optionally filtered by text."""
    statement = select(Caption)
    count_statement = select(func.count(Caption.id))

    if q:
        pattern = f"%{q.strip()}%"
        condition = or_(
            Caption.caption_text.ilike(pattern),
            func.lower(Caption.caption_text).like(pattern),
        )
        statement = statement.where(condition)
        count_statement = count_statement.where(condition)

    total = session.execute(count_statement).scalar_one()
    rows = (
        session.execute(
            statement.order_by(Caption.created_at.desc(), Caption.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )

    items = [
        CaptionItem(
            id=row.id,
            image_id=row.image_id,
            caption=row.caption_text,
            words=list(row.words or []),
            attention=reshape_attention(row.attention or []),
            time_ms=row.generation_time_ms,
            created_at=_as_aware(row.created_at),
            image_url=IMAGE_URL_TEMPLATE.format(image_id=row.image_id),
        )
        for row in rows
    ]
    return CaptionPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/{caption_id}", response_model=CaptionItem)
def get_caption(caption_id: str, session: Session = Depends(get_db)) -> CaptionItem:
    """Fetch one caption by id."""
    row = _require_caption_row(session, caption_id)
    return CaptionItem(
        id=row.id,
        image_id=row.image_id,
        caption=row.caption_text,
        words=list(row.words or []),
        attention=reshape_attention(row.attention or []),
        time_ms=row.generation_time_ms,
        created_at=_as_aware(row.created_at),
        image_url=IMAGE_URL_TEMPLATE.format(image_id=row.image_id),
    )


def _as_aware(value: datetime) -> datetime:
    """Attach UTC to a naive timestamp.

    SQLite hands back naive datetimes even for a timezone-aware column; treating
    them as UTC keeps the API consistent across backends.
    """
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
