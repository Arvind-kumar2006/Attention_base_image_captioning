"""Serving stored uploads."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ...db.base import get_db
from ...db.models import Image as ImageModel
from ...services.image_utils import resolve_stored_path

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])


@router.get("/images/{image_id}")
def get_image(image_id: str, session: Session = Depends(get_db)) -> FileResponse:
    """Serve a stored upload.

    The stored bytes are returned as received, so the file on disk is what the
    client gets back.
    """
    row = session.get(ImageModel, image_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no image with id {image_id!r}")

    try:
        path = resolve_stored_path(row.image_path)
    except HTTPException:
        raise
    if not path.is_file():
        logger.warning("image row %s points at a missing file: %s", image_id, row.image_path)
        raise HTTPException(status_code=404, detail="image file is missing from storage")

    return FileResponse(path, media_type=row.content_type, filename=row.filename)
