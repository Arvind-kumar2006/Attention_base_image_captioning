"""v1 API router."""

from __future__ import annotations

from fastapi import APIRouter

from . import captions, images, stats

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(captions.router)
api_router.include_router(images.router)
api_router.include_router(stats.router)

__all__ = ["api_router"]
