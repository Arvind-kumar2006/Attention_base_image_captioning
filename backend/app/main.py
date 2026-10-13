"""FastAPI application.

The model is loaded once in the lifespan handler and kept on ``app.state``.
Loading a ResNet-50 checkpoint takes seconds; doing it per request would make
the API unusable.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import api_router
from .core.config import settings
from .db.session import create_all, ping
from .schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

VERSION = "0.1.0"


def load_model() -> Any:
    """Load the caption generator, or return ``None`` if it cannot be loaded.

    Failing to load is not fatal: ``/health`` reports it and the caption
    endpoints return 503, so the rest of the API stays reachable and the problem
    is visible instead of the process crash-looping on startup.
    """
    from .services.caption_service import CaptionService

    try:
        service = CaptionService.from_checkpoint(
            str(settings.checkpoint_path), device=settings.device
        )
    except Exception as error:
        logger.error(
            "could not load %s: %r — caption endpoints will return 503",
            settings.checkpoint_path,
            error,
        )
        return None

    logger.info("model loaded, vocab %d words", service.vocab_size)
    return service


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create tables and load the model once, before serving."""
    logger.info("starting up: db=%s uploads=%s", settings.database_url.split("@")[-1], settings.upload_dir)

    try:
        create_all()
    except Exception as error:
        logger.error("database setup failed: %r", error)

    application.state.caption_service = load_model()
    application.state.settings = settings

    try:
        yield
    finally:
        application.state.caption_service = None
        logger.info("shut down")


def create_app() -> FastAPI:
    """Build the application. Tests call this with the model already stubbed."""
    application = FastAPI(
        title="Lumina Captioning API",
        description=(
            "Attention-based image captioning. Uploads an image and returns a "
            "caption with the per-word attention map the model used."
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    origins = settings.cors_origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS allowed origins: %s", origins)

    application.include_router(api_router)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Liveness plus whether the model is ready to serve captions."""
        service = getattr(application.state, "caption_service", None)
        loaded = bool(service is not None and service.is_ready)
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            database=ping(),
            version=VERSION,
        )

    return application


app = create_app()
