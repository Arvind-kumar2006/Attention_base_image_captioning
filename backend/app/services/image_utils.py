"""Upload validation and storage.

The content type alone is not trusted: a renamed ``.exe`` arrives as
``image/jpeg`` and is only caught when Pillow fails to decode it. Everything
here is CPU-bound and blocking, so it runs in FastAPI's threadpool.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..core.config import MAX_IMAGE_PIXELS, settings

logger = logging.getLogger(__name__)

# Pillow is only used to verify and normalise, never to serve untrusted data
# to a parser that cannot cope, so the decompression-bomb guard is on.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass
class StoredImage:
    """An image that passed validation and is on disk."""

    id: str
    path: Path
    relative_path: str
    filename: str
    content_type: str
    size_bytes: int

    @property
    def url_path(self) -> str:
        """Path component for the image-serving endpoint."""
        return self.id


async def read_upload(file: UploadFile, max_bytes: int | None = None) -> bytes:
    """Read the upload, refusing anything over the size limit.

    Raises:
        HTTPException: 400 for a disallowed content type, 413 when too large.
    """
    limit = max_bytes if max_bytes is not None else settings.max_upload_bytes

    content_type = (file.content_type or "").lower().strip()
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported content type {content_type or 'unknown'!r}; "
                f"allowed: {', '.join(settings.allowed_content_types)}"
            ),
        )

    # Read one byte past the limit so an oversize body is detected without
    # buffering the whole thing.
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"file is larger than the {limit // (1024 * 1024)} MB limit",
        )
    if not data:
        raise HTTPException(status_code=400, detail="file is empty")

    return data


def decode_image(data: bytes) -> Image.Image:
    """Decode and normalise an upload, rejecting anything that is not a real image.

    Applies EXIF rotation, converts to RGB and drops the EXIF payload, so
    orientation from a phone photo comes out right and no metadata is stored.

    Raises:
        HTTPException: 400 when Pillow cannot decode the bytes, or when the
            result is implausibly large.
    """
    import io

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decode now so a truncated file fails here
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"could not decode the file as an image: {error}",
        ) from error

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"image is {image.width}x{image.height} pixels, over the "
                f"{MAX_IMAGE_PIXELS:,} pixel limit"
            ),
        )

    # transpose() applies the EXIF orientation tag, but leaves the remaining
    # metadata in place. Rebuild the image so no EXIF is carried into what we
    # store or serve: it can carry GPS coordinates and device names.
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    # Image.new + paste rather than getdata(): putdata is deprecated in Pillow
    # 12 and removed in 14, and paste works across versions.
    flattened = Image.new("RGB", image.size)
    flattened.paste(image.convert("RGB"))
    return flattened


def save_image(
    image: Image.Image,
    data: bytes,
    original_filename: str,
    content_type: str,
    upload_dir: Path | None = None,
) -> StoredImage:
    """Write the *original* bytes to disk under a uuid name.

    The original is stored rather than the re-encoded image so the stored file
    is byte-identical to what the client sent, and so re-encoding never loses
    quality.
    """
    directory = Path(upload_dir) if upload_dir is not None else settings.upload_dir
    directory.mkdir(parents=True, exist_ok=True)

    extension = ".png" if content_type == "image/png" else ".jpg"
    image_id = uuid.uuid4().hex
    target = directory / f"{image_id}{extension}"
    target.write_bytes(data)

    logger.info("stored %s (%d bytes) as %s", original_filename, len(data), target.name)
    return StoredImage(
        id=image_id,
        path=target,
        relative_path=target.name,
        filename=original_filename or target.name,
        content_type=content_type,
        size_bytes=len(data),
    )


def resolve_stored_path(relative_path: str, upload_dir: Path | None = None) -> Path:
    """Turn a stored relative path back into an absolute one.

    Rejects anything that tries to escape the upload directory, so a tampered
    database row cannot be used to read arbitrary files.
    """
    directory = (Path(upload_dir) if upload_dir is not None else settings.upload_dir).resolve()
    candidate = (directory / relative_path).resolve()

    if not candidate.is_relative_to(directory):
        raise HTTPException(status_code=404, detail="image not found")

    return candidate
