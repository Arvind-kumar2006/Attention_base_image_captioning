"""Unit tests for upload validation and attention reshaping."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from backend.app.core.config import MAX_UPLOAD_BYTES, settings
from backend.app.services.caption_service import (
    flatten_attention,
    reshape_attention,
    round_triples,
)
from backend.app.services.image_utils import (
    decode_image,
    read_upload,
    resolve_stored_path,
    save_image,
)

GRID = 7
NUM_REGIONS = 49


def png_bytes(width: int = 20, height: int = 15, colour=(10, 200, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def make_upload(data: bytes, filename: str = "a.png", content_type: str = "image/png") -> UploadFile:
    """An UploadFile carrying the content type, the way FastAPI builds one."""
    from starlette.datastructures import Headers

    headers = Headers({"content-type": content_type})
    return UploadFile(filename=filename, file=io.BytesIO(data), headers=headers)


# --------------------------------------------------------------------------
# read_upload
# --------------------------------------------------------------------------


def test_read_upload_rejects_wrong_content_type():
    with pytest.raises(HTTPException) as error:
        import asyncio

        asyncio.run(read_upload(make_upload(b"%PDF", "a.pdf", "application/pdf")))
    assert error.value.status_code == 400


def test_read_upload_rejects_oversize():
    import asyncio

    with pytest.raises(HTTPException) as error:
        asyncio.run(read_upload(make_upload(b"x" * 100), max_bytes=10))
    assert error.value.status_code == 413


def test_read_upload_rejects_empty():
    import asyncio

    with pytest.raises(HTTPException) as error:
        asyncio.run(read_upload(make_upload(b"")))
    assert error.value.status_code == 400


def test_read_upload_accepts_jpeg_and_png():
    import asyncio

    for content_type, data in (
        ("image/png", png_bytes()),
        ("image/jpeg", b"\xff\xd8\xff\xe0" + b"\x00" * 40),
    ):
        result = asyncio.run(
            read_upload(make_upload(data, "a.bin", content_type))
        )
        assert result == data


def test_max_upload_is_ten_megabytes():
    assert MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    assert settings.max_upload_bytes == MAX_UPLOAD_BYTES


# --------------------------------------------------------------------------
# decode_image
# --------------------------------------------------------------------------


def test_decode_rejects_renamed_binary():
    with pytest.raises(HTTPException) as error:
        decode_image(b"MZ\x90\x00" + b"\x00" * 500)
    assert error.value.status_code == 400


def test_decode_rejects_truncated_image():
    with pytest.raises(HTTPException) as error:
        decode_image(png_bytes()[:40])
    assert error.value.status_code == 400


def test_decode_converts_to_rgb():
    buffer = io.BytesIO()
    Image.new("RGBA", (12, 8), (1, 2, 3, 255)).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB"


def test_decode_converts_grayscale_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (12, 8), 128).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB"


def test_decode_applies_exif_rotation():
    """An image tagged orientation 6 comes back rotated 90 degrees."""
    buffer = io.BytesIO()
    # 40 wide x 20 tall, tagged as needing a 90 degree rotation
    original = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = original.getexif()
    exif[274] = 6  # Orientation: rotate 90 CW
    original.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(buffer.getvalue())
    # the stored pixels are rotated, so the axes swap
    assert (decoded.width, decoded.height) == (20, 40)


def test_decode_drops_exif_metadata():
    buffer = io.BytesIO()
    image = Image.new("RGB", (30, 20), (10, 20, 30))
    exif = image.getexif()
    exif[271] = "TestCamera"  # Make
    image.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(buffer.getvalue())
    assert "TestCamera" not in str(dict(decoded.getexif()))


# --------------------------------------------------------------------------
# save_image / resolve_stored_path
# --------------------------------------------------------------------------


def test_save_writes_original_bytes_under_uuid(tmp_path: Path):
    data = png_bytes()
    stored = save_image(
        decode_image(data), data, "photo.png", "image/png", upload_dir=tmp_path
    )

    assert stored.path.is_file()
    assert stored.path.read_bytes() == data
    assert stored.path.name.startswith(stored.id)
    assert stored.path.suffix == ".png"
    assert stored.size_bytes == len(data)


def test_save_png_extension_for_png(tmp_path: Path):
    data = png_bytes()
    stored = save_image(decode_image(data), data, "a.png", "image/png", upload_dir=tmp_path)
    assert stored.path.suffix == ".png"


def test_save_jpg_extension_for_jpeg(tmp_path: Path):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), (1, 2, 3)).save(buffer, format="JPEG")
    data = buffer.getvalue()
    stored = save_image(decode_image(data), data, "a.jpg", "image/jpeg", upload_dir=tmp_path)
    assert stored.path.suffix == ".jpg"


def test_resolve_returns_the_stored_file(tmp_path: Path):
    data = png_bytes()
    stored = save_image(decode_image(data), data, "a.png", "image/png", upload_dir=tmp_path)
    assert resolve_stored_path(stored.relative_path, upload_dir=tmp_path) == stored.path.resolve()


def test_resolve_blocks_path_traversal(tmp_path: Path):
    """A tampered path must not escape the upload directory."""
    with pytest.raises(HTTPException) as error:
        resolve_stored_path("../../etc/passwd", upload_dir=tmp_path)
    assert error.value.status_code == 404


def test_resolve_blocks_absolute_escape(tmp_path: Path):
    with pytest.raises(HTTPException):
        resolve_stored_path("/etc/passwd", upload_dir=tmp_path)


# --------------------------------------------------------------------------
# attention reshaping
# --------------------------------------------------------------------------


def test_flatten_49_columns_from_7x7():
    attention = np.arange(3 * GRID * GRID, dtype=np.float32).reshape(3, GRID, GRID)
    flat = flatten_attention(attention)

    assert len(flat) == 3
    assert all(len(row) == NUM_REGIONS for row in flat)
    # row-major order preserved
    assert flat[0][:3] == [0.0, 1.0, 2.0]


def test_flatten_rounds_to_four_decimals():
    attention = np.full((1, GRID, GRID), 1.0 / NUM_REGIONS, dtype=np.float32)
    flat = flatten_attention(attention)
    for value in flat[0]:
        assert value == round(value, 4)
    assert sum(flat[0]) == pytest.approx(1.0, abs=1e-3)


def test_flatten_handles_empty():
    assert flatten_attention(np.zeros((0, GRID, GRID), dtype=np.float32)) == []


def test_reshape_round_trips_flatten():
    attention = np.random.default_rng(0).random((4, GRID, GRID)).astype(np.float32)
    flat = flatten_attention(attention)
    restored = reshape_attention(flat)

    assert np.array(restored).shape == (4, GRID, GRID)
    assert np.allclose(np.array(restored), np.round(attention, 4), atol=1e-4)


def test_reshape_handles_empty():
    assert reshape_attention([]) == []


def test_reshape_handles_flat_single_word():
    out = reshape_attention([0.5] * NUM_REGIONS)
    assert len(out) == 1 and len(out[0]) == GRID and len(out[0][0]) == GRID


def test_round_triples_shape_and_rounding():
    attention = np.full((2, GRID, GRID), 1.0 / NUM_REGIONS, dtype=np.float32)
    triples = round_triples(attention)
    assert len(triples) == 2
    assert all(len(grid) == GRID for grid in triples)
    for grid in triples:
        for row in grid:
            for value in row:
                assert value == round(value, 4)
