"""Backend API tests.

A fake CaptionGenerator stands in for the real one, so the suite runs with no
checkpoint, no torch weights and no model download. The database is a temporary
SQLite file per test.

Environment setup lives in ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image

from backend.app import main as main_module
from backend.app.core.config import settings
from backend.app.db import base as db_base
from backend.app.db.models import Base
from backend.app.services.caption_service import CaptionService

GRID = 7
NUM_REGIONS = 49


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeResult:
    """Stands in for captioning.generator.CaptionResult."""

    def __init__(self, words: list[str], time_ms: float = 12.5) -> None:
        self.caption = " ".join(words)
        self.words = words
        self.attention = np.full((len(words), GRID, GRID), 1.0 / NUM_REGIONS, dtype=np.float32)
        self.time_ms = time_ms


class FakeGenerator:
    """Stands in for captioning.generator.CaptionGenerator."""

    def __init__(self, words: list[str] | None = None) -> None:
        self.vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.config = type("Config", (), {"max_len": 50})()
        self.strategy = "greedy"
        self.calls = 0
        self._words = words or ["a", "dog", "runs"]

    def generate(self, image: Image.Image) -> FakeResult:
        self.calls += 1
        return FakeResult(list(self._words))


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator["TestClient"]:  # noqa: F821
    """A TestClient with the model stubbed and a fresh database per test."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module, "load_model", lambda: CaptionService(FakeGenerator()))

    application = main_module.create_app()

    # a clean database for every test
    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)

    with TestClient(application) as test_client:
        yield test_client

    Base.metadata.drop_all(bind=db_base.engine)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_png(width: int = 40, height: int = 30, colour=(200, 60, 30)) -> bytes:
    """Real PNG bytes."""
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg(width: int = 40, height: int = 30) -> bytes:
    """Real JPEG bytes."""
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 60, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(client, data: bytes, filename: str = "photo.png", content_type: str = "image/png"):
    """POST a file to the captions endpoint."""
    return client.post(
        "/api/v1/captions",
        files={"file": (filename, data, content_type)},
    )


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database"] is True


def test_health_degraded_without_model(monkeypatch: pytest.MonkeyPatch):
    """A model that fails to load must not take the whole app down."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module, "load_model", lambda: None)
    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)

    with TestClient(main_module.create_app()) as test_client:
        body = test_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_captions_503_when_model_missing(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module, "load_model", lambda: None)
    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)

    with TestClient(main_module.create_app()) as test_client:
        response = test_client.post(
            "/api/v1/captions",
            files={"file": ("a.png", make_png(), "image/png")},
        )
    assert response.status_code == 503
    assert "detail" in response.json()


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------


def test_upload_returns_201_with_caption_and_attention(client):
    response = upload(client, make_png())

    assert response.status_code == 201
    body = response.json()

    assert body["id"]
    assert body["caption"] == "a dog runs"
    assert body["words"] == ["a", "dog", "runs"]
    assert body["time_ms"] == pytest.approx(12.5)
    assert body["image_url"] == f"/api/v1/images/{body['image_url'].rsplit('/', 1)[1]}"


def test_attention_shape_is_words_by_7_by_7(client):
    body = upload(client, make_png()).json()
    attention = body["attention"]

    assert len(attention) == len(body["words"]) == 3
    for grid in attention:
        assert len(grid) == GRID
        assert all(len(row) == GRID for row in grid)
        # every word's map is a distribution over the 49 regions
        total = sum(sum(row) for row in grid)
        assert total == pytest.approx(1.0, abs=1e-3)


def test_attention_stored_flat_as_49_columns(client):
    """The database column holds num_words x 49, not 7x7."""
    from sqlalchemy import select

    from backend.app.db.models import Caption

    upload(client, make_png())

    with db_base.SessionLocal() as session:
        row = session.execute(select(Caption)).scalars().one()
        assert len(row.attention) == len(row.words) == 3
        assert all(len(values) == NUM_REGIONS for values in row.attention)
        # rounded to 4 decimals
        for values in row.attention:
            for value in values:
                assert value == round(value, 4)
        # the stored rows still sum to 1
        for values in row.attention:
            assert sum(values) == pytest.approx(1.0, abs=1e-3)


def test_jpeg_upload_works(client):
    response = upload(client, make_jpeg(), filename="photo.jpg", content_type="image/jpeg")
    assert response.status_code == 201


def test_upload_stores_the_original_bytes(client):
    """The file on disk is byte-identical to what was sent."""
    data = make_png()
    body = upload(client, data).json()
    image_id = body["image_url"].rsplit("/", 1)[1]

    stored = list(Path(settings.upload_dir).glob(f"{image_id}.*"))
    assert len(stored) == 1
    assert stored[0].read_bytes() == data


def test_stored_image_is_served_back(client):
    body = upload(client, make_png()).json()
    response = client.get(body["image_url"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == make_png()


def test_repeated_uploads_create_separate_images(client):
    first = upload(client, make_png()).json()
    second = upload(client, make_png()).json()
    assert first["id"] != second["id"]
    assert first["image_url"] != second["image_url"]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_renamed_exe_returns_400(client):
    """A file that lies about its content type is caught by decoding it."""
    fake_exe = b"MZ\x90\x00" + b"\x00" * 2000
    response = upload(client, fake_exe, filename="virus.jpg", content_type="image/jpeg")

    assert response.status_code == 400
    assert "detail" in response.json()


def test_truncated_png_returns_400(client):
    truncated = make_png()[:60]
    response = upload(client, truncated, filename="broken.png", content_type="image/png")
    assert response.status_code == 400


def test_text_file_returns_400(client):
    response = upload(client, b"just some text, not an image at all", filename="a.png", content_type="image/png")
    assert response.status_code == 400


def test_oversize_file_returns_413(client):
    """Over the 10 MB limit -> 413, not 400."""
    # Incompressible noise: a 1600x1600 RGB PNG lands comfortably past 10 MB,
    # where a solid-colour image would compress to a few kilobytes.
    import io

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(2600, 2600, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise, mode="RGB").save(buffer, format="PNG")
    oversize = buffer.getvalue()

    assert len(oversize) > 10 * 1024 * 1024, f"only {len(oversize)} bytes"

    response = upload(client, oversize, filename="huge.png", content_type="image/png")
    assert response.status_code == 413
    assert "detail" in response.json()


def test_disallowed_content_type_returns_400(client):
    response = upload(client, b"%PDF-1.4", filename="doc.pdf", content_type="application/pdf")
    assert response.status_code == 400
    assert "content type" in response.json().get("detail", "").lower()


def test_empty_file_returns_400(client):
    response = upload(client, b"", filename="empty.png", content_type="image/png")
    assert response.status_code == 400


def test_missing_file_returns_422(client):
    response = client.post("/api/v1/captions")
    assert response.status_code == 422


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------


def test_history_is_newest_first(client):
    upload(client, make_png(colour=(10, 10, 10)))
    upload(client, make_png(colour=(20, 20, 20)))
    upload(client, make_png(colour=(30, 30, 30)))

    body = client.get("/api/v1/captions").json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    # all three captions are identical here, so check the timestamps instead
    times = [item["created_at"] for item in body["items"]]
    assert times == sorted(times, reverse=True)


def test_history_limit_and_offset(client):
    for _ in range(5):
        upload(client, make_png())

    first = client.get("/api/v1/captions", params={"limit": 2, "offset": 0}).json()
    second = client.get("/api/v1/captions", params={"limit": 2, "offset": 2}).json()

    assert first["total"] == 5
    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    assert {i["id"] for i in first["items"]}.isdisjoint({i["id"] for i in second["items"]})


def test_history_search_filters_by_text(monkeypatch: pytest.MonkeyPatch):
    """q matches on caption text, case-insensitively."""
    from fastapi.testclient import TestClient

    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)

    # two services, so the corpus has two different captions to search
    monkeypatch.setattr(
        main_module, "load_model", lambda: CaptionService(FakeGenerator(["a", "cat", "sits"]))
    )
    with TestClient(main_module.create_app()) as cat_client:
        cat_client.post("/api/v1/captions", files={"file": ("a.png", make_png(), "image/png")})

    monkeypatch.setattr(
        main_module, "load_model", lambda: CaptionService(FakeGenerator(["two", "dogs", "running"]))
    )
    with TestClient(main_module.create_app()) as dog_client:
        dog_client.post("/api/v1/captions", files={"file": ("b.png", make_png(), "image/png")})

        assert dog_client.get("/api/v1/captions").json()["total"] == 2

        upper = dog_client.get("/api/v1/captions", params={"q": "CAT"}).json()
        assert upper["total"] == 1
        assert upper["items"][0]["caption"] == "a cat sits"

        lower = dog_client.get("/api/v1/captions", params={"q": "dogs"}).json()
        assert lower["total"] == 1
        assert lower["items"][0]["caption"] == "two dogs running"

        both = dog_client.get("/api/v1/captions", params={"q": "a"}).json()
        assert both["total"] == 1  # "a cat sits" only

        assert dog_client.get("/api/v1/captions", params={"q": "zzz"}).json()["total"] == 0


def test_history_search_with_no_match(client):
    upload(client, make_png())
    body = client.get("/api/v1/captions", params={"q": "zebra"}).json()
    assert body["total"] == 0
    assert body["items"] == []


def test_history_rejects_bad_pagination(client):
    assert client.get("/api/v1/captions", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/captions", params={"limit": 500}).status_code == 422
    assert client.get("/api/v1/captions", params={"offset": -1}).status_code == 422


def test_get_caption_by_id(client):
    created = upload(client, make_png()).json()
    response = client.get(f"/api/v1/captions/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["caption"] == "a dog runs"
    assert len(body["attention"]) == 3


def test_get_caption_unknown_id_returns_404(client):
    response = client.get("/api/v1/captions/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_image_unknown_id_returns_404(client):
    assert client.get("/api/v1/images/nope").status_code == 404


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def test_stats_empty(client):
    body = client.get("/api/v1/stats").json()
    assert body["total_captions"] == 0
    assert body["avg_generation_time_ms"] == 0.0
    assert body["evaluation"] is None


def test_stats_after_uploads(client):
    upload(client, make_png())
    upload(client, make_png())

    body = client.get("/api/v1/stats").json()
    assert body["total_captions"] == 2
    assert body["avg_generation_time_ms"] == pytest.approx(12.5, abs=0.1)
    assert body["evaluation"] is None


def test_stats_includes_latest_evaluation(client):
    from backend.app.db.models import EvaluationRun

    with db_base.SessionLocal() as session:
        session.add(
            EvaluationRun(
                checkpoint="ml/weights/best.pt",
                split="test",
                tokenizer="PTBTokenizer",
                num_images=5000,
                scores={"Bleu_4": 0.31, "CIDEr": 1.02, "METEOR": 0.27},
            )
        )
        session.commit()

    body = client.get("/api/v1/stats").json()
    assert body["evaluation"] is not None
    assert body["evaluation"]["split"] == "test"
    assert body["evaluation"]["tokenizer"] == "PTBTokenizer"
    assert body["evaluation"]["num_images"] == 5000
    assert body["evaluation"]["scores"]["CIDEr"] == pytest.approx(1.02)


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------


def test_cors_allows_the_frontend_origin(client):
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_openapi_lists_every_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    for expected in (
        "/health",
        "/api/v1/captions",
        "/api/v1/captions/{caption_id}",
        "/api/v1/images/{image_id}",
        "/api/v1/stats",
    ):
        assert expected in paths, expected
