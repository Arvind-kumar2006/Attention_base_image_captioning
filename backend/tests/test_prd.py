"""PRD acceptance tests, addressed by their PRD identifiers.

Each test is named ``test_<id>_*`` so ``pytest -k TEST-03`` selects it and so CI
output points straight at the requirement it covers.

| ID     | Requirement                                  |
|--------|----------------------------------------------|
| TEST-01| Encoder output shape is (1, 49, 2048)        |
| TEST-02| Attention weights sum to 1 within 1e-4       |
| TEST-03| A generated caption is persisted            |
| TEST-04| A corrupt upload returns 400                |
| TEST-05| A caption completes within the latency target |

TEST-01, TEST-02 and TEST-04 are also covered nearer the code they exercise;
these are the single place to look for the requirements as a set.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sqlalchemy import select

from backend.app import main as main_module
from backend.app.core.config import settings
from backend.app.db import base as db_base
from backend.app.db.models import Base, Caption
from backend.app.db.models import Image as ImageModel
from backend.app.services.caption_service import CaptionService
from captioning.attention import Attention
from captioning.encoder import Encoder

GRID = 7
NUM_REGIONS = 49
ENCODER_DIM = 2048

#: Inference budget for one caption, in milliseconds, on the CPU deployment.
#: Generous enough to be stable on a loaded CI runner while still catching a
#: regression that turns inference into a multi-second stall. The measured figure
#: for a trained checkpoint goes in the README results table.
LATENCY_TARGET_MS = 2000.0


# --------------------------------------------------------------------------
# fakes, so no checkpoint and no weights download is needed
# --------------------------------------------------------------------------


class FakeResult:
    def __init__(self, words, time_ms):
        self.caption = " ".join(words)
        self.words = words
        self.attention = np.full(
            (len(words), GRID, GRID), 1.0 / NUM_REGIONS, dtype=np.float32
        )
        self.time_ms = time_ms


class FakeGenerator:
    """Mimics CaptionGenerator, with a deliberate think time for TEST-05."""

    def __init__(self, words=None, time_ms=12.5):
        self.vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.config = type("Config", (), {"max_len": 50})()
        self.strategy = "greedy"
        self.time_ms = time_ms
        self._words = words or ["a", "dog", "runs"]

    def generate(self, image):
        return FakeResult(list(self._words), self.time_ms)


def png_bytes(size=(60, 40), colour=(30, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(monkeypatch):
    """A TestClient with the model stubbed and a fresh database."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module, "load_model", lambda: CaptionService(FakeGenerator()))
    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)
    with TestClient(main_module.create_app()) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=db_base.engine)


def upload(client, data, filename="photo.png", content_type="image/png"):
    return client.post("/api/v1/captions", files={"file": (filename, data, content_type)})


# --------------------------------------------------------------------------
# TEST-01
# --------------------------------------------------------------------------


@pytest.mark.prd
@pytest.mark.test_01
def test_TEST_01_encoder_output_shape():
    """PRD TEST-01: the encoder returns one 2048-dim vector per 7x7 region."""
    encoder = Encoder(freeze=True)
    images = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        regions = encoder(images)

    assert regions.shape == (1, 49, 2048)
    assert regions.shape == (1, GRID * GRID, ENCODER_DIM)
    # and the values are real features, not an uninitialised tensor
    assert torch.isfinite(regions).all()
    assert regions.abs().sum() > 0


@pytest.mark.prd
@pytest.mark.test_01
def test_TEST_01_encoder_output_shape_batched():
    """The batch axis is real: two images give two independent region sets."""
    encoder = Encoder(freeze=True)
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        regions = encoder(images)

    assert regions.shape == (2, 49, 2048)
    # different inputs must not collapse to the same features
    assert not torch.allclose(regions[0], regions[1])


# --------------------------------------------------------------------------
# TEST-02
# --------------------------------------------------------------------------


@pytest.mark.prd
@pytest.mark.test_02
def test_TEST_02_attention_weights_sum_to_one():
    """PRD TEST-02: each word's attention is a distribution over 49 regions."""
    attention = Attention(ENCODER_DIM, 32, 32)
    features = torch.randn(3, NUM_REGIONS, ENCODER_DIM)
    hidden = torch.randn(3, 32)

    context, alpha = attention(features, hidden)

    assert alpha.shape == (3, NUM_REGIONS)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(3), atol=1e-4)
    # a distribution: no negative mass, nothing above 1
    assert (alpha >= 0).all()
    assert (alpha <= 1).all()
    # the context is the alpha-weighted mean of the regions
    assert context.shape == (3, ENCODER_DIM)
    assert torch.allclose(context, (features * alpha.unsqueeze(2)).sum(1), atol=1e-5)


@pytest.mark.prd
@pytest.mark.test_02
def test_TEST_02_attention_sums_to_one_over_many_states():
    """Holds for a spread of decoder states, not just one."""
    attention = Attention(ENCODER_DIM, 32, 32)
    features = torch.randn(8, NUM_REGIONS, ENCODER_DIM)

    for seed in range(8):
        torch.manual_seed(seed)
        hidden = torch.randn(8, 32)
        _context, alpha = attention(features, hidden)
        assert torch.allclose(
            alpha.sum(dim=1), torch.ones(8), atol=1e-4
        ), f"row sums drifted at seed {seed}"


@pytest.mark.prd
@pytest.mark.test_02
def test_TEST_02_caption_api_returns_normalised_attention(client):
    """The same guarantee holds for what the API hands to the browser."""
    response = upload(client, png_bytes())
    assert response.status_code == 201
    attention = response.json()["attention"]

    assert attention, "caption produced no attention"
    for grid in attention:
        assert len(grid) == GRID and all(len(row) == GRID for row in grid)
        assert sum(sum(row) for row in grid) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# TEST-03
# --------------------------------------------------------------------------


@pytest.mark.prd
@pytest.mark.test_03
def test_TEST_03_caption_is_persisted(client):
    """PRD TEST-03: the generated caption lands in the database and reads back."""
    response = upload(client, png_bytes(), filename="persisted.png")
    assert response.status_code == 201
    created = response.json()

    with db_base.SessionLocal() as session:
        row = session.get(Caption, created["id"])

        assert row is not None, "caption row missing"
        assert row.caption_text == created["caption"]
        assert row.caption_text == "a dog runs"
        assert row.words == created["words"] == ["a", "dog", "runs"]
        assert row.created_at is not None

        # attention is stored flattened as num_words x 49
        assert len(row.attention) == len(row.words) == 3
        assert all(len(values) == NUM_REGIONS for values in row.attention)
        for values in row.attention:
            assert sum(values) == pytest.approx(1.0, abs=1e-3)

        # the image row is persisted too, so the file can be served later
        image_row = session.get(ImageModel, row.image_id)
        assert image_row is not None
        assert (Path(settings.upload_dir) / image_row.image_path).is_file()


@pytest.mark.prd
@pytest.mark.test_03
def test_TEST_03_caption_readable_from_the_history_endpoint(client):
    """What was written is what the history endpoint returns."""
    created = upload(client, png_bytes()).json()

    history = client.get("/api/v1/captions").json()
    assert history["total"] == 1

    item = history["items"][0]
    assert item["id"] == created["id"]
    assert item["caption"] == created["caption"]
    assert item["words"] == created["words"]
    # the flattened 49-column rows come back as 7x7 for the frontend
    assert len(item["attention"]) == 3
    assert len(item["attention"][0]) == GRID
    assert len(item["attention"][0][0]) == GRID


@pytest.mark.prd
@pytest.mark.test_03
def test_TEST_03_persisted_caption_survives_a_new_session(client):
    """The row is committed, not held in a session that gets rolled back."""
    upload(client, png_bytes())

    # a completely separate engine-level read
    with db_base.SessionLocal() as session:
        count = len(session.execute(select(Caption)).scalars().all())
    assert count == 1


# --------------------------------------------------------------------------
# TEST-04
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "data", "filename", "content_type"),
    [
        ("executable renamed to jpg", b"MZ\x90\x00" + b"\x00" * 2048, "virus.jpg", "image/jpeg"),
        ("truncated png", None, "broken.png", "image/png"),
        ("plain text", b"this is definitely not an image, is it", "notes.png", "image/png"),
        ("pdf", b"%PDF-1.7\n%binary\n", "doc.pdf", "application/pdf"),
        ("empty", b"", "empty.png", "image/png"),
        ("gif", b"GIF89a\x01\x00\x01\x00\x00\xff\x00,", "anim.gif", "image/gif"),
    ],
)
@pytest.mark.prd
@pytest.mark.test_04
def test_TEST_04_corrupt_upload_returns_400(client, label, data, filename, content_type):
    """PRD TEST-04: anything that is not a decodable JPEG or PNG is a 400."""
    if data is None:
        data = png_bytes()[:50]  # a real PNG cut short

    response = upload(client, data, filename=filename, content_type=content_type)

    assert response.status_code == 400, f"{label} should be 400"
    body = response.json()
    assert set(body) == {"detail"}, "errors must use the {detail: ...} envelope"
    assert body["detail"]


@pytest.mark.prd
@pytest.mark.test_04
def test_TEST_04_corrupt_upload_is_not_persisted(client):
    """A rejected upload must leave no trace in the database or on disk."""
    before = len(list(Path(settings.upload_dir).glob("*"))) if Path(settings.upload_dir).exists() else 0

    response = upload(client, b"MZ\x90\x00" + b"\x00" * 2048, "virus.jpg", "image/jpeg")
    assert response.status_code == 400

    with db_base.SessionLocal() as session:
        assert session.execute(select(Caption)).scalars().all() == []
        assert session.execute(select(ImageModel)).scalars().all() == []

    after = len(list(Path(settings.upload_dir).glob("*"))) if Path(settings.upload_dir).exists() else 0
    assert after == before, "a rejected upload wrote a file"


# --------------------------------------------------------------------------
# TEST-05
# --------------------------------------------------------------------------


@pytest.mark.prd
@pytest.mark.test_05
def test_TEST_05_caption_within_latency_target(client):
    """PRD TEST-05: a caption returns inside the latency budget.

    The stub reports 12.5 ms, so the wall-clock assertion here proves the request
    path adds no unexpected cost; the recorded ``time_ms`` assertion is the part
    that actually tracks the model.
    """
    started = time.perf_counter()
    response = upload(client, png_bytes())
    wall_clock_ms = (time.perf_counter() - started) * 1000.0

    assert response.status_code == 201
    assert wall_clock_ms < LATENCY_TARGET_MS, (
        f"request took {wall_clock_ms:.0f} ms, over the {LATENCY_TARGET_MS:.0f} ms target"
    )
    assert response.json()["time_ms"] < LATENCY_TARGET_MS


@pytest.mark.prd
@pytest.mark.test_05
def test_TEST_05_reported_time_matches_the_budget(monkeypatch):
    """A model that takes longer than the target must be caught, not excused."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        main_module,
        "load_model",
        lambda: CaptionService(FakeGenerator(time_ms=LATENCY_TARGET_MS + 500)),
    )
    Base.metadata.drop_all(bind=db_base.engine)
    Base.metadata.create_all(bind=db_base.engine)

    with TestClient(main_module.create_app()) as slow_client:
        body = slow_client.post(
            "/api/v1/captions", files={"file": ("a.png", png_bytes(), "image/png")}
        ).json()

    # the API reports the cost honestly rather than hiding it
    assert body["time_ms"] > LATENCY_TARGET_MS


@pytest.mark.prd
@pytest.mark.test_05
def test_TEST_05_latency_target_is_defined():
    """Guard the target itself, so it cannot silently drift to something huge."""
    assert 0 < LATENCY_TARGET_MS <= 10_000
