"""Captioning service wrapping :class:`captioning.generator.CaptionGenerator`.

Model inference is CPU/GPU-bound and blocking, so the router that calls this
module must be a plain ``def`` endpoint. FastAPI then runs it in the threadpool
and the event loop stays free to serve other requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..core.config import ATTENTION_DECIMALS

logger = logging.getLogger(__name__)

#: The encoder emits a 7x7 grid of regions.
GRID = 7
NUM_REGIONS = GRID * GRID


@dataclass
class CaptionOutput:
    """A caption plus the attention map reshaped for the API response."""

    caption: str
    words: list[str]
    #: ``(num_words, 7, 7)`` floats.
    attention: list[list[list[float]]]
    #: Flattened ``(num_words, 49)``, rounded for storage.
    flat_attention: list[list[float]] = field(default_factory=list)
    time_ms: float = 0.0

    @property
    def num_words(self) -> int:
        return len(self.words)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is available."""


def flatten_attention(attention: np.ndarray) -> list[list[float]]:
    """Flatten ``(num_words, 7, 7)`` into ``(num_words, 49)`` rows.

    Rounded to four decimals: the values are weights summing to 1, so full
    float precision is noise in a JSON column and inflates the row size.
    """
    array = np.asarray(attention, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 2:
        array = array.reshape(array.shape[0], GRID, GRID)
    if array.ndim != 3 or array.shape[1:] != (GRID, GRID):
        raise ValueError(
            f"expected attention of shape (num_words, {GRID}, {GRID}), got {array.shape}"
        )

    flattened = array.reshape(array.shape[0], NUM_REGIONS)
    return [
        [round(float(value), ATTENTION_DECIMALS) for value in row]
        for row in flattened
    ]


def reshape_attention(flat: list[list[float]] | list[float]) -> list[list[list[float]]]:
    """Turn stored ``(num_words, 49)`` rows back into ``(num_words, 7, 7)``.

    Also accepts a bare 49-value list, which is how a single-word caption is
    stored, and returns that as a 7x7 grid rather than a flat row.
    """
    if not flat:
        return []
    if not isinstance(flat[0], (list, tuple)):
        flat = [flat]  # a single caption stored as one flat row
    return [
        [
            [float(row[index * GRID + column]) for column in range(GRID)]
            for index in range(GRID)
        ]
        for row in flat
    ]


def round_triples(attention: np.ndarray) -> list[list[list[float]]]:
    """``(num_words, 7, 7)`` as plain nested lists, rounded for the response.

    Rounding happens on Python floats, not on the float32 array: ``np.round``
    followed by ``tolist()`` re-expands to the float32 value, so a response would
    carry 0.020400000736117363 instead of 0.0204.
    """
    array = np.asarray(attention, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, GRID, GRID)
    if array.ndim == 2:
        array = array.reshape(array.shape[0], GRID, GRID)
    return [
        [[round(float(value), ATTENTION_DECIMALS) for value in row] for row in grid]
        for grid in array
    ]


class CaptionService:
    """Loads the generator once and captions images with it."""

    def __init__(self, generator: Any) -> None:
        self._generator = generator

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: str = "cpu"
    ) -> "CaptionService":
        """Build a service by loading a checkpoint from disk."""
        from captioning.generator import CaptionGenerator

        logger.info("loading caption model from %s on %s", checkpoint_path, device)
        return cls(CaptionGenerator(checkpoint_path, device=device))

    @property
    def is_ready(self) -> bool:
        """True when a generator is present."""
        return self._generator is not None

    @property
    def vocab_size(self) -> int:
        """Number of words the model knows, or 0 when not loaded."""
        return len(getattr(self._generator, "vocab", {}) or {})

    def set_strategy(self, strategy: Any) -> None:
        """Swap the decoding strategy, e.g. greedy for beam search."""
        self._generator.strategy = strategy

    def generate(self, image: Image.Image) -> CaptionOutput:
        """Caption one image.

        Raises:
            ModelNotLoadedError: when no generator is attached.
        """
        if self._generator is None:
            raise ModelNotLoadedError("the caption model is not loaded")

        result = self._generator.generate(image)

        words = [str(word) for word in result.words]
        attention = np.asarray(result.attention, dtype=np.float32)
        if attention.size:
            expected = len(words) * GRID * GRID
            if attention.size != expected:
                raise ValueError(
                    f"attention has {attention.size} values but {len(words)} words "
                    f"imply {expected}"
                )

        return CaptionOutput(
            caption=str(result.caption),
            words=words,
            attention=round_triples(attention),
            flat_attention=flatten_attention(attention),
            time_ms=round(float(result.time_ms), 2),
        )

    def describe(self) -> Mapping[str, Any]:
        """Small summary for the health endpoint."""
        return {
            "vocab_size": self.vocab_size,
            "max_len": getattr(getattr(self._generator, "config", None), "max_len", None),
        }
