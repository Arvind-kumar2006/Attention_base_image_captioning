"""End-to-end image captioning for a saved checkpoint.

Loads a checkpoint written by ``ml/train.py``, preprocesses an image the same
way the encoder expects at eval time, runs a decoding strategy, and returns the
caption together with its per-word attention as a ``(num_words, 7, 7)`` array
ready to be turned into a heatmap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .data import IMAGENET_MEAN, IMAGENET_STD
from .decoding import (
    DecodingStrategy,
    GreedyDecodingStrategy,
)
from .model import load_checkpoint
from .vocab import decode, invert_vocab

#: ResNet-50 with avgpool+fc removed yields a 7x7 grid of regions.
GRID = 7


@dataclass
class CaptionResult:
    """One caption and its attention.

    Attributes:
        caption: the caption as a single string.
        words: the same caption split into words, with ``<start>``/``<end>``
            removed and OOV words rendered as ``<unk>``.
        attention: ``(num_words, 7, 7)`` float array; every row sums to 1.
        time_ms: wall-clock milliseconds spent encoding and decoding.
    """

    caption: str
    words: List[str]
    attention: np.ndarray
    time_ms: float

    @property
    def num_words(self) -> int:
        """Number of words in the caption."""
        return len(self.words)


def build_eval_transform() -> T.Compose:
    """Resize to 224x224 with no crop, then ImageNet-normalise."""
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def reshape_attention(alphas: torch.Tensor) -> np.ndarray:
    """Turn ``(num_words, num_regions)`` attention into ``(num_words, 7, 7)``."""
    if alphas.dim() != 2:
        raise ValueError(
            f"expected attention of shape (num_words, num_regions), got {tuple(alphas.shape)}"
        )
    num_regions = alphas.size(1)
    side = round(num_regions ** 0.5)
    if side * side != num_regions:
        raise ValueError(
            f"num_regions must be a perfect square to reshape, got {num_regions}"
        )
    return alphas.detach().cpu().float().reshape(-1, side, side).numpy()


class CaptionGenerator:
    """Turns a PIL image into a caption plus attention maps.

    Args:
        checkpoint_path: a ``.pt`` from ``save_checkpoint`` or ``ml/train.py``.
        device: ``"cpu"``, ``"cuda"`` or a ``torch.device``.
        strategy: a decoding strategy; defaults to greedy when omitted.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        strategy: Optional[DecodingStrategy] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.vocab, self.metadata = load_checkpoint(
            Path(checkpoint_path), map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        self.id_to_word: Dict[int, str] = invert_vocab(self.vocab)
        self.transform = build_eval_transform()
        self.max_len = int(self.model.config.max_len)

        if strategy is None:
            strategy = GreedyDecodingStrategy(self.model.decoder)
        self.strategy = strategy

    @property
    def config(self) -> Any:
        """The checkpoint's model config."""
        return self.model.config

    @torch.no_grad()
    def generate(self, pil_image: Image.Image) -> CaptionResult:
        """Caption one PIL image.

        Args:
            pil_image: any image; it is converted to RGB and resized to 224x224.

        Returns:
            A :class:`CaptionResult` with the caption, its words, the
            ``(num_words, 7, 7)`` attention array and the elapsed milliseconds.
        """
        if not isinstance(pil_image, Image.Image):
            raise TypeError(f"expected a PIL image, got {type(pil_image).__name__}")

        started = time.perf_counter()

        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0)
        tensor = tensor.to(self.device)
        features = self.model.encode(tensor)

        token_ids, alphas = self.strategy.decode(features, max_len=self.max_len)

        words = decode(token_ids, self.id_to_word, strip_specials=True)
        attention = reshape_attention(alphas)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return CaptionResult(
            caption=" ".join(words),
            words=words,
            attention=attention,
            time_ms=elapsed_ms,
        )

    @torch.no_grad()
    def generate_attention_overlay(
        self,
        pil_image: Image.Image,
        alpha: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Per-word attention upsampled to the image size, for a heatmap UI.

        Args:
            pil_image: the image that was captioned.
            alpha: opacity of the attention overlay.

        Returns:
            One dict per word with ``word``, ``heatmap`` ``(H, W)`` and ``alpha``.
        """
        result = self.generate(pil_image)
        width, height = pil_image.size
        maps: List[Dict[str, Any]] = []
        for index, word in enumerate(result.words):
            grid = torch.from_numpy(result.attention[index]).unsqueeze(0).unsqueeze(0)
            heat = F.interpolate(
                grid, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]
            heat = _minmax_normalise(heat)
            maps.append({"word": word, "heatmap": heat.numpy(), "alpha": alpha})
        return maps


def _minmax_normalise(heat: torch.Tensor) -> torch.Tensor:
    """Scale a heatmap to ``[0, 1]``.

    The guard is relative to the value range rather than an absolute epsilon:
    attention can be near-uniform, and a fixed ``1e-8`` would then dominate the
    denominator and leave the map almost entirely dark.
    """
    low = heat.min()
    spread = heat.max() - low
    if float(spread) <= 0.0:
        return torch.zeros_like(heat)
    return (heat - low) / (spread + spread * 1e-6)
