"""End-to-end captioning model: encoder + decoder, config and checkpoints.

The checkpoint layout follows the project contract::

    {
        "model_state": {...},        # combined encoder + decoder state dict
        "vocab": {word: id},         # so a checkpoint is self-describing
        "config": {...},             # architecture hyper-parameters
        "epoch": int,
        "val_bleu4": float,
    }

``vocab.json`` is always written next to the ``.pt`` file so the backend can load
the vocabulary without unpickling torch tensors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import ENCODER_DIM, NUM_REGIONS, Encoder


@dataclass
class ModelConfig:
    """Architecture hyper-parameters stored in every checkpoint."""

    vocab_size: int
    embed_dim: int = 512
    decoder_dim: int = 512
    attention_dim: int = 512
    encoder_dim: int = ENCODER_DIM
    dropout: float = 0.5
    max_len: int = 50
    num_regions: int = NUM_REGIONS
    freeze_encoder: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for the checkpoint."""
        return asdict(self)


class CaptioningModel(nn.Module):
    """Wraps the frozen ResNet encoder with the attention LSTM decoder."""

    def __init__(self, encoder: Encoder, decoder: Decoder, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    @classmethod
    def from_config(cls, config: ModelConfig) -> "CaptioningModel":
        """Build encoder + decoder from a config."""
        encoder = Encoder(freeze=config.freeze_encoder)
        decoder = Decoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            decoder_dim=config.decoder_dim,
            attention_dim=config.attention_dim,
            encoder_dim=config.encoder_dim,
            dropout=config.dropout,
        )
        return cls(encoder=encoder, decoder=decoder, config=config)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Region features ``(B, 49, 2048)`` for a batch of images."""
        return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images then teacher-force the decoder over the captions."""
        features = self.encode(images)
        return self.decoder(features, captions, lengths)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that receive gradients, honouring the encoder freeze flag."""
        params = list(self.decoder.parameters())
        params.extend(self.encoder.trainable_parameters())
        return params

    def set_finetune(self, flag: bool) -> None:
        """Toggle fine-tuning of ``layer3`` / ``layer4`` in the encoder."""
        self.encoder.set_finetune(flag)

    def state_dict(self) -> Dict[str, Any]:
        """Combined state dict for the whole model."""
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        """Load a combined state dict as produced by :meth:`state_dict`."""
        self.encoder.load_state_dict(state["encoder"], strict=strict)
        self.decoder.load_state_dict(state["decoder"], strict=strict)


def save_checkpoint(
    path: Path,
    model: CaptioningModel,
    vocab: Mapping[str, int],
    epoch: int,
    val_bleu4: float,
) -> Path:
    """Write a ``.pt`` checkpoint plus a sibling ``vocab.json``.

    Args:
        path: destination ``.pt`` path.
        model: the model to persist.
        vocab: ``word -> id`` mapping, stored in the checkpoint and on disk.
        epoch: training epoch the weights come from.
        val_bleu4: validation BLEU-4 at this epoch.

    Returns:
        The checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "vocab": dict(vocab),
        "config": model.config.to_dict(),
        "epoch": epoch,
        "val_bleu4": val_bleu4,
    }
    torch.save(payload, path)

    vocab_path = path.with_suffix(".json")
    with open(vocab_path, "w") as handle:
        json.dump(dict(vocab), handle)

    return path


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> Tuple[CaptioningModel, Dict[str, int], Dict[str, Any]]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: the ``.pt`` file to read.
        map_location: torch device mapping for the stored tensors.

    Returns:
        ``(model, vocab, metadata)`` where metadata carries ``epoch`` and
        ``val_bleu4`` alongside the config.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    config_dict = dict(payload["config"])
    config = ModelConfig(**config_dict)
    model = CaptioningModel.from_config(config)
    model.load_state_dict(payload["model_state"])

    metadata = {
        "epoch": payload.get("epoch", 0),
        "val_bleu4": payload.get("val_bleu4", 0.0),
        "config": config_dict,
    }
    return model, payload["vocab"], metadata


def infer_vocab_size(vocab_path: Path) -> int:
    """Number of entries in a ``vocab.json`` file."""
    with open(vocab_path) as handle:
        return len(json.load(handle))
