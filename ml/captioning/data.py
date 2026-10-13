"""Image / caption dataset, transforms and batching.

Mirrors Part 2 of ``ml/notebooks/captioning_Image.ipynb``: COCO images resized
to 224x224 for eval (no crop, so the 7x7 ResNet grid covers the whole frame) and
random-cropped to 224x224 for training, with ImageNet normalisation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms as T

from .vocab import PAD, encode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_NORMALIZE = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def get_transform(train: bool) -> T.Compose:
    """Return the torchvision transform pipeline for ``train`` or eval."""
    if train:
        return T.Compose([T.Resize(256), T.RandomCrop(224), T.ToTensor(), _NORMALIZE])
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), _NORMALIZE])


class CocoCaptions(Dataset):
    """One item = (image tensor, caption ids).

    Training picks a random one of the image's captions; eval always takes the
    first so that validation is deterministic.
    """

    def __init__(
        self,
        items: Sequence[Mapping[str, Any]],
        vocab: Mapping[str, int],
        root: Path | str,
        train: bool,
    ) -> None:
        self.items = items
        self.vocab = vocab
        self.root = Path(root)
        self.train = train
        self.transform = get_transform(train)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Tokens -> ids using this dataset's vocabulary."""
        return encode(tokens, self.vocab)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = self.transform(Image.open(self.root / item["path"]).convert("RGB"))
        captions: Sequence[Sequence[str]] = item["caps"]
        caption = random.choice(captions) if self.train else captions[0]
        return image, torch.tensor(self.encode(caption), dtype=torch.long)


def collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack images, right-pad caption ids with ``<pad>`` and report true lengths.

    Returns ``(images, captions, lengths)`` with images ``(B, 3, 224, 224)``,
    captions ``(B, max_len)`` and lengths ``(B,)``.
    """
    images, captions = zip(*batch)
    lengths = torch.tensor([len(caption) for caption in captions], dtype=torch.long)
    padded = pad_sequence(captions, batch_first=True, padding_value=PAD)
    return torch.stack(images), padded, lengths
