"""ResNet-50 CNN encoder producing 7x7 region features.

Mirrors Part 1 of ``ml/notebooks/captioning_Image.ipynb``: the classification
head (avgpool + fc) is dropped so the network outputs a spatial map that is
flattened into 49 region vectors for the attention module.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

ENCODER_DIM = 2048
NUM_REGIONS = 49
INPUT_SIZE = 224


class Encoder(nn.Module):
    """ResNet-50 without avgpool/fc: ``(B, 3, 224, 224) -> (B, 49, 2048)``.

    The backbone is frozen by default. :meth:`set_finetune` optionally unfreezes
    only ``layer3`` and ``layer4``; BatchNorm statistics stay frozen in eval mode
    either way, since a tiny fine-tuning batch would otherwise wreck them.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        full = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # children are conv1,bn1,relu,maxpool,layer1..layer4,avgpool,fc
        self.backbone = nn.Sequential(*list(full.children())[:-2])
        self.finetune = False
        self.set_finetune(not freeze)

    @property
    def layer3(self) -> nn.Module:
        """The stage unfrozen by :meth:`set_finetune`."""
        return self.backbone[6]

    @property
    def layer4(self) -> nn.Module:
        """The stage unfrozen by :meth:`set_finetune`."""
        return self.backbone[7]

    def set_finetune(self, flag: bool) -> None:
        """Freeze everything, or unfreeze only ``layer3`` and ``layer4``."""
        self.finetune = bool(flag)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.finetune:
            for module in (self.layer3, self.layer4):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Parameters that currently receive gradients."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> "Encoder":
        """Keep every BatchNorm in eval mode, including during fine-tuning."""
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into region features ``(B, 49, 2048)``."""
        feature_map = self.backbone(images)  # (B, 2048, 7, 7)
        return feature_map.flatten(2).transpose(1, 2)  # (B, 49, 2048)
