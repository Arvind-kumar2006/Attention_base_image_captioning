"""Additive soft attention over the 49 image regions.

Mirrors Part 3 of ``ml/notebooks/captioning_Image.ipynb`` (Xu et al. 2015):

    e_i     = w^T tanh(W_enc a_i + W_dec h)
    alpha   = softmax over the 49 regions
    context = sum_i alpha_i a_i
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ENCODER_DIM, NUM_REGIONS


class Attention(nn.Module):
    """Deterministic (soft) attention conditioned on the decoder hidden state.

    ``forward(features, h)`` takes region features ``(B, 49, 2048)`` and the
    decoder state ``(B, decoder_dim)``, returning the attended context
    ``(B, 2048)`` and the region weights ``(B, 49)`` which sum to 1.
    """

    def __init__(
        self,
        encoder_dim: int = ENCODER_DIM,
        decoder_dim: int = 512,
        attention_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)  # W_enc
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)  # W_dec
        self.score = nn.Linear(attention_dim, 1)  # w

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over the regions.

        Args:
            features: ``(B, num_regions, encoder_dim)`` region features.
            hidden: ``(B, decoder_dim)`` decoder hidden state.

        Returns:
            ``(context, alpha)`` where context is ``(B, encoder_dim)`` and alpha
            is ``(B, num_regions)`` with rows summing to 1.
        """
        # (B, 49, A) + (B, 1, A) -> (B, 49, A) -> (B, 49)
        energies = self.score(
            torch.tanh(self.encoder_proj(features) + self.decoder_proj(hidden).unsqueeze(1))
        ).squeeze(2)

        alpha = F.softmax(energies, dim=1)  # weights over regions
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (B, 2048)
        return context, alpha

    def extra_repr(self) -> str:
        return (
            f"num_regions={NUM_REGIONS}, "
            f"encoder_dim={self.encoder_proj.in_features}, "
            f"decoder_dim={self.decoder_proj.in_features}, "
            f"attention_dim={self.encoder_proj.out_features}"
        )
