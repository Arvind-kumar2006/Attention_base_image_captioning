"""LSTM decoder with attention and a gated context.

Mirrors Part 4 of ``ml/notebooks/captioning_Image.ipynb``::

    <start> -> Embedding -> [ Attention -> LSTM ] -> Linear -> word

``h0`` / ``c0`` come from the mean of the encoder features and a sigmoid gate
``beta`` scales the context before it is concatenated with the word embedding.

Training uses teacher forcing. Because the batch is sorted by length descending,
step ``t`` only advances the rows whose captions are still running, so finished
rows cost nothing.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .attention import Attention
from .encoder import ENCODER_DIM
from .vocab import END, PAD, START, mask_generated_specials


class Decoder(nn.Module):
    """Attention-based LSTMCell decoder over region features."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        encoder_dim: int = ENCODER_DIM,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # initial state from mean feature
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gate on the context
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initial LSTM state ``(h0, c0)`` from the mean encoder feature."""
        mean = features.mean(dim=1)  # (B, encoder_dim)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def step(
        self,
        word_ids: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step.

        Args:
            word_ids: ``(B,)`` previous word ids (``<start>`` at t=0).
            features: ``(B, num_regions, encoder_dim)`` encoder output.
            hidden: ``(B, decoder_dim)`` LSTM hidden state.
            cell: ``(B, decoder_dim)`` LSTM cell state.

        Returns:
            ``(logits, hidden, cell, alpha)`` with logits ``(B, vocab_size)``
            and alpha ``(B, num_regions)``.
        """
        embedded = self.embedding(word_ids)  # (B, embed_dim)
        context, alpha = self.attention(features, hidden)  # look at the image
        context = torch.sigmoid(self.f_beta(hidden)) * context  # gate
        hidden, cell = self.lstm(torch.cat([embedded, context], dim=1), (hidden, cell))
        logits = self.fc(self.dropout(hidden))
        return logits, hidden, cell, alpha

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced decoding over a length-sorted batch.

        Args:
            features: ``(B, num_regions, encoder_dim)`` encoder output.
            captions: ``(B, max_len)`` ids including ``<start>`` / ``<end>`` / ``<pad>``.
            lengths: ``(B,)`` true caption lengths, ``<pad>`` excluded.

        Returns:
            ``(logits, sorted_captions, decode_lengths, alphas, sort_indices)``
            where logits is ``(B, max_len - 1, vocab_size)`` predicting
            ``sorted_captions[:, 1:]``, alphas is ``(B, max_len - 1, num_regions)``,
            and ``sort_indices`` maps sorted rows back to input order.
        """
        batch_size, max_len = captions.shape
        num_regions = features.size(1)

        # Longest first so that rows finish early and get skipped.
        sorted_lengths, sort_indices = torch.sort(lengths, descending=True)
        captions = captions[sort_indices]
        features = features[sort_indices]
        decode_lengths = sorted_lengths - 1  # steps still to run per row

        hidden, cell = self.init_hidden_state(features)

        # Because the batch is sorted by length, the active rows are always a
        # prefix, so the still-running count is all we need. Per-row lists are
        # padded at the end rather than written into a tensor in place: an
        # in-place write would invalidate the autograd graph of the state.
        logits_per_row: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        alphas_per_row: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for step in range(max_len - 1):
            num_active = int((decode_lengths > step).sum())
            if num_active == 0:
                break
            step_logits, step_hidden, step_cell, step_alpha = self.step(
                captions[:num_active, step],
                features[:num_active],
                hidden[:num_active],
                cell[:num_active],
            )
            for row in range(num_active):
                logits_per_row[row].append(step_logits[row])
                alphas_per_row[row].append(step_alpha[row])
            # Out-of-place so the autograd graph of the previous state stays valid.
            hidden = torch.cat([step_hidden, hidden[num_active:]], dim=0)
            cell = torch.cat([step_cell, cell[num_active:]], dim=0)

        steps = max_len - 1
        if all(not row for row in logits_per_row):
            # Degenerate batch: no caption has a target step (<start> only).
            logits = features.new_zeros(batch_size, steps, self.vocab_size)
            alphas = features.new_zeros(batch_size, steps, num_regions)
            return logits, captions, decode_lengths, alphas, sort_indices

        logits = pad_sequence(
            [torch.stack(row) for row in logits_per_row],
            batch_first=True,
            padding_value=0.0,
        )
        alphas = pad_sequence(
            [torch.stack(row) for row in alphas_per_row],
            batch_first=True,
            padding_value=0.0,
        )

        return logits, captions, decode_lengths, alphas, sort_indices

    @torch.no_grad()
    def generate(
        self, features: torch.Tensor, max_len: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding.

        Args:
            features: ``(B, num_regions, encoder_dim)`` encoder output.
            max_len: maximum number of words to generate.

        Returns:
            ``(ids, alphas)`` with ids ``(B, <= max_len)`` and alphas
            ``(B, <= max_len, num_regions)``. Rows are *not* right-padded;
            stopping at ``<end>`` happens for the whole batch at once.
        """
        batch_size = features.size(0)
        hidden, cell = self.init_hidden_state(features)
        word = torch.full(
            (batch_size,), START, dtype=torch.long, device=features.device
        )
        ids: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        for _ in range(max_len):
            logits, hidden, cell, alpha = self.step(word, features, hidden, cell)
            word = mask_generated_specials(logits).argmax(dim=1)
            ids.append(word)
            alphas.append(alpha)
            if (word == END).all():
                break
        return torch.stack(ids, dim=1), torch.stack(alphas, dim=1)


def caption_loss(
    logits: torch.Tensor,
    alphas: torch.Tensor,
    captions: torch.Tensor,
    alpha_c: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over non-padded positions + doubly stochastic regularisation.

    Args:
        logits: ``(B, T - 1, vocab_size)`` predictions for ``captions[:, 1:]``.
        alphas: ``(B, T - 1, num_regions)`` attention weights.
        captions: ``(B, T)`` sorted captions containing ``<start>`` / ``<end>``.
        alpha_c: weight on the attention regularisation term.

    Returns:
        ``(total_loss, cross_entropy)``.
    """
    targets = captions[:, 1:]
    cross_entropy = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD,
    )
    # Only real (non-pad) words contribute attention mass.
    mask = (targets != PAD).float().unsqueeze(2)
    alpha_sum = (alphas * mask).sum(dim=1)  # (B, num_regions)
    regularization = ((1.0 - alpha_sum) ** 2).mean()
    return cross_entropy + alpha_c * regularization, cross_entropy
