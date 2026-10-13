"""Decoding strategies: greedy and beam search.

Both take encoder features ``(1, 49, 2048)`` and return the chosen token ids
together with the attention weights for that sequence, so the caller can render
a per-word heatmap.

The strategies only use the decoder's ``init_hidden_state`` and ``step``, so
they work with any model exposing those two methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .vocab import END, START, mask_generated_specials

MAX_LEN = 50
NEG_INF = float("-inf")


@dataclass
class _Beam:
    """One partial caption and the decoder state that goes with it."""

    score: float
    token_ids: List[int] = field(default_factory=list)
    alphas: List[torch.Tensor] = field(default_factory=list)
    hidden: torch.Tensor | None = None
    cell: torch.Tensor | None = None

    def extended(
        self,
        token: int,
        score: float,
        alpha: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> "_Beam":
        """A copy of this beam with one more token appended."""
        return _Beam(
            score=score,
            token_ids=[*self.token_ids, token],
            alphas=[*self.alphas, alpha],
            hidden=hidden,
            cell=cell,
        )


class DecodingStrategy(ABC):
    """Turns encoder features into a token sequence plus its attention."""

    @abstractmethod
    def decode(
        self, features: torch.Tensor, max_len: int = MAX_LEN
    ) -> Tuple[List[int], torch.Tensor]:
        """Decode a single image.

        Args:
            features: ``(1, num_regions, encoder_dim)`` encoder output.
            max_len: maximum number of tokens to produce.

        Returns:
            ``(token_ids, alphas)`` where ``token_ids`` is a list of ints with
            ``<start>`` / ``<end>`` removed, and ``alphas`` is
            ``(num_tokens, num_regions)`` with rows summing to 1.
        """

    @staticmethod
    def _validate(features: torch.Tensor) -> None:
        """Reject anything that is not a single-image feature tensor."""
        if features.dim() != 3 or features.size(0) != 1:
            raise ValueError(
                "expected features of shape (1, num_regions, encoder_dim), got "
                f"{tuple(features.shape)}"
            )

    @staticmethod
    def _stack_alphas(
        alphas: List[torch.Tensor], features: torch.Tensor
    ) -> torch.Tensor:
        """Stack per-token attention into ``(num_tokens, num_regions)``."""
        if not alphas:
            return features.new_zeros((0, features.size(1)))
        return torch.stack(alphas, dim=0)


class GreedyDecodingStrategy(DecodingStrategy):
    """Always extend with the highest-scoring token."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.decoder = decoder

    @torch.no_grad()
    def decode(
        self, features: torch.Tensor, max_len: int = MAX_LEN
    ) -> Tuple[List[int], torch.Tensor]:
        self._validate(features)

        hidden, cell = self.decoder.init_hidden_state(features)
        token = START
        token_ids: List[int] = []
        alphas: List[torch.Tensor] = []

        for _ in range(max_len):
            words = torch.full(
                (1,), token, dtype=torch.long, device=features.device
            )
            logits, hidden, cell, alpha = self.decoder.step(
                words, features, hidden, cell
            )
            logits = mask_generated_specials(logits)
            token = int(logits.argmax(dim=1).item())
            if token == END:
                break
            token_ids.append(token)
            alphas.append(alpha[0])

        return token_ids, self._stack_alphas(alphas, features)


class BeamSearchDecodingStrategy(DecodingStrategy):
    """Keep ``beam_size`` partial captions, scored by summed log-probability.

    Beams that emit ``<end>`` are banked and never extended again. When the
    search runs out of steps the highest-scoring banked beam wins, falling back
    to the best live beam if nothing terminated.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        beam_size: int = 3,
        length_penalty: float = 0.0,
    ) -> None:
        """Args:
            decoder: any model exposing ``init_hidden_state`` and ``step``.
            beam_size: how many partial captions to keep.
            length_penalty: exponent applied to the caption length when ranking
                beams. ``0.0`` is plain summed log-probability, which favours
                short captions; ``1.0`` divides by the length and favours long
                ones. Set it explicitly when one behaviour is wanted.
        """
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}")
        self.decoder = decoder
        self.beam_size = beam_size
        self.length_penalty = length_penalty

    @torch.no_grad()
    def decode(
        self, features: torch.Tensor, max_len: int = MAX_LEN
    ) -> Tuple[List[int], torch.Tensor]:
        self._validate(features)

        hidden, cell = self.decoder.init_hidden_state(features)
        beams = [_Beam(score=0.0, hidden=hidden[0], cell=cell[0])]
        finished: List[_Beam] = []

        for _ in range(max_len):
            batch = len(beams)
            stacked_features = features.expand(batch, -1, -1)
            stacked_hidden = torch.stack([beam.hidden for beam in beams])
            stacked_cell = torch.stack([beam.cell for beam in beams])
            words = torch.tensor(
                [beam.token_ids[-1] if beam.token_ids else START for beam in beams],
                dtype=torch.long,
                device=features.device,
            )

            logits, next_hidden, next_cell, alpha = self.decoder.step(
                words, stacked_features, stacked_hidden, stacked_cell
            )
            log_probs = F.log_softmax(
                mask_generated_specials(logits), dim=1
            )  # (batch, vocab)

            # Rank every (beam, token) pair together, so a promising token from
            # one beam can outrank a mediocre one from another.
            candidate_scores = (
                torch.tensor([beam.score for beam in beams], device=features.device)
                .unsqueeze(1)
                + log_probs
            )
            top_scores, top_indices = candidate_scores.view(-1).topk(
                min(self.beam_size * 2, candidate_scores.numel())
            )

            next_beams: List[_Beam] = []
            for flat_index, score in zip(
                top_indices.tolist(), top_scores.tolist()
            ):
                source = flat_index // log_probs.size(1)
                token = flat_index % log_probs.size(1)

                if token == END:
                    # An empty caption is never a useful answer, so a beam that
                    # would end before emitting anything is dropped rather than
                    # banked; otherwise its short, high score wins every search.
                    if beams[source].token_ids:
                        finished.append(
                            _Beam(
                                score=score,
                                token_ids=list(beams[source].token_ids),
                                alphas=list(beams[source].alphas),
                            )
                        )
                    continue

                next_beams.append(
                    beams[source].extended(
                        token=token,
                        score=score,
                        alpha=alpha[source],
                        hidden=next_hidden[source],
                        cell=next_cell[source],
                    )
                )
                if len(next_beams) == self.beam_size:
                    break

            if not next_beams:
                break
            beams = next_beams

        # Beams still running at max_len compete with the ones that reached
        # <end>. Excluding them would make beam search lose to greedy on any
        # caption the length cap truncated, since a long beam always carries a
        # lower summed log-probability than a short one. Ranking is what decides
        # the trade-off, and length_penalty controls how strongly.
        best = max([*finished, *beams], key=self._rank)

        return best.token_ids, self._stack_alphas(best.alphas, features)

    def _rank(self, beam: _Beam) -> float:
        """Ranking score for a finished beam.

        With ``length_penalty == 0`` this is the summed log-probability. A
        non-zero penalty divides by ``length ** length_penalty``, the usual
        fix for beam search's bias towards short sequences.
        """
        if self.length_penalty == 0.0:
            return beam.score
        length = max(len(beam.token_ids), 1)
        return beam.score / (length ** self.length_penalty)
