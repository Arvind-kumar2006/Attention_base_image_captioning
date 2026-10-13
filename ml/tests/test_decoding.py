"""Tests for the decoding strategies and the caption generator.

A tiny randomly initialised model is used throughout, so no training and no
ImageNet download are needed: the tests care about shapes, attention
normalisation and determinism, not about caption quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captioning.decoding import (
    MAX_LEN,
    BeamSearchDecodingStrategy,
    DecodingStrategy,
    GreedyDecodingStrategy,
)
from captioning.generator import (
    GRID,
    CaptionGenerator,
    CaptionResult,
    reshape_attention,
)
from captioning.model import ModelConfig, save_checkpoint
from captioning.vocab import END, PAD, SPECIAL_TOKENS, START

VOCAB_SIZE = 10
NUM_REGIONS = 49
EMBED = 16
DIM = 16


# --------------------------------------------------------------------------
# tiny model
# --------------------------------------------------------------------------


class TinyDecoder(torch.nn.Module):
    """LSTMCell decoder with attention, small enough to run instantly."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        from captioning.attention import Attention
        from captioning.vocab import PAD as PAD_ID

        self.embed = torch.nn.Embedding(vocab_size, EMBED, padding_idx=PAD_ID)
        self.attention = Attention(2048, DIM, DIM)
        self.init_h = torch.nn.Linear(2048, DIM)
        self.init_c = torch.nn.Linear(2048, DIM)
        self.lstm = torch.nn.LSTMCell(EMBED + 2048, DIM)
        self.out = torch.nn.Linear(DIM, vocab_size)

    def init_hidden_state(self, features):
        mean = features.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def step(self, word_ids, features, hidden, cell):
        embedded = self.embed(word_ids)
        context, alpha = self.attention(features, hidden)
        hidden, cell = self.lstm(torch.cat([embedded, context], dim=1), (hidden, cell))
        return self.out(hidden), hidden, cell, alpha


class TinyEncoder(torch.nn.Module):
    """Deterministic stand-in for the ResNet encoder.

    Pools to a small grid before projecting, so the parameter count stays in the
    tens of thousands. Projecting straight from a flattened 3x224x224 image would
    mean ~300M parameters and a 1.2 GB checkpoint per test.
    """

    def __init__(self, seed: int = 1) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.proj = torch.nn.Linear(3 * 2 * 2, 2048)

    def forward(self, images):
        pooled = self.pool(images).flatten(1)
        features = self.proj(pooled)  # (B, 2048)
        return features.unsqueeze(1).expand(-1, NUM_REGIONS, -1)


@pytest.fixture()
def decoder() -> TinyDecoder:
    return TinyDecoder()


@pytest.fixture()
def features() -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randn(1, NUM_REGIONS, 2048)


# --------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------


def test_strategies_subclass_the_interface(decoder):
    assert issubclass(GreedyDecodingStrategy, DecodingStrategy)
    assert issubclass(BeamSearchDecodingStrategy, DecodingStrategy)


def test_max_len_default_is_fifty():
    assert MAX_LEN == 50


def test_beam_size_must_be_positive(decoder):
    with pytest.raises(ValueError):
        BeamSearchDecodingStrategy(decoder, beam_size=0)


def test_decode_rejects_batched_features(decoder):
    strategy = GreedyDecodingStrategy(decoder)
    with pytest.raises(ValueError):
        strategy.decode(torch.randn(2, NUM_REGIONS, 2048))
    with pytest.raises(ValueError):
        strategy.decode(torch.randn(NUM_REGIONS, 2048))


# --------------------------------------------------------------------------
# greedy
# --------------------------------------------------------------------------


def test_greedy_return_types_and_shapes(decoder, features):
    strategy = GreedyDecodingStrategy(decoder)
    token_ids, alphas = strategy.decode(features)

    assert isinstance(token_ids, list)
    assert all(isinstance(token, int) for token in token_ids)
    assert isinstance(alphas, torch.Tensor)
    assert alphas.shape == (len(token_ids), NUM_REGIONS)
    assert alphas.dtype == torch.float32


def test_greedy_attention_rows_sum_to_one(decoder, features):
    token_ids, alphas = GreedyDecodingStrategy(decoder).decode(features)
    assert len(token_ids) > 0
    sums = alphas.sum(dim=1)
    assert torch.allclose(sums, torch.ones(len(token_ids)), atol=1e-4)


def test_greedy_never_emits_special_tokens(decoder, features):
    token_ids, _alphas = GreedyDecodingStrategy(decoder).decode(features)
    assert START not in token_ids
    assert END not in token_ids
    assert PAD not in token_ids


def test_greedy_is_deterministic(decoder, features):
    strategy = GreedyDecodingStrategy(decoder)
    first_ids, first_alphas = strategy.decode(features)
    for _ in range(3):
        ids, alphas = strategy.decode(features)
        assert ids == first_ids
        assert torch.equal(alphas, first_alphas)


def test_greedy_respects_max_len(decoder, features):
    token_ids, alphas = GreedyDecodingStrategy(decoder).decode(features, max_len=3)
    assert len(token_ids) <= 3
    assert alphas.shape == (len(token_ids), NUM_REGIONS)


def test_greedy_stops_at_end_token():
    """A decoder that always emits <end> must return nothing after step 0."""

    class AlwaysEnd(TinyDecoder):
        def step(self, word_ids, features, hidden, cell):
            logits, _h, _c, alpha = super().step(word_ids, features, hidden, cell)
            logits = torch.full_like(logits, -1e4)
            logits[:, END] = 1e4
            return logits, _h, _c, alpha

    strategy = GreedyDecodingStrategy(AlwaysEnd())
    token_ids, alphas = strategy.decode(torch.randn(1, NUM_REGIONS, 2048))
    assert token_ids == []
    assert alphas.shape == (0, NUM_REGIONS)


# --------------------------------------------------------------------------
# beam search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beam_size", [1, 2, 3, 5])
def test_beam_return_types_and_shapes(decoder, features, beam_size):
    strategy = BeamSearchDecodingStrategy(decoder, beam_size=beam_size)
    token_ids, alphas = strategy.decode(features)

    assert isinstance(token_ids, list)
    assert all(isinstance(token, int) for token in token_ids)
    assert isinstance(alphas, torch.Tensor)
    assert alphas.shape == (len(token_ids), NUM_REGIONS)


@pytest.mark.parametrize("beam_size", [1, 3, 5])
def test_beam_attention_rows_sum_to_one(decoder, features, beam_size):
    token_ids, alphas = BeamSearchDecodingStrategy(
        decoder, beam_size=beam_size
    ).decode(features)
    assert len(token_ids) > 0
    assert torch.allclose(
        alphas.sum(dim=1), torch.ones(len(token_ids)), atol=1e-4
    )


def test_beam_never_emits_special_tokens(decoder, features):
    token_ids, _alphas = BeamSearchDecodingStrategy(decoder, beam_size=3).decode(features)
    assert END not in token_ids
    assert START not in token_ids


def test_beam_respects_max_len(decoder, features):
    token_ids, alphas = BeamSearchDecodingStrategy(decoder, beam_size=3).decode(
        features, max_len=2
    )
    assert len(token_ids) <= 2
    assert alphas.shape == (len(token_ids), NUM_REGIONS)


def test_beam_size_one_equals_greedy(decoder, features):
    """With a single beam, beam search is greedy search."""
    greedy_ids, greedy_alphas = GreedyDecodingStrategy(decoder).decode(features)
    beam_ids, beam_alphas = BeamSearchDecodingStrategy(decoder, beam_size=1).decode(
        features
    )
    assert beam_ids == greedy_ids
    assert torch.allclose(beam_alphas, greedy_alphas, atol=1e-6)


def test_beam_is_deterministic(decoder, features):
    strategy = BeamSearchDecodingStrategy(decoder, beam_size=3)
    first_ids, first_alphas = strategy.decode(features)
    for _ in range(2):
        ids, alphas = strategy.decode(features)
        assert ids == first_ids
        assert torch.equal(alphas, first_alphas)


def test_wider_beam_scores_at_least_as_well_as_greedy(decoder, features):
    """Under the default no-penalty ranking, a wider beam cannot score worse.

    Greedy commits to the locally best token at each step. Beam search explores
    alternatives, so the sequence it returns must have a summed log-probability
    at least as high as greedy's under the same metric.
    """
    import torch.nn.functional as F

    def summed_log_prob(token_ids):
        """Score a token sequence by replaying it through the decoder."""
        hidden, cell = decoder.init_hidden_state(features)
        total = 0.0
        word = START
        for token in token_ids:
            logits, hidden, cell, _alpha = decoder.step(
                torch.tensor([word]), features, hidden, cell
            )
            total += float(F.log_softmax(logits, dim=1)[0, token].item())
            word = token
        return total

    greedy_ids, _ = GreedyDecodingStrategy(decoder).decode(features)
    strategy = BeamSearchDecodingStrategy(decoder, beam_size=5)
    assert strategy.length_penalty == 0.0
    wide_ids, _ = strategy.decode(features)

    assert summed_log_prob(wide_ids) >= summed_log_prob(greedy_ids) - 1e-6


def test_beam_search_escapes_a_greedy_dead_end(features):
    """With a length penalty, beam search outruns greedy on a trapped decoder.

    The decoder makes token 4 look marginally best at step 0 (logit 1.0 vs
    0.95) but forces an immediate <end> after it, while token 5 loops at high
    probability. Greedy commits to 4 and stops after one word; a beam that
    ranks by average log-probability keeps 5 alive and returns the long caption.
    """
    from captioning.vocab import END as END_ID
    from captioning.vocab import START as START_ID

    class DeadEnd(TinyDecoder):
        def step(self, word_ids, features, hidden, cell):
            logits, _h, _c, alpha = super().step(word_ids, features, hidden, cell)
            out = torch.full_like(logits, -1e4)
            for index in range(logits.size(0)):
                token = int(word_ids[index])
                if token == START_ID:
                    out[index, 4] = 1.0
                    out[index, 5] = 0.95
                elif token == 4:
                    out[index, END_ID] = 0.0  # forced to stop
                else:
                    out[index, 5] = 1.0  # loops at high probability
            return out, _h, _c, alpha

    decoder = DeadEnd()
    greedy_ids, _alphas = GreedyDecodingStrategy(decoder).decode(features)
    assert greedy_ids == [4]

    beam_ids, beam_alphas = BeamSearchDecodingStrategy(
        decoder, beam_size=3, length_penalty=1.0
    ).decode(features)
    assert len(beam_ids) > len(greedy_ids)
    assert beam_ids[0] == 5
    assert END_ID not in beam_ids
    assert beam_alphas.shape == (len(beam_ids), NUM_REGIONS)


def test_beam_length_penalty_zero_prefers_shorter(features):
    """The default (no penalty) ranks by summed log-probability."""
    from captioning.vocab import END as END_ID
    from captioning.vocab import START as START_ID

    class DeadEnd(TinyDecoder):
        def step(self, word_ids, features, hidden, cell):
            logits, _h, _c, alpha = super().step(word_ids, features, hidden, cell)
            out = torch.full_like(logits, -1e4)
            for index in range(logits.size(0)):
                token = int(word_ids[index])
                if token == START_ID:
                    out[index, 4] = 1.0
                    out[index, 5] = 0.95
                elif token == 4:
                    out[index, END_ID] = 0.0
                else:
                    out[index, 5] = 1.0
            return out, _h, _c, alpha

    strategy = BeamSearchDecodingStrategy(DeadEnd(), beam_size=3)
    assert strategy.length_penalty == 0.0
    token_ids, _alphas = strategy.decode(features, max_len=6)
    # summed log-prob favours the short branch, matching greedy
    assert token_ids == [4]


def test_beam_with_zero_capacity_stops_cleanly(decoder, features):
    """No beam may reach <end>; the best live beam is returned."""

    class NeverEnd(TinyDecoder):
        def step(self, word_ids, features, hidden, cell):
            logits, _h, _c, alpha = super().step(word_ids, features, hidden, cell)
            logits = torch.full_like(logits, -1e4)
            logits[:, 5] = 1e4  # never <end>
            return logits, _h, _c, alpha

    token_ids, alphas = BeamSearchDecodingStrategy(
        NeverEnd(), beam_size=3
    ).decode(features, max_len=4)
    assert len(token_ids) == 4
    assert alphas.shape == (4, NUM_REGIONS)


# --------------------------------------------------------------------------
# attention reshaping
# --------------------------------------------------------------------------


def test_reshape_attention_produces_7x7_grid():
    alphas = torch.rand(5, 49)
    grid = reshape_attention(alphas)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (5, GRID, GRID)
    assert np.allclose(grid.sum(axis=(1, 2)), alphas.sum(dim=1).numpy(), atol=1e-6)


def test_reshape_attention_handles_empty():
    grid = reshape_attention(torch.zeros(0, 49))
    assert grid.shape == (0, GRID, GRID)


def test_reshape_attention_rejects_bad_shapes():
    with pytest.raises(ValueError):
        reshape_attention(torch.rand(49))
    with pytest.raises(ValueError):
        reshape_attention(torch.rand(3, 50))  # not a perfect square


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

VOCAB = dict(SPECIAL_TOKENS)
VOCAB.update({f"w{i}": i + 4 for i in range(VOCAB_SIZE - 4)})


@pytest.fixture()
def checkpoint(tmp_path, monkeypatch):
    """A checkpoint holding the tiny encoder/decoder, plus the patched loader."""
    import captioning.generator as generator_module

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = TinyEncoder()
            self.decoder = TinyDecoder()
            self.config = ModelConfig(vocab_size=VOCAB_SIZE, max_len=50)

        def encode(self, images):
            return self.encoder(images)

        def state_dict(self):
            return {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}

        def load_state_dict(self, state, strict=True):
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])

    model = TinyModel()
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, VOCAB, epoch=0, val_bleu4=0.0)

    def fake_load(path, map_location="cpu"):
        fresh = TinyModel()
        fresh.to(map_location)
        return fresh, dict(VOCAB), {"epoch": 0, "val_bleu4": 0.0, "config": {}}

    monkeypatch.setattr(generator_module, "load_checkpoint", fake_load)
    return path


@pytest.fixture()
def image():
    from PIL import Image

    return Image.new("RGB", (300, 200), color=(40, 90, 160))


def test_generator_result_types_and_shapes(checkpoint, image):
    result = CaptionGenerator(checkpoint).generate(image)

    assert isinstance(result, CaptionResult)
    assert isinstance(result.caption, str)
    assert isinstance(result.words, list)
    assert all(isinstance(word, str) for word in result.words)
    assert isinstance(result.attention, np.ndarray)
    assert result.attention.dtype == np.float32
    assert result.attention.ndim == 3
    assert result.attention.shape == (len(result.words), GRID, GRID)
    assert isinstance(result.time_ms, float)
    assert result.time_ms > 0.0
    assert result.num_words == len(result.words)


def test_generator_attention_rows_sum_to_one(checkpoint, image):
    result = CaptionGenerator(checkpoint).generate(image)
    assert result.attention.shape[0] > 0
    sums = result.attention.sum(axis=(1, 2))
    assert np.allclose(sums, np.ones(len(result.words)), atol=1e-4)


def test_generator_caption_matches_words(checkpoint, image):
    result = CaptionGenerator(checkpoint).generate(image)
    assert result.caption == " ".join(result.words)


def test_generator_strips_special_tokens(checkpoint, image):
    result = CaptionGenerator(checkpoint).generate(image)
    for token in ("<start>", "<end>", "<pad>"):
        assert token not in result.words
        assert token not in result.caption


def test_generator_is_deterministic(checkpoint, image):
    first = CaptionGenerator(checkpoint).generate(image)
    for _ in range(2):
        again = CaptionGenerator(checkpoint).generate(image)
        assert again.caption == first.caption
        assert np.array_equal(again.attention, first.attention)


def test_generator_rejects_non_image(checkpoint):
    with pytest.raises(TypeError):
        CaptionGenerator(checkpoint).generate("not-an-image")


def test_generator_handles_greyscale_and_rgba(checkpoint):
    from PIL import Image

    for mode in ("L", "RGBA"):
        result = CaptionGenerator(checkpoint).generate(Image.new(mode, (64, 64)))
        assert result.attention.shape[1:] == (GRID, GRID)


def test_generator_uses_beam_strategy_when_given(checkpoint, image):
    from captioning.decoding import BeamSearchDecodingStrategy

    model_holder = {}
    generator = CaptionGenerator(checkpoint)
    model_holder["greedy"] = generator.generate(image).caption

    beam_generator = CaptionGenerator(
        checkpoint,
        strategy=BeamSearchDecodingStrategy(generator.model.decoder, beam_size=3),
    )
    beam_result = beam_generator.generate(image)
    assert beam_result.attention.shape == (len(beam_result.words), GRID, GRID)
    assert isinstance(beam_generator.strategy, BeamSearchDecodingStrategy)


def test_generator_defaults_to_greedy(checkpoint):
    generator = CaptionGenerator(checkpoint)
    assert isinstance(generator.strategy, GreedyDecodingStrategy)


def test_generator_model_is_in_eval_mode(checkpoint):
    generator = CaptionGenerator(checkpoint)
    assert generator.model.training is False
    assert all(not module.training for module in generator.model.modules())


def test_generator_overlay_shapes(checkpoint, image):
    maps = CaptionGenerator(checkpoint).generate_attention_overlay(image)
    assert len(maps) > 0
    for entry in maps:
        assert isinstance(entry["word"], str)
        assert entry["heatmap"].shape == (200, 300)  # matches the image
        assert entry["alpha"] == pytest.approx(0.55)
        assert entry["heatmap"].min() >= 0.0
        assert entry["heatmap"].max() <= 1.0
        # min-max normalised, so the peak reaches ~1.0 even for near-uniform
        # attention, where the value range is tiny
        assert entry["heatmap"].max() > 0.99


# --- <pad> / <start> must never be generated -------------------------------
#
# Regression: a trained model can put its highest logit on <pad> mid-caption.
# decode(strip_specials=True) drops that id, so the caption came back shorter
# than the attention rows and the API failed with
#   ValueError: attention has 2450 values but 28 words imply 1372
# Both are generated by masking those two ids out of the logits.


class PadEmittingDecoder(TinyDecoder):
    """A decoder whose argmax is <pad> on every step unless it is masked."""

    def step(self, word_ids, features, hidden, cell):
        logits, _h, _c, alpha = super().step(word_ids, features, hidden, cell)
        out = torch.full_like(logits, -1e4)
        out[:, PAD] = 1e4  # <pad> wins outright
        out[:, 5] = 2.0  # a real word, and the only other candidate
        return out, _h, _c, alpha


def test_mask_generated_specials_blocks_pad_and_start():
    from captioning.vocab import START as START_ID
    from captioning.vocab import mask_generated_specials

    logits = torch.zeros(1, 6)
    logits[0, PAD] = 50.0  # <pad> is the top logit
    logits[0, START_ID] = 40.0
    logits[0, 5] = 1.0

    masked = mask_generated_specials(logits)

    assert masked[0, PAD] == float("-inf")
    assert masked[0, START_ID] == float("-inf")
    assert int(masked.argmax()) == 5
    # the original is untouched
    assert logits[0, PAD] == 50.0


@pytest.mark.parametrize("beam_size", [1, 3, 5])
def test_strategies_never_generate_pad(beam_size):
    features = torch.randn(1, NUM_REGIONS, 2048)
    decoder = PadEmittingDecoder()
    token_ids, alphas = BeamSearchDecodingStrategy(
        decoder, beam_size=beam_size
    ).decode(features, max_len=6)

    assert PAD not in token_ids
    assert START not in token_ids
    # the invariant the API depends on: one attention row per word
    assert alphas.shape == (len(token_ids), NUM_REGIONS)


def test_greedy_never_generates_pad():
    features = torch.randn(1, NUM_REGIONS, 2048)
    token_ids, alphas = GreedyDecodingStrategy(PadEmittingDecoder()).decode(
        features, max_len=6
    )
    assert PAD not in token_ids
    assert alphas.shape == (len(token_ids), NUM_REGIONS)


def test_real_decoder_generate_never_generates_pad():
    """The production Decoder.generate, with <pad> pushed to the top logit."""
    from captioning.decoder import Decoder

    decoder = Decoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED, decoder_dim=DIM, attention_dim=DIM
    )
    with torch.no_grad():
        decoder.fc.bias[PAD] = 100.0  # <pad> wins every step unless masked
        decoder.fc.bias[5] = 1.0

    features = torch.randn(1, NUM_REGIONS, 2048)
    ids, alphas = decoder.generate(features, max_len=6)

    assert (ids != PAD).all()
    assert (ids != START).all()
    assert alphas.shape[0] == ids.shape[0]


def test_generator_keeps_words_and_attention_aligned(checkpoint):
    """End to end: a pad-happy decoder must still yield a consistent result.

    This is the exact shape the backend rejects with a 500, so it is asserted
    through CaptionGenerator rather than only at the strategy level.
    """
    from PIL import Image

    from captioning.generator import CaptionGenerator

    generator = CaptionGenerator(checkpoint)
    generator.strategy = GreedyDecodingStrategy(PadEmittingDecoder())

    result = generator.generate(Image.new("RGB", (60, 40)))

    assert len(result.words) > 0
    assert result.attention.shape == (len(result.words), GRID, GRID)
    assert PAD not in result.words
    assert START not in result.words
