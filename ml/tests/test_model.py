"""Tests for the encoder, attention, decoder and checkpoint round-trip.

Shapes are checked against the contract in ``rule/project.mdc``:
features ``(B, 49, 2048)``, logits ``(B, max_len - 1, vocab)``,
alphas ``(B, max_len - 1, 49)``.

The ResNet weights are fetched from the torchvision hub on first run. Set
``TORCH_HOME`` to a warm cache to keep the suite offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from captioning.attention import Attention
from captioning.decoder import Decoder, caption_loss
from captioning.encoder import Encoder
from captioning.model import (
    CaptioningModel,
    ModelConfig,
    load_checkpoint,
    save_checkpoint,
)
from captioning.vocab import END, PAD, SPECIAL_TOKENS, START

VOCAB_SIZE = 12
EMBED_DIM = 32
DECODER_DIM = 32
ATTENTION_DIM = 32
DROPOUT = 0.5


@pytest.fixture(scope="module")
def encoder() -> Encoder:
    """ResNet-50 encoder; downloads ImageNet weights once per module."""
    return Encoder(freeze=True)


@pytest.fixture()
def decoder() -> Decoder:
    """Small decoder so the tests stay fast."""
    torch.manual_seed(0)
    return Decoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    )


@pytest.fixture()
def features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(3, 49, 2048)


@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A padded caption batch with two different true lengths."""
    torch.manual_seed(2)
    captions = torch.tensor(
        [
            [START, 4, 5, END, PAD, PAD],
            [START, 6, 7, 8, 9, END],
        ]
    )
    lengths = torch.tensor([4, 6])
    return captions, lengths, captions


# --- encoder --------------------------------------------------------------


def test_encoder_output_shape(encoder):
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = encoder(images)
    assert out.shape == (1, 49, 2048)


def test_encoder_batch_shape(encoder):
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = encoder(images)
    assert out.shape == (2, 49, 2048)


def test_encoder_is_frozen_by_default(encoder):
    encoder.set_finetune(False)
    assert encoder.trainable_parameters() == []
    assert all(not p.requires_grad for p in encoder.parameters())


def test_set_finetune_unfreezes_only_layer3_and_layer4(encoder):
    encoder.set_finetune(True)
    trainable = {id(p) for p in encoder.trainable_parameters()}
    assert trainable, "fine-tuning should expose some parameters"

    for name, param in encoder.named_parameters():
        is_late = name.startswith(("backbone.6.", "backbone.7."))
        assert param.requires_grad == is_late, name

    encoder.set_finetune(False)
    assert encoder.trainable_parameters() == []


def test_batchnorm_stays_in_eval_mode_when_finetuning(encoder):
    encoder.set_finetune(True)
    encoder.train()
    norms = [
        m
        for m in encoder.backbone.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert norms, "ResNet-50 should contain BatchNorm layers"
    assert all(not m.training for m in norms)
    # the trainable conv layers above them are still in train mode
    assert encoder.layer4[0].conv1.training
    encoder.set_finetune(False)
    encoder.eval()


def test_encoder_dropout_free_eval_is_deterministic(encoder):
    encoder.eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        first = encoder(images)
        second = encoder(images)
    assert torch.allclose(first, second, atol=1e-6)


# --- attention ------------------------------------------------------------


def test_attention_alpha_sums_to_one(features):
    attention = Attention(2048, DECODER_DIM, ATTENTION_DIM)
    hidden = torch.randn(3, DECODER_DIM)
    context, alpha = attention(features, hidden)

    assert context.shape == (3, 2048)
    assert alpha.shape == (3, 49)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(3), atol=1e-4)


def test_attention_alpha_is_non_negative(features):
    attention = Attention(2048, DECODER_DIM, ATTENTION_DIM)
    _context, alpha = attention(features, torch.randn(3, DECODER_DIM))
    assert (alpha >= 0).all()


def test_attention_context_is_weighted_mean(features):
    attention = Attention(2048, DECODER_DIM, ATTENTION_DIM)
    torch.manual_seed(3)
    context, alpha = attention(features, torch.randn(3, DECODER_DIM))
    expected = (features * alpha.unsqueeze(2)).sum(dim=1)
    assert torch.allclose(context, expected, atol=1e-5)


def test_attention_is_permutation_consistent(features):
    """Reordering the regions must reorder alpha and leave the context alone."""
    attention = Attention(2048, DECODER_DIM, ATTENTION_DIM)
    torch.manual_seed(4)
    hidden = torch.randn(1, DECODER_DIM)
    context, alpha = attention(features, hidden)

    perm = torch.randperm(49)
    permuted_context, permuted_alpha = attention(
        features[:, perm], hidden
    )
    assert torch.allclose(context, permuted_context, atol=1e-5)
    assert torch.allclose(alpha[:, perm], permuted_alpha, atol=1e-5)


# --- decoder --------------------------------------------------------------


def test_decoder_forward_shapes(decoder, features, batch):
    captions, lengths, _ = batch
    decoder.eval()
    with torch.no_grad():
        logits, sorted_caps, decode_lengths, alphas, sort_indices = decoder(
            features, captions, lengths
        )

    assert logits.shape == (2, 5, VOCAB_SIZE)  # (B, max_len - 1, V)
    assert alphas.shape == (2, 5, 49)  # (B, max_len - 1, num_regions)
    assert sorted_caps.shape == captions.shape
    assert decode_lengths.shape == (2,)
    assert sort_indices.shape == (2,)


def test_decoder_sorts_by_length_descending(decoder, features, batch):
    captions, lengths, _ = batch
    decoder.eval()
    with torch.no_grad():
        _logits, _sorted_caps, decode_lengths, _alphas, sort_indices = decoder(
            features, captions, lengths
        )
    # the short caption (4) must be sorted after the long one (6)
    assert sort_indices.tolist() == [1, 0]
    assert decode_lengths.tolist() == [5, 3]
    assert torch.all(decode_lengths[1:] <= decode_lengths[:-1])


def test_decoder_active_rows_only(decoder, features):
    """Finished rows must not be advanced, so their later logits stay zero."""
    captions = torch.tensor([[START, 4, END, PAD, PAD], [START, 5, 6, 7, END]])
    lengths = torch.tensor([3, 5])
    decoder.eval()
    with torch.no_grad():
        logits, _sorted_caps, decode_lengths, _alphas, sort_indices = decoder(
            features[:2], captions, lengths
        )
    # sorting puts the length-5 caption first and the length-3 caption second
    assert sort_indices.tolist() == [1, 0]
    assert decode_lengths.tolist() == [4, 2]
    # row 1 (length 3 -> 2 steps) is finished, so steps 2 and 3 stay zero
    assert logits[1, 2:].abs().sum() == 0
    # row 0 runs the full 4 steps
    assert logits[0, :4].abs().sum() > 0


def test_decoder_degenerate_start_only_batch(decoder, features):
    """A batch of <start>-only captions has no target step; it must not crash."""
    captions = torch.tensor([[START], [START]])
    lengths = torch.tensor([1, 1])
    decoder.eval()
    with torch.no_grad():
        logits, sorted_caps, decode_lengths, alphas, _idx = decoder(
            features[:2], captions, lengths
        )
    assert logits.shape == (2, 0, VOCAB_SIZE)
    assert alphas.shape == (2, 0, 49)
    assert decode_lengths.tolist() == [0, 0]


def test_decoder_padding_is_zero_beyond_decode_length(decoder, features):
    """Steps past a row's decode_length must be exactly zero, not stale values."""
    raw = [
        [START, 4, 5, END],
        [START, 7, END],
        [START, 8, 9, 10, 11, END],
    ]
    width = max(len(row) for row in raw)
    captions = torch.tensor([row + [PAD] * (width - len(row)) for row in raw])
    lengths = torch.tensor([len(row) for row in raw])

    decoder.eval()
    with torch.no_grad():
        logits, _sorted_caps, decode_lengths, alphas, _idx = decoder(
            features[:3], captions, lengths
        )

    for row, decode_length in enumerate(decode_lengths.tolist()):
        assert logits[row, decode_length:].abs().sum() == 0
        assert alphas[row, decode_length:].abs().sum() == 0


def test_decoder_sort_indices_restore_original_order(decoder, features):
    """Unsorting with sort_indices must recover the input captions exactly."""
    captions = torch.tensor(
        [
            [START, 4, END, PAD, PAD],
            [START, 5, 6, 7, END],
            [START, 8, END, PAD, PAD],
        ]
    )
    lengths = torch.tensor([3, 5, 3])
    decoder.eval()
    with torch.no_grad():
        _logits, sorted_caps, _dl, _alphas, sort_indices = decoder(
            features[:3], captions, lengths
        )
    restored = torch.empty_like(sorted_caps)
    restored[sort_indices] = sorted_caps
    assert torch.equal(restored, captions)


def test_decoder_single_caption(decoder, features):
    captions = torch.tensor([[START, 4, END]])
    lengths = torch.tensor([3])
    decoder.eval()
    with torch.no_grad():
        logits, _sorted_caps, decode_lengths, alphas, _idx = decoder(
            features[:1], captions, lengths
        )
    assert logits.shape == (1, 2, VOCAB_SIZE)
    assert alphas.shape == (1, 2, 49)
    assert decode_lengths.tolist() == [2]


def test_init_hidden_state_shape(decoder, features):
    hidden, cell = decoder.init_hidden_state(features)
    assert hidden.shape == (3, DECODER_DIM)
    assert cell.shape == (3, DECODER_DIM)


def test_context_gate_is_sigmoid_bounded(decoder, features):
    """beta must stay in (0, 1) so it can only scale the context."""
    decoder.eval()
    hidden, _cell = decoder.init_hidden_state(features)
    beta = torch.sigmoid(decoder.f_beta(hidden))
    assert (beta > 0).all() and (beta < 1).all()
    assert beta.shape == (3, 2048)


def test_gradients_reach_decoder_and_skip_frozen_encoder(decoder, features, batch):
    captions, lengths, _ = batch
    decoder.train()
    logits, sorted_caps, _dl, alphas, _idx = decoder(features, captions, lengths)
    loss, _ce = caption_loss(logits, alphas, sorted_caps)
    loss.backward()

    assert decoder.embedding.weight.grad is not None
    assert decoder.attention.score.weight.grad is not None
    assert decoder.lstm.weight_ih.grad is not None


def test_generate_shapes(decoder, features):
    decoder.eval()
    ids, alphas = decoder.generate(features, max_len=6)
    assert ids.shape[0] == 3
    assert ids.shape[1] <= 6
    assert alphas.shape == (ids.shape[0], ids.shape[1], 49)
    assert (ids < VOCAB_SIZE).all() and (ids >= 0).all()


# --- loss -----------------------------------------------------------------


def test_caption_loss_matches_manual_cross_entropy(decoder, features, batch):
    """Cross-entropy must average only over non-pad target positions."""
    captions, lengths, _ = batch
    decoder.eval()
    with torch.no_grad():
        logits, sorted_caps, _dl, alphas, _idx = decoder(features, captions, lengths)
        _total, ce = caption_loss(logits, alphas, sorted_caps)

        targets = sorted_caps[:, 1:]
        mask = targets != PAD
        # index only the real positions, so CE sees a flat (N, V) / (N,) pair
        rows, cols = mask.nonzero(as_tuple=True)
        picked = logits[rows, cols]
        manual = F.cross_entropy(picked, targets[rows, cols])
    assert torch.allclose(ce, manual, atol=1e-5)


def test_caption_loss_ignores_extra_padding(decoder, features):
    """Padding a short caption further must not change the cross-entropy."""
    captions = torch.tensor([[START, 4, END, PAD], [START, 5, 6, END]])
    lengths = torch.tensor([3, 4])
    decoder.eval()
    with torch.no_grad():
        logits, sorted_caps, _dl, alphas, _idx = decoder(features[:2], captions, lengths)
        short_ce, _ = caption_loss(logits, alphas, sorted_caps)

        # same real tokens, but the short row carries extra pad columns
        wider = torch.cat(
            [sorted_caps, sorted_caps.new_full((2, 2), PAD)], dim=1
        )
        # logits/alphas need the matching extra steps; reuse by padding them too
        padded_logits = torch.cat(
            [logits, logits.new_zeros(2, 2, VOCAB_SIZE)], dim=1
        )
        padded_alphas = torch.cat([alphas, alphas.new_zeros(2, 2, 49)], dim=1)
        wide_ce, _ = caption_loss(padded_logits, padded_alphas, wider)
    assert torch.allclose(short_ce, wide_ce, atol=1e-6)


def test_caption_loss_alpha_c_zero_drops_regularization(decoder, features, batch):
    captions, lengths, _ = batch
    decoder.eval()
    with torch.no_grad():
        logits, sorted_caps, _dl, alphas, _idx = decoder(features, captions, lengths)
        total, ce = caption_loss(logits, alphas, sorted_caps, alpha_c=0.0)
    assert torch.allclose(total, ce, atol=1e-6)


# --- learning -------------------------------------------------------------


def test_loss_decreases_on_tiny_batch(decoder):
    """Overfit a single random batch: 20 Adam steps must reduce the loss."""
    torch.manual_seed(7)
    batch_size = 4
    feats = torch.randn(batch_size, 49, 2048)
    captions = torch.tensor(
        [
            [START, 4, 5, 6, END, PAD],
            [START, 7, 8, END, PAD, PAD],
            [START, 9, 10, 11, 5, END],
            [START, 6, END, PAD, PAD, PAD],
        ]
    )
    lengths = torch.tensor([5, 4, 6, 3])

    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    first = last = None
    for step in range(20):
        logits, sorted_caps, _dl, alphas, _idx = decoder(feats, captions, lengths)
        loss, _ce = caption_loss(logits, alphas, sorted_caps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    # also well under the ln(V) a random model sits at
    assert last < torch.log(torch.tensor(float(VOCAB_SIZE))).item()


# --- config + checkpoints -------------------------------------------------


def test_model_config_defaults():
    config = ModelConfig(vocab_size=VOCAB_SIZE)
    assert config.encoder_dim == 2048
    assert config.num_regions == 49
    assert config.max_len == 50
    assert config.dropout == 0.5
    assert config.to_dict()["vocab_size"] == VOCAB_SIZE


def test_model_config_rejects_bad_vocab_size():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=0)


def test_model_from_config_shapes():
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
    )
    model = CaptioningModel.from_config(config)
    assert model.encode(torch.randn(1, 3, 224, 224)).shape == (1, 49, 2048)


def test_trainable_parameters_respects_freeze():
    config = ModelConfig(vocab_size=VOCAB_SIZE, freeze_encoder=True)
    model = CaptioningModel.from_config(config)
    assert all(p.requires_grad for p in model.trainable_parameters())
    assert model.encoder.trainable_parameters() == []

    model.set_finetune(True)
    decoder_params = sum(1 for _ in model.decoder.parameters())
    assert len(model.trainable_parameters()) > decoder_params


def test_checkpoint_round_trip(tmp_path: Path):
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
    )
    model = CaptioningModel.from_config(config)
    vocab = dict(SPECIAL_TOKENS)
    vocab.update({f"w{i}": i + 4 for i in range(VOCAB_SIZE - 4)})

    path = save_checkpoint(tmp_path / "weights" / "model.pt", model, vocab, epoch=3, val_bleu4=0.31)
    assert path.is_file()
    # vocab.json sits next to the .pt
    vocab_json = path.with_suffix(".json")
    assert vocab_json.is_file()
    assert json.loads(vocab_json.read_text()) == vocab

    restored, restored_vocab, metadata = load_checkpoint(path)
    assert restored_vocab == vocab
    assert metadata["epoch"] == 3
    assert metadata["val_bleu4"] == pytest.approx(0.31)
    assert metadata["config"]["encoder_dim"] == 2048
    assert metadata["config"]["num_regions"] == 49

    model.eval()
    restored.eval()
    images = torch.randn(1, 3, 224, 224)
    captions = torch.tensor([[START, 4, 5, END]])
    lengths = torch.tensor([4])
    with torch.no_grad():
        before = model(images, captions, lengths)[0]
        after = restored(images, captions, lengths)[0]
    assert torch.allclose(before, after, atol=1e-6)


def test_checkpoint_payload_keys(tmp_path: Path):
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
    )
    model = CaptioningModel.from_config(config)
    vocab = dict(SPECIAL_TOKENS)
    path = save_checkpoint(tmp_path / "m.pt", model, vocab, epoch=0, val_bleu4=0.0)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {"model_state", "vocab", "config", "epoch", "val_bleu4"}
    assert set(payload["model_state"]) == {"encoder", "decoder"}
    for key in (
        "encoder_dim",
        "attention_dim",
        "embed_dim",
        "decoder_dim",
        "dropout",
        "max_len",
        "num_regions",
    ):
        assert key in payload["config"], key
