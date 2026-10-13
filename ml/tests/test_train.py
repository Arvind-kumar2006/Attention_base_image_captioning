"""Tests for the ml.train entry point.

Everything runs against a 3-image fake dataset in a temp folder with a tiny
vocab, and a stub model so no ResNet weights are downloaded and no real training
happens.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from captioning.vocab import invert_vocab
from ml import train as train_module
from ml.train import (
    FINETUNE_LR,
    GRAD_CLIP,
    append_metrics,
    apply_finetune,
    build_dataloaders,
    build_optimizer,
    first_caption_only,
    greedy_bleu4,
    load_split,
    load_vocab,
    main,
    parse_args,
    restore,
    run_epoch,
    save_resume_state,
    take_subset,
)

VOCAB = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3, "a": 4, "dog": 5, "runs": 6}


# --------------------------------------------------------------------------
# tiny fake dataset
# --------------------------------------------------------------------------


def _image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color=(10, 200, 90)).save(path)


@pytest.fixture()
def fake_data(tmp_path: Path) -> dict:
    coco_root = tmp_path / "coco"
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)

    caps = [["a", "dog", "runs"], ["a", "dog", "sleeps"], ["a", "dog", "eats"]]
    train, val = [], []
    for index in range(3):
        name = f"train2014/t{index}.jpg"
        _image(coco_root / name)
        train.append({"path": name, "caps": caps})
    for index in range(2):
        name = f"val2014/v{index}.jpg"
        _image(coco_root / name)
        val.append({"path": name, "caps": [["a", "dog", "runs"]]})

    (data_dir / "train.json").write_text(json.dumps(train))
    (data_dir / "val.json").write_text(json.dumps(val))
    (data_dir / "test.json").write_text(json.dumps([]))
    (data_dir / "vocab.json").write_text(json.dumps(VOCAB))
    return {"coco_root": coco_root, "data_dir": data_dir, "out_dir": tmp_path / "weights"}


# --------------------------------------------------------------------------
# stub model, so the tests need no ResNet weights
# --------------------------------------------------------------------------


class StubDecoder(torch.nn.Module):
    """Mimics the parts of ``Decoder`` that the training loop uses."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.out = torch.nn.Linear(2048, vocab_size)

    def forward(self, features, captions, lengths):
        batch = features.size(0)
        steps = captions.size(1) - 1
        logits = self.out(features.mean(dim=1)).unsqueeze(1).expand(
            batch, steps, self.out.out_features
        )
        alphas = features.new_full((batch, steps, features.size(1)), 1.0 / features.size(1))
        return logits, captions, lengths - 1, alphas, torch.arange(batch)

    def generate(self, features, max_len: int = 50):
        batch = features.size(0)
        ids = torch.full((batch, max_len), 5, dtype=torch.long)
        alphas = features.new_zeros(batch, max_len, features.size(1))
        return ids, alphas


class StubEncoder(torch.nn.Module):
    """Mimics the real Encoder: frozen by default, BN always in eval."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2048, 3, stride=32)
        self.bn = torch.nn.BatchNorm2d(2048)
        self.finetune = False
        self.backbone = torch.nn.Sequential(self.conv, self.bn)
        self.set_finetune(False)

    def set_finetune(self, flag: bool) -> None:
        self.finetune = bool(flag)
        for param in self.parameters():
            param.requires_grad = False
        if flag:
            for param in self.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True) -> "StubEncoder":
        super().train(mode)
        self.bn.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, images):
        return self.backbone(images).flatten(2).transpose(1, 2)


@pytest.fixture()
def stub_model(monkeypatch):
    """Replace CaptioningModel.from_config with a tiny stand-in."""
    import ml.captioning.encoder as encoder_module

    class StubModel(train_module.CaptioningModel):
        def __init__(self, vocab_size: int) -> None:
            from captioning.model import ModelConfig

            torch.nn.Module.__init__(self)
            self.encoder = StubEncoder()
            self.decoder = StubDecoder(vocab_size)
            self.config = ModelConfig(vocab_size=vocab_size)

        @classmethod
        def from_config(cls, config):
            return cls(config.vocab_size)

    monkeypatch.setattr(train_module, "CaptioningModel", StubModel)
    del encoder_module
    return StubModel


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def test_defaults():
    args = parse_args(
        ["--coco-root", "c", "--data-dir", "d", "--out-dir", "o"]
    )
    assert args.batch_size == 32
    assert args.lr == 4e-4
    assert args.epochs == 5
    assert args.patience == 3
    assert args.subset is None
    assert args.overfit is None
    assert args.finetune_after is None
    assert args.resume is None
    assert args.amp is False
    assert args.workers >= 1


def test_all_flags_parse():
    args = parse_args(
        [
            "--coco-root", "c", "--data-dir", "d", "--out-dir", "o",
            "--subset", "100", "--epochs", "2", "--batch-size", "8",
            "--lr", "1e-3", "--workers", "0", "--amp",
            "--resume", "r.pt", "--overfit", "4",
            "--finetune-after", "2", "--patience", "5",
        ]
    )
    assert (args.subset, args.epochs, args.batch_size) == (100, 2, 8)
    assert args.lr == 1e-3
    assert args.workers == 0
    assert args.amp is True
    assert str(args.resume) == "r.pt"
    assert args.overfit == 4
    assert args.finetune_after == 2
    assert args.patience == 5


def test_required_args_enforced():
    with pytest.raises(SystemExit):
        parse_args([])


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------


def test_load_split_and_vocab(fake_data):
    assert len(load_split(fake_data["data_dir"], "train")) == 3
    assert load_vocab(fake_data["data_dir"]) == VOCAB


def test_load_split_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        load_split(tmp_path, "train")


def test_take_subset():
    items = [{"path": str(i), "caps": [["a"]]} for i in range(5)]
    assert len(take_subset(items, None)) == 5
    assert [i["path"] for i in take_subset(items, 2)] == ["0", "1"]


def test_first_caption_only():
    items = [{"path": "a", "caps": [["x"], ["y"], ["z"]]}]
    assert first_caption_only(items) == [{"path": "a", "caps": [["x"]]}]


def test_build_dataloaders(fake_data):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "2", "--workers", "0",
        ]
    )
    train_loader, val_loader, bleu_loader, vocab, train_items = build_dataloaders(args)
    assert len(train_items) == 3
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 2
    assert len(bleu_loader.dataset) == 2
    assert vocab == VOCAB

    images, captions, lengths = next(iter(train_loader))
    assert images.shape == (2, 3, 224, 224)
    assert captions.shape[0] == 2
    assert lengths.shape == (2,)


def test_overfit_mode_uses_first_caption_and_no_augmentation(fake_data):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--overfit", "2", "--batch-size", "2", "--workers", "0",
        ]
    )
    train_loader, _val, _bleu, _vocab, train_items = build_dataloaders(args)
    assert len(train_items) == 2
    assert all(len(item["caps"]) == 1 for item in train_items)
    dataset = train_loader.dataset
    # eval transform -> deterministic pixels and the same caption every time
    assert dataset.train is False
    first = dataset[0][1]
    assert torch.equal(first, dataset[0][1])


def test_subset_limits_training_images(fake_data):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--subset", "2", "--batch-size", "2", "--workers", "0",
        ]
    )
    _train, _val, _bleu, _vocab, train_items = build_dataloaders(args)
    assert len(train_items) == 2


# --------------------------------------------------------------------------
# epoch loop
# --------------------------------------------------------------------------


def _scaler(enabled: bool = False) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cpu", enabled=enabled)


def test_run_epoch_trains_and_reduces_loss(fake_data, stub_model):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "3", "--workers", "0", "--overfit", "3",
        ]
    )
    train_loader, _val, _bleu, _vocab, _items = build_dataloaders(args)
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, args.lr)
    device = torch.device("cpu")

    first = run_epoch(model, train_loader, device, optimizer, _scaler(), False, True)
    for _ in range(20):
        last = run_epoch(model, train_loader, device, optimizer, _scaler(), False, True)
    assert last < first


def test_run_epoch_eval_does_not_change_weights(fake_data, stub_model):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "2", "--workers", "0",
        ]
    )
    _train, val_loader, _bleu, _vocab, _items = build_dataloaders(args)
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    before = [p.detach().clone() for p in model.parameters()]
    run_epoch(model, val_loader, torch.device("cpu"), None, _scaler(), False, False)
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_grad_clip_is_five():
    assert GRAD_CLIP == 5.0


def test_bleu_proxy_is_in_range(fake_data, stub_model):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "2", "--workers", "0",
        ]
    )
    _train, _val, bleu_loader, vocab, _items = build_dataloaders(args)
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    score = greedy_bleu4(model, bleu_loader, invert_vocab(vocab), 5, torch.device("cpu"))
    assert 0.0 <= score <= 1.0


def test_bleu_proxy_is_deterministic(fake_data, stub_model):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "2", "--workers", "0",
        ]
    )
    _train, _val, bleu_loader, vocab, _items = build_dataloaders(args)
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    kwargs = (model, bleu_loader, invert_vocab(vocab), 5, torch.device("cpu"))
    assert greedy_bleu4(*kwargs) == greedy_bleu4(*kwargs)


# --------------------------------------------------------------------------
# finetune switching
# --------------------------------------------------------------------------


def test_apply_finetune_adds_encoder_at_lower_lr(stub_model):
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, 4e-4)
    assert len(optimizer.param_groups) == 1

    optimizer = apply_finetune(model, optimizer, 4e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 4e-4
    assert optimizer.param_groups[1]["lr"] == FINETUNE_LR == 1e-4
    assert model.encoder.trainable_parameters()
    assert all(p.requires_grad for p in model.decoder.parameters())


def test_apply_finetune_preserves_decoder_adam_moments(fake_data, stub_model):
    args = parse_args(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--batch-size", "3", "--workers", "0", "--overfit", "3",
        ]
    )
    train_loader, *_ = build_dataloaders(args)
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, 4e-4)
    run_epoch(model, train_loader, torch.device("cpu"), optimizer, _scaler(), False, True)

    # out.weight is the only decoder tensor this stub actually differentiates
    decoder_param = model.decoder.out.weight
    exp_avg_before = optimizer.state[decoder_param]["exp_avg"].clone()
    assert exp_avg_before.abs().sum() > 0

    new_optimizer = apply_finetune(model, optimizer, 4e-4)
    assert torch.allclose(
        new_optimizer.state[decoder_param]["exp_avg"], exp_avg_before
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_append_metrics_header_and_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    result = train_module.EpochResult(1.5, 1.4, 0.2, 90.0, 1024.0)

    append_metrics(path, 1, result, [4e-4])
    append_metrics(path, 2, result, [4e-4, FINETUNE_LR])

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["val_bleu4"]) == pytest.approx(0.2)
    assert float(rows[0]["minutes"]) == pytest.approx(1.5)
    assert float(rows[0]["gpu_memory_mb"]) == pytest.approx(1024.0)
    assert rows[0]["lr"] == "0.00040000"
    # both group lrs are recorded once the encoder is unfrozen
    assert rows[1]["lr"] == "0.00040000|0.00010000"
    assert set(rows[0]) == set(train_module.METRICS_FIELDS)


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_resume_restores_epoch_and_best_score(tmp_path, fake_data, stub_model):
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, 4e-4)
    path = tmp_path / "last_state.pt"
    save_resume_state(path, model, optimizer, epoch=5, best_bleu4=0.42, vocab=VOCAB)

    fresh = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    state = restore(path, fresh, VOCAB, 4e-4, torch.device("cpu"))

    assert state.epoch == 5
    assert state.best_bleu4 == pytest.approx(0.42)
    for a, b in zip(model.decoder.parameters(), fresh.decoder.parameters()):
        assert torch.equal(a, b)


def test_resume_state_file_has_optimizer(tmp_path, stub_model):
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, 4e-4)
    path = tmp_path / "s.pt"
    save_resume_state(path, model, optimizer, epoch=1, best_bleu4=0.1, vocab=VOCAB)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer_state" in payload
    assert payload["epoch"] == 1
    assert payload["finetune_applied"] is False


def test_resume_across_finetune_switch(tmp_path, stub_model):
    """Adam state must still land on the right tensors after the group change."""
    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    optimizer = build_optimizer(model, 4e-4)
    # apply_finetune returns a rebuilt optimiser, so the result must be used
    optimizer = apply_finetune(model, optimizer, 4e-4)
    assert len(optimizer.param_groups) == 2

    path = tmp_path / "s.pt"
    save_resume_state(path, model, optimizer, epoch=3, best_bleu4=0.3, vocab=VOCAB)

    fresh = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    state = restore(path, fresh, VOCAB, 4e-4, torch.device("cpu"))

    assert state.finetune_applied is True
    assert len(state.optimizer.param_groups) == 2
    assert state.optimizer.param_groups[1]["lr"] == FINETUNE_LR
    # the encoder group holds exactly the unfrozen encoder tensors
    encoder_ids = {id(p) for p in state.optimizer.param_groups[1]["params"]}
    assert encoder_ids == {id(p) for p in fresh.encoder.trainable_parameters()}


def test_resume_rebuilds_encoder_group_from_pretrained_checkpoint(
    tmp_path, stub_model
):
    """A plain save_checkpoint .pt has no optimiser state; restore must not crash."""
    from captioning.model import save_checkpoint

    model = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, VOCAB, epoch=2, val_bleu4=0.2)

    fresh = stub_model.from_config(train_module.ModelConfig(vocab_size=len(VOCAB)))
    state = restore(path, fresh, VOCAB, 4e-4, torch.device("cpu"))

    assert state.epoch == 2
    assert state.best_bleu4 == pytest.approx(0.2)
    assert state.finetune_applied is False
    assert len(state.optimizer.param_groups) == 1
    for a, b in zip(model.decoder.parameters(), fresh.decoder.parameters()):
        assert torch.equal(a, b)


def test_resume_missing_file_exits(fake_data, stub_model):
    with pytest.raises(SystemExit):
        main(
            [
                "--coco-root", str(fake_data["coco_root"]),
                "--data-dir", str(fake_data["data_dir"]),
                "--out-dir", str(fake_data["out_dir"]),
                "--resume", str(fake_data["out_dir"] / "nope.pt"),
                "--batch-size", "2", "--workers", "0", "--epochs", "1",
            ]
        )


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_main_writes_all_outputs(fake_data, stub_model, monkeypatch):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    exit_code = main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "2", "--batch-size", "2", "--workers", "0",
            "--overfit", "3",
        ]
    )
    assert exit_code == 0

    out = fake_data["out_dir"]
    assert (out / "last.pt").is_file()
    assert (out / "best.pt").is_file()
    assert (out / "last_state.pt").is_file()
    assert (out / "metrics.csv").is_file()
    # vocab.json next to each checkpoint, per the contract
    assert (out / "last.json").is_file()
    assert json.loads((out / "last.json").read_text()) == VOCAB

    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert [row["epoch"] for row in rows] == ["1", "2"]

    payload = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert set(payload) == {"model_state", "vocab", "config", "epoch", "val_bleu4"}
    assert payload["epoch"] == 2


def test_main_finetune_switch_creates_two_groups(fake_data, stub_model, monkeypatch):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "2", "--batch-size", "2", "--workers", "0",
            "--finetune-after", "2", "--overfit", "3",
        ]
    )
    payload = torch.load(
        fake_data["out_dir"] / "last_state.pt", map_location="cpu", weights_only=False
    )
    assert payload["finetune_applied"] is True
    groups = payload["optimizer_state"]["param_groups"]
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(4e-4)
    assert groups[1]["lr"] == pytest.approx(FINETUNE_LR)


def test_main_records_encoder_lr_in_metrics(fake_data, stub_model, monkeypatch):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "2", "--batch-size", "2", "--workers", "0",
            "--finetune-after", "2", "--overfit", "3",
        ]
    )
    rows = list(csv.DictReader((fake_data["out_dir"] / "metrics.csv").open()))
    assert rows[0]["lr"] == "0.00040000"  # frozen encoder, decoder only
    assert rows[1]["lr"] == "0.00040000|0.00010000"


def test_main_early_stops_on_no_improvement(fake_data, stub_model, monkeypatch):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "10", "--batch-size", "2", "--workers", "0",
            "--patience", "1", "--overfit", "3",
        ]
    )
    rows = list(csv.DictReader((fake_data["out_dir"] / "metrics.csv").open()))
    # patience 1 with a deterministic proxy BLEU stops after the second epoch
    assert len(rows) < 10


def test_main_resume_continues_from_epoch(fake_data, stub_model, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    base = [
        "--coco-root", str(fake_data["coco_root"]),
        "--data-dir", str(fake_data["data_dir"]),
        "--out-dir", str(fake_data["out_dir"]),
        "--epochs", "1", "--batch-size", "2", "--workers", "0", "--overfit", "3",
    ]
    main(base)
    capsys.readouterr()

    # Extend the target so the resumed run has an epoch left to do.
    main(
        [
            *base,
            "--epochs", "2",
            "--resume", str(fake_data["out_dir"] / "last_state.pt"),
        ]
    )
    out = capsys.readouterr().out
    assert "resumed from" in out
    assert "next epoch 2" in out
    rows = list(csv.DictReader((fake_data["out_dir"] / "metrics.csv").open()))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert torch.load(
        fake_data["out_dir"] / "last.pt", map_location="cpu", weights_only=False
    )["epoch"] == 2


def test_main_resume_past_target_does_nothing(fake_data, stub_model, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    base = [
        "--coco-root", str(fake_data["coco_root"]),
        "--data-dir", str(fake_data["data_dir"]),
        "--out-dir", str(fake_data["out_dir"]),
        "--epochs", "1", "--batch-size", "2", "--workers", "0", "--overfit", "3",
    ]
    main(base)
    capsys.readouterr()

    main([*base, "--resume", str(fake_data["out_dir"] / "last_state.pt")])
    assert "nothing to do" in capsys.readouterr().out
    rows = list(csv.DictReader((fake_data["out_dir"] / "metrics.csv").open()))
    assert [row["epoch"] for row in rows] == ["1"]


def test_main_overfit_prints_captions(fake_data, stub_model, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "1", "--batch-size", "2", "--workers", "0",
            "--overfit", "2",
        ]
    )
    out = capsys.readouterr().out
    assert "generated:" in out
    assert "reference: a dog runs" in out


def test_main_amp_on_cpu_warns(fake_data, stub_model, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "CaptioningModel", stub_model)
    main(
        [
            "--coco-root", str(fake_data["coco_root"]),
            "--data-dir", str(fake_data["data_dir"]),
            "--out-dir", str(fake_data["out_dir"]),
            "--epochs", "1", "--batch-size", "2", "--workers", "0",
            "--amp", "--overfit", "3",
        ]
    )
    if torch.cuda.is_available():
        pytest.skip("cuda present, so the CPU warning path is not exercised")
    assert "--amp ignored" in capsys.readouterr().out
