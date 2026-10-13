"""Tests for the vocab / data pipeline against a tiny fake COCO dataset.

Builds 3 images plus a synthetic Karpathy ``dataset_coco.json`` in a temp
folder, so nothing here touches the real 20 GB download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from captioning.data import CocoCaptions, collate, get_transform
from captioning.vocab import (
    END,
    MAX_LEN,
    PAD,
    SPECIAL_TOKENS,
    START,
    UNK,
    decode,
    encode,
    invert_vocab,
)
from ml.data_prep import build_splits, write_outputs

# Word frequencies in the *train* split (restval + train images):
#   "a" x5 -> kept (>= MIN_FREQ of 5)
#   "cat" x3, "bird"/"rare"/... x1 -> dropped, i.e. they encode as <unk>
CAPS_A = [["a", "dog", "runs"]]                                  # val split only
CAPS_B = [["a", "cat", "sits"], ["a", "cat", "jumps"], ["a", "cat", "purrs"]]
CAPS_C = [["a", "bird", "flies"], ["a", "rare", "word"]]


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 30, 200)).save(path)


@pytest.fixture()
def fake_dataset(tmp_path: Path) -> dict:
    """A 3-image fake COCO tree plus a matching Karpathy annotation file.

    Layout:
      * ``val2014/val_a.jpg``  -> split "val"     (1 caption pair)
      * ``train2014/train_b.jpg`` -> split "restval", merged into train
      * ``train2014/train_c.jpg`` -> split "train"
      * ``val2014/missing.jpg`` -> split "val" but the file is absent on disk
    """
    coco_root = tmp_path / "coco"
    _write_image(coco_root / "val2014" / "val_a.jpg")
    _write_image(coco_root / "train2014" / "train_b.jpg")
    _write_image(coco_root / "train2014" / "train_c.jpg")
    # "val2014/missing.jpg" is intentionally never written to disk.

    karpathy = tmp_path / "dataset_coco.json"
    karpathy.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "filepath": "val2014",
                        "filename": "val_a.jpg",
                        "split": "val",
                        "sentences": [{"tokens": CAPS_A[0]}],
                    },
                    {
                        "filepath": "train2014",
                        "filename": "train_b.jpg",
                        "split": "restval",
                        "sentences": [{"tokens": c} for c in CAPS_B],
                    },
                    {
                        "filepath": "train2014",
                        "filename": "train_c.jpg",
                        "split": "train",
                        "sentences": [{"tokens": c} for c in CAPS_C],
                    },
                    {
                        "filepath": "val2014",
                        "filename": "missing.jpg",
                        "split": "val",
                        "sentences": [{"tokens": ["a", "ghost"]}],
                    },
                ]
            }
        )
    )
    return {"karpathy": karpathy, "coco_root": coco_root, "out_dir": tmp_path / "processed"}


@pytest.fixture()
def splits(fake_dataset: dict):
    return build_splits(fake_dataset["karpathy"], fake_dataset["coco_root"], verbose=False)


# --- split building -------------------------------------------------------


def test_split_sizes_and_missing_count(splits):
    built, vocab, counter, missing = splits
    # restval merged into train -> 2 train images, 1 val, 0 test
    assert [len(built[name]) for name in ("train", "val", "test")] == [2, 1, 0]
    # the one entry whose image file was missing
    assert missing == 1


def test_restval_is_merged_into_train(splits):
    built, _vocab, _counter, _missing = splits
    paths = [item["path"] for item in built["train"]]
    assert "train2014/train_b.jpg" in paths  # came from restval
    assert "train2014/train_c.jpg" in paths  # came from train
    assert all(path.startswith("train2014/") for path in paths)


def test_long_train_captions_are_dropped(fake_dataset):
    long_caption = ["a"] + ["pad"] * MAX_LEN  # MAX_LEN + 1 tokens -> too long
    entries = json.loads(fake_dataset["karpathy"].read_text())
    for image in entries["images"]:
        # restval is merged into train, so it is subject to the same limit
        if image["split"] in ("train", "restval"):
            image["sentences"] = [{"tokens": long_caption}]
    fake_dataset["karpathy"].write_text(json.dumps(entries))

    built, _vocab, counter, _missing = build_splits(
        fake_dataset["karpathy"], fake_dataset["coco_root"], verbose=False
    )
    # every train caption was dropped -> no captions left -> images left the split
    assert built["train"] == []
    assert counter == {}  # nothing to count for the vocabulary


def test_val_captions_are_not_length_limited(fake_dataset):
    entries = json.loads(fake_dataset["karpathy"].read_text())
    long_caption = ["a"] + ["pad"] * MAX_LEN
    for image in entries["images"]:
        if image["split"] == "val" and image["filename"] == "val_a.jpg":
            image["sentences"] = [{"tokens": long_caption}]
    fake_dataset["karpathy"].write_text(json.dumps(entries))

    built, _vocab, _counter, _missing = build_splits(
        fake_dataset["karpathy"], fake_dataset["coco_root"], verbose=False
    )
    # the MAX_LEN drop is a training-time rule only
    assert len(built["val"]) == 1
    assert built["val"][0]["caps"] == [long_caption]


def test_paths_are_relative_posix(fake_dataset, splits):
    built, _vocab, _counter, _missing = splits
    for item in built["train"]:
        assert "\\" not in item["path"]
        assert (fake_dataset["coco_root"] / item["path"]).is_file()


# --- vocab ----------------------------------------------------------------


def test_special_token_ids():
    assert (PAD, START, END, UNK) == (0, 1, 2, 3)
    assert SPECIAL_TOKENS == {
        "<pad>": 0,
        "<start>": 1,
        "<end>": 2,
        "<unk>": 3,
    }


def test_build_vocab_drops_rare_words(splits):
    _built, vocab, counter, _missing = splits
    # "a" occurs 6x in train -> kept; "rare" occurs 1x -> absent
    assert vocab["a"] >= 4
    assert counter["rare"] == 1
    assert "rare" not in vocab
    assert "dog" not in vocab  # 3 occurrences < MIN_FREQ of 5
    assert len(vocab) == len(SPECIAL_TOKENS) + 1


def test_vocab_ids_are_contiguous(splits):
    _built, vocab, _counter, _missing = splits
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_encode_decode_round_trip(splits):
    _built, vocab, _counter, _missing = splits
    id_to_word = invert_vocab(vocab)

    tokens = ["a", "unknownword"]
    ids = encode(tokens, vocab)
    assert ids[0] == START and ids[-1] == END
    assert ids[1] == vocab["a"]
    assert ids[2] == UNK  # out-of-vocabulary word

    assert decode(ids, id_to_word) == ["<start>", "a", "<unk>"]
    assert decode(ids, id_to_word, strip_specials=True) == ["a", "<unk>"]


def test_encode_is_deterministic_and_stops_at_end(splits):
    _built, vocab, _counter, _missing = splits
    id_to_word = invert_vocab(vocab)
    ids = encode(["a"], vocab)
    # anything after <end> is ignored by decode
    assert decode([*ids, vocab["a"], vocab["a"]], id_to_word) == ["<start>", "a"]


# --- dataset + collate ----------------------------------------------------


def test_get_transform_output_size():
    image = Image.new("RGB", (300, 200), color=(10, 20, 30))
    assert get_transform(train=False)(image).shape == (3, 224, 224)
    assert get_transform(train=True)(image).shape == (3, 224, 224)


def test_dataset_item_shapes(fake_dataset, splits):
    built, vocab, _counter, _missing = splits
    dataset = CocoCaptions(built["train"], vocab, fake_dataset["coco_root"], train=True)
    assert len(dataset) == 2

    image, caption = dataset[0]
    assert image.shape == (3, 224, 224)
    assert caption.dtype == torch.long
    assert caption[0].item() == START
    assert caption[-1].item() == END


def test_eval_dataset_is_deterministic(fake_dataset, splits):
    built, vocab, _counter, _missing = splits
    dataset = CocoCaptions(built["val"], vocab, fake_dataset["coco_root"], train=False)
    first = dataset[0][1]
    assert all(torch.equal(first, dataset[0][1]) for _ in range(3))


def test_collate_shapes_and_padding(fake_dataset, splits):
    built, vocab, _counter, _missing = splits
    dataset = CocoCaptions(built["train"], vocab, fake_dataset["coco_root"], train=True)
    torch.manual_seed(0)

    images, captions, lengths = collate([dataset[i] for i in range(len(dataset))])

    assert images.shape == (2, 3, 224, 224)
    assert captions.shape[0] == 2
    assert lengths.shape == (2,)
    assert captions.shape[1] == int(lengths.max())
    assert lengths.dtype == torch.long

    # padded tail is <pad> and the true length matches the non-pad prefix
    for row, length in zip(captions, lengths.tolist()):
        assert row[length:].tolist() == [PAD] * (row.numel() - length)
        assert row[0].item() == START
        assert row[length - 1].item() == END


def test_dataloader_end_to_end(fake_dataset, splits):
    from torch.utils.data import DataLoader

    built, vocab, _counter, _missing = splits
    dataset = CocoCaptions(built["train"], vocab, fake_dataset["coco_root"], train=True)
    images, captions, lengths = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))

    assert images.shape[0] == captions.shape[0] == lengths.shape[0] == 2
    assert images.shape[1:] == (3, 224, 224)
    assert lengths.min() >= 2  # at minimum <start> + <end>


# --- output files ---------------------------------------------------------


def test_write_outputs_files(fake_dataset, splits):
    built, vocab, _counter, _missing = splits
    write_outputs(built, vocab, fake_dataset["out_dir"])

    for name in ("train", "val", "test", "vocab"):
        assert (fake_dataset["out_dir"] / f"{name}.json").is_file()

    train = json.loads((fake_dataset["out_dir"] / "train.json").read_text())
    assert len(train) == 2
    assert json.loads((fake_dataset["out_dir"] / "vocab.json").read_text()) == dict(vocab)
