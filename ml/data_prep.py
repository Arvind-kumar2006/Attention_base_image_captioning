"""Build the Karpathy train/val/test splits and the vocabulary.

Reads the Karpathy ``dataset_coco.json``, drops images that are not on disk,
merges ``restval`` into ``train``, drops training captions longer than
``MAX_LEN`` tokens, counts training word frequencies and writes ``train.json``,
``val.json``, ``test.json`` and ``vocab.json``.

Usage::

    python -m ml.data_prep \\
        --karpathy-json data/dataset_coco.json \\
        --coco-root data/coco \\
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from tqdm.auto import tqdm

try:  # allow both `python ml/data_prep.py` and `python -m ml.data_prep`
    from .captioning.vocab import MAX_LEN, MIN_FREQ, build_vocab
except ImportError:  # pragma: no cover - direct script execution
    from captioning.vocab import MAX_LEN, MIN_FREQ, build_vocab

SPLIT_NAMES = ("train", "val", "test")


def build_splits(
    karpathy_json: Path,
    coco_root: Path,
    max_len: int = MAX_LEN,
    verbose: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], collections.Counter, int]:
    """Build the splits and vocabulary from the Karpathy annotation file.

    Returns ``(splits, vocab, counter, missing)`` where ``missing`` is the number
    of entries skipped because the image file was not found under ``coco_root``.
    """
    with open(karpathy_json) as handle:
        images = json.load(handle)["images"]

    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    counter: collections.Counter = collections.Counter()
    missing = 0

    entries = tqdm(images, desc="building splits") if verbose else images
    for image in entries:
        if not (coco_root / image["filepath"] / image["filename"]).exists():
            missing += 1  # e.g. train2014 was never downloaded
            continue

        # Karpathy restval is a validation remainder -> merge it into train.
        split = "train" if image["split"] in ("train", "restval") else image["split"]
        captions: List[List[str]] = [s["tokens"] for s in image["sentences"]]

        if split == "train":
            captions = [caption for caption in captions if len(caption) <= max_len]
            for caption in captions:
                counter.update(caption)

        if captions:
            splits[split].append(
                {
                    "path": f"{image['filepath']}/{image['filename']}",
                    "caps": captions,
                }
            )

    vocab = build_vocab(counter)
    return splits, vocab, counter, missing


def write_outputs(
    splits: Mapping[str, List[Dict[str, Any]]],
    vocab: Mapping[str, int],
    out_dir: Path,
) -> None:
    """Write ``train.json``, ``val.json``, ``test.json`` and ``vocab.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        with open(out_dir / f"{name}.json", "w") as handle:
            json.dump(splits.get(name, []), handle)
    with open(out_dir / "vocab.json", "w") as handle:
        json.dump(dict(vocab), handle)


def summarise(
    splits: Mapping[str, List[Dict[str, Any]]],
    vocab: Mapping[str, int],
    counter: Mapping[str, int],
    missing: int,
) -> None:
    """Print split sizes, vocabulary size and the most frequent words."""
    print(f"entries skipped (image not on disk): {missing}")
    print()
    for name in SPLIT_NAMES:
        items = splits.get(name, [])
        caption_count = sum(len(item["caps"]) for item in items)
        print(f"{name:5s}: {len(items):>8,} images | {caption_count:>9,} captions")
    print(f"\nvocabulary size : {len(vocab):,}")
    print("10 most frequent:", collections.Counter(counter).most_common(10))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Karpathy splits and vocabulary for COCO captioning.",
    )
    parser.add_argument(
        "--karpathy-json",
        type=Path,
        required=True,
        help="Path to dataset_coco.json (the Karpathy split file).",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="Folder containing train2014/ and val2014/ with the .jpg files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where train.json, val.json, test.json and vocab.json are written.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN,
        help=f"Drop training captions longer than this many tokens (default: {MAX_LEN}).",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=MIN_FREQ,
        help=f"Keep words seen at least this often in train (default: {MIN_FREQ}).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    for path, label in ((args.karpathy_json, "karpathy json"), (args.coco_root, "coco root")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    splits, vocab, counter, missing = build_splits(
        args.karpathy_json, args.coco_root, max_len=args.max_len
    )
    if args.min_freq != MIN_FREQ:
        vocab = build_vocab(counter, min_freq=args.min_freq)

    write_outputs(splits, vocab, args.out_dir)
    summarise(splits, vocab, counter, missing)
    print(f"\nwrote {args.out_dir}/train.json, val.json, test.json, vocab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
