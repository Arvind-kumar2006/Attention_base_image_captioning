"""Evaluate a trained checkpoint on a Karpathy split.

Generates a caption for every image with the chosen decoding strategy, then
scores the corpus with pycocoevalcap (BLEU-1..4, METEOR, CIDEr) against all
reference captions of that image.

Usage::

    python -m ml.evaluate \\
        --checkpoint ml/weights/best.pt \\
        --data-dir data/processed \\
        --coco-root data/coco \\
        --split test \\
        --beam-size 3 \\
        --out results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    # The installed package name, so the modules are the same objects the rest
    # of the project imports (isinstance checks depend on that).
    from captioning.decoding import (
        BeamSearchDecodingStrategy,
        GreedyDecodingStrategy,
    )
    from captioning.generator import CaptionGenerator
    from captioning.metrics import (
        FALLBACK_TOKENIZER,
        PTB_TOKENIZER,
        per_image_cider,
        score_captions,
    )
except ImportError:  # pragma: no cover - not installed yet
    from .captioning.decoding import (
        BeamSearchDecodingStrategy,
        GreedyDecodingStrategy,
    )
    from .captioning.generator import CaptionGenerator
    from .captioning.metrics import (
        FALLBACK_TOKENIZER,
        PTB_TOKENIZER,
        per_image_cider,
        score_captions,
    )

BEST_WORST_COUNT = 10


def load_split(data_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Read ``<split>.json`` written by ``ml/data_prep.py``."""
    path = data_dir / f"{split}.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} - run `python -m ml.data_prep` first")
    with open(path) as handle:
        return json.load(handle)


def build_strategy(
    model: torch.nn.Module, beam_size: Optional[int]
) -> Any:
    """Greedy when ``beam_size`` is 0, 1 or None; otherwise beam search.

    A beam width of 1 is mathematically identical to greedy, so it is routed
    there rather than paying for the search machinery.
    """
    if not beam_size or beam_size <= 1:
        return GreedyDecodingStrategy(model.decoder)
    return BeamSearchDecodingStrategy(model.decoder, beam_size=beam_size)


def generate_captions(
    generator: CaptionGenerator,
    items: Sequence[Dict[str, Any]],
    coco_root: Path,
    description: str = "captions",
) -> List[Dict[str, Any]]:
    """Caption every item, collecting the caption and its references.

    Every caption of an image becomes a reference, matching how the Karpathy
    split is used: COCO supplies five human captions per image and all five are
    valid answers.
    """
    records: List[Dict[str, Any]] = []
    for index, item in enumerate(tqdm(items, desc=description)):
        image = Image.open(coco_root / item["path"]).convert("RGB")
        result = generator.generate(image)
        records.append(
            {
                "index": index,
                "image_id": index,
                "path": item["path"],
                "caption": result.caption,
                "references": [" ".join(caption) for caption in item["caps"]],
                "num_words": result.num_words,
                "time_ms": round(result.time_ms, 3),
            }
        )
    return records


def build_scoring_dicts(
    records: Sequence[Dict[str, Any]]
) -> tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Turn records into the ``gts``/``res`` mapping pair pycocoevalcap wants."""
    gts = {record["image_id"]: list(record["references"]) for record in records}
    res = {record["image_id"]: [record["caption"]] for record in records}
    return gts, res


def rank_records(
    records: Sequence[Dict[str, Any]], cider_per_image: Dict[int, float]
) -> List[Dict[str, Any]]:
    """Attach each record's CIDEr, sorted best first.

    Ties are broken by caption so the ordering is stable across runs.
    """
    ranked: List[Dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["cider"] = float(cider_per_image.get(record["image_id"], 0.0))
        ranked.append(enriched)
    ranked.sort(key=lambda item: (-item["cider"], item["caption"]))
    return ranked


def print_ranked(
    title: str, records: Sequence[Dict[str, Any]], count: int = BEST_WORST_COUNT
) -> None:
    """Print the top or bottom ``count`` captions with their CIDEr score."""
    print(f"\n{title}")
    print("-" * 78)
    if not records:
        print("(none)")
        return
    for record in records[:count]:
        print(f"CIDEr {record['cider']:.4f}  {record['caption'] or '(empty)'}")
        for reference in record["references"]:
            print(f"{'':>13}ref: {reference}")
        print()


def build_results(
    scores: Dict[str, float],
    per_image: Dict[str, List[float]],
    records: Sequence[Dict[str, Any]],
    generator: CaptionGenerator,
    args: argparse.Namespace,
    split: str,
    tokenizer: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Assemble the results.json payload."""
    return {
        "scores": scores,
        "per_image_summary": {
            name: {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for name, values in per_image.items()
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "epoch": generator.metadata.get("epoch"),
            "val_bleu4_at_checkpoint": generator.metadata.get("val_bleu4"),
            "config": generator.config.to_dict()
            if hasattr(generator.config, "to_dict")
            else dict(generator.config),
            "vocab_size": len(generator.vocab),
        },
        "decoding": {
            "strategy": (
                "greedy"
                if isinstance(generator.strategy, GreedyDecodingStrategy)
                else "beam_search"
            ),
            "beam_size": (
                None
                if isinstance(generator.strategy, GreedyDecodingStrategy)
                else args.beam_size
            ),
            "max_len": generator.max_len,
        },
        "dataset": {
            "split": split,
            "coco_root": str(args.coco_root),
            "data_dir": str(args.data_dir),
            "num_images": len(records),
            "limit": args.limit,
        },
        "tokenizer": tokenizer,
        "tokenizer_note": (
            "PTBTokenizer (Java) as used by the COCO leaderboard"
            if tokenizer == PTB_TOKENIZER
            else f"{FALLBACK_TOKENIZER}: Java unavailable, so captions were "
            "lower-cased and split on whitespace without punctuation stripping"
        ),
        "runtime": {
            "device": str(generator.device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
        "captions": list(records),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a captioning checkpoint with pycocoevalcap.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint from ml/train.py (best.pt or last.pt).")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with <split>.json and vocab.json.")
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="Folder containing train2014/ and val2014/.")
    parser.add_argument("--split", default="test",
                        help="Karpathy split to score (default: test).")
    parser.add_argument("--beam-size", type=int, default=3,
                        help="Beam width; 0 or 1 uses greedy decoding (default: 3).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N images.")
    parser.add_argument("--out", type=Path, default=Path("results.json"),
                        help="Where to write the results (default: results.json).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluation entry point."""
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    items = load_split(args.data_dir, args.split)
    if not items:
        raise SystemExit(f"split '{args.split}' is empty")
    if args.limit is not None:
        items = items[: args.limit]
    print(f"split '{args.split}': {len(items)} images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load once to reach the decoder, then hand the same model to the generator
    # so the checkpoint is read a single time.
    generator = CaptionGenerator(args.checkpoint, device=device)
    strategy = build_strategy(generator.model, args.beam_size)
    generator.strategy = strategy
    is_greedy = isinstance(generator.strategy, GreedyDecodingStrategy)
    print(
        f"device: {device} | decoding: "
        + ("greedy" if is_greedy else f"beam search k={args.beam_size}")
    )

    started = time.time()
    records = generate_captions(generator, items, args.coco_root)
    elapsed = time.time() - started

    empty = [record for record in records if not record["caption"]]
    if empty:
        print(f"WARNING: {len(empty)} images produced an empty caption")

    gts, res = build_scoring_dicts(records)
    print("\nscoring with pycocoevalcap ...")
    result = score_captions(gts, res)
    if result.tokenizer == FALLBACK_TOKENIZER:
        print(
            "PTBTokenizer needs a JVM and none was found on PATH: "
            "captions were lower-cased and whitespace-tokenised instead. "
            "METEOR and BLEU can differ slightly from leaderboard numbers."
        )
    else:
        print("tokenised with PTBTokenizer (Java)")

    print("\nscores")
    print("-" * 78)
    for name in ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "CIDEr"):
        if name in result.scores:
            print(f"{name:>8}: {result.scores[name]:.4f}")

    cider_per_image = per_image_cider(gts, res, use_ptb=result.tokenizer == PTB_TOKENIZER)
    ranked = rank_records(records, cider_per_image)
    print_ranked(f"best {BEST_WORST_COUNT} captions by CIDEr", ranked, BEST_WORST_COUNT)
    print_ranked(
        f"worst {BEST_WORST_COUNT} captions by CIDEr",
        list(reversed(ranked)),
        BEST_WORST_COUNT,
    )

    payload = build_results(
        result.scores,
        result.per_image,
        ranked,
        generator,
        args,
        args.split,
        result.tokenizer,
        elapsed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nwrote {args.out} ({len(records)} captions, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
