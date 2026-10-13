"""Train the attention-based captioner.

Loss is cross-entropy over the non-padded positions plus the doubly stochastic
attention regulariser::

    loss = CE(logits, captions[:, 1:]) + 1.0 * mean((1 - sum_t alpha_t)^2)

Optimisation is Adam with gradient clipping at 5.0 and optional
``torch.autocast`` for mixed precision on GPU.

The encoder stays frozen unless ``--finetune-after E`` unfreezes ``layer3`` and
``layer4`` from epoch ``E`` onwards, at which point the encoder runs at 1e-4.

The BLEU-4 reported here is a *quick proxy*: nltk corpus BLEU over greedy
decoding on a fixed sample of validation images. Final numbers come from
``ml/evaluate.py``, which scores the Karpathy test split with pycocoevalcap.

Usage::

    python -m ml.train \\
        --coco-root data/coco \\
        --data-dir data/processed \\
        --out-dir ml/weights \\
        --epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:  # allow both `python ml/train.py` and `python -m ml.train`
    from .captioning.data import CocoCaptions, collate, get_transform
    from .captioning.decoder import caption_loss
    from .captioning.model import (
        CaptioningModel,
        ModelConfig,
        save_checkpoint,
    )
    from .captioning.vocab import END, PAD, START, invert_vocab
except ImportError:  # pragma: no cover - not installed yet
    from captioning.data import CocoCaptions, collate, get_transform
    from captioning.decoder import caption_loss
    from captioning.model import (
        CaptioningModel,
        ModelConfig,
        save_checkpoint,
    )
    from captioning.vocab import END, PAD, START, invert_vocab

ALPHA_C = 1.0  # weight on the attention regularisation term
GRAD_CLIP = 5.0
FINETUNE_LR = 1e-4

#: Fixed validation sample used for the BLEU-4 proxy, so epochs are comparable.
BLEU_SAMPLE_SIZE = 1000

METRICS_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_bleu4",
    "lr",
    "epoch_seconds",
    "minutes",
    "gpu_memory_mb",
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(data_dir: Path, name: str) -> List[Dict[str, Any]]:
    """Read ``<name>.json`` written by ``ml/data_prep.py``."""
    path = data_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(
            f"missing {path} - run `python -m ml.data_prep` to build the splits first"
        )
    with open(path) as handle:
        return json.load(handle)


def load_vocab(data_dir: Path) -> Dict[str, int]:
    """Read ``vocab.json`` written by ``ml/data_prep.py``."""
    path = data_dir / "vocab.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} - run `python -m ml.data_prep` first")
    with open(path) as handle:
        return json.load(handle)


def take_subset(
    items: Sequence[Dict[str, Any]], count: Optional[int]
) -> List[Dict[str, Any]]:
    """First ``count`` items, or everything when ``count`` is None."""
    if count is None:
        return list(items)
    return list(items[:count])


def first_caption_only(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the first caption of each image, for the overfit mode."""
    return [{"path": item["path"], "caps": [item["caps"][0]]} for item in items]


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[
    DataLoader, DataLoader, DataLoader, Dict[str, int], List[Dict[str, Any]]
]:
    """Build the train, val and BLEU-probe loaders.

    Returns ``(train_loader, val_loader, bleu_loader, vocab, train_items)``.
    """
    vocab = load_vocab(args.data_dir)
    train_items = take_subset(load_split(args.data_dir, "train"), args.subset)
    val_items = load_split(args.data_dir, "val")

    if not train_items:
        raise SystemExit("training split is empty after applying --subset")

    if args.overfit is not None:
        # One fixed caption per image, and the eval transform so no augmentation
        # and no random caption choice varies between passes.
        train_items = first_caption_only(train_items[: args.overfit])
        train_dataset = CocoCaptions(
            train_items, vocab, args.coco_root, train=False
        )
    else:
        train_dataset = CocoCaptions(train_items, vocab, args.coco_root, train=True)

    val_dataset = CocoCaptions(val_items, vocab, args.coco_root, train=False)

    loader_kwargs: Dict[str, Any] = {
        "num_workers": args.workers,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.overfit is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    # Fixed slice, eval transform, first caption: fully deterministic per epoch.
    bleu_loader = DataLoader(
        CocoCaptions(
            val_items[:BLEU_SAMPLE_SIZE], vocab, args.coco_root, train=False
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, bleu_loader, vocab, train_items


# --------------------------------------------------------------------------
# BLEU-4 proxy
# --------------------------------------------------------------------------


def greedy_bleu4(
    model: CaptioningModel,
    loader: DataLoader,
    id_to_word: Mapping[int, str],
    max_len: int,
    device: torch.device,
) -> float:
    """Corpus BLEU-4 over the fixed validation sample, using nltk.

    A quick signal for checkpoint selection and early stopping only. The
    authoritative BLEU/METEOR/CIDEr numbers come from ``ml/evaluate.py``.
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    model.eval()
    hypotheses: List[List[str]] = []
    references: List[List[str]] = []
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for images, captions, _lengths in loader:
            features = model.encode(images.to(device, non_blocking=True))
            ids, _alphas = model.decoder.generate(features, max_len=max_len)
            for row in ids.tolist():
                hypotheses.append(
                    [
                        id_to_word.get(index, "<unk>")
                        for index in row
                        if index not in (PAD, END)
                    ]
                )
            # The eval dataset uses the first caption, so one reference per row.
            for target in captions.tolist():
                references.append(
                    [
                        id_to_word.get(index, "<unk>")
                        for index in target
                        if index not in (PAD, START, END)
                    ]
                )

    if not hypotheses:
        return 0.0
    return float(corpus_bleu(references, hypotheses, smoothing_function=smoothing))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class EpochResult:
    """Losses and BLEU for one epoch, plus duration and peak GPU memory."""

    train_loss: float
    val_loss: float
    val_bleu4: float
    seconds: float
    gpu_memory_mb: float


def _encoder_is_training(model: CaptioningModel) -> bool:
    """True when the encoder is unfrozen, so it needs a gradient graph."""
    return bool(model.encoder.trainable_parameters())


def run_epoch(
    model: CaptioningModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    training: bool,
) -> float:
    """One pass over ``loader``; returns the mean cross-entropy loss.

    ``optimizer=None`` runs evaluation only.
    """
    model.train(training)
    total = 0.0
    batches = 0
    # A frozen encoder needs no autograd graph; unfreezing it does.
    encoder_context = nullcontext() if _encoder_is_training(model) else torch.no_grad()

    for images, captions, lengths in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                with encoder_context:
                    features = model.encode(images)
                logits, sorted_captions, _dl, alphas, _idx = model.decoder(
                    features, captions, lengths
                )
            # back to fp32 so the loss and the log-softmax are stable
            loss, cross_entropy = caption_loss(
                logits.float(), alphas.float(), sorted_captions, ALPHA_C
            )

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()

        total += float(cross_entropy.detach())
        batches += 1

    return total / max(batches, 1)


def gpu_memory_mb(device: torch.device) -> float:
    """Peak allocated GPU memory in MiB, or 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_gpu_memory(device: torch.device) -> None:
    """Clear the peak-memory counter at the start of an epoch."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def print_epoch_summary(epoch: int, epochs: int, result: EpochResult) -> None:
    """Log one epoch's losses, BLEU proxy, duration and memory."""
    print(
        f"epoch {epoch}/{epochs} | train loss {result.train_loss:.4f} | "
        f"val loss {result.val_loss:.4f} | val BLEU-4 {result.val_bleu4:.4f} | "
        f"{result.seconds / 60:.2f} min | "
        f"gpu mem {result.gpu_memory_mb:.0f} MiB"
    )


def append_metrics(
    path: Path, epoch: int, result: EpochResult, lrs: Sequence[float]
) -> None:
    """Append one row to ``metrics.csv``, writing the header on first use.

    ``lrs`` is every param-group learning rate, so the encoder's 1e-4 shows up
    once fine-tuning starts.
    """
    new_file = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{result.train_loss:.6f}",
                "val_loss": f"{result.val_loss:.6f}",
                "val_bleu4": f"{result.val_bleu4:.6f}",
                "lr": "|".join(f"{lr:.8f}" for lr in lrs),
                "epoch_seconds": f"{result.seconds:.2f}",
                "minutes": f"{result.seconds / 60:.2f}",
                "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
            }
        )


def build_optimizer(
    model: CaptioningModel, lr: float
) -> torch.optim.Adam:
    """Adam over the decoder, plus the encoder only once it is unfrozen."""
    encoder_lr = FINETUNE_LR if model.encoder.trainable_parameters() else lr
    groups = [{"params": list(model.decoder.parameters()), "lr": lr}]
    encoder_params = model.encoder.trainable_parameters()
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    return torch.optim.Adam(groups, lr=lr)


def apply_finetune(
    model: CaptioningModel, optimizer: torch.optim.Adam, lr: float
) -> torch.optim.Adam:
    """Unfreeze ``layer3``/``layer4`` and add them to the optimiser at 1e-4.

    The optimiser is rebuilt because the encoder parameters were not previously
    registered in it. The decoder's Adam moments are carried across so the
    switch does not reset its state.
    """
    model.set_finetune(True)
    new_optimizer = build_optimizer(model, lr)
    _carry_over_state(optimizer, new_optimizer, model.decoder.parameters())
    return new_optimizer


def _carry_over_state(
    source: torch.optim.Optimizer,
    target: torch.optim.Optimizer,
    params: Sequence[torch.nn.Parameter],
) -> None:
    """Copy Adam's per-parameter state for ``params`` from one optimiser to another."""
    for param in params:
        if param in source.state:
            target.state[param] = source.state[param]


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def save_resume_state(
    path: Path,
    model: CaptioningModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_bleu4: float,
    vocab: Mapping[str, int],
) -> None:
    """Write a training-state file with the optimiser and the best score.

    Kept separate from ``last.pt`` so that file still matches the checkpoint
    contract exactly (model_state / vocab / config / epoch / val_bleu4).
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": dict(vocab),
            "config": model.config.to_dict(),
            "epoch": epoch,
            "best_bleu4": best_bleu4,
            "finetune_applied": bool(model.encoder.trainable_parameters()),
        },
        path,
    )


@dataclass
class ResumeState:
    """Training state restored from a training-state file."""

    epoch: int
    best_bleu4: float
    optimizer: torch.optim.Optimizer
    finetune_applied: bool


def restore(
    path: Path,
    model: CaptioningModel,
    vocab: Mapping[str, int],
    lr: float,
    device: torch.device,
) -> ResumeState:
    """Restore model weights, optimiser, epoch and best score.

    Accepts either a training-state file (with ``optimizer_state``) or a plain
    ``.pt`` from :func:`save_checkpoint`, in which case the optimiser restarts
    fresh and only the weights are restored.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)

    # The saved config wins if it disagrees, so a resumed run keeps its shapes.
    saved_config = payload.get("config")
    if saved_config is not None:
        model.config = ModelConfig(**saved_config)

    saved_vocab = payload.get("vocab")
    if saved_vocab and dict(saved_vocab) != dict(vocab):
        print(
            f"WARNING: {path} carries a vocab of {len(saved_vocab)} entries but "
            f"--data-dir has {len(vocab)}; using the checkpoint's"
        )

    # Re-create the same param-group layout as at save time, otherwise Adam's
    # positional state mapping lands on the wrong tensors.
    finetune_applied = bool(payload.get("finetune_applied", False))
    if finetune_applied:
        model.set_finetune(True)
    optimizer = build_optimizer(model, lr)

    if "optimizer_state" in payload:
        # The encoder group changes the group count at the fine-tune switch;
        # align the saved group sizes before handing the state over.
        saved_groups = payload["optimizer_state"].get("param_groups", [])
        optimizer = _align_groups(optimizer, saved_groups, model, lr)
        optimizer.load_state_dict(payload["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        print(f"restored optimiser state ({len(optimizer.state)} tensors)")

    return ResumeState(
        epoch=int(payload.get("epoch", 0)),
        # A plain checkpoint stores val_bleu4, not the running best.
        best_bleu4=float(
            payload.get("best_bleu4", payload.get("val_bleu4", 0.0))
        ),
        optimizer=optimizer,
        finetune_applied=finetune_applied,
    )


def _align_groups(
    optimizer: torch.optim.Optimizer,
    saved_groups: Sequence[Mapping[str, Any]],
    model: CaptioningModel,
    lr: float,
) -> torch.optim.Optimizer:
    """Match the optimiser's param groups to the saved layout by tensor count.

    Adam's ``state_dict`` maps each parameter's moments by its position in
    ``param_groups``. A fine-tune switch appends an encoder group, so a resumed
    run has to rebuild the same groups before ``load_state_dict`` is safe.
    """
    live = [
        (group["lr"], len(group["params"])) for group in optimizer.param_groups
    ]
    saved = [
        (group.get("lr", lr), len(group.get("params", []))) for group in saved_groups
    ]
    if live == saved:
        return optimizer

    # The saved run had a different layout: rebuild from the saved group sizes,
    # taking parameters from the decoder first, then the unfrozen encoder.
    decoder_params = list(model.decoder.parameters())
    encoder_params = model.encoder.trainable_parameters()
    pool = {"decoder": decoder_params, "encoder": encoder_params}

    rebuilt: List[Dict[str, Any]] = []
    cursor = {"decoder": 0, "encoder": 0}
    for group_lr, count in saved:
        if count == 0:
            continue
        # Prefer whichever pool still has enough parameters left.
        source = "encoder" if (
            cursor["encoder"] + count <= len(encoder_params)
            and cursor["decoder"] >= len(decoder_params)
        ) else "decoder"
        if cursor[source] + count > len(pool[source]):
            source = "encoder" if source == "decoder" else "decoder"
        start = cursor[source]
        rebuilt.append(
            {"params": pool[source][start : start + count], "lr": group_lr}
        )
        cursor[source] = start + count

    aligned = torch.optim.Adam(rebuilt, lr=lr)
    _carry_over_state(optimizer, aligned, decoder_params + encoder_params)
    return aligned


def _to_device(value: Any, device: torch.device) -> Any:
    """Move tensors inside optimiser state to ``device``, leaving others alone."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


# --------------------------------------------------------------------------
# overfit demo
# --------------------------------------------------------------------------


def print_overfit_captions(
    model: CaptioningModel,
    items: Sequence[Dict[str, Any]],
    id_to_word: Mapping[int, str],
    coco_root: Path,
    device: torch.device,
    max_len: int,
) -> None:
    """Print greedy captions beside their references for the overfit items."""
    from PIL import Image

    model.eval()
    transform = get_transform(train=False)
    print("\ngreedy captions (untrained architecture, or overfit demo):")
    with torch.no_grad():
        for item in items:
            image = transform(
                Image.open(coco_root / item["path"]).convert("RGB")
            )
            features = model.encode(image.unsqueeze(0).to(device))
            ids, _alphas = model.decoder.generate(features, max_len=max_len)
            words = [
                id_to_word.get(index, "<unk>")
                for index in ids[0].tolist()
                if index != END
            ]
            print(f"  generated: {' '.join(words)}")
            for reference in item["caps"]:
                print(f"  reference: {' '.join(reference)}")
            print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the attention-based image captioner.",
    )
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="Folder containing train2014/ and val2014/.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with train.json, val.json and vocab.json.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where last.pt, best.pt, vocab.json and metrics.csv go.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N training images.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Decoder learning rate.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2),
                        help="DataLoader worker processes.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable torch.autocast mixed precision on GPU.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Training-state file to resume from (last_state.pt).")
    parser.add_argument("--overfit", type=int, default=None,
                        help="Overfit N images with one fixed caption each, then print captions.")
    parser.add_argument("--finetune-after", type=int, default=None, metavar="E",
                        help="Unfreeze encoder layer3/layer4 from epoch E at lr 1e-4.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many epochs without BLEU-4 improvement.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Training entry point."""
    args = parse_args(argv)
    random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"device: {device} | amp: {use_amp}")
    if args.amp and device.type == "cpu":
        print("WARNING: --amp ignored, autocast needs a CUDA device.")

    train_loader, val_loader, bleu_loader, vocab, train_items = build_dataloaders(args)
    id_to_word = invert_vocab(vocab)
    print(
        f"train images: {len(train_items):,} | batches/epoch: {len(train_loader)} | "
        f"val images: {len(val_loader.dataset):,} | "
        f"bleu sample: {len(bleu_loader.dataset)} | vocab: {len(vocab):,}"
    )

    model = CaptioningModel.from_config(ModelConfig(vocab_size=len(vocab)))
    model.to(device)

    optimizer = build_optimizer(model, args.lr)
    start_epoch = 1
    best_bleu4 = -1.0
    finetune_applied = False

    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"resume file not found: {args.resume}")
        resumed = restore(args.resume, model, vocab, args.lr, device)
        optimizer = resumed.optimizer
        start_epoch = resumed.epoch + 1
        best_bleu4 = resumed.best_bleu4
        finetune_applied = resumed.finetune_applied
        print(
            f"resumed from {args.resume} | next epoch {start_epoch} | "
            f"best BLEU-4 {best_bleu4:.4f}"
        )
        if start_epoch > args.epochs:
            print("nothing to do: already trained for the requested number of epochs")
            return 0

    if args.finetune_after is not None:
        print(
            f"encoder layer3/layer4 unfreeze at epoch {args.finetune_after} "
            f"with lr {FINETUNE_LR}"
        )

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.finetune_after is not None
            and epoch >= args.finetune_after
            and not finetune_applied
        ):
            optimizer = apply_finetune(model, optimizer, args.lr)
            finetune_applied = True
            print(f"epoch {epoch}: encoder fine-tuning enabled (lr {FINETUNE_LR})")

        reset_gpu_memory(device)
        started = time.time()

        train_loss = run_epoch(
            model, train_loader, device, optimizer, scaler, use_amp, training=True
        )
        val_loss = run_epoch(
            model, val_loader, device, None, scaler, use_amp, training=False
        )
        val_bleu4 = greedy_bleu4(
            model, bleu_loader, id_to_word, model.config.max_len, device
        )
        seconds = time.time() - started

        result = EpochResult(
            train_loss, val_loss, val_bleu4, seconds, gpu_memory_mb(device)
        )
        print_epoch_summary(epoch, args.epochs, result)
        append_metrics(
            metrics_path,
            epoch,
            result,
            [group["lr"] for group in optimizer.param_groups],
        )

        save_checkpoint(
            args.out_dir / "last.pt", model, vocab, epoch, val_bleu4
        )
        save_resume_state(
            args.out_dir / "last_state.pt",
            model,
            optimizer,
            epoch,
            max(best_bleu4, val_bleu4),
            vocab,
        )

        if val_bleu4 > best_bleu4:
            best_bleu4 = val_bleu4
            epochs_without_improvement = 0
            save_checkpoint(
                args.out_dir / "best.pt", model, vocab, epoch, val_bleu4
            )
            print(f"  new best val BLEU-4 {best_bleu4:.4f} -> best.pt")
        else:
            epochs_without_improvement += 1
            print(
                f"  no improvement "
                f"({epochs_without_improvement}/{args.patience} without BLEU-4 gain)"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping: {args.patience} epochs without BLEU-4 improvement "
                f"(best {best_bleu4:.4f})"
            )
            break

    print(f"\nbest val BLEU-4 (nltk proxy): {best_bleu4:.4f}")
    print(f"checkpoints in {args.out_dir}/ | metrics in {metrics_path}")

    if args.overfit is not None:
        print_overfit_captions(
            model,
            train_items,
            id_to_word,
            args.coco_root,
            device,
            model.config.max_len,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
