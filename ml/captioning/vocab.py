"""Vocabulary and caption <-> id conversion.

Mirrors Part 2 of ``ml/notebooks/captioning_Image.ipynb``:

* special ids ``<pad>=0 <start>=1 <end>=2 <unk>=3``
* a word joins the vocabulary only when it occurs at least ``MIN_FREQ`` times in
  the training captions; everything else becomes ``<unk>``
* captions longer than ``MAX_LEN`` tokens are dropped while building the splits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of the import
    import torch

PAD, START, END, UNK = 0, 1, 2, 3

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

#: Insertion order matters: ids are assigned from 0 upwards.
SPECIAL_TOKENS: Dict[str, int] = {
    PAD_TOKEN: PAD,
    START_TOKEN: START,
    END_TOKEN: END,
    UNK_TOKEN: UNK,
}

#: Keep words seen >= 5 times in the training captions.
MIN_FREQ = 5

#: Drop training captions longer than this many tokens.
MAX_LEN = 50


def build_vocab(counter: Mapping[str, int], min_freq: int = MIN_FREQ) -> Dict[str, int]:
    """Build the ``word -> id`` mapping from a training word counter.

    Specials come first with fixed ids, then every word whose count is at least
    ``min_freq``, ordered alphabetically for a stable vocabulary across runs.
    """
    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    for word, count in sorted(counter.items()):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def invert_vocab(vocab: Mapping[str, int]) -> Dict[int, str]:
    """Turn a ``word -> id`` mapping into an ``id -> word`` mapping."""
    return {index: word for word, index in vocab.items()}


def encode(tokens: Sequence[str], vocab: Mapping[str, int]) -> List[int]:
    """Tokens -> ids, wrapped in ``<start>`` / ``<end>``; unknown words map to ``<unk>``."""
    return [START] + [vocab.get(word, UNK) for word in tokens] + [END]


def decode(
    ids: Iterable[int],
    id_to_word: Mapping[int, str],
    strip_specials: bool = False,
) -> List[str]:
    """Ids -> words, stopping at ``<end>``.

    Ids with no entry in ``id_to_word`` decode to ``<unk>``. With
    ``strip_specials`` the surrounding ``<start>`` / ``<end>`` markers are removed,
    which is what the inference path wants.
    """
    words: List[str] = []
    for index in ids:
        if index == END:
            break
        word = id_to_word.get(index, UNK_TOKEN)
        if strip_specials and index in (PAD, START):
            continue
        words.append(word)
    return words


def mask_generated_specials(logits: "torch.Tensor") -> "torch.Tensor":
    """Forbid ``<pad>`` and ``<start>`` from ever being *generated*.

    ``decode(strip_specials=True)`` drops those two ids wherever they appear, so
    if the decoder were allowed to emit one mid-sequence the returned words would
    be shorter than the attention rows collected alongside them -- the frontend
    then gets a heatmap it cannot line up with the caption.

    A caption legitimately contains neither token, so they are masked out of the
    logits during decoding instead. This is generation-only: during teacher
    forcing ``<pad>`` is a real *input* for short captions and must stay visible.

    Returns a new tensor; the input is left untouched.
    """
    masked = logits.clone()
    masked[..., PAD] = float("-inf")
    masked[..., START] = float("-inf")
    return masked
