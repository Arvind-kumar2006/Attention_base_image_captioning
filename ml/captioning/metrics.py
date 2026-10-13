"""Caption metrics: BLEU-1..4, METEOR and CIDEr via pycocoevalcap.

pycocoevalcap scores a corpus in two stages. Captions must be tokenised into
whitespace-joined strings, and the scorers want the reference corpus to match.
Tokenisation uses ``PTBTokenizer`` (a Java jar) when a JVM is available, which
is what the COCO leaderboard uses; otherwise we fall back to lower-casing and
whitespace splitting. The two are close but not identical, and METEOR in
particular shifts a little, so the tokeniser is recorded in the results.

COCO-style scores are high-is-better. METEOR and CIDEr additionally use corpus
level aggregation, so they are not the mean of per-image values.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

PTB_TOKENIZER = "PTBTokenizer"
FALLBACK_TOKENIZER = "lowercase-whitespace"

#: Metrics reported by this module, in the order they appear in results.json.
METRIC_NAMES = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "CIDEr")


class TokenizerUnavailable(RuntimeError):
    """Raised when the requested tokenizer cannot be used."""


@dataclass
class CorpusScores:
    """Corpus-level scores plus the tokeniser that produced them."""

    scores: Dict[str, float] = field(default_factory=dict)
    per_image: Dict[str, List[float]] = field(default_factory=dict)
    tokenizer: str = FALLBACK_TOKENIZER

    def __getitem__(self, name: str) -> float:
        return self.scores[name]

    def as_dict(self) -> Dict[str, Any]:
        """The scores only, ready for JSON."""
        return dict(self.scores)


def java_available() -> bool:
    """True when a JVM is on PATH, which PTBTokenizer needs."""
    return shutil.which("java") is not None


def simple_tokenize(text: str) -> str:
    """Lower-case and split on whitespace, re-joined by single spaces.

    The fallback for when Java is unavailable. It does not strip punctuation, so
    ``"grass."`` stays one token rather than becoming ``"grass"``.
    """
    return " ".join(text.lower().split())


def _to_caption_dicts(
    captions: Mapping[int, Sequence[str]]
) -> Dict[int, List[Dict[str, str]]]:
    """PTBTokenizer wants ``{id: [{"caption": text}]}``, not plain strings."""
    return {
        image_id: [{"caption": text} for text in texts]
        for image_id, texts in captions.items()
    }


def _unwrap_captions(tokenized: Mapping[int, Sequence[Any]]) -> Dict[int, List[str]]:
    """Normalise PTBTokenizer's output to ``{id: [str]}``.

    The tokenizer takes ``{"caption": text}`` entries but returns plain strings
    on the way out; this handles either shape so a pycocoevalcap version that
    changes its return format does not break scoring.
    """
    result: Dict[int, List[str]] = {}
    for image_id, entries in tokenized.items():
        texts: List[str] = []
        for entry in entries:
            texts.append(entry["caption"] if isinstance(entry, dict) else str(entry))
        result[image_id] = texts
    return result


def _prepare_captions(
    gts: Mapping[int, Sequence[str]],
    res: Mapping[int, Sequence[str]],
) -> tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Run both caption sets through PTBTokenizer.

    The tokenizer lower-cases for us, matching the fallback path.
    """
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tokenizer = PTBTokenizer()
    tokenized_gts = _unwrap_captions(tokenizer.tokenize(_to_caption_dicts(gts)))
    tokenized_res = _unwrap_captions(tokenizer.tokenize(_to_caption_dicts(res)))
    return tokenized_gts, tokenized_res


def score_captions(
    gts: Mapping[int, Sequence[str]],
    res: Mapping[int, Sequence[str]],
    use_ptb: Optional[bool] = None,
) -> CorpusScores:
    """Score generated captions against multiple references.

    Args:
        gts: ``{image_id: [reference captions]}``.
        res: ``{image_id: [generated captions]}``, one per image.
        use_ptb: force the Java tokenizer on or off. ``None`` auto-detects a JVM.

    Returns:
        A :class:`CorpusScores` holding BLEU-1..4, METEOR and CIDEr.

    Raises:
        ValueError: if the two dicts cover different image ids, or if either is
            empty. pycocoevalcap silently mis-scores mismatched input.
    """
    gts = {int(key): list(value) for key, value in gts.items()}
    res = {int(key): list(value) for key, value in res.items()}

    if not gts:
        raise ValueError("no references given")
    if not res:
        raise ValueError("no generated captions given")
    if set(gts) != set(res):
        missing = sorted(set(gts) - set(res))[:5]
        extra = sorted(set(res) - set(gts))[:5]
        raise ValueError(
            "references and captions cover different images "
            f"(missing captions for {missing}, unexpected captions for {extra})"
        )

    gts, res, tokenizer_name = _tokenise(gts, res, use_ptb)
    scores, per_image = _run_scorers(gts, res)
    return CorpusScores(
        scores=scores, per_image=per_image, tokenizer=tokenizer_name
    )


def _tokenise(
    gts: Mapping[int, Sequence[str]],
    res: Mapping[int, Sequence[str]],
    use_ptb: Optional[bool],
) -> tuple[Dict[int, List[str]], Dict[int, List[str]], str]:
    """Tokenise both caption sets, reporting which tokeniser was used.

    A PTBTokenizer failure of any kind falls back to the simple tokeniser rather
    than aborting the evaluation, since a missing JVM should degrade the score's
    precision, not lose the run.
    """
    if use_ptb is None:
        use_ptb = java_available()

    if use_ptb:
        try:
            gts, res = _prepare_captions(gts, res)
            return gts, res, PTB_TOKENIZER
        except Exception as error:
            print(
                f"PTBTokenizer unavailable ({error!r}); "
                "falling back to lowercase-whitespace tokenisation"
            )

    gts = {k: [simple_tokenize(t) for t in v] for k, v in gts.items()}
    res = {k: [simple_tokenize(t) for t in v] for k, v in res.items()}
    return gts, res, FALLBACK_TOKENIZER


def _run_scorers(
    gts: Dict[int, List[str]], res: Dict[int, List[str]]
) -> tuple[Dict[str, float], Dict[str, List[float]]]:
    """Invoke the pycocoevalcap scorers and collect corpus plus per-image values."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor

    scores: Dict[str, float] = {}
    per_image: Dict[str, List[float]] = {}

    bleu, _bleu_per_image = Bleu(4).compute_score(gts, res)
    # pycocoevalcap returns BLEU as [b1, b2, b3, b4, bp]; the brevity penalty
    # is not a reported metric here.
    for index, value in enumerate(bleu[:4]):
        scores[f"Bleu_{index + 1}"] = float(value)

    _meteor, meteor_per_image = Meteor().compute_score(gts, res)
    scores["METEOR"] = float(_meteor)
    per_image["METEOR"] = [float(value) for value in meteor_per_image]

    _cider, cider_per_image = Cider().compute_score(gts, res)
    scores["CIDEr"] = float(_cider)
    per_image["CIDEr"] = [float(value) for value in cider_per_image]

    return scores, per_image


def per_image_cider(
    gts: Mapping[int, Sequence[str]],
    res: Mapping[int, Sequence[str]],
    use_ptb: Optional[bool] = None,
) -> Dict[int, float]:
    """Per-image CIDEr, used to rank the best and worst captions.

    The corpus CIDEr score is a document-frequency weighted sum, so it is not
    the mean of these; the ranking is what matters here.
    """
    gts = {int(key): list(value) for key, value in gts.items()}
    res = {int(key): list(value) for key, value in res.items()}
    if set(gts) != set(res):
        raise ValueError("references and captions cover different images")

    gts, res, _tokenizer_name = _tokenise(gts, res, use_ptb)

    from pycocoevalcap.cider.cider import Cider

    _corpus, per_image = Cider().compute_score(gts, res)
    return {
        image_id: float(per_image[index])
        for index, image_id in enumerate(res.keys())
    }
