"""Tests for the metrics helpers and the evaluate entry point.

pycocoevalcap needs Java for PTBTokenizer and ships METEOR's jar with the
package. The tests here use a stub scorer where the real one would be needed, so
the suite runs without a JVM, and exercise the real pycocoevalcap path when it
is importable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from captioning.decoding import BeamSearchDecodingStrategy, GreedyDecodingStrategy
from captioning.metrics import (
    FALLBACK_TOKENIZER,
    METRIC_NAMES,
    PTB_TOKENIZER,
    CorpusScores,
    java_available,
    per_image_cider,
    score_captions,
    simple_tokenize,
)
from captioning.model import ModelConfig, save_checkpoint
from captioning.vocab import SPECIAL_TOKENS
from ml import evaluate as evaluate_module
from ml.evaluate import (
    build_results,
    build_scoring_dicts,
    build_strategy,
    generate_captions,
    load_split,
    main,
    parse_args,
    print_ranked,
    rank_records,
)

VOCAB_SIZE = 8
VOCAB = dict(SPECIAL_TOKENS)
VOCAB.update({f"w{i}": i + 4 for i in range(VOCAB_SIZE - 4)})

REFS = {
    0: ["a dog runs in the grass", "a dog is running on grass"],
    1: ["a cat sits on a mat", "a cat is sitting"],
}
CAPS = {
    0: ["a dog runs in the grass"],
    1: ["a cat sits on a sofa"],
}


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------


def test_simple_tokenize_lowercases_and_splits():
    assert simple_tokenize("A Dog  RUNS") == "a dog runs"
    assert simple_tokenize("  leading and trailing  ") == "leading and trailing"


def test_simple_tokenize_does_not_strip_punctuation():
    # documents the difference from PTBTokenizer
    assert simple_tokenize("grass.") == "grass."


def test_java_available_returns_bool():
    assert isinstance(java_available(), bool)


def test_tokenizer_names_are_distinct():
    assert PTB_TOKENIZER != FALLBACK_TOKENIZER


def test_metric_names():
    assert METRIC_NAMES == (
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "CIDEr",
    )


# --------------------------------------------------------------------------
# score_captions validation
# --------------------------------------------------------------------------


def test_score_captions_rejects_empty_references():
    with pytest.raises(ValueError):
        score_captions({}, CAPS)


def test_score_captions_rejects_empty_candidates():
    with pytest.raises(ValueError):
        score_captions(REFS, {})


def test_score_captions_rejects_id_mismatch():
    with pytest.raises(ValueError) as error:
        score_captions(REFS, {0: ["a dog runs"]})
    assert "different images" in str(error.value)


def test_score_captions_uses_fallback_when_java_missing(monkeypatch):
    """With use_ptb=False the scorers run on simple-tokenised text."""
    monkeypatch.setattr(
        "captioning.metrics._run_scorers",
        lambda gts, res: ({"Bleu_1": 0.5, "CIDEr": 0.1}, {"CIDEr": [0.1, 0.0]}),
    )
    result = score_captions(REFS, CAPS, use_ptb=False)
    assert result.tokenizer == FALLBACK_TOKENIZER
    assert result.scores["Bleu_1"] == 0.5
    assert result["CIDEr"] == 0.1


def test_score_captions_falls_back_when_ptb_raises(monkeypatch):
    def explode(_gts, _res):
        raise RuntimeError("java not found")

    monkeypatch.setattr("captioning.metrics._prepare_captions", explode)
    monkeypatch.setattr(
        "captioning.metrics._run_scorers",
        lambda gts, res: ({"Bleu_4": 0.25}, {}),
    )
    result = score_captions(REFS, CAPS, use_ptb=True)
    assert result.tokenizer == FALLBACK_TOKENIZER
    assert result.scores["Bleu_4"] == 0.25


def test_score_captions_uses_ptb_when_available(monkeypatch):
    monkeypatch.setattr(
        "captioning.metrics._prepare_captions",
        lambda gts, res: (gts, res),
    )
    monkeypatch.setattr(
        "captioning.metrics._run_scorers",
        lambda gts, res: ({"Bleu_4": 0.4}, {}),
    )
    result = score_captions(REFS, CAPS, use_ptb=True)
    assert result.tokenizer == PTB_TOKENIZER


def test_corpus_scores_mapping_access():
    scores = CorpusScores(scores={"CIDEr": 0.9}, tokenizer=FALLBACK_TOKENIZER)
    assert scores["CIDEr"] == 0.9
    assert scores.as_dict() == {"CIDEr": 0.9}


def test_per_image_cider_returns_every_image(monkeypatch):
    """Every scored image gets a CIDEr value back, keyed by image id.

    pycocoevalcap is not installed in the unit-test environment, so a stub Cider
    is injected. The end-to-end check with the real scorers is
    test_real_pycocoevalcap_scores, which skips when the package is absent.
    """
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")

    import pycocoevalcap.cider.cider as cider_module

    original = cider_module.Cider

    def counting_factory(*args, **kwargs):
        class Counting(original):
            def compute_score(self, gts, res):
                # record the tokenised text so the test can check it
                scored["res"] = dict(res)
                return super().compute_score(gts, res)

        return Counting(*args, **kwargs)

    scored: dict = {}
    monkeypatch.setattr(cider_module, "Cider", counting_factory)
    scores = per_image_cider(REFS, CAPS, use_ptb=False)
    assert set(scores) == {0, 1}
    # the fallback tokeniser lower-cased the candidates before scoring
    assert scored["res"] == {
        0: ["a dog runs in the grass"],
        1: ["a cat sits on a sofa"],
    }


def test_simple_tokenize_applied_before_scoring(monkeypatch):
    """The fallback path lower-cases both sides before the scorers see them."""
    monkeypatch.setattr(
        "captioning.metrics._run_scorers",
        lambda gts, res: ({"Bleu_1": 1.0}, {}),
    )
    captured = {}

    def capture(gts, res):
        captured["gts"] = gts
        captured["res"] = res
        return {"Bleu_1": 1.0}, {}

    monkeypatch.setattr("captioning.metrics._run_scorers", capture)
    score_captions(
        {0: ["A Dog Runs"]}, {0: ["A Dog Runs"]}, use_ptb=False
    )
    assert captured["gts"] == {0: ["a dog runs"]}
    assert captured["res"] == {0: ["a dog runs"]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(
        ["--checkpoint", "c.pt", "--data-dir", "d", "--coco-root", "r"]
    )
    assert args.split == "test"
    assert args.beam_size == 3
    assert args.limit is None
    assert args.out == Path("results.json")


def test_parse_args_all_flags():
    args = parse_args(
        [
            "--checkpoint", "c.pt", "--data-dir", "d", "--coco-root", "r",
            "--split", "val", "--beam-size", "5", "--limit", "100",
            "--out", "out/res.json",
        ]
    )
    assert (args.split, args.beam_size, args.limit) == ("val", 5, 100)
    assert args.out == Path("out/res.json")


def test_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args([])


def test_build_strategy_greedy_for_zero_beam():
    model = type("M", (), {"decoder": torch.nn.Linear(1, 1)})()
    for beam_size in (0, None, 1):
        strategy = build_strategy(model, beam_size)
        assert isinstance(strategy, GreedyDecodingStrategy), beam_size


def test_build_strategy_beam():
    model = type("M", (), {"decoder": torch.nn.Linear(1, 1)})()
    strategy = build_strategy(model, 4)
    assert isinstance(strategy, BeamSearchDecodingStrategy)
    assert strategy.beam_size == 4


def test_load_split_missing(tmp_path):
    with pytest.raises(SystemExit):
        load_split(tmp_path, "test")


# --------------------------------------------------------------------------
# fake dataset + generator
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_setup(tmp_path, monkeypatch):
    """A 3-image fake split plus a stubbed CaptionGenerator."""
    coco_root = tmp_path / "coco"
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)

    split = []
    for index in range(3):
        name = f"val2014/v{index}.jpg"
        path = coco_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), color=(10, 20, 30)).save(path)
        split.append(
            {
                "path": name,
                "caps": [
                    ["a", "dog", "runs"],
                    ["a", "dog", "sleeps"],
                ],
            }
        )
    (data_dir / "test.json").write_text(json.dumps(split))
    (data_dir / "val.json").write_text(json.dumps(split[:1]))

    # a real checkpoint file, so the CLI's existence check passes
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = ModelConfig(vocab_size=VOCAB_SIZE)

        def state_dict(self):
            return {}

        def load_state_dict(self, state, strict=True):
            return None

    checkpoint = tmp_path / "c.pt"
    save_checkpoint(checkpoint, TinyModel(), VOCAB, epoch=0, val_bleu4=0.0)

    class FakeResult:
        def __init__(self, caption):
            self.caption = caption
            self.words = caption.split()
            self.num_words = len(self.words)
            self.time_ms = 1.5

    class FakeGenerator:
        def __init__(self, checkpoint, device="cpu", strategy=None):
            self.checkpoint = checkpoint
            self.device = torch.device(device)
            self.model = type("M", (), {"decoder": torch.nn.Linear(1, 1)})()
            self.strategy = strategy or GreedyDecodingStrategy(self.model.decoder)
            self.vocab = dict(VOCAB)
            self.config = ModelConfig(vocab_size=VOCAB_SIZE)
            self.max_len = 50
            self.metadata = {"epoch": 4, "val_bleu4": 0.22}

        def generate(self, image):
            return FakeResult("a dog runs")

    monkeypatch.setattr(evaluate_module, "CaptionGenerator", FakeGenerator)
    return {
        "coco_root": coco_root,
        "data_dir": data_dir,
        "checkpoint": checkpoint,
        "out": tmp_path / "results.json",
    }


def test_generate_captions_keeps_all_references(fake_setup, monkeypatch):
    from captioning.generator import CaptionResult

    # a generator that always returns the same caption
    class G:
        max_len = 50

        def generate(self, image):
            return CaptionResult(
                caption="a dog runs",
                words=["a", "dog", "runs"],
                attention=__import__("numpy").zeros((3, 7, 7), dtype="float32"),
                time_ms=2.0,
            )

    items = load_split(fake_setup["data_dir"], "test")
    records = generate_captions(G(), items, fake_setup["coco_root"])
    assert len(records) == 3
    # both human captions kept as references
    assert records[0]["references"] == ["a dog runs", "a dog sleeps"]
    assert records[0]["caption"] == "a dog runs"
    assert records[0]["image_id"] == 0
    assert records[0]["num_words"] == 3
    assert records[0]["time_ms"] == 2.0


def test_build_scoring_dicts_shapes(fake_setup):
    items = load_split(fake_setup["data_dir"], "test")

    class G:
        def generate(self, image):
            import numpy as np

            from captioning.generator import CaptionResult

            return CaptionResult("a dog runs", ["a", "dog", "runs"], np.zeros((3, 7, 7), dtype="float32"), 1.0)

    records = generate_captions(G(), items, fake_setup["coco_root"])
    gts, res = build_scoring_dicts(records)
    assert set(gts) == set(res) == {0, 1, 2}
    assert all(len(v) == 2 for v in gts.values())  # two references
    assert all(len(v) == 1 for v in res.values())  # one candidate


def test_rank_records_sorts_best_first():
    records = [
        {"image_id": 0, "caption": "b", "references": []},
        {"image_id": 1, "caption": "a", "references": []},
        {"image_id": 2, "caption": "c", "references": []},
    ]
    ranked = rank_records(records, {0: 0.1, 1: 0.9, 2: 0.5})
    assert [r["image_id"] for r in ranked] == [1, 2, 0]
    assert ranked[0]["cider"] == pytest.approx(0.9)


def test_rank_records_breaks_ties_deterministically():
    records = [
        {"image_id": 0, "caption": "zebra"},
        {"image_id": 1, "caption": "apple"},
    ]
    first = rank_records(records, {0: 0.5, 1: 0.5})
    second = rank_records(list(reversed(records)), {0: 0.5, 1: 0.5})
    assert [r["caption"] for r in first] == [r["caption"] for r in second] == [
        "apple",
        "zebra",
    ]


def test_rank_records_missing_score_defaults_to_zero():
    ranked = rank_records([{"image_id": 0, "caption": "x", "references": []}], {})
    assert ranked[0]["cider"] == 0.0


def test_print_ranked_prints_references(capsys):
    records = [
        {
            "caption": "a dog runs",
            "references": ["a dog runs fast"],
            "cider": 1.2,
        }
    ]
    print_ranked("best", records, count=1)
    out = capsys.readouterr().out
    assert "best" in out
    assert "CIDEr 1.2000" in out
    assert "ref: a dog runs fast" in out


def test_print_ranked_handles_empty(capsys):
    print_ranked("best", [], count=3)
    assert "(none)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# results payload
# --------------------------------------------------------------------------


def test_build_results_contains_everything(fake_setup, monkeypatch):
    class FakeGenerator:
        device = torch.device("cpu")
        vocab = dict(VOCAB)
        max_len = 50
        metadata = {"epoch": 4, "val_bleu4": 0.22}
        config = ModelConfig(vocab_size=VOCAB_SIZE)
        model = type("M", (), {"decoder": torch.nn.Linear(1, 1)})()
        strategy = BeamSearchDecodingStrategy(model.decoder, beam_size=3)

    args = parse_args(
        [
            "--checkpoint", "w/best.pt",
            "--data-dir", str(fake_setup["data_dir"]),
            "--coco-root", str(fake_setup["coco_root"]),
            "--split", "test",
            "--beam-size", "3",
            "--out", str(fake_setup["out"]),
        ]
    )
    records = [
        {
            "index": 0,
            "image_id": 0,
            "path": "val2014/v0.jpg",
            "caption": "a dog runs",
            "references": ["a dog runs", "a dog sleeps"],
            "num_words": 3,
            "time_ms": 2.0,
            "cider": 0.8,
        }
    ]
    scores = {"Bleu_1": 0.3, "Bleu_4": 0.1, "METEOR": 0.2, "CIDEr": 0.5}
    payload = build_results(
        scores,
        {"CIDEr": [0.8]},
        records,
        FakeGenerator(),
        args,
        "test",
        FALLBACK_TOKENIZER,
        elapsed_seconds=1.25,
    )

    assert payload["scores"] == scores
    assert payload["model"]["epoch"] == 4
    assert payload["model"]["config"]["encoder_dim"] == 2048
    assert payload["model"]["vocab_size"] == len(VOCAB)
    assert payload["decoding"] == {
        "strategy": "beam_search",
        "beam_size": 3,
        "max_len": 50,
    }
    assert payload["dataset"]["split"] == "test"
    assert payload["dataset"]["num_images"] == 1
    assert payload["tokenizer"] == FALLBACK_TOKENIZER
    assert "whitespace" in payload["tokenizer_note"]
    assert payload["captions"][0]["caption"] == "a dog runs"
    assert payload["captions"][0]["references"]
    assert payload["runtime"]["elapsed_seconds"] == 1.25
    assert "per_image_summary" in payload
    # serialisable
    json.dumps(payload)


def test_build_results_notes_ptb_tokenizer():
    class FakeGenerator:
        device = torch.device("cpu")
        vocab = dict(VOCAB)
        max_len = 50
        metadata = {}
        config = ModelConfig(vocab_size=VOCAB_SIZE)
        model = type("M", (), {"decoder": torch.nn.Linear(1, 1)})()
        strategy = GreedyDecodingStrategy(model.decoder)

    args = parse_args(
        ["--checkpoint", "c.pt", "--data-dir", "d", "--coco-root", "r", "--beam-size", "0"]
    )
    payload = build_results(
        {}, {}, [], FakeGenerator(), args, "test", PTB_TOKENIZER, 0.0
    )
    assert "COCO leaderboard" in payload["tokenizer_note"]
    assert payload["decoding"]["strategy"] == "greedy"


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_main_writes_results_json(fake_setup, monkeypatch, capsys):
    """Full run against a stubbed generator and a stubbed scorer."""
    monkeypatch.setattr(
        evaluate_module,
        "score_captions",
        lambda gts, res, use_ptb=None: CorpusScores(
            scores={
                "Bleu_1": 0.31,
                "Bleu_2": 0.2,
                "Bleu_3": 0.14,
                "Bleu_4": 0.1,
                "METEOR": 0.28,
                "CIDEr": 0.44,
            },
            per_image={"CIDEr": [0.9, 0.3, 0.1], "METEOR": [0.3, 0.3, 0.2]},
            tokenizer=FALLBACK_TOKENIZER,
        ),
    )
    monkeypatch.setattr(
        evaluate_module,
        "per_image_cider",
        lambda gts, res, use_ptb=None: {0: 0.9, 1: 0.3, 2: 0.1},
    )

    exit_code = main(
        [
            "--checkpoint", str(fake_setup["checkpoint"]),
            "--data-dir", str(fake_setup["data_dir"]),
            "--coco-root", str(fake_setup["coco_root"]),
            "--out", str(fake_setup["out"]),
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "CIDEr" in out
    assert "best 10 captions" in out
    assert "worst 10 captions" in out
    assert "PTBTokenizer needs a JVM" in out

    payload = json.loads(fake_setup["out"].read_text())
    assert payload["scores"]["CIDEr"] == 0.44
    assert len(payload["captions"]) == 3
    assert all(len(c["references"]) == 2 for c in payload["captions"])
    assert payload["decoding"]["beam_size"] == 3
    assert payload["tokenizer"] == FALLBACK_TOKENIZER


def test_main_respects_limit(fake_setup, monkeypatch):
    monkeypatch.setattr(
        evaluate_module,
        "score_captions",
        lambda gts, res, use_ptb=None: CorpusScores(
            scores={"CIDEr": 0.1}, per_image={"CIDEr": [0.1]}, tokenizer=FALLBACK_TOKENIZER
        ),
    )
    monkeypatch.setattr(
        evaluate_module, "per_image_cider", lambda gts, res, use_ptb=None: {0: 0.1}
    )
    main(
        [
            "--checkpoint", str(fake_setup["checkpoint"]),
            "--data-dir", str(fake_setup["data_dir"]),
            "--coco-root", str(fake_setup["coco_root"]),
            "--limit", "1",
            "--out", str(fake_setup["out"]),
        ]
    )
    payload = json.loads(fake_setup["out"].read_text())
    assert len(payload["captions"]) == 1
    assert payload["dataset"]["num_images"] == 1
    assert payload["dataset"]["limit"] == 1


def test_main_missing_checkpoint_exits(tmp_path, fake_setup):
    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint", str(tmp_path / "nope.pt"),
                "--data-dir", str(fake_setup["data_dir"]),
                "--coco-root", str(fake_setup["coco_root"]),
            ]
        )


def test_main_missing_split_exits(fake_setup):
    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint", str(fake_setup["checkpoint"]),
                "--data-dir", str(fake_setup["data_dir"]),
                "--coco-root", str(fake_setup["coco_root"]),
                "--split", "val2",
            ]
        )


def test_real_pycocoevalcap_scores():
    """Exercise the actual BLEU/METEOR/CIDEr scorers.

    Skipped unless pycocoevalcap is importable; it is in ``ml/requirements.txt``
    but not always present in a bare environment.
    """
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")

    result = score_captions(REFS, CAPS, use_ptb=False)

    assert result.tokenizer == FALLBACK_TOKENIZER
    for name in METRIC_NAMES:
        assert name in result.scores, name
        assert isinstance(result.scores[name], float)

    # image 0 matches a reference exactly, image 1 does not, so BLEU-1 must be
    # strictly higher than 0 and CIDEr must reflect the mismatch
    assert 0.0 < result.scores["Bleu_1"] < 1.0
    assert result.scores["Bleu_1"] >= result.scores["Bleu_4"]
    assert result.scores["CIDEr"] > 0.0


def test_real_pycocoevalcap_perfect_match_scores_one():
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")

    perfect = {0: ["a dog runs in the grass"], 1: ["a cat sits on a mat"]}
    result = score_captions(REFS, perfect, use_ptb=False)
    assert result.scores["Bleu_1"] == pytest.approx(1.0, abs=1e-6)
    assert result.scores["Bleu_4"] == pytest.approx(1.0, abs=1e-6)


def test_real_pycocoevalcap_empty_captions_score_zero():
    """A caption set of empty strings must not crash the scorers."""
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")

    blanks = {0: [""], 1: [""]}
    result = score_captions(REFS, blanks, use_ptb=False)
    assert result.scores["Bleu_1"] == pytest.approx(0.0, abs=1e-6)
    assert result.scores["CIDEr"] == pytest.approx(0.0, abs=1e-6)


def test_real_pycocoevalcap_per_image_cider_covers_all():
    pytest.importorskip("pycocoevalcap", reason="pycocoevalcap not installed")

    scores = per_image_cider(REFS, CAPS, use_ptb=False)
    assert set(scores) == {0, 1}
    # the mismatched caption scores strictly worse
    assert scores[0] > scores[1]


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "ml.evaluate", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0
    assert "--beam-size" in result.stdout
    assert "--split" in result.stdout
    assert "--out" in result.stdout
