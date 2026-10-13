"""Smoke-run a notebook's Python cells against a local fake environment.

Rewrites the Kaggle paths (REPO, WORK, COCO candidates) to a temp tree, then
executes the data-prep, training and plotting cells for real. Verifies the
notebook works rather than just parsing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_notebook import strip_magics
from PIL import Image

CAPTIONS = [["a", "dog", "runs"], ["a", "dog", "sleeps"], ["a", "cat", "sits"]]


def build_fake_coco(root: Path, karpathy: Path, count: int = 4) -> list[dict]:
    """A tiny COCO tree plus a matching Karpathy annotation file.

    Returns the entries so a test split can be written from them.
    """
    entries = []
    for index in range(count):
        sub = "val2014" if index % 2 == 0 else "train2014"
        folder = root / sub
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (30, 120, 200)).save(folder / f"COCO_{index}.jpg")
        entries.append(
            {
                "filepath": sub,
                "filename": f"COCO_{index}.jpg",
                "split": "val" if index % 2 else "restval",
                "sentences": [{"tokens": caption} for caption in CAPTIONS],
            }
        )
    karpathy.write_text(json.dumps({"images": entries}))
    return entries


def run(notebook_path: Path, cells: list[int]) -> dict:
    """Execute the given cells in order, sharing one namespace."""
    repo = str(Path(__file__).resolve().parent.parent)
    work = Path(tempfile.mkdtemp()) / "working"
    coco = Path(tempfile.mkdtemp()) / "input" / "coco2014"
    work.mkdir(parents=True)
    build_fake_coco(coco, work / "dataset_coco.json")

    notebook = json.loads(notebook_path.read_text())
    namespace: dict = {}
    substitutions = [
        (r'REPO\s*=\s*"/kaggle/working/repo"', f"REPO = {repo!r}"),
        (r'WORK\s*=\s*Path\("/kaggle/working"\)', f"WORK = Path({str(work)!r})"),
        (r'"epochs":\s*\d+', '"epochs": 1'),
        (r'"workers":\s*\d+', '"workers": 0'),
        (r'"batch_size":\s*\d+', '"batch_size": 2'),
    ]

    for index in cells:
        source = strip_magics("".join(notebook["cells"][index]["source"]))
        if not source.strip():
            continue
        for pattern, replacement in substitutions:
            source = re.sub(pattern, replacement, source)
        source = re.sub(
            r"COCO_ROOT_CANDIDATES = \[.*?\]",
            f"COCO_ROOT_CANDIDATES = [{str(coco)!r}]",
            source,
            flags=re.S,
        )
        exec(compile(source, f"cell{index}", "exec"), namespace)
        print(f"cell {index:2d} ok")

    namespace["_work"] = work
    namespace["_coco"] = coco
    return namespace


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "ml/notebooks/02_train_kaggle.ipynb")
    selected = [4, 6, 8, 10, 13, 14, 18, 19, 24, 25]
    state = run(target, selected)

    run_dir = state["RUN_DIR"]
    print("\nartifacts:", sorted(p.name for p in run_dir.iterdir()))
    metrics = run_dir / "metrics.csv"
    if metrics.is_file():
        print("metrics.csv:")
        print(metrics.read_text().strip())
    data_dir = state["DATA_DIR"]
    print("\nsplits:", sorted(p.name for p in data_dir.iterdir()))
