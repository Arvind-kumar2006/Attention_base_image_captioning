"""Backend test setup.

Two jobs:

1. Point the ML package at ``ml/`` so ``import captioning`` resolves without an
   editable install having been run.
2. Redirect every settings value at a temp directory *before* anything imports
   ``backend.app.core.config``, since ``Settings`` reads the environment at
   import time. Keeping that in conftest means the individual test modules do
   not each have to remember to do it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="lumina-backend-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))
os.environ.setdefault("FRONTEND_ORIGIN", "http://localhost:5173")
# Deliberately points at nothing: a real checkpoint must never be loaded by a
# unit test, which would silently download ResNet weights.
os.environ.setdefault("CHECKPOINT_PATH", str(_TMP / "missing.pt"))
