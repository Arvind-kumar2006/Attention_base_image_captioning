"""Make ``ml/`` importable as the ``captioning`` package without installing it.

Lets ``pytest ml/tests`` run straight from a clone, before
``pip install -e ./ml`` has been run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
