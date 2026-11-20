"""Ensure repository root is on sys.path for both `pytest` and `python -m pytest`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
