"""Pytest path setup for the reproduction package.

Puts the package root (for ``backtest`` and ``scripts``) and ``src`` (for the
``strategies`` namespace) on ``sys.path`` so the verification suite runs from a
clean checkout with no install step.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
