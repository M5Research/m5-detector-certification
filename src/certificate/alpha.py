"""Geometric error spending for evidentiary certificate revisions."""
from __future__ import annotations

ALPHA_BUDGET = 0.05


def evidentiary_alpha(k: int, budget: float = ALPHA_BUDGET) -> float:
    """Return alpha allocated to evidentiary revision k.

    ``alpha_k = budget * 2^(-(k+1))``. The infinite sum is exactly ``budget``.
    Metadata patches retain ``k`` and must not call this as a new spend.
    """
    if k < 0:
        raise ValueError("evidence revision must be >= 0")
    return float(budget) * (2.0 ** -(k + 1))
