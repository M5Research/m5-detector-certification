"""Finite-sample confidence procedures shared by certificate gates."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

from certificate.information import primary_block_length, stationary_block_indices


def exact_binomial_bounds(successes: int, draws: int, *, alpha: float) -> dict[str, float]:
    """One-sided Clopper--Pearson bounds with tail probability ``alpha``."""
    if draws <= 0 or not 0 <= successes <= draws:
        raise ValueError("require 0 <= successes <= draws and draws > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, draws - successes + 1))
    upper = (
        1.0
        if successes == draws
        else float(beta.ppf(1.0 - alpha, successes + 1, draws - successes))
    )
    return {"lower": lower, "upper": upper}


def stationary_mean_interval(
    values: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
    mean_block: int | None = None,
) -> tuple[float, float]:
    """Percentile interval for a dependent-series mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or n_boot <= 0:
        raise ValueError("values and n_boot must be non-empty/positive")
    block = mean_block or primary_block_length(int(arr.size))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        means[index] = float(np.mean(arr[stationary_block_indices(arr.size, block, rng)]))
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in original order."""
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, original in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[original]))
        adjusted[original] = running
    return adjusted
