"""Finite-sample confidence and multiplicity helpers."""
from __future__ import annotations

import numpy as np


def test_exact_one_sided_binomial_bounds_cover_edge_counts() -> None:
    from certificate.statistics import exact_binomial_bounds

    zero = exact_binomial_bounds(0, 10_000, alpha=0.025)
    all_ = exact_binomial_bounds(2_000, 2_000, alpha=0.025)

    assert zero["lower"] == 0.0
    assert 0.0 < zero["upper"] < 0.001
    assert 0.99 < all_["lower"] < 1.0
    assert all_["upper"] == 1.0


def test_stationary_mean_interval_is_deterministic_and_ordered() -> None:
    from certificate.statistics import stationary_mean_interval

    values = np.sin(np.arange(200) / 5.0)
    first = stationary_mean_interval(values, n_boot=199, alpha=0.05, seed=7)
    second = stationary_mean_interval(values, n_boot=199, alpha=0.05, seed=7)

    assert first == second
    assert first[0] <= float(np.mean(values)) <= first[1]


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    from certificate.statistics import holm_adjust

    adjusted = holm_adjust([0.01, 0.03, 0.02, 0.8])

    assert adjusted == [0.04, 0.06, 0.06, 0.8]
