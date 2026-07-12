"""Unit tests for scripts/wp1/gate_analysis.py.

Seven tests:
  1. test_constant_series_is_degenerate   -- constant pred flagged as degenerate
  2. test_kw_epsilon_sq_known_groups      -- clearly-separated groups -> eps_sq > 0.5
  3. test_non_overlapping_sampler         -- stride extractor returns correct length + offset
  4. test_gate_report_schema              -- run_gate dict has required keys
  5. test_min_gap_corrected_sampler       -- SC1: WR-01-corrected sampler gap >= stride
  6. test_epsilon_sq_boot_ci              -- SC2: block-bootstrap CI on grouped epsilon-sq
  7. test_permutation_null_range          -- SC2: permutation-null p in (0, 1]

All seeded with np.random.default_rng (never np.random.seed).
No xfail markers -- these are live tests as of plan 00-04.
"""
from __future__ import annotations

import numpy as np
from scripts.wp1.gate_analysis import check_degeneracy, non_overlapping_samples, run_gate


def test_constant_series_is_degenerate() -> None:
    """A constant predictability series (all zeros) must be flagged as degenerate.

    check_degeneracy(np.zeros(500)) -> is_degenerate == True.
    A constant series has std == 0, which immediately triggers the degeneracy flag
    (pre-registered condition: std == 0 => degenerate, per D-04).
    """
    pred = np.zeros(500)
    result = check_degeneracy(pred)
    assert result["is_degenerate"] is True


def test_kw_epsilon_sq_known_groups() -> None:
    """KW + epsilon-squared on clearly separated synthetic groups returns epsilon-sq > 0.5.

    Three groups drawn from well-separated Gaussians:
      g0 ~ N(0.0, 0.1), g1 ~ N(0.5, 0.1), g2 ~ N(1.0, 0.1)
    The effect is so large (Cohen analogues: 0.01 small, 0.06 medium, 0.14 large)
    that epsilon-squared must far exceed 0.5.

    Uses np.random.default_rng(7) for determinism.
    """
    rng = np.random.default_rng(7)
    g0 = rng.normal(0.0, 0.1, 200)
    g1 = rng.normal(0.5, 0.1, 200)  # clearly separated from g0
    g2 = rng.normal(1.0, 0.1, 200)  # clearly separated from g0 and g1

    pred_series = np.concatenate([g0, g1, g2])
    # Regime labels: 0 for first 200, 1 for next 200, 2 for last 200
    regime = np.array([0] * 200 + [1] * 200 + [2] * 200, dtype=np.int8)

    # stride=1 (default): all 600 samples used as non-overlapping
    result = run_gate(
        pred_series={("W200", "q5"): pred_series},
        regime=regime,
        wq_grid=[("W200", "q5")],
    )
    assert result["per_wq"][0]["epsilon_sq"] > 0.5  # large separation expected


def test_non_overlapping_sampler() -> None:
    """Non-overlapping sampler returns the correct number and offset.

    For a 600-element array with stride=120:
      - length == 5  (600 / 120 = 5 windows)
      - first element == 119  (stride-1 offset: the last bar of the first window)

    This verifies Pitfall 3 mitigation: adjacent samples share no overlapping bars.
    """
    arr = np.arange(600, dtype=float)
    out = non_overlapping_samples(arr, stride=120)

    assert len(out) == 5   # 600 / 120
    assert out[0] == 119   # stride-1 offset (last element of first window)


def test_min_gap_corrected_sampler() -> None:
    """SC1: WR-01-corrected sampler guarantees gap >= stride in original index.

    Uses a series with 7 warmup bars (not a multiple of stride=120), so the
    buggy compacted-array sampler and the corrected sampler return DIFFERENT
    first sample positions.  The corrected sampler puts the first sample at
    original index stride-1=119; the buggy sampler would put it at orig[126].

    Both assertions (gap >= stride AND retained_idx[0] == stride-1) are required
    by 01-PREREGISTRATION.md §12 to distinguish the corrected from the buggy sampler.
    """
    from strategies.vol_regime_switch.regime_population import (
        non_overlapping_samples as nl_sampler,
    )

    stride = 120
    N = 600
    # 7 warmup bars (not a multiple of stride), then valid bars in three regimes
    regime = np.array([-1] * 7 + [0] * 400 + [1] * 153 + [2] * 40, dtype=np.int8)
    series = np.ones(N, dtype=np.float64)

    pred_nl, regime_nl, retained_idx = nl_sampler(series, regime, stride)

    # SC1 assertion 1: all consecutive gaps >= stride in original index
    gaps = np.diff(retained_idx)
    assert len(gaps) == 0 or int(gaps.min()) >= stride, (
        f"WR-01 FAIL: minimum gap {gaps.min()} < stride {stride}"
    )

    # SC1 assertion 2: first retained sample is at original index stride-1=119
    # (the buggy compacted sampler returns orig[7+119]=orig[126] instead)
    assert retained_idx[0] == stride - 1, (
        f"WR-01 FAIL: first retained index {retained_idx[0]} != {stride - 1} "
        "(expected stride-1 in original time, not stride-1 in compacted array)"
    )

    assert len(pred_nl) > 0
    assert len(pred_nl) == len(regime_nl)


def test_epsilon_sq_boot_ci() -> None:
    """SC2: block-bootstrap CI on epsilon-sq uses joint (pred, regime) resampling.

    Three well-separated synthetic groups (N=200 each, sigma=0.1):
      g0 ~ N(0.0, 0.1), g1 ~ N(0.5, 0.1), g2 ~ N(1.0, 0.1)

    Pitfall-1 guard: joint resampling must be used — point estimate from
    epsilon_sq_boot_ci must equal _epsilon_sq_kw to within 1e-9.
    """
    from strategies.vol_regime_switch.regime_population import (
        _epsilon_sq_kw,
        epsilon_sq_boot_ci,
    )

    rng = np.random.default_rng(7)
    n = 200
    g0 = rng.normal(0.0, 0.1, n)
    g1 = rng.normal(0.5, 0.1, n)
    g2 = rng.normal(1.0, 0.1, n)
    pred_nl = np.concatenate([g0, g1, g2])
    regime_nl = np.array([0] * n + [1] * n + [2] * n, dtype=np.int8)

    point, lo, hi = epsilon_sq_boot_ci(pred_nl, regime_nl, block=10, n_boot=2000, seed=43)

    assert np.isfinite(point), "point estimate must be finite"
    assert np.isfinite(lo), "CI lower bound must be finite"
    assert np.isfinite(hi), "CI upper bound must be finite"
    assert lo <= point <= hi, f"CI order violated: lo={lo}, point={point}, hi={hi}"

    # Pitfall-1: joint resampling — point must equal _epsilon_sq_kw exactly
    kw_point = _epsilon_sq_kw(pred_nl, regime_nl)
    assert abs(point - kw_point) < 1e-9, (
        f"Pitfall-1 guard: point={point} != _epsilon_sq_kw={kw_point} (joint resampling broken)"
    )


def test_permutation_null_range() -> None:
    """SC2: regime-label permutation null returns p in (0, 1] for 2+ groups.

    Uses the same three-group fixture as test_epsilon_sq_boot_ci.
    Degenerate single-group input must return NaN (not crash).
    """
    from strategies.vol_regime_switch.regime_population import (
        regime_label_permutation_null,
    )

    rng = np.random.default_rng(7)
    n = 200
    g0 = rng.normal(0.0, 0.1, n)
    g1 = rng.normal(0.5, 0.1, n)
    g2 = rng.normal(1.0, 0.1, n)
    pred_nl = np.concatenate([g0, g1, g2])
    regime_nl = np.array([0] * n + [1] * n + [2] * n, dtype=np.int8)

    p = regime_label_permutation_null(pred_nl, regime_nl, n_perm=5000, seed=42)

    assert np.isfinite(p), f"p must be finite for well-separated groups, got {p}"
    # +1/+1 Phipson-Smyth correction guarantees p >= 1/(n_perm+1) > 0
    assert p > 0, f"p must be strictly > 0 (Phipson-Smyth correction), got {p}"
    assert p <= 1.0, f"p must be <= 1, got {p}"

    # Degenerate: all same group -> NaN (not crash)
    regime_single = np.zeros(n, dtype=np.int8)
    p_degen = regime_label_permutation_null(pred_nl[:n], regime_single, n_perm=100, seed=42)
    assert np.isnan(p_degen), f"single-group input should return NaN, got {p_degen}"


def test_gate_report_schema() -> None:
    """run_gate return dict contains required top-level keys and per-wq row schema.

    Verifies that run_gate returns a dict with 'verdict' and 'per_wq', and that
    each per_wq row contains W, q, is_degenerate, epsilon_sq, kw_h, kw_pval --
    the fields required by Task 3 verdict ratification.

    Uses np.random.default_rng(3) for determinism.
    """
    rng = np.random.default_rng(3)
    pred = rng.normal(0.5, 0.2, 300)
    regime = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int8)

    result = run_gate(
        pred_series={("W120", "q5"): pred},
        regime=regime,
        wq_grid=[("W120", "q5")],
    )

    assert "verdict" in result
    assert "per_wq" in result
    for row in result["per_wq"]:
        assert "W" in row and "q" in row and "is_degenerate" in row
        assert "epsilon_sq" in row and "kw_h" in row and "kw_pval" in row
