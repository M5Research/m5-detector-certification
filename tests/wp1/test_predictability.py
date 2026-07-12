"""Unit and cross-validation tests for scripts/wp1/predictability.py.

Tests are the RED gate for plan 00-02: they contain real assertions and import
from scripts.wp1.predictability, which does NOT exist yet.  The suite must fail
with ImportError/ModuleNotFoundError — that is the expected RED signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.wp1.predictability import VREstimator, compute_rolling_predictability


def test_vr_m2_matches_arch_reference() -> None:
    """Cross-validate hand-rolled VR M2 against arch.unitroot.VarianceRatio(robust=True).

    Guards the arch import with pytest.importorskip so collection succeeds
    even if arch is absent.
    """
    arch_unitroot = pytest.importorskip(
        "arch.unitroot", reason="arch not installed — skipping VR cross-val"
    )

    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.001, 240)
    q = 5

    # Our estimator (single-window call on the full length-240 window)
    est = VREstimator(W=240, q=q)
    vr_ours, _ = est._vr_m2(r)

    # Reference: arch VR test takes price levels; robust=True -> M2
    prices = np.exp(np.cumsum(r))
    vr_ref = arch_unitroot.VarianceRatio(prices, lags=q, robust=True)
    assert abs(vr_ours - vr_ref.vr) < 1e-6, (
        f"VR mismatch: ours={vr_ours:.10f} vs arch={vr_ref.vr:.10f}"
    )


def test_vr_m2_zstat_matches_arch_stat() -> None:
    """Cross-validate the M2 HAC-robust z-statistic against arch.VarianceRatio.stat.

    The existing test_vr_m2_matches_arch_reference only checks the VR point
    estimate, discarding z_m2.  This test closes the gap: a sign error, an
    off-by-one in the phi loop, a wrong scale normalization, or a missing nq
    multiplier would pass the VR check but fail here.

    Guards with pytest.importorskip so collection succeeds if arch is absent.
    Two seeds and two horizons exercise the (1-k/q)^2-weighted phi sum at
    different q values.
    """
    arch_unitroot = pytest.importorskip(
        "arch.unitroot", reason="arch not installed — skipping M2 z-stat cross-val"
    )
    from scripts.wp1.predictability import _vr_m2_kernel  # noqa: PLC0415

    # --- Seed 42, q=5, W=240 ---
    rng = np.random.default_rng(42)
    r = rng.normal(0, 0.001, 240)
    vr_ours, z_ours = _vr_m2_kernel(r, 5)
    arch_obj = arch_unitroot.VarianceRatio(np.exp(np.cumsum(r)), lags=5, robust=True)
    assert abs(vr_ours - arch_obj.vr) < 1e-6, (
        f"VR mismatch (seed42,q5): ours={vr_ours:.10f} vs arch={arch_obj.vr:.10f}"
    )
    assert abs(z_ours - arch_obj.stat) < 1e-6, (
        f"M2 z-stat mismatch (seed42,q5): ours={z_ours:.10f} vs arch={arch_obj.stat:.10f}"
    )

    # --- Seed 7, q=15, W=240 (exercises phi sum at wider horizon) ---
    rng2 = np.random.default_rng(7)
    r2 = rng2.normal(0, 0.001, 240)
    vr_ours2, z_ours2 = _vr_m2_kernel(r2, 15)
    arch_obj2 = arch_unitroot.VarianceRatio(np.exp(np.cumsum(r2)), lags=15, robust=True)
    assert abs(vr_ours2 - arch_obj2.vr) < 1e-6, (
        f"VR mismatch (seed7,q15): ours={vr_ours2:.10f} vs arch={arch_obj2.vr:.10f}"
    )
    assert abs(z_ours2 - arch_obj2.stat) < 1e-6, (
        f"M2 z-stat mismatch (seed7,q15): ours={z_ours2:.10f} vs arch={arch_obj2.stat:.10f}"
    )


def test_predictability_no_lookahead() -> None:
    """Spike at the last bar must not change any prior predictability_t value.

    Mirrors tests/vol_regime_switch/test_regime_detector.py::test_lookahead.
    """
    rng = np.random.default_rng(0)
    close_base = np.cumprod(1.0 + rng.normal(0.0, 0.001, 500))
    close_spiked = close_base.copy()
    close_spiked[-1] = close_base[-1] * 10.0  # massive spike at last bar only

    pred_base = compute_rolling_predictability(close_base, W=120, q=5)
    pred_spiked = compute_rolling_predictability(close_spiked, W=120, q=5)

    # All values up to the second-to-last bar must be identical (causal guarantee)
    np.testing.assert_array_equal(pred_base[:-1], pred_spiked[:-1])


def test_predictability_warmup_is_nan() -> None:
    """predictability_t is NaN for warmup bars (indices 0..W-2); finite at W-1.

    Mirrors the warmup guard in regime_detector.py (rolling_sum, lines 20-29).
    """
    rng = np.random.default_rng(1)
    close = np.cumprod(1.0 + rng.normal(0.0, 0.001, 300))
    pred = compute_rolling_predictability(close, W=120, q=5)

    assert len(pred) == len(close)
    assert np.all(np.isnan(pred[:119]))  # indices 0..W-2 are warmup
    assert np.isfinite(pred[119])  # first valid value at index W-1


def test_ljungbox_matches_statsmodels() -> None:
    """Ljung-Box Q-statistic from predictability.py matches statsmodels reference.

    Compares the Q statistic (lb_stat) within 1e-9.
    Uses seeded synthetic returns so the comparison is deterministic.
    Skips cleanly if statsmodels is not installed so collection of the
    dependency-free tests (no-lookahead, warmup-NaN) is unaffected.
    """
    pytest.importorskip("statsmodels", reason="statsmodels not installed — skipping Ljung-Box cross-val")
    from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: PLC0415

    from scripts.wp1.predictability import ljungbox_in_window  # noqa: PLC0415

    rng = np.random.default_rng(99)
    returns = rng.normal(0.0, 0.001, 120)
    lags = 5

    lb_stat_ours, _ = ljungbox_in_window(returns, lags=lags)
    ref = acorr_ljungbox(returns, lags=[lags], return_df=True)
    lb_stat_ref = float(ref["lb_stat"].iloc[-1])

    assert abs(lb_stat_ours - lb_stat_ref) < 1e-9, (
        f"Ljung-Box Q mismatch: ours={lb_stat_ours:.12f} vs statsmodels={lb_stat_ref:.12f}"
    )
