import numpy as np
import pytest

# Faster bootstrap in unit tests; production default remains 500 (pre-reg §5.3).
N_BOOT_TEST = 25

from scripts.wp1.mutual_information import (
    build_forward_sign_pairs,
    build_lag_pairs,
    build_sign_pairs,
    estimate_mi_from_returns,
)


def _simulate_garch11(
    n: int,
    alpha: float = 0.05,
    beta: float = 0.90,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GARCH(1,1) with unit unconditional variance and zero mean."""
    if rng is None:
        rng = np.random.default_rng(43)
    omega = 1.0 - alpha - beta
    r = np.empty(n, dtype=np.float64)
    sigma2 = 1.0
    for t in range(n):
        z = rng.standard_normal()
        r[t] = np.sqrt(sigma2) * z
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    return r


def _garch_returns(n: int, alpha: float, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Prefer arch-backed simulation when available; otherwise hand-rolled GARCH(1,1)."""
    try:
        from arch.univariate import ConstantMean, GARCH, Normal

        pytest.importorskip("arch")
        omega = 1.0 - alpha - beta
        model = ConstantMean(np.zeros(n))
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = Normal()
        params = np.array([0.0, omega, alpha, beta])
        sim = model.simulate(params, nobs=n, initial_value={"volatility": 1.0})
        return np.asarray(sim["data"], dtype=np.float64)
    except Exception:
        return _simulate_garch11(n, alpha=alpha, beta=beta, rng=rng)


def test_ksg_returns_tuple():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(3_000)
    result = estimate_mi_from_returns(r, q=5, method="signs", n_boot=N_BOOT_TEST)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mi, stderr = result
    assert np.isfinite(mi)
    assert np.isfinite(stderr)


def test_independent_near_zero():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(3_000)
    mi, stderr = estimate_mi_from_returns(r, q=5, method="signs", n_boot=N_BOOT_TEST)
    assert abs(mi) < 0.01 or abs(mi) <= 2 * stderr


def test_ar1_gaussian_mi():
    rng = np.random.default_rng(43)
    phi = 0.3
    n = 3_000
    eps = rng.standard_normal(n)
    r = np.empty(n, dtype=np.float64)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]

    mi, _ = estimate_mi_from_returns(r, q=1, method="raw", n_boot=N_BOOT_TEST)
    analytic = -0.5 * np.log(1.0 - phi**2)
    assert abs(mi - analytic) < 0.05


def test_sign_garch_near_zero():
    rng = np.random.default_rng(43)
    r = _garch_returns(3_000, alpha=0.05, beta=0.90, rng=rng)
    mi, stderr = estimate_mi_from_returns(r, q=5, method="signs", n_boot=N_BOOT_TEST)
    assert abs(mi) <= max(2 * stderr, 0.01)


def test_raw_garch_upper_bound():
    rng = np.random.default_rng(43)
    # Stronger volatility clustering so KSG on raw lag-5 pairs exceeds the zero clamp.
    r = _simulate_garch11(8_000, alpha=0.15, beta=0.80, rng=rng)
    mi, _ = estimate_mi_from_returns(r, q=5, method="raw", n_boot=N_BOOT_TEST)
    assert mi > 0.0


def test_build_lag_pairs_length():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(100)
    x, y = build_lag_pairs(r, q=5)
    assert len(x) == len(r) - 5
    assert len(y) == len(r) - 5


def test_build_sign_pairs_values():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(100)
    x, y = build_sign_pairs(r, q=5)
    assert set(np.unique(x)).issubset({-1.0, 0.0, 1.0})
    assert set(np.unique(y)).issubset({-1.0, 0.0, 1.0})


def test_build_forward_sign_pairs_signal_precedes_target():
    r = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    signal, target = build_forward_sign_pairs(r, q=2)
    np.testing.assert_array_equal(signal, np.array([1.0, -1.0, 1.0]))
    np.testing.assert_array_equal(target, np.array([1.0, -1.0, 1.0]))


def test_forward_orientation_matches_serial_mi_value_for_signs():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(3_000)
    mi_serial, se_serial = estimate_mi_from_returns(
        r, q=5, method="signs", n_boot=N_BOOT_TEST, orientation="serial"
    )
    mi_forward, se_forward = estimate_mi_from_returns(
        r, q=5, method="signs", n_boot=N_BOOT_TEST, orientation="forward"
    )
    assert mi_forward == mi_serial
    assert se_forward == se_serial


def test_ksg_deterministic():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(5_000)
    mi1, se1 = estimate_mi_from_returns(r, q=5, method="signs", n_boot=N_BOOT_TEST, seed=43)
    mi2, se2 = estimate_mi_from_returns(r, q=5, method="signs", n_boot=N_BOOT_TEST, seed=43)
    assert mi1 == mi2
    assert se1 == se2
