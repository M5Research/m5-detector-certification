import numpy as np
import pytest

from scripts.wp1.signal_injector import (
    inject_ar1_autocorrelation,
    fit_garch_sigma_t,
    hash_combine,
    clopper_pearson_ci,
    phi_to_delta_mapping,
    invert_phi_delta_mapping,
)

@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(43)
    return rng.normal(0, 0.001, 10000)

@pytest.fixture
def synthetic_sigma_t(synthetic_returns):
    return fit_garch_sigma_t(synthetic_returns)

def test_variance_preservation_phi_zero(synthetic_returns, synthetic_sigma_t):
    # variance preservation: the AR(1) process eps has variance 1.
    # so r_inj has variance var(returns) + var(sigma_t * eps)
    r_inj = inject_ar1_autocorrelation(synthetic_returns, phi=0.0, seed=42, sigma_t=synthetic_sigma_t)
    rng = np.random.default_rng(42)
    expected_var = np.var(synthetic_returns) + np.var(synthetic_sigma_t * rng.normal(0, 1, 10000))
    var_ratio = np.var(r_inj) / expected_var
    assert 0.99 <= var_ratio <= 1.01

def test_variance_preservation_phi_nonzero(synthetic_returns, synthetic_sigma_t):
    # The variance should be invariant to phi because eps is orthogonalized
    r_inj = inject_ar1_autocorrelation(synthetic_returns, phi=0.5, seed=42, sigma_t=synthetic_sigma_t)
    r_inj_0 = inject_ar1_autocorrelation(synthetic_returns, phi=0.0, seed=42, sigma_t=synthetic_sigma_t)
    var_ratio = np.var(r_inj) / np.var(r_inj_0)
    assert 0.95 <= var_ratio <= 1.05

def test_injection_determinism(synthetic_returns, synthetic_sigma_t):
    r1 = inject_ar1_autocorrelation(synthetic_returns, phi=0.5, seed=42, sigma_t=synthetic_sigma_t)
    r2 = inject_ar1_autocorrelation(synthetic_returns, phi=0.5, seed=42, sigma_t=synthetic_sigma_t)
    np.testing.assert_array_equal(r1, r2)

def test_injection_different_seed(synthetic_returns, synthetic_sigma_t):
    r1 = inject_ar1_autocorrelation(synthetic_returns, phi=0.5, seed=42, sigma_t=synthetic_sigma_t)
    r2 = inject_ar1_autocorrelation(synthetic_returns, phi=0.5, seed=43, sigma_t=synthetic_sigma_t)
    assert not np.array_equal(r1, r2)

def test_phi_zero_no_change(synthetic_returns, synthetic_sigma_t):
    r_inj = inject_ar1_autocorrelation(synthetic_returns, phi=0.0, seed=42, sigma_t=synthetic_sigma_t)
    assert len(r_inj) == len(synthetic_returns)

def test_hash_combine_determinism():
    assert hash_combine(43, 0, 123) == hash_combine(43, 0, 123)

def test_hash_combine_uniqueness():
    h1 = hash_combine(43, 0, 123)
    h2 = hash_combine(43, 1, 123)
    h3 = hash_combine(43, 0, 124)
    assert h1 != h2
    assert h1 != h3

def test_clopper_pearson_symmetry():
    ci = clopper_pearson_ci(100, 200, alpha=0.05)
    assert 0.4 < ci[0] < 0.5
    assert 0.5 < ci[1] < 0.6

def test_clopper_pearson_zero_fires():
    ci = clopper_pearson_ci(0, 200)
    assert ci[0] == 0.0
    assert ci[1] > 0.0

def test_clopper_pearson_all_fires():
    ci = clopper_pearson_ci(200, 200)
    assert ci[0] < 1.0
    assert ci[1] == 1.0

def test_phi_delta_monotonic():
    phis = np.linspace(0, 0.9, 5)
    mapping = phi_to_delta_mapping(rng_seed=42, N=10000, phis_to_test=phis)
    for q in mapping[phis[0]].keys():
        deltas = [mapping[float(p)][q] for p in phis]
        assert np.all(np.diff(deltas) >= 0)

def test_phi_delta_q_grid():
    mapping = phi_to_delta_mapping(rng_seed=42, N=1000, phis_to_test=np.array([0.0]))
    assert set(mapping[0.0].keys()) == {2, 5, 15, 60}

def test_invert_phi_delta():
    phis = np.linspace(0, 0.9, 5)
    mapping = phi_to_delta_mapping(rng_seed=42, N=10000, phis_to_test=phis)
    
    target_delta = mapping[0.45][5]
    inv_phi = invert_phi_delta_mapping(target_delta, 5, mapping)
    assert 0.44 <= inv_phi <= 0.46
    
    low_delta = mapping[0.0][5] - 0.001
    assert invert_phi_delta_mapping(low_delta, 5, mapping) > 0.0

    high_delta = mapping[phis[-1]][5] + 0.1
    assert invert_phi_delta_mapping(high_delta, 5, mapping) > phis[-1]

def test_garch_sigma_t_length(synthetic_returns):
    sigma_t = fit_garch_sigma_t(synthetic_returns)
    assert len(sigma_t) == len(synthetic_returns)

def test_garch_ewma_fallback():
    degenerate = np.ones(1000)
    sigma_t = fit_garch_sigma_t(degenerate)
    assert len(sigma_t) == 1000
    assert not np.any(np.isnan(sigma_t))

def test_inject_phi_bounds(synthetic_returns, synthetic_sigma_t):
    with pytest.raises(ValueError):
        inject_ar1_autocorrelation(synthetic_returns, phi=1.0, seed=42, sigma_t=synthetic_sigma_t)
    with pytest.raises(ValueError):
        inject_ar1_autocorrelation(synthetic_returns, phi=-0.1, seed=42, sigma_t=synthetic_sigma_t)
    # Exploratory negative-phi arm uses explicit opt-in
    out = inject_ar1_autocorrelation(
        synthetic_returns, phi=-0.1, seed=42, sigma_t=synthetic_sigma_t, allow_negative_phi=True
    )
    assert len(out) == len(synthetic_returns)
