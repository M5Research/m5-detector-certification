import numpy as np
from scripts.wp1.stats import (
    deflated_sharpe_ratio,
    sharpe_ratio,
    spearman_ic,
)


def test_sharpe_zero_for_constant_equity():
    eq = np.full(300, 10000.0)
    ts = 1704067200000 + np.arange(300) * 86400000  # daily
    assert sharpe_ratio(eq, ts) == 0.0


def test_sharpe_positive_for_upward_drift():
    rng = np.random.default_rng(1)
    daily = 0.001 + rng.normal(0, 0.0005, 365)
    eq = 10000 * np.cumprod(1 + daily)
    ts = 1704067200000 + np.arange(365) * 86400000
    assert sharpe_ratio(eq, ts) > 1.0


def test_spearman_ic_detects_known_signal():
    rng = np.random.default_rng(2)
    sig = rng.normal(0, 1, 5000)
    fwd = 0.7 * sig + rng.normal(0, 1, 5000)  # strong positive rank relation
    ic, n = spearman_ic(sig, fwd)
    assert ic > 0.4 and n == 5000


def test_deflated_sharpe_shrinks_with_trials():
    # Same observed Sharpe, more trials -> lower deflated probability.
    p_few = deflated_sharpe_ratio(observed_sr=0.5, n_obs=252, n_trials=1, sr_variance=0.25)
    p_many = deflated_sharpe_ratio(observed_sr=0.5, n_obs=252, n_trials=200, sr_variance=0.25)
    assert 0.0 <= p_many < p_few <= 1.0
