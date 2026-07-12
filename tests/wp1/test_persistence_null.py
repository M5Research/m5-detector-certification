import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from scripts.wp1.persistence_null import (
    MAX_HORIZON,
    PERSIST_DIR,
    VERDICT_CONSISTENT,
    VERDICT_DEVIATION,
    arcsin_survival,
    persistence_probability,
)

pytest.importorskip("scripts.wp1.persistence_figures")
from scripts.wp1.persistence_figures import generate_survival_figure

N_BOOT_TEST = 25
N_CLOSE_RW = 8_000
N_CLOSE_AR1 = 10_000


def _random_walk_close(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))


def _ar1_close(n: int, phi: float = 0.4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    innovations = rng.normal(0, 0.001, n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + np.sqrt(1 - phi**2) * innovations[t]
    return 100.0 * np.exp(np.cumsum(r))


def test_persistence_probability_rw():
    close = _random_walk_close(N_CLOSE_RW)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    required_keys = [
        "S_empirical",
        "theta",
        "theta_ci",
        "ks_statistic",
        "bootstrap_p_value",
        "n_windows",
        "verdict",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    assert result["verdict"] in (VERDICT_CONSISTENT, VERDICT_DEVIATION)
    assert len(result["S_empirical"]) == MAX_HORIZON
    assert result["n_windows"] > 0


def test_arcsin_survival():
    t_grid = np.array([1.0, 2.0, 4.0, 120.0])
    s0 = arcsin_survival(t_grid)
    assert abs(s0[0] - 1.0) < 1e-9
    assert abs(s0[1] - 0.5) < 1e-9
    assert s0[2] < s0[1]
    assert s0[3] > 0.0
    assert all(np.isfinite(s0))


def test_theta_rw():
    close = _random_walk_close(N_CLOSE_RW)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    assert abs(result["theta"] - 0.5) < 0.15, f"theta={result['theta']:.3f} too far from 0.5"


def test_ks_fails_rw():
    close = _random_walk_close(N_CLOSE_RW)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    assert result["bootstrap_p_value"] > 0.05, (
        f"KS rejected for RW: p={result['bootstrap_p_value']:.4f}"
    )


def test_ks_rejects_ar1():
    close = _ar1_close(N_CLOSE_AR1)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    assert result["bootstrap_p_value"] < 0.05, (
        f"KS failed to reject AR(1): p={result['bootstrap_p_value']:.4f}"
    )


def test_theta_ar1():
    """PERSIST-04: theta != 0.5 (descriptive) for AR(1) phi=0.4."""
    close = _ar1_close(N_CLOSE_AR1)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    assert abs(result["theta"] - 0.5) > 0.1, (
        f"theta={result['theta']:.3f} too close to 0.5 for AR(1) phi=0.4"
    )


def test_figure_generation(tmp_path):
    close = _random_walk_close(N_CLOSE_RW)
    result = persistence_probability(close, n_bootstrap=N_BOOT_TEST)
    result["run_timestamp"] = "test_run"
    out = generate_survival_figure(result, output_dir=tmp_path)
    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 1000
