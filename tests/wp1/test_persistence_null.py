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


# ---------------------------------------------------------------------------
# Calibration of the null generator
#
# These are the tests that would have caught the original defect: the
# reference distribution was built by circular block-resampling the OBSERVED
# returns, which preserves the observed dependence instead of imposing the
# martingale-difference hypothesis. A p-value computed against that reference
# is not calibrated, in either direction.
# ---------------------------------------------------------------------------

N_BOOT_CALIB = 200


def test_sign_flip_null_is_calibrated_under_a_random_walk() -> None:
    """Under H0 the p-value must vary across samples, not sit at a fixed point.

    A valid p-value is approximately uniform under the null, so across
    independent random walks it should have a mean near 0.5 AND real spread.
    The spread assertion is the one with teeth: a reference distribution that
    tracks the observed statistic produces p-values piled up near a single
    value, which passes a mean check and fails this.
    """
    ps = [
        persistence_probability(
            _random_walk_close(4_000, seed=s),
            n_bootstrap=N_BOOT_CALIB,
            seed=s,
        )["bootstrap_p_value"]
        for s in range(12)
    ]
    assert 0.15 < float(np.mean(ps)) < 0.85, f"mean p = {np.mean(ps):.3f}"
    assert float(np.std(ps)) > 0.1, f"p-values barely vary: std = {np.std(ps):.3f}"


def test_sign_flip_null_has_power_against_a_persistent_alternative() -> None:
    """A strongly autocorrelated series must be rejected."""
    result = persistence_probability(
        _ar1_close(N_CLOSE_AR1, phi=0.4, seed=1),
        n_bootstrap=N_BOOT_CALIB,
        seed=7,
    )
    assert result["bootstrap_p_value"] < 0.05
    assert result["verdict"] == VERDICT_DEVIATION


def test_monte_carlo_p_value_is_never_exactly_zero() -> None:
    """The (1+k)/(1+B) convention: the observed statistic is itself a draw."""
    result = persistence_probability(
        _ar1_close(N_CLOSE_AR1, phi=0.6, seed=3),
        n_bootstrap=50,
        seed=11,
    )
    assert result["bootstrap_p_value"] > 0.0
    assert result["bootstrap_p_value"] >= 1.0 / 51.0


def test_null_generator_is_recorded_and_validated() -> None:
    result = persistence_probability(
        _random_walk_close(2_000, seed=5), n_bootstrap=20, seed=5
    )
    assert result["null_generator"] == "sign_flip"

    with pytest.raises(ValueError, match="unknown null generator"):
        persistence_probability(
            _random_walk_close(2_000, seed=5),
            n_bootstrap=20,
            seed=5,
            null="not_a_null",
        )


def test_legacy_block_observed_null_is_still_reachable() -> None:
    """The published artifact's behaviour must stay reproducible."""
    result = persistence_probability(
        _random_walk_close(4_000, seed=2),
        n_bootstrap=40,
        seed=2,
        null="block_observed",
    )
    assert result["null_generator"] == "block_observed"


def test_legacy_null_has_no_power_and_sign_flip_does() -> None:
    """The defect, demonstrated side by side on the same data.

    Block-resampling the observed returns preserves the very dependence the
    arcsine law is a hypothesis about, so KS_boot inherits it and tracks
    KS_obs. Against a strongly autocorrelated series the legacy reference
    therefore returns p near 0.5 and never rejects, while the sign-flip null
    rejects consistently.

    This is the discriminating case. Under a plain random walk the two
    generators agree, because an iid series already satisfies H0 and
    resampling it is resampling under the null; the difference only appears
    once the observed process actually carries dependence.
    """
    closes = [_ar1_close(N_CLOSE_AR1, phi=0.4, seed=s) for s in range(4)]

    p_sign = [
        persistence_probability(c, n_bootstrap=N_BOOT_CALIB, seed=i, null="sign_flip")[
            "bootstrap_p_value"
        ]
        for i, c in enumerate(closes)
    ]
    p_block = [
        persistence_probability(
            c, n_bootstrap=N_BOOT_CALIB, seed=i, null="block_observed"
        )["bootstrap_p_value"]
        for i, c in enumerate(closes)
    ]

    assert float(np.mean(p_sign)) < 0.10, (
        f"sign-flip null should reject a phi=0.4 series; mean p = {np.mean(p_sign):.3f}"
    )
    assert float(np.mean(p_block)) > 0.25, (
        "legacy block_observed null unexpectedly gained power; the published "
        f"artifact's caveat may need revisiting. mean p = {np.mean(p_block):.3f}"
    )
