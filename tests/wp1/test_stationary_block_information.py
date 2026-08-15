"""Stationary-block mutual information: determinism, type I, power, no permutation_p."""
from __future__ import annotations

import math

import numpy as np
from certificate.information import (
    miller_madow_mutual_information,
    plugin_mutual_information,
    primary_block_length,
    seed_from_spec_hash,
    stationary_block_information,
)

SPEC_HASH = "ab" * 32


def test_primary_block_length_is_cube_root() -> None:
    assert primary_block_length(1) == 1
    assert primary_block_length(8) == 2
    assert primary_block_length(1000) == 10
    assert primary_block_length(1001) == 11


def test_plugin_and_miller_madow_agree_on_independent_uniform() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, size=4000)
    y = rng.integers(0, 3, size=4000)
    plugin = plugin_mutual_information(x, y)
    mm = miller_madow_mutual_information(x, y)
    assert plugin >= 0.0
    assert mm < 0.02
    assert math.isfinite(mm)


def test_coupled_series_have_positive_information() -> None:
    x = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
    y = x.copy()
    assert plugin_mutual_information(x, y) > 0.3


def test_stationary_block_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    latent = rng.normal(size=400)
    x = (latent > 0).astype(np.int64)
    y = ((0.8 * latent + 0.2 * rng.normal(size=400)) > 0).astype(np.int64)
    a = stationary_block_information(x, y, n_boot=199, spec_hash=SPEC_HASH)
    b = stationary_block_information(x, y, n_boot=199, spec_hash=SPEC_HASH)
    assert a == b
    assert "permutation_p" not in a
    assert "p_value" in a
    assert "plugin_mi" in a
    assert "miller_madow_mi" in a
    assert "ci" in a


def test_type_i_under_independent_autocorrelated_series() -> None:
    rng = np.random.default_rng(7)
    n = 800

    def ar1(rho: float) -> np.ndarray:
        eps = rng.normal(size=n)
        out = np.empty(n)
        out[0] = eps[0]
        for i in range(1, n):
            out[i] = rho * out[i - 1] + eps[i]
        return out

    x = (ar1(0.6) > 0).astype(np.int64)
    y = (ar1(0.6) > 0).astype(np.int64)
    result = stationary_block_information(x, y, n_boot=399, spec_hash=SPEC_HASH)
    assert result["p_value"] > 0.05
    assert result["gate_state"] in {"fail", "pass"}


def test_power_under_coupled_autocorrelated_series() -> None:
    rng = np.random.default_rng(11)
    n = 800
    eps = rng.normal(size=n)
    latent = np.empty(n)
    latent[0] = eps[0]
    for i in range(1, n):
        latent[i] = 0.7 * latent[i - 1] + eps[i]
    x = (latent > 0).astype(np.int64)
    y = ((latent + 0.05 * rng.normal(size=n)) > 0).astype(np.int64)
    result = stationary_block_information(x, y, n_boot=399, spec_hash=SPEC_HASH, alpha=0.05)
    assert result["p_value"] <= 0.05
    assert result["gate_state"] == "pass"


def test_reversed_block_length_sensitivity_is_invalid() -> None:
    x = np.array([0, 1, 0, 1] * 30, dtype=np.int64)
    y = np.array([0, 0, 1, 1] * 30, dtype=np.int64)
    result = stationary_block_information(
        x,
        y,
        n_boot=49,
        spec_hash=SPEC_HASH,
        alpha=0.05,
        force_unstable=True,
    )
    assert result["gate_state"] == "invalid"
    assert result["reason"] == "unstable_reference"


def test_seed_is_domain_separated_from_spec_hash() -> None:
    first = seed_from_spec_hash(SPEC_HASH, "information:primary")
    repeat = seed_from_spec_hash(SPEC_HASH, "information:primary")
    other = seed_from_spec_hash(SPEC_HASH, "information:null")

    assert first == repeat
    assert first != other


def test_revision_alpha_controls_confidence_interval_width() -> None:
    rng = np.random.default_rng(17)
    x = rng.integers(0, 3, size=300)
    y = (x + rng.integers(0, 2, size=300)) % 3
    loose = stationary_block_information(
        x,
        y,
        n_boot=199,
        spec_hash=SPEC_HASH,
        alpha=0.05,
    )
    strict = stationary_block_information(
        x,
        y,
        n_boot=199,
        spec_hash=SPEC_HASH,
        alpha=0.025,
    )

    loose_width = loose["ci"][1] - loose["ci"][0]
    strict_width = strict["ci"][1] - strict["ci"][0]
    assert strict_width >= loose_width
    assert strict["confidence_level"] == 0.975
