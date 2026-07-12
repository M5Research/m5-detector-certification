"""Integration tests for the full injection cascade pipeline."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from scripts.wp1.signal_injector import (
    inject_ar1_autocorrelation,
    invert_phi_delta_mapping,
    hash_combine,
    fit_garch_sigma_t,
    clopper_pearson_ci,
)
from scripts.wp1.signal_injection import process_grid_point
from scripts.wp1.precheck_b import run_precheck_b_cascade

@pytest.fixture(scope="session")
def synthetic_data():
    """Generate synthetic BTC-like returns for testing."""
    rng = np.random.default_rng(43)
    N = 50000
    returns = rng.normal(0, 0.001, N)
    sigma_t = fit_garch_sigma_t(returns)
    timestamps = np.arange(
        1609459200000,
        1609459200000 + N * 60000,
        60000,
        dtype=np.int64,
    )
    return {
        "returns": returns,
        "sigma_t": sigma_t,
        "timestamps": timestamps,
    }

@pytest.mark.slow
def test_no_false_fires_at_delta_zero(synthetic_data, tmp_path):
    """At delta=0 (phi=0), cascade should fire at approximately Holm-adjusted alpha."""
    r = synthetic_data
    delta = 0.0
    q = 5
    W = 120
    grid_hash = hash((delta, q, W))
    N_mc_test = 200

    phi = invert_phi_delta_mapping(delta, q)

    n_fires = 0
    for mc_idx in range(N_mc_test):
        mc_seed = hash_combine(43, mc_idx, grid_hash)
        r_inj = inject_ar1_autocorrelation(r["returns"], phi, mc_seed, r["sigma_t"])
        close_inj = np.exp(np.cumsum(r_inj))
        result = run_precheck_b_cascade(close_inj, r["timestamps"], mc_seed)
        if result["cascade_fired"]:
            n_fires += 1

    observed_rate = n_fires / N_mc_test
    ci_lo, ci_hi = clopper_pearson_ci(n_fires, N_mc_test, alpha=0.05)
    print(f"delta=0: P_det={observed_rate:.4f} ({n_fires}/{N_mc_test}) CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    
    assert observed_rate < 0.15, (
        f"P_det at delta=0 is {observed_rate:.4f} ({n_fires}/{N_mc_test}) -- "
        f"substantially above the expected 0.05 false-positive rate"
    )

@pytest.mark.slow
def test_pdet_monotonic(synthetic_data, tmp_path):
    """P_det should be monotonic in delta for a small grid."""
    r = synthetic_data
    deltas = [0.001, 0.01, 0.10]
    q = 5
    W = 120
    N_mc_test = 50

    p_dets = []
    for delta in deltas:
        gp = {"delta": delta, "q": q, "W": W, "grid_hash": hash((delta, q, W))}
        result = process_grid_point(
            gp,
            r_real=r["returns"],
            sigma_t=r["sigma_t"],
            timestamps=r["timestamps"],
            N_mc=N_mc_test,
            injection_dir=tmp_path,
        )
        p_dets.append(result.get("P_det", 0))

    diffs = np.diff(p_dets)
    assert np.all(diffs >= 0), f"P_det not monotonic in delta: deltas={deltas}, P_dets={p_dets}"
    print(f"P_det monotonic: deltas={deltas}, P_dets={[f'{p:.4f}' for p in p_dets]}")

def test_hash_seed_diversity():
    """Seeds for all 96x200 draws must be unique."""
    seeds = set()
    for delta in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
        for q in [2, 5, 15, 60]:
            for W in [60, 120, 240]:
                grid_hash = hash((delta, q, W))
                for mc in range(200):
                    seed = hash_combine(43, mc, grid_hash)
                    seeds.add(seed)

    expected = 8 * 4 * 3 * 200
    assert len(seeds) == expected, f"Expected {expected} unique seeds, got {len(seeds)}"

def test_injection_cascade_end_to_end(synthetic_data):
    """Full pipeline: GARCH fit -> inject -> cascade -> result."""
    r = synthetic_data

    phi = 0.5
    r_inj = inject_ar1_autocorrelation(r["returns"], phi, 42, r["sigma_t"])
    close_inj = np.exp(np.cumsum(r_inj))

    result = run_precheck_b_cascade(close_inj, r["timestamps"], 42)

    assert "cascade_fired" in result
    assert "per_wq" in result
    assert "holm_correction" in result
    assert len(result["per_wq"]) == 3

    print(f"e2e pipeline OK: cascade_fired={result['cascade_fired']}")
