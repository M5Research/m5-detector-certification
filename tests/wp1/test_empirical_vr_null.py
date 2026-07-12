import numpy as np

from scripts.wp1.empirical_vr_null import (
    circular_block_bootstrap_indices,
    empirical_pvalue,
    holm_adjust_pvalues,
    moving_block_bootstrap_indices,
    run_empirical_vr_null,
    stationary_block_bootstrap_indices,
)


def test_circular_block_bootstrap_indices_wrap_and_length():
    rng = np.random.default_rng(1)
    idx = circular_block_bootstrap_indices(10, 4, rng)
    assert len(idx) == 10
    assert np.min(idx) >= 0
    assert np.max(idx) < 10


def test_moving_block_bootstrap_indices_stay_in_bounds():
    rng = np.random.default_rng(2)
    idx = moving_block_bootstrap_indices(10, 4, rng)
    assert len(idx) == 10
    assert np.min(idx) >= 0
    assert np.max(idx) < 10


def test_stationary_block_bootstrap_indices_are_deterministic_for_seed():
    a_rng = np.random.default_rng(3)
    b_rng = np.random.default_rng(3)
    a = stationary_block_bootstrap_indices(20, 5, a_rng)
    b = stationary_block_bootstrap_indices(20, 5, b_rng)
    assert len(a) == 20
    assert np.array_equal(a, b)


def test_empirical_pvalue_plus_one_correction():
    null = np.array([-1.0, 0.0, 1.0])
    assert empirical_pvalue(null, 2.0, alternative="two-sided") == 0.25
    assert empirical_pvalue(null, 0.5, alternative="greater") == 0.5


def test_holm_adjust_pvalues_uses_full_family():
    adjusted = holm_adjust_pvalues([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_empirical_vr_null_report_shape_is_deterministic():
    rng = np.random.default_rng(43)
    returns = rng.standard_t(df=5, size=2_000) * 0.001
    kwargs = {
        "returns": returns,
        "cells": ((60, 2),),
        "n_boot": 3,
        "block": 30,
        "seed": 7,
    }
    a = run_empirical_vr_null(**kwargs)
    b = run_empirical_vr_null(**kwargs)

    assert a["year_2026_loaded"] is False
    assert a["manifest"]["n_boot"] == 3
    assert len(a["results"]) == 1
    assert len(a["results"][0]["cells"]) == 1
    assert a["results"][0]["cells"][0]["W"] == 60
    assert a["results"][0]["cells"][0]["q"] == 2
    assert a == b


def test_empirical_vr_null_full_manifest_and_family_adjustment():
    rng = np.random.default_rng(44)
    returns = rng.standard_t(df=6, size=2_000) * 0.001
    report = run_empirical_vr_null(
        returns,
        cells=((60, 2), (60, 5)),
        n_boot=3,
        blocks=(30,),
        seed=8,
        null_methods=("circular-block", "wild", "student-t", "rank-sign"),
    )

    manifest = report["manifest"]
    assert manifest["cells"] == [{"W": 60, "q": 2}, {"W": 60, "q": 5}]
    assert manifest["null_methods"] == ["circular-block", "wild", "student-t", "rank-sign"]
    assert manifest["block_lengths"] == [30]
    assert len(report["results"]) == 4
    for null_result in report["results"]:
        assert len(null_result["cells"]) == 2
        for row in null_result["cells"]:
            assert "empirical_two_sided_p_vs_null" in row
            assert "holm_adjusted_two_sided_p" in row
            assert "empirical_size_alpha_0.05_asymptotic_two_sided" in row["null_summary"]
