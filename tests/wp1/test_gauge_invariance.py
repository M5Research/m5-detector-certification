import numpy as np
import pytest

from scripts.wp1 import vr_significance
import scripts.wp1.gauge_invariance as gauge_invariance
from scripts.wp1.gauge_invariance import (
    PRACTICAL_TOST_EPSILON,
    VERDICT_INVARIANT,
    VERDICT_VIOLATION,
    all_bars_non_overlapping,
    calibrate_wq_calendar,
    compute_tost_epsilon,
)


def _mock_pipeline(medians: dict[int, float], n_nl: int = 1000) -> dict:
    per_wq_primary_q = [
        {
            "q": q,
            "median_vr_dep": med,
            "n_nl": n_nl,
            "ci_95_lo": med - 0.001,
            "ci_95_hi": med + 0.001,
        }
        for q, med in medians.items()
    ]
    return {"per_wq_primary_q": per_wq_primary_q, "per_wq": []}


def _mock_pipeline_with_ci(
    medians: dict[int, float],
    ci_half_width: float,
    n_nl: int = 1000,
) -> dict:
    per_wq_primary_q = [
        {
            "q": q,
            "median_vr_dep": med,
            "n_nl": n_nl,
            "ci_95_lo": med - ci_half_width,
            "ci_95_hi": med + ci_half_width,
        }
        for q, med in medians.items()
    ]
    return {"per_wq_primary_q": per_wq_primary_q, "per_wq": []}


def test_tost_equivalent_medians():
    gauges = {
        g: _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001})
        for g in ("clock", "volume", "intrinsic")
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.01)
    assert VERDICT_INVARIANT in result["verdict"]


def test_tost_violation_large_spread():
    gauges = {
        "clock": _mock_pipeline({2: 0.05, 5: 0.05, 15: 0.05, 60: 0.05}),
        "volume": _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001}),
        "intrinsic": _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001}),
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.001)
    assert VERDICT_VIOLATION in result["verdict"]


def test_tost_holm_adjustment():
    gauges = {
        g: _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001})
        for g in ("clock", "volume", "intrinsic")
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.01)
    for row in result["per_q_comparisons"]:
        assert row["holm_adjusted"] >= row["p_equiv"] - 1e-12


def test_tost_holm_uses_actual_pairwise_family_size():
    gauges = {
        "clock": _mock_pipeline({2: 0.05, 5: 0.05, 15: 0.05, 60: 0.05}),
        "volume": _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001}),
        "intrinsic": _mock_pipeline({2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001}),
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.001)
    for row in result["per_q_comparisons"]:
        assert row["holm_adjusted"] >= row["p_equiv"] - 1e-12


def test_tost_requires_ci_inside_equivalence_margin():
    gauges = {
        g: _mock_pipeline_with_ci(
            {2: 0.001, 5: 0.001, 15: 0.001, 60: 0.001},
            ci_half_width=0.10,
            n_nl=100_000,
        )
        for g in ("clock", "volume", "intrinsic")
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.01)
    assert result["verdict"] == VERDICT_VIOLATION
    assert any(not row["ci_within_margin"] for row in result["per_q_comparisons"])


def test_compute_tost_epsilon_uses_practical_margin_not_mde():
    n_nl = 500
    clock = {
        "per_wq": [{"W": 120, "q": 5, "n_nl": n_nl}],
        "per_wq_primary_q": [],
    }
    eps = compute_tost_epsilon(clock)
    mde = vr_significance.compute_mde_vr(n_nl)["mde_vr_departure"]
    assert eps == PRACTICAL_TOST_EPSILON
    assert eps != mde


def test_verdict_strings_frozen():
    gauges = {
        g: _mock_pipeline({5: 0.001})
        for g in ("clock", "volume", "intrinsic")
    }
    result = gauge_invariance.test_gauge_invariance(gauges, epsilon=0.01)
    assert result["verdict"] in (VERDICT_INVARIANT, VERDICT_VIOLATION)


def test_all_bars_non_overlapping_no_regime():
    series = np.arange(600, dtype=np.float64)
    pred, idx = all_bars_non_overlapping(series, stride=120)
    assert len(pred) > 0
    assert len(idx) == len(pred)


def test_calibrate_wq_calendar_floors():
    rows = calibrate_wq_calendar([(120, 5)], median_duration_min=12.0)
    assert rows[0]["q_adj"] >= 2
    assert rows[0]["W_adj"] >= rows[0]["q_adj"] + 10
