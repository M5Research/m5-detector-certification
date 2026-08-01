"""Pin the duplicated implementations to each other.

Several pieces of load-bearing logic exist in more than one copy across this
package: Holm correction (four), the pre-registration gate guard (three), the
non-overlapping sampler (five), and `_median_se_for_q` (two).

The duplication is not hypothetical debt. It has already produced a real bug:
the two copies of `_median_se_for_q` diverged, one was fixed and the other was
not, and the unfixed copy in `gauge_invariance` carried an unreachable return
that made it fall off the end and return `None` for `n_nl == 0` — in the
function that computes the transport margin.

Consolidating them into one module would be the textbook fix. It is
deliberately NOT what this file does. These functions sit inside frozen
analysis paths whose outputs are published artifacts, and a refactor that
changes any of them by a floating-point ulp is a far worse outcome for a
certification package than the duplication itself. What actually caused harm
was *divergence*, not duplication, so these tests fail the moment two copies
stop agreeing. That is the property worth enforcing.

If the implementations are ever genuinely unified, these tests become
tautologies and can go.
"""
from __future__ import annotations

import numpy as np
import pytest

# Deliberately NOT marked `artifact` at module level: these compare code paths
# against each other. The one test that reads a committed artifact carries the
# marker itself.


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------


HOLM_FAMILIES = [
    [0.01, 0.02, 0.03, 0.04],
    [0.001, 0.5, 0.9, 1.0],
    [0.05, 0.05, 0.05, 0.05],
    [0.0, 0.0, 1.0, 1.0],
    [1e-12, 0.2, 0.7, 0.99],
]


@pytest.mark.parametrize("pvalues", HOLM_FAMILIES)
def test_holm_implementations_agree_on_a_family_of_four(pvalues: list[float]) -> None:
    """gauge_invariance._holm_adjust, empirical_vr_null.holm_adjust_pvalues and
    vr_significance.apply_holm_b must return identical adjusted families.

    All three correct a family of four at the frozen Pre-Check B denominator,
    so they are interchangeable by construction and must stay that way.
    """
    from scripts.wp1.empirical_vr_null import holm_adjust_pvalues
    from scripts.wp1.gauge_invariance import _holm_adjust
    from scripts.wp1.vr_significance import apply_holm_b

    a = _holm_adjust(list(pvalues))
    b = holm_adjust_pvalues(list(pvalues))
    c = apply_holm_b(list(pvalues))

    np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg="gauge vs empirical_vr_null")
    np.testing.assert_allclose(a, c, rtol=0, atol=0, err_msg="gauge vs vr_significance")


def test_holm_is_monotone_and_bounded() -> None:
    """Properties every copy must satisfy, checked on random families."""
    from scripts.wp1.gauge_invariance import _holm_adjust

    rng = np.random.default_rng(43)
    for _ in range(200):
        pvalues = list(rng.uniform(0.0, 1.0, size=4))
        adjusted = _holm_adjust(pvalues)
        assert all(0.0 <= p <= 1.0 for p in adjusted)
        # Adjusted p-values are never smaller than the raw ones.
        for raw, adj in zip(pvalues, adjusted, strict=True):
            assert adj >= raw - 1e-12
        # Step-down monotonicity in sorted order.
        order = np.argsort(pvalues)
        sorted_adj = [adjusted[i] for i in order]
        assert all(
            sorted_adj[i] <= sorted_adj[i + 1] + 1e-12
            for i in range(len(sorted_adj) - 1)
        )


@pytest.mark.artifact
def test_gauge_holm_matches_the_published_transport_family() -> None:
    """The 12-row transport family reproduces from its own raw p-values."""
    import json

    from backtest.utils import PROJECT_ROOT
    from scripts.wp1.gauge_invariance import _holm_adjust

    path = (
        PROJECT_ROOT
        / "backtest_results"
        / "gauge_invariance"
        / "gauge_report_mde_margin.json"
    )
    tost = json.loads(path.read_text(encoding="utf-8"))["tost"]
    raw = [c["p_tost_raw"] for c in tost["per_q_comparisons"]]
    np.testing.assert_allclose(_holm_adjust(raw), tost["holm_adjusted"], rtol=1e-12)


# ---------------------------------------------------------------------------
# Non-overlapping samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2, 5, 17, 120])
@pytest.mark.parametrize("n", [1, 7, 100, 1001])
def test_non_overlapping_samplers_agree(n: int, stride: int) -> None:
    """gauge_invariance and holdout_confirmatory sample the same indices.

    Both take every `stride`-th element. The holdout driver keeps its own copy;
    if the two ever disagree, the holdout and the gauge would be computed over
    different effective sample sizes, which silently changes every standard
    error downstream.
    """
    from scripts.wp1.gauge_invariance import all_bars_non_overlapping
    from scripts.wp1.holdout_confirmatory import _all_bars_non_overlapping

    series = np.arange(n, dtype=np.float64)
    values_a, idx_a = all_bars_non_overlapping(series, stride=stride)
    values_b, idx_b = _all_bars_non_overlapping(series, stride=stride)

    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_array_equal(values_a, values_b)


@pytest.mark.parametrize("stride", [2, 5, 120])
def test_non_overlapping_sampler_really_is_non_overlapping(stride: int) -> None:
    from scripts.wp1.gauge_invariance import all_bars_non_overlapping

    _, idx = all_bars_non_overlapping(np.arange(1000, dtype=np.float64), stride=stride)
    gaps = np.diff(idx)
    assert gaps.size == 0 or gaps.min() >= stride, (
        f"stride={stride} produced overlapping samples: min gap {gaps.min()}"
    )


# ---------------------------------------------------------------------------
# _median_se_for_q — the pair that already diverged
# ---------------------------------------------------------------------------


def _pipeline(median: float, n_nl: int, ci: tuple[float, float] | None) -> dict:
    cell = {"q": 5, "W": 120, "median_vr_dep": median, "n_nl": n_nl}
    if ci is not None:
        cell["ci_95_lo"], cell["ci_95_hi"] = ci
    return {"per_wq_primary_q": [], "per_wq": [cell]}


@pytest.mark.parametrize(
    "median,n_nl,ci",
    [
        (0.15, 20126, (0.14, 0.16)),
        (0.15, 20126, None),
        (0.15, 454, None),
        (0.20, 1, None),
    ],
)
def test_median_se_copies_agree(median: float, n_nl: int, ci) -> None:
    """The two copies of _median_se_for_q must return the same standard error.

    This is the exact pair that diverged. gauge_invariance returns a triple and
    migrate_gauge_tost_ci returns the scalar, so only the SE is comparable.
    """
    from scripts.wp1.gauge_invariance import _median_se_for_q as gauge_version
    from scripts.wp1.migrate_gauge_tost_ci import _median_se_for_q as migrate_version

    pipeline = _pipeline(median, n_nl, ci)
    _, _, se_gauge = gauge_version(pipeline, q=5)
    se_migrate = migrate_version(pipeline, q=5)

    assert se_gauge == pytest.approx(se_migrate, rel=0, abs=0), (
        f"the two _median_se_for_q copies disagree: {se_gauge} vs {se_migrate}"
    )


def test_median_se_handles_a_missing_cell_without_returning_none() -> None:
    """The C1 regression: n_nl == 0 used to fall off the end and return None.

    The caller unpacks a triple, so None raised TypeError rather than
    degrading. The margin computation depends on this function.
    """
    from scripts.wp1.gauge_invariance import _median_se_for_q

    empty = {"per_wq_primary_q": [], "per_wq": []}
    median, n_nl, se = _median_se_for_q(empty, q=5)
    assert np.isnan(median)
    assert n_nl == 0
    assert np.isinf(se)

    zero_n = _pipeline(0.15, 0, None)
    median, n_nl, se = _median_se_for_q(zero_n, q=5)
    assert n_nl == 0
    assert np.isinf(se), "n_nl == 0 must give an infinite SE, not None"
