"""Tests for the recentered-reference repair (§4.8, claim C9).

The repair supports the only positive result in the paper and the protocol's
first admissible certificate, so it gets the same treatment as the negative
results: an executable generator and tests over the frozen artifact.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from backtest.utils import PROJECT_ROOT
from scripts.wp1.reference_repair import (
    ALPHA,
    FROZEN_ARTIFACT,
    estimate_empirical_null,
    fires,
    oos_size,
    p_det,
    recenter,
    run_repair,
    threshold_at,
    verify_frozen,
)

pytestmark = pytest.mark.artifact


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN_ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The frozen artifact
# ---------------------------------------------------------------------------


def test_frozen_artifact_passes_every_internal_check(artifact: dict) -> None:
    assert verify_frozen(artifact) == []


def test_frozen_artifact_matches_the_manuscript(artifact: dict) -> None:
    """The values the manuscript quotes for the primary window."""
    q2 = artifact["cells"]["W120_q2"]
    q5 = artifact["cells"]["W120_q5"]

    assert q2["cmde_orig_d90"] == 0.15
    assert q2["cmde_recentered_d90"] == 0.02
    assert q2["oos_size"] == 0.02

    assert q5["cmde_orig_d90"] == 0.30
    assert q5["cmde_recentered_d90"] == 0.10
    assert q5["oos_size"] == 0.07


def test_out_of_sample_size_is_controlled_only_at_the_primary_window(
    artifact: dict,
) -> None:
    """Size control holds at W=120 and fails at the other q=2 windows.

    The admissible certificate is scoped to W=120, where OOS size is 0.02. At
    W=60 it is 0.12 and at W=240 it is 0.08, both above the nominal 0.05. That
    scoping is therefore load-bearing and not incidental: the repaired
    instrument is not size-controlled across the window grid, only at the
    window the certificate declares.
    """
    sizes = {k: v["oos_size"] for k, v in artifact["cells"].items()}
    assert sizes["W120_q2"] <= ALPHA
    assert sizes["W60_q2"] > ALPHA
    assert sizes["W240_q2"] > ALPHA


# ---------------------------------------------------------------------------
# The repair procedure
# ---------------------------------------------------------------------------


def test_empirical_null_recovers_a_known_location_and_scale() -> None:
    rng = np.random.default_rng(43)
    draws = rng.normal(loc=-0.10, scale=0.02, size=20_000)
    mu_0, sigma_0 = estimate_empirical_null(draws)
    assert mu_0 == pytest.approx(-0.10, abs=5e-4)
    assert sigma_0 == pytest.approx(0.02, rel=5e-2)


def test_recentring_makes_the_null_standard_normal() -> None:
    """The whole point: a mis-centred reference becomes a calibrated one."""
    rng = np.random.default_rng(7)
    draws = rng.normal(loc=-0.106, scale=0.0087, size=50_000)
    mu_0, sigma_0 = estimate_empirical_null(draws)
    z = recenter(draws, mu_0, sigma_0)
    assert float(np.mean(z)) == pytest.approx(0.0, abs=0.02)
    assert float(np.std(z)) == pytest.approx(1.0, rel=0.02)
    # Size under the null lands near alpha, which the uncorrected reference
    # would not: the raw statistic is ~12 null SDs below zero.
    assert float(np.mean(fires(z, ALPHA))) == pytest.approx(ALPHA, abs=0.01)


def test_uncorrected_reference_would_never_fire() -> None:
    """Why the repair was needed at all.

    Judged against N(0,1), a statistic distributed around -0.106 with SD 0.0087
    is roughly 12 SDs from where the reference expects it, so a one-sided
    positive test never fires regardless of signal.
    """
    rng = np.random.default_rng(11)
    draws = rng.normal(loc=-0.106, scale=0.0087, size=10_000)
    assert float(np.mean(fires(draws, ALPHA))) == 0.0


def test_p_det_rises_with_injected_amplitude() -> None:
    rng = np.random.default_rng(3)
    mu_0, sigma_0 = -0.10, 0.01
    previous = -1.0
    for shift in (0.0, 0.01, 0.02, 0.04):
        draws = rng.normal(loc=mu_0 + shift, scale=sigma_0, size=4_000)
        value = p_det(draws, mu_0, sigma_0)
        assert value >= previous
        previous = value
    assert previous > 0.9


def test_threshold_at_takes_the_smallest_qualifying_amplitude() -> None:
    table = {0.01: 0.10, 0.02: 0.55, 0.05: 0.92, 0.10: 0.99}
    assert threshold_at(table, 0.50) == 0.02
    assert threshold_at(table, 0.90) == 0.05
    assert threshold_at(table, 1.01) is None


def test_oos_size_is_estimated_out_of_sample() -> None:
    """Splitting by draw index means the reported size is not the in-sample one."""
    rng = np.random.default_rng(5)
    draws = rng.normal(loc=-0.10, scale=0.01, size=20_000)
    assert oos_size(draws) == pytest.approx(ALPHA, abs=0.015)


def test_run_repair_reproduces_a_synthetic_ground_truth() -> None:
    """End-to-end: the procedure recovers a known detectability threshold."""
    rng = np.random.default_rng(43)
    mu_0, sigma_0 = -0.106, 0.0087
    shifts = {0.0005: 0.0, 0.01: 0.01, 0.02: 0.02, 0.05: 0.05}
    per_draw = {
        (120, 5, delta): rng.normal(mu_0 + shift, sigma_0, size=2_000)
        for delta, shift in shifts.items()
    }

    result = run_repair(per_draw)
    cell = result["cells"]["W120_q5"]

    assert cell["null_center"] == pytest.approx(mu_0, abs=1e-3)
    assert cell["null_sd"] == pytest.approx(sigma_0, rel=0.05)
    assert cell["oos_size"] == pytest.approx(ALPHA, abs=0.02)
    # A shift of 2 null SDs is not yet 90% power one-sided at alpha=0.05;
    # roughly 3 SDs is. 0.05 / 0.0087 is about 5.7 SDs, so d90 lands there.
    assert cell["cmde_recentered_d90"] == 0.05
    assert cell["cmde_recentered_d50"] == 0.02
    assert result["provenance"]["type"] == "exploratory_post_freeze_repair"


def test_run_repair_requires_a_calibration_cell() -> None:
    rng = np.random.default_rng(1)
    per_draw = {(120, 5, 0.05): rng.normal(0.0, 1.0, size=100)}
    with pytest.raises(KeyError, match="calibration cell"):
        run_repair(per_draw)


def test_recenter_rejects_a_degenerate_scale() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        recenter(np.array([1.0, 2.0]), 0.0, 0.0)
