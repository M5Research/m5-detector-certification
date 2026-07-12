from pathlib import Path

import numpy as np
import pytest

from scripts.wp1.holdout_confirmatory import (
    HOLDOUT_Q,
    HOLDOUT_W,
    assert_holdout_gate,
    compute_holdout_primary,
    validate_holdout_timestamps,
)


def test_holdout_gate_requires_flag_and_marker(tmp_path):
    marker = tmp_path / ".jef_draft_complete"
    with pytest.raises(PermissionError):
        assert_holdout_gate(confirm_jef_draft=False, marker_path=marker)
    with pytest.raises(FileNotFoundError):
        assert_holdout_gate(confirm_jef_draft=True, marker_path=marker)
    marker.write_text("complete\n", encoding="utf-8")
    assert_holdout_gate(confirm_jef_draft=True, marker_path=marker)


def test_validate_holdout_timestamps_rejects_pre_2026():
    ts = np.array([1_735_689_599_000, 1_735_689_600_000], dtype=np.int64)
    with pytest.raises(ValueError):
        validate_holdout_timestamps(ts)


def test_compute_holdout_primary_reports_primary_cell():
    rng = np.random.default_rng(43)
    n = 2_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    ts = np.arange(n, dtype=np.int64) * 60_000 + 1_767_225_600_000
    row = compute_holdout_primary(close, ts)
    assert row["W"] == HOLDOUT_W
    assert row["q"] == HOLDOUT_Q
    assert row["n_nl"] > 0
    assert row["observed_vr_dep"] >= 0.0
    assert np.isfinite(row["median_z_m2"])
