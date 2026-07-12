import numpy as np
import pytest
from scripts.wp1 import eth_replication


def test_eth_replication_defaults_keep_btc_primary_scope():
    assert eth_replication.DEFAULT_SYMBOL == "ETHUSDT"
    assert eth_replication.PRIMARY_ASSET == "BTCUSDT"
    assert eth_replication.OUTPUT_DIR.name == "asset_replication"


def test_validate_in_sample_span_rejects_2026():
    ts = np.array([1_767_225_600_000], dtype=np.int64)
    with pytest.raises(ValueError, match="in-sample.*2026"):
        eth_replication.validate_in_sample_span(ts)


def test_validate_holdout_span_marks_partial_holdout():
    ts = np.array([1_767_225_600_000, 1_772_323_200_000], dtype=np.int64)
    span = eth_replication.validate_holdout_span(ts)
    assert span["year_2026_loaded"] is True
    assert span["partial_holdout"] is True
    assert span["start"] == "2026-01-01"


def test_build_report_keeps_injection_boundary_explicit():
    in_sample = {
        "data_span": {"start": "2021-05-29", "end": "2025-12-31", "n_bars": 10},
        "precheck": {"primary_cell": {"observed_vr_dep": 0.1}},
        "gauge": {"verdict": "gauge equivalence not certified"},
        "information": {"headline_verdict": "demon_runs_at_loss"},
        "persistence": {"verdict": "persistence_deviation_detected"},
    }
    holdout = {
        "data_span": {
            "start": "2026-01-01",
            "end": "2026-05-29",
            "n_bars": 10,
            "year_2026_loaded": True,
            "partial_holdout": True,
        },
        "primary_cell": {"observed_vr_dep": 0.2},
    }

    report = eth_replication.build_eth_replication_report(
        symbol="ETHUSDT",
        in_sample=in_sample,
        holdout=holdout,
        run_id="test",
        code_commit="abc",
        prereg_commit="def",
    )

    assert report["symbol"] == "ETHUSDT"
    assert report["primary_asset"] == "BTCUSDT"
    assert report["scope"] == "external_asset_replication"
    assert report["injection_calibration"]["eth_specific_injection_grid"] is False
    assert "not reused" in report["injection_calibration"]["btc_injection_grid_policy"]
