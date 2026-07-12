"""Tests for the harmonized detector benchmark core.

These tests define the benchmark contract before the implementation exists:
frozen mappings, validity gates, agreement metrics, timestamp intersection, and
artifact shape.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_collapse_to_two_state_preserves_warmup_and_merges_risk_states() -> None:
    from scripts.wp1.harmonized_benchmark import collapse_to_two_state

    labels = np.array([-1, 0, 1, 2, 0, 2], dtype=np.int8)

    mapped = collapse_to_two_state(labels)

    np.testing.assert_array_equal(
        mapped,
        np.array([-1, 0, 1, 1, 0, 1], dtype=np.int8),
    )


def test_degenerate_stream_is_instrument_failure_not_valid_disagreement() -> None:
    from scripts.wp1.harmonized_benchmark import (
        STATUS_INSTRUMENT_FAILURE,
        validate_label_stream,
    )

    labels = np.zeros(240, dtype=np.int8)

    result = validate_label_stream(
        "hmm",
        labels,
        n_states=2,
        stride=4,
        min_nonoverlap=10,
        variance_separation=(1.20,),
        convergence_ok=True,
        deterministic=True,
    )

    assert result["status"] == STATUS_INSTRUMENT_FAILURE
    assert any("state_1" in reason for reason in result["reasons"])


def test_ms_style_detector_fails_when_variance_separation_is_too_small() -> None:
    from scripts.wp1.harmonized_benchmark import (
        STATUS_INSTRUMENT_FAILURE,
        validate_label_stream,
    )

    labels = np.tile(np.array([0, 1], dtype=np.int8), 200)

    result = validate_label_stream(
        "ms",
        labels,
        n_states=2,
        stride=4,
        min_nonoverlap=10,
        variance_separation=(1.04,),
        convergence_ok=True,
        deterministic=True,
    )

    assert result["status"] == STATUS_INSTRUMENT_FAILURE
    assert "variance_separation_below_1.10" in result["reasons"]


def test_agreement_metrics_identify_perfect_and_permuted_labels() -> None:
    from scripts.wp1.harmonized_benchmark import pairwise_agreement

    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    perfect = pairwise_agreement(labels, labels, n_states=3)
    permuted = pairwise_agreement(labels, np.array([2, 2, 1, 1, 0, 0]), n_states=3)

    assert perfect["cohens_kappa"] == 1.0
    assert perfect["ari"] == 1.0
    assert perfect["nmi"] == 1.0
    assert perfect["variation_of_information"] == 0.0
    assert permuted["ari"] == 1.0
    assert permuted["nmi"] == 1.0
    assert permuted["cohens_kappa"] < 1.0


def test_shared_intersection_keeps_only_valid_common_timestamps() -> None:
    from scripts.wp1.harmonized_benchmark import DetectorOutput, shared_intersection

    a = DetectorOutput(
        name="a",
        timestamps=np.array([10, 20, 30, 40]),
        labels=np.array([-1, 0, 1, 1], dtype=np.int8),
    )
    b = DetectorOutput(
        name="b",
        timestamps=np.array([20, 30, 40, 50]),
        labels=np.array([0, -1, 1, 1], dtype=np.int8),
    )

    intersected = shared_intersection([a, b])

    assert intersected["timestamps"].tolist() == [20, 40]
    np.testing.assert_array_equal(intersected["labels"]["a"], np.array([0, 1]))
    np.testing.assert_array_equal(intersected["labels"]["b"], np.array([0, 1]))


def test_real_benchmark_window_preserves_one_minute_frequency() -> None:
    from scripts.wp1.harmonized_benchmark import slice_1m_window

    start = np.datetime64("2022-10-01T00:00:00")
    timestamps = (
        start.astype("datetime64[ms]").astype(np.int64)
        + np.arange(10, dtype=np.int64) * 60_000
    )
    close = np.linspace(100.0, 101.0, len(timestamps))

    close_w, timestamps_w, span = slice_1m_window(
        close,
        timestamps,
        start_date="2022-10-01",
        end_date="2022-10-01T00:04:00",
        symbol="BTCUSDT",
    )

    assert len(close_w) == 5
    assert len(timestamps_w) == 5
    assert span["symbol"] == "BTCUSDT"
    assert span["bar_frequency"] == "1min"
    assert span["window_policy"] == "contiguous_time_domain_truncation_no_downsampling"
    assert span["downsampled"] is False
    assert span["q_units"] == "bars_at_native_1min_frequency"
    assert span["year_2026_loaded"] is False


def test_detector_contingent_information_discretizes_soft_outputs() -> None:
    from scripts.wp1.harmonized_benchmark import (
        DetectorOutput,
        discrete_labels_for_task,
    )

    det = DetectorOutput(
        name="hmm",
        timestamps=np.arange(4),
        labels=np.array([-1, -1, -1, -1], dtype=np.int8),
        soft_probabilities=np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.51, 0.49],
                [0.1, 0.9],
            ]
        ),
    )

    labels = discrete_labels_for_task(det, task="2-state")

    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1], dtype=np.int8))


def test_real_hmm_profile_is_explicit_and_recorded(monkeypatch) -> None:
    import scripts.wp1.harmonized_benchmark as hb

    class FakeHMM:
        def __init__(self, k_regimes: int, em_iter: int, search_reps: int) -> None:
            self.k_regimes = k_regimes
            self.em_iter = em_iter
            self.search_reps = search_reps
            self.convergence_ok_ = True
            self.filtered_probs_ = np.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                    [0.6, 0.4],
                    [0.1, 0.9],
                ]
            )
            self.sigma2_sorted_ = np.array([1.0, 1.3])
            self.hmm_fallback_ = "none"

        def fit(self, close: np.ndarray) -> np.ndarray:
            assert len(close) == 4
            return np.array([0, 1, 0, 1], dtype=np.int8)

    monkeypatch.setattr(hb, "HMMDetector", FakeHMM)

    outputs = hb.build_real_detectors(
        np.array([100.0, 101.0, 100.5, 102.0]),
        np.array([1, 2, 3, 4], dtype=np.int64),
        ["hmm"],
        W=2,
        q=1,
        hmm_regimes=2,
        hmm_em_iter=30,
        hmm_search_reps=3,
    )

    diagnostics = outputs[0].diagnostics
    assert diagnostics["hmm_regimes"] == 2
    assert diagnostics["hmm_em_iter"] == 30
    assert diagnostics["hmm_search_reps"] == 3
    assert diagnostics["hmm_profile"] == "explicit_2_state_harmonized_profile"


def test_detector_contingent_information_reports_cost_adjusted_bound() -> None:
    from scripts.wp1.harmonized_benchmark import detector_contingent_information

    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    forward_returns = np.array([-0.02, -0.01, 0.03, 0.02, -0.03, 0.04, -0.01, 0.05])

    result = detector_contingent_information(
        labels,
        forward_returns,
        n_states=2,
        cost_bps=1.0,
        n_boot=20,
        n_perm=20,
        seed=7,
    )

    assert result["mi_nats"] > 0.0
    assert result["gross_bound_bps"] > result["net_bound_bps"]
    assert result["bootstrap_ci_nats"][0] <= result["mi_nats"] <= result["bootstrap_ci_nats"][1]
    assert set(result["conditional_forward_return"].keys()) == {"0", "1"}


def test_constant_detector_labels_have_zero_information() -> None:
    from scripts.wp1.harmonized_benchmark import (
        detector_contingent_information,
        discrete_entropy,
    )

    labels = np.ones(8, dtype=np.int8)
    forward_returns = np.array([-0.02, 0.01, -0.03, 0.02, 0.01, -0.01, 0.04, -0.02])

    result = detector_contingent_information(
        labels,
        forward_returns,
        n_states=2,
        cost_bps=10.0,
        n_boot=10,
        n_perm=10,
        seed=9,
    )

    assert discrete_entropy(labels) == 0.0
    assert result["mi_nats"] == 0.0
    assert result["gross_bound_bps"] == 0.0
    assert result["net_bound_bps"] == -10.0


def test_vr_holm_trigger_information_uses_binary_discrete_labels(monkeypatch) -> None:
    import scripts.wp1.vr_significance as vr_significance
    from scripts.wp1.harmonized_benchmark import vr_holm_trigger_information

    z_by_q = {
        2: np.array([np.nan, np.nan, 3.0, -3.0, 0.1]),
        5: np.array([np.nan, np.nan, 0.1, 0.1, 0.1]),
        15: np.array([np.nan, np.nan, 0.1, 0.1, 0.1]),
        60: np.array([np.nan, np.nan, 0.1, 0.1, 0.1]),
    }

    def fake_compute(close: np.ndarray, W: int, q: int, stride: int | None = None):
        assert stride == 2
        return np.zeros(len(close), dtype=np.float64), z_by_q[q]

    monkeypatch.setattr(vr_significance, "compute_rolling_vr_and_z_strided", fake_compute)

    close = np.array([100.0, 101.0, 102.0, 101.0, 103.0])
    timestamps = np.arange(len(close), dtype=np.int64) * 60_000
    result = vr_holm_trigger_information(
        close,
        timestamps,
        W=120,
        target_q=2,
        stride=2,
        n_boot=0,
        n_perm=0,
    )

    assert result["n_labels"] == 3
    assert result["n_triggers"] == 1
    assert result["trigger_rate"] == 1 / 3
    assert result["label_entropy_nats"] > 0.0
    assert result["mi_nats"] >= 0.0


def test_smoke_runner_writes_benchmark_artifact(tmp_path: Path) -> None:
    out_path = tmp_path / "harmonized_benchmark_smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/wp1/harmonized_benchmark.py",
            "--smoke",
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert out_path.exists(), completed.stdout + completed.stderr
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["benchmark"]["protocol_id"] == "harmonized-regime-benchmark-mvp-v1"
    assert artifact["data_span"]["year_2026_loaded"] is False
    assert set(artifact["tasks"]) == {"2-state", "3-state"}
    assert artifact["disposition"]["classification"] in {
        "harmonized_convergence",
        "persistent_disagreement",
        "mixed_family_structure",
        "instrument_failure",
    }
    assert any(
        detector["validity"]["status"] == "instrument_failure"
        for detector in artifact["tasks"]["3-state"]["detectors"].values()
    )
    assert math.isfinite(artifact["tasks"]["2-state"]["pairwise"]["rolling_quantile__hmm"]["nmi"])
