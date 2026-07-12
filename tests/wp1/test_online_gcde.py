from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_alpha_budget_is_by_gate_not_by_window() -> None:
    from scripts.wp1.online_gcde import DEFAULT_ALPHA_BUDGET

    assert DEFAULT_ALPHA_BUDGET == {
        "size": 0.02,
        "sensitivity": 0.015,
        "detector_information": 0.015,
    }
    assert sum(DEFAULT_ALPHA_BUDGET.values()) == 0.05


def test_max_stat_threshold_uses_path_maxima_not_window_quantiles() -> None:
    from scripts.wp1.online_gcde import calibrate_max_stat_thresholds

    bootstrap_paths = {
        "size": np.array(
            [
                [0.0, 0.0, 10.0],
                [5.0, 5.0, 5.0],
                [6.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    thresholds = calibrate_max_stat_thresholds(bootstrap_paths, {"size": 0.5})

    assert thresholds["size"] == 7.0


def test_state_path_admits_and_revokes_after_signal_changes() -> None:
    from scripts.wp1.online_gcde import build_state_path

    stats = [
        {"date": "2021-01-01", "size": 0.1, "sensitivity": 0.1, "detector_information": 0.1},
        {"date": "2021-02-01", "size": 0.1, "sensitivity": 0.1, "detector_information": 0.1},
        {"date": "2021-03-01", "size": 2.0, "sensitivity": 2.0, "detector_information": 2.0},
        {"date": "2021-04-01", "size": 2.1, "sensitivity": 2.1, "detector_information": 2.1},
        {"date": "2021-05-01", "size": 0.2, "sensitivity": 0.2, "detector_information": 0.2},
        {"date": "2021-06-01", "size": 0.2, "sensitivity": 0.2, "detector_information": 0.2},
    ]

    path = build_state_path(
        stats,
        {"size": 1.0, "sensitivity": 1.0, "detector_information": 1.0},
        min_consecutive_passes=2,
        revoke_after_misses=2,
    )

    assert [row["A_t"] for row in path] == [
        "warmup",
        "non_admitted",
        "non_admitted",
        "admissible",
        "admissible",
        "revoked",
    ]
    assert path[3]["passes_all_gates"] is True
    assert path[-1]["tau_revoke"] == "2021-06-01"


def test_transport_uncertified_is_not_core_failure() -> None:
    from scripts.wp1.online_gcde import initial_online_state

    state = initial_online_state(
        gauge_scope="clock_only",
        transport_status="gauge_uncertified",
    )

    assert state["A_t"] == "warmup"
    assert state["gauge_scope"] == "clock_only"
    assert state["transport_status"] == "gauge_uncertified"


def test_replay_windows_are_causal_and_targets_are_matured() -> None:
    from scripts.wp1.online_gcde import iter_replay_windows

    timestamps = np.array(
        [
            np.datetime64(f"2021-01-{day:02d}", "ms").astype("datetime64[ms]").astype(np.int64)
            for day in range(1, 11)
        ],
        dtype=np.int64,
    )

    windows = list(
        iter_replay_windows(
            timestamps,
            start="2021-01-05",
            end="2021-01-08",
            window_days=3,
            stride_days=1,
            target_horizon_bars=2,
        )
    )

    assert len(windows) == 4
    for window in windows:
        assert window["window_start_idx"] <= window["matured_target_end_idx"]
        assert window["matured_target_end_idx"] <= window["decision_idx"] - 2
        assert timestamps[window["window_end_idx"]] <= window["decision_timestamp_ms"]
        assert timestamps[window["decision_idx"]] == window["decision_timestamp_ms"]


def test_artifact_schema_is_deterministic_and_gauge_uncertified() -> None:
    from scripts.wp1.online_gcde import build_online_gcde_artifact

    kwargs = {
        "demo_stats": [
            {"date": "2021-01-01", "size": 0.1, "sensitivity": 0.1, "detector_information": 0.1},
            {"date": "2021-02-01", "size": 2.0, "sensitivity": 2.0, "detector_information": 2.0},
            {"date": "2021-03-01", "size": 2.0, "sensitivity": 2.0, "detector_information": 2.0},
        ],
        "real_stats": [
            {"date": "2021-01-01", "size": 0.1, "sensitivity": 0.1, "detector_information": 0.1},
            {"date": "2021-02-01", "size": 0.2, "sensitivity": 0.2, "detector_information": 0.2},
        ],
        "thresholds": {"size": 1.0, "sensitivity": 1.0, "detector_information": 1.0},
        "generated_utc": "2026-06-25T00:00:00+00:00",
        "params": {"mode": "both", "smoke": True},
    }
    first = build_online_gcde_artifact(**kwargs)
    second = build_online_gcde_artifact(**kwargs)

    assert first == second
    assert first["schema_version"] == 1
    assert first["sequential_control"] == "maxT"
    assert first["gauge_scope"] == "clock_only"
    assert first["transport_status"] == "gauge_uncertified"
    assert first["alpha_budget_total"] == 0.05
    assert set(first["thresholds"]) == {"size", "sensitivity", "detector_information"}
    assert first["controlled_dynamic_injection"]["state_path"][-1]["A_t"] == "admissible"
    assert first["real_btc_replay"]["state_path"][-1]["A_t"] == "non_admitted"


def test_cli_smoke_writes_online_replay_artifact(tmp_path: Path) -> None:
    out_path = tmp_path / "online_gcde_replay.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/wp1/online_gcde.py",
            "--mode",
            "both",
            "--smoke",
            "--n-boot",
            "9",
            "--n-mc",
            "12",
            "--n-perm",
            "20",
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
    demo_states = [row["A_t"] for row in artifact["controlled_dynamic_injection"]["state_path"]]
    real_states = [row["A_t"] for row in artifact["real_btc_replay"]["state_path"]]

    assert "admissible" in demo_states
    assert "revoked" in demo_states
    assert real_states[-1] == "non_admitted"
    assert artifact["controlled_dynamic_injection"]["cell"] == {"W": 120, "q": 2}
    assert artifact["real_btc_replay"]["cell"] == {"W": 120, "q": 5}
