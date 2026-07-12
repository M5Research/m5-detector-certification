"""Tests for VR detector-contingent MI artifact helpers."""
from __future__ import annotations

import json
from pathlib import Path


def test_saturated_injection_cell_has_zero_trigger_entropy(tmp_path: Path) -> None:
    from scripts.wp1.vr_detector_mi import injection_cell_trigger_summary

    cell = {
        "W": 120,
        "q": 5,
        "delta_target": 0.15,
        "N_mc": 3,
        "n_fires": 0,
        "P_det": 0.0,
        "per_draw": [
            {"mc_idx": 0, "cascade_fired": False},
            {"mc_idx": 1, "cascade_fired": False},
            {"mc_idx": 2, "cascade_fired": False},
        ],
    }
    path = tmp_path / "inj_d0.15_q5_W120.json"
    path.write_text(json.dumps(cell), encoding="utf-8")

    summary = injection_cell_trigger_summary(path)

    assert summary["n_draws"] == 3
    assert summary["n_fires"] == 0
    assert summary["trigger_entropy_nats"] == 0.0
    assert summary["mi_upper_bound_nats"] == 0.0
    assert summary["mi_nats_if_conditioning_within_saturated_cell"] == 0.0
