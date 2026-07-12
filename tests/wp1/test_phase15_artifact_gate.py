"""Phase 15 Wave 0 artifact completeness gate (filesystem only, no heavy compute)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest.utils import PROJECT_ROOT

INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
THERMO_DIR = PROJECT_ROOT / "backtest_results" / "thermodynamic_bound"
GAUGE_REPORT = (
    PROJECT_ROOT
    / "backtest_results"
    / "gauge_invariance"
    / "gauge_report_20260624_221834.json"
)
PERSISTENCE_REPORT = (
    PROJECT_ROOT
    / "backtest_results"
    / "persistence"
    / "persistence_report_20260612_214557.json"
)
PRIMARY_INJECTION = INJECTION_DIR / "inj_d0.005_q5_W120.json"
CONFIRMATORY_GRID_CELLS = 96
EXPLORATORY_ADDENDUM_CELLS = 8


def _load_injection_artifacts() -> list[tuple[Path, dict]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(INJECTION_DIR.glob("inj_*.json"))
    ]


def test_injection_grid_complete() -> None:
    artifacts = _load_injection_artifacts()
    confirmatory = [
        item for item in artifacts
        if item[1].get("provenance", {}).get("exploratory_addendum") is not True
    ]
    exploratory = [
        item for item in artifacts
        if item[1].get("provenance", {}).get("exploratory_addendum") is True
    ]

    assert len(confirmatory) == CONFIRMATORY_GRID_CELLS, (
        f"expected {CONFIRMATORY_GRID_CELLS} confirmatory injection JSONs, "
        f"got {len(confirmatory)}"
    )
    assert len(exploratory) == EXPLORATORY_ADDENDUM_CELLS, (
        f"expected {EXPLORATORY_ADDENDUM_CELLS} exploratory addendum JSONs, "
        f"got {len(exploratory)}"
    )
    exploratory_keys = {
        (
            float(data["delta_target"]),
            int(data["q"]),
            int(data["W"]),
        )
        for _, data in exploratory
    }
    assert exploratory_keys == {
        (0.15, 2, 60),
        (0.15, 2, 120),
        (0.15, 2, 240),
        (0.15, 5, 60),
        (0.15, 5, 120),
        (0.15, 5, 240),
        (0.20, 5, 120),
        (0.30, 5, 120),
    }


def test_injection_json_schema() -> None:
    assert PRIMARY_INJECTION.exists(), f"missing primary cell artifact: {PRIMARY_INJECTION}"
    data = json.loads(PRIMARY_INJECTION.read_text(encoding="utf-8"))
    assert data.get("injected") is True
    assert "P_det" in data
    assert data.get("N_mc") == 200


def test_thermo_report_exists() -> None:
    reports = sorted(THERMO_DIR.glob("thermo_report_*.json"))
    assert reports, "no thermo_report_*.json under backtest_results/thermodynamic_bound/"
    latest = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert latest.get("headline_verdict") in (
        "demon_runs_at_a_loss",
        "bound_exceeds_costs",
    )
    taker_floor = min(latest["cost_schedule"]["taker_rt_bps"])
    for row in latest["per_q_results"]:
        expected = (
            "demon_runs_at_loss"
            if row["ksg_gmax_bps"] < taker_floor
            else "bound_exceeds_costs"
        )
        assert row["verdict_sign_pair_vs_taker"] == expected
        assert row["verdict_vs_taker"] == row["verdict_sign_pair_vs_taker"]
        assert "verdict_gaussian_vs_taker" in row


def test_ratified_peer_artifacts() -> None:
    assert GAUGE_REPORT.is_file(), f"missing ratified gauge report: {GAUGE_REPORT}"
    gauge = json.loads(GAUGE_REPORT.read_text(encoding="utf-8"))
    comparisons = gauge["tost"]["per_q_comparisons"]
    assert comparisons
    for row in comparisons:
        assert "ci_90_lo" in row
        assert "ci_90_hi" in row
        assert "ci_within_margin" in row
    assert PERSISTENCE_REPORT.is_file(), (
        f"missing ratified persistence report: {PERSISTENCE_REPORT}"
    )


def test_holdout_not_in_dev_artifacts() -> None:
    paths = list(INJECTION_DIR.glob("*.json")) + list(THERMO_DIR.glob("*.json"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert '"year_2026_loaded": true' not in text, (
            f"D-07 violation: {path} contains year_2026_loaded=true"
        )
