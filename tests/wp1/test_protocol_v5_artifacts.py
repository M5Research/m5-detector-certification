"""Protocol-v5 immutable evidence checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BTC = ROOT / "evidence" / "gates" / "btcusdt-vr-q2-v5-r0" / "evidence.json"
BTC_CERTIFICATE = ROOT / "certificates" / "btcusdt-vr-q2-v5-r0.json"
EURUSD = ROOT / "evidence" / "gates" / "eurusd-rq-v5-r0" / "evidence.json"


@pytest.mark.artifact
def test_btc_successor_uses_full_frozen_budgets_and_no_permutation() -> None:
    artifact = json.loads(BTC.read_text(encoding="utf-8"))

    assert artifact["calibration"]["size"]["draws"] == 10_000
    assert len(artifact["calibration"]["power"]) == 9
    assert all(row["draws"] == 2_000 for row in artifact["calibration"]["power"])
    assert artifact["information"]["q2"]["n_boot"] == 49_999
    assert artifact["information"]["q5"]["n_boot"] == 49_999
    assert "permutation" not in BTC.read_text(encoding="utf-8").lower()


@pytest.mark.artifact
def test_btc_successor_gate_values_are_self_consistent() -> None:
    artifact = json.loads(BTC.read_text(encoding="utf-8"))
    gates = {row["gate"]: row for row in artifact["gate_results"]}

    assert gates["size"]["state"] == artifact["calibration"]["size"]["state"]
    assert gates["power"]["estimand"] == artifact["calibration"]["cmde_90"]
    assert gates["transport"]["state"] == artifact["transport"]["state"]
    assert gates["information"]["estimand"] == artifact["information"]["q2"]["e_value"]


@pytest.mark.artifact
def test_btc_successor_certificate_is_complete_and_mechanically_withheld() -> None:
    record = json.loads(BTC_CERTIFICATE.read_text(encoding="utf-8"))

    assert record["completeness"] is True
    assert record["disposition"] == "target_mismatched"
    assert {row["gate"] for row in record["gates"]} == set(record["spec"]["required_gates"])


@pytest.mark.artifact
def test_eurusd_certificate_evidence_uses_every_frozen_budget() -> None:
    artifact = json.loads(EURUSD.read_text(encoding="utf-8"))

    assert artifact["calibration"]["size"]["draws"] == 10_000
    assert len(artifact["calibration"]["power"]) == 5
    assert all(row["draws"] == 2_000 for row in artifact["calibration"]["power"])
    assert artifact["transport"]["n_boot"] == 9_999
    assert artifact["information"]["n_boot"] == 49_999
    assert artifact["value"]["n_boot"] == 49_999
    assert "permutation" not in EURUSD.read_text(encoding="utf-8").lower()


@pytest.mark.artifact
def test_eurusd_evidence_contains_all_seven_gates_and_four_transport_tests() -> None:
    artifact = json.loads(EURUSD.read_text(encoding="utf-8"))

    assert {row["gate"] for row in artifact["gate_results"]} == {
        "instrument",
        "target",
        "size",
        "transport",
        "power",
        "information",
        "value",
    }
    assert len(artifact["transport"]["comparisons"]) == 4
