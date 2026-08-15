"""Protocol-v5 immutable evidence checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BTC = ROOT / "evidence" / "gates" / "btcusdt-vr-q2-v5-r0" / "evidence.json"


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
