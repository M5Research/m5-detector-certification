"""Certificate disposition precedence is independent of detector code."""
from __future__ import annotations

import math

import pytest

from certificate.disposition import (
    DISPOSITION_PRECEDENCE,
    classify_disposition,
)
from certificate.errors import InvalidGateStateError
from certificate.models import GateResult


REQUIRED = (
    "instrument",
    "target",
    "size",
    "transport",
    "power",
    "information",
    "value",
)


def _gate(name: str, state: str, estimand: float | None = 1.0, reason: str = "ok") -> GateResult:
    return GateResult(
        gate=name,
        state=state,  # type: ignore[arg-type]
        estimand=estimand,
        threshold=0.0,
        uncertainty={"alpha": 0.025},
        artifacts=[],
        reason=reason,
    )


def _all_pass() -> list[GateResult]:
    return [_gate(name, "pass") for name in REQUIRED]


def test_precedence_tuple_is_frozen() -> None:
    assert DISPOSITION_PRECEDENCE == (
        "instrument_failed",
        "target_mismatched",
        "incomplete",
        "size_distorted",
        "transport_uncertified",
        "underpowered",
        "information_unsupported",
        "value_negative",
        "admissible",
    )


def test_all_required_gates_passing_is_admissible() -> None:
    result = classify_disposition(_all_pass(), required_gates=REQUIRED)
    assert result.disposition == "admissible"
    assert result.completeness is True
    assert result.failure_profile == []


def test_missing_required_gate_overrides_measured_failure() -> None:
    gates = [
        _gate("instrument", "pass"),
        _gate("target", "pass"),
        _gate("size", "fail", reason="false-event bound exceeds alpha"),
        _gate("transport", "pass"),
        _gate("power", "pass"),
        _gate("value", "fail", reason="utility bound <= 0"),
    ]
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "incomplete"
    assert result.completeness is False
    failed = {entry["gate"] for entry in result.failure_profile}
    assert failed == {"size", "value"}


def test_not_run_required_gate_is_incomplete_even_when_size_fails() -> None:
    gates = _all_pass()
    gates[2] = _gate("size", "fail", reason="size fail")
    gates[5] = _gate("information", "not_run", estimand=None, reason="not executed")
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "incomplete"
    assert result.completeness is False
    assert any(entry["gate"] == "size" for entry in result.failure_profile)


def test_instrument_failure_precedes_incomplete_and_size() -> None:
    gates = [
        _gate("instrument", "fail", reason="validity metadata rejected"),
        _gate("target", "pass"),
        _gate("size", "fail", reason="size fail"),
    ]
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "instrument_failed"
    assert result.completeness is False
    failed = {entry["gate"] for entry in result.failure_profile}
    assert "instrument" in failed
    assert "size" in failed


def test_target_mismatch_precedes_incomplete() -> None:
    gates = [
        _gate("instrument", "pass"),
        _gate("target", "fail", reason="claim target differs"),
        _gate("size", "fail", reason="size fail"),
    ]
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "target_mismatched"
    assert result.completeness is False


def test_size_failure_with_complete_gates() -> None:
    gates = _all_pass()
    gates[2] = _gate("size", "fail", reason="upper false-event bound > alpha")
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "size_distorted"
    assert result.completeness is True
    assert [entry["gate"] for entry in result.failure_profile] == ["size"]


def test_later_failures_are_retained_under_earlier_disposition() -> None:
    gates = _all_pass()
    gates[2] = _gate("size", "fail", reason="size")
    gates[3] = _gate("transport", "fail", reason="transport")
    gates[6] = _gate("value", "fail", reason="value")
    result = classify_disposition(gates, required_gates=REQUIRED)
    assert result.disposition == "size_distorted"
    assert [entry["gate"] for entry in result.failure_profile] == [
        "size",
        "transport",
        "value",
    ]


def test_information_and_value_have_distinct_dispositions() -> None:
    info = _all_pass()
    info[5] = _gate("information", "fail", reason="MI unsupported")
    assert classify_disposition(info, required_gates=REQUIRED).disposition == (
        "information_unsupported"
    )

    value = _all_pass()
    value[6] = _gate("value", "fail", reason="utility <= 0")
    assert classify_disposition(value, required_gates=REQUIRED).disposition == (
        "value_negative"
    )


def test_unknown_gate_state_is_rejected() -> None:
    with pytest.raises(InvalidGateStateError):
        classify_disposition(
            [_gate("instrument", "skipped")],
            required_gates=("instrument",),
        )


def test_invalid_gate_is_rejected_as_execution_failure() -> None:
    gates = _all_pass()
    gates[5] = _gate(
        "information",
        "invalid",
        estimand=None,
        reason="unstable_reference",
    )
    with pytest.raises(InvalidGateStateError, match="unstable_reference"):
        classify_disposition(gates, required_gates=REQUIRED)


def test_non_finite_estimand_without_invalid_reason_rejected_at_model() -> None:
    with pytest.raises(Exception):
        GateResult(
            gate="size",
            state="fail",
            estimand=math.nan,
            threshold=0.025,
            uncertainty={},
            artifacts=[],
            reason="",
        )
