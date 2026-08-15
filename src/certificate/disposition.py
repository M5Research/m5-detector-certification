"""Disposition classifier. Independent of detector and gate implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from certificate.errors import InvalidGateStateError
from certificate.models import KNOWN_GATE_STATES, GateResult

DISPOSITION_PRECEDENCE: tuple[str, ...] = (
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

GATE_FAIL_DISPOSITION: dict[str, str] = {
    "instrument": "instrument_failed",
    "target": "target_mismatched",
    "size": "size_distorted",
    "transport": "transport_uncertified",
    "power": "underpowered",
    "information": "information_unsupported",
    "value": "value_negative",
}


@dataclass(frozen=True)
class DispositionResult:
    disposition: str
    completeness: bool
    failure_profile: list[dict[str, object]]


def _profile_entry(gate: GateResult) -> dict[str, object]:
    return {
        "gate": gate.gate,
        "state": gate.state,
        "reason": gate.reason,
        "estimand": gate.estimand,
        "threshold": gate.threshold,
    }


def classify_disposition(
    gates: Iterable[GateResult],
    required_gates: Iterable[str],
) -> DispositionResult:
    """Apply frozen precedence. Missing required gates yield incomplete.

    Instrument and target failures still outrank incomplete. Every observed
    failure remains in the failure profile.
    """
    required = list(required_gates)
    by_name: dict[str, GateResult] = {}
    for gate in gates:
        if gate.state not in KNOWN_GATE_STATES:
            raise InvalidGateStateError(f"unknown gate state {gate.state!r} for {gate.gate}")
        if gate.state == "invalid":
            raise InvalidGateStateError(gate.reason or f"{gate.gate} is invalid")
        by_name[gate.gate] = gate

    failure_profile = [
        _profile_entry(gate)
        for gate in by_name.values()
        if gate.state == "fail"
    ]

    instrument = by_name.get("instrument")
    if instrument is not None and instrument.state == "fail":
        missing = any(
            name not in by_name or by_name[name].state == "not_run" for name in required
        )
        return DispositionResult(
            disposition="instrument_failed",
            completeness=not missing,
            failure_profile=failure_profile,
        )

    target = by_name.get("target")
    if target is not None and target.state == "fail":
        missing = any(
            name not in by_name or by_name[name].state == "not_run" for name in required
        )
        return DispositionResult(
            disposition="target_mismatched",
            completeness=not missing,
            failure_profile=failure_profile,
        )

    missing_required = [
        name for name in required if name not in by_name or by_name[name].state == "not_run"
    ]
    if missing_required:
        return DispositionResult(
            disposition="incomplete",
            completeness=False,
            failure_profile=failure_profile,
        )

    for name in ("size", "transport", "power", "information", "value"):
        gate = by_name.get(name)
        if gate is not None and gate.state == "fail":
            return DispositionResult(
                disposition=GATE_FAIL_DISPOSITION[name],
                completeness=True,
                failure_profile=failure_profile,
            )

    return DispositionResult(
        disposition="admissible",
        completeness=True,
        failure_profile=failure_profile,
    )
