"""Detector-specific gate-workflow registry used by the certificate CLI."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from certificate.errors import CertificateError
from certificate.models import CertificateSpec, GateResult

GateWorkflow = Callable[[CertificateSpec, Path], list[GateResult]]
_WORKFLOWS: dict[str, GateWorkflow] = {}
_BUILTINS: dict[str, tuple[str, str]] = {
    "btc_vr_recentered_q2": ("scripts.wp1.btc_successor", "load_gate_results"),
    "eurusd_rolling_quantile": ("scripts.wp1.eurusd_certificate", "load_gate_results"),
}


def register_gate_workflow(detector: str, workflow: GateWorkflow) -> None:
    if detector in _WORKFLOWS:
        raise CertificateError(f"gate workflow already registered for {detector}")
    _WORKFLOWS[detector] = workflow


def evaluate_gates(spec: CertificateSpec, repo: Path) -> list[GateResult]:
    try:
        workflow = _WORKFLOWS[spec.detector]
    except KeyError:
        try:
            module_name, function_name = _BUILTINS[spec.detector]
        except KeyError as exc:
            raise CertificateError(f"no gate workflow registered for {spec.detector}") from exc
        workflow = getattr(import_module(module_name), function_name)
    gates = workflow(spec, repo)
    if {gate.gate for gate in gates} != set(spec.required_gates):
        raise CertificateError("gate workflow did not return exactly the required gates")
    return gates
