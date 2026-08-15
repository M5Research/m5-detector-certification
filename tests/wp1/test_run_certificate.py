"""Create-only certificate runner and preregistration chronology guards."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from certificate.errors import (
    ArtifactExistsError,
    DuplicateCertificateIdError,
    LineageError,
    PreregistrationError,
)
from certificate.models import CertificateSpec, GateResult
from certificate.runner import run_certificate, validate_gate_artifacts
from certificate.serialize import spec_hash
from scripts.wp1 import run_certificate as cli_module


REQUIRED = [
    "instrument",
    "target",
    "size",
    "transport",
    "power",
    "information",
    "value",
]


def _spec_dict(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "certificate_id": "fixture-admitted",
        "claim_id": "fixture-claim",
        "evidence_revision": 0,
        "detector": "fixture_detector",
        "instrument": "BTCUSDT",
        "venue": "binance_usd_m",
        "target": "positive_short_horizon_serial_dependence",
        "schemes": ["calendar", "volume"],
        "horizon": 2,
        "signal_family": "ar1_injection",
        "null_family": "stationary_block",
        "margins": {"transport": 0.019748},
        "costs": {},
        "required_gates": REQUIRED,
        "preregistration_commit": "deadbee",
        "alpha": 0.025,
        "protocol_id": "protocol-v5",
        "thresholds": {},
    }
    payload.update(overrides)
    return payload


def _pass_gates() -> list[GateResult]:
    return [
        GateResult(
            gate=name,
            state="pass",
            estimand=1.0,
            threshold=0.0,
            uncertainty={},
            artifacts=[],
            reason="pass",
        )
        for name in REQUIRED
    ]


def test_admitted_fixture_writes_new_artifact(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    record = run_certificate(
        spec_path=spec_path,
        out_path=out_path,
        gate_results=_pass_gates(),
        preregistration_ok=True,
        existing_ids=set(),
    )
    assert record.disposition == "admissible"
    assert out_path.is_file()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["disposition"] == "admissible"
    assert "permutation_p" not in json.dumps(saved)


def test_withheld_size_failure_is_exit_success(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(
        json.dumps(_spec_dict(certificate_id="fixture-withheld")),
        encoding="utf-8",
    )
    gates = _pass_gates()
    gates[2] = GateResult(
        gate="size",
        state="fail",
        estimand=0.08,
        threshold=0.025,
        uncertainty={},
        artifacts=[],
        reason="false-event bound exceeds alpha",
    )
    record = run_certificate(
        spec_path=spec_path,
        out_path=out_path,
        gate_results=gates,
        preregistration_ok=True,
        existing_ids=set(),
    )
    assert record.disposition == "size_distorted"
    assert record.completeness is True


def test_incomplete_fixture_when_required_gate_missing(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(
        json.dumps(_spec_dict(certificate_id="fixture-incomplete")),
        encoding="utf-8",
    )
    gates = [g for g in _pass_gates() if g.gate != "information"]
    gates[2] = GateResult(
        gate="size",
        state="fail",
        estimand=0.08,
        threshold=0.025,
        uncertainty={},
        artifacts=[],
        reason="size fail",
    )
    record = run_certificate(
        spec_path=spec_path,
        out_path=out_path,
        gate_results=gates,
        preregistration_ok=True,
        existing_ids=set(),
    )
    assert record.disposition == "incomplete"
    assert record.completeness is False


def test_invalid_gate_does_not_write_artifact(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(
        json.dumps(_spec_dict(certificate_id="fixture-invalid")),
        encoding="utf-8",
    )
    gates = _pass_gates()
    gates[5] = GateResult(
        gate="information",
        state="invalid",
        estimand=None,
        threshold=0.0,
        uncertainty={},
        artifacts=[],
        reason="unstable_reference",
    )
    with pytest.raises(Exception):
        run_certificate(
            spec_path=spec_path,
            out_path=out_path,
            gate_results=gates,
            preregistration_ok=True,
            existing_ids=set(),
        )
    assert not out_path.exists()


def test_refuses_in_place_replacement(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    out_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactExistsError):
        run_certificate(
            spec_path=spec_path,
            out_path=out_path,
            gate_results=_pass_gates(),
            preregistration_ok=True,
            existing_ids=set(),
        )


def test_duplicate_certificate_id_is_rejected(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    with pytest.raises(DuplicateCertificateIdError):
        run_certificate(
            spec_path=spec_path,
            out_path=out_path,
            gate_results=_pass_gates(),
            preregistration_ok=True,
            existing_ids={"fixture-admitted"},
        )


def test_duplicate_certificate_id_is_discovered_from_output_directory(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_dir = tmp_path / "certificates"
    out_dir.mkdir()
    out_path = out_dir / "new.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    (out_dir / "existing.json").write_text(
        json.dumps({"spec": _spec_dict()}),
        encoding="utf-8",
    )

    with pytest.raises(DuplicateCertificateIdError):
        run_certificate(
            spec_path=spec_path,
            out_path=out_path,
            gate_results=_pass_gates(),
            preregistration_ok=True,
        )


def test_runner_validates_parent_spec_hash_against_existing_record(tmp_path: Path) -> None:
    out_dir = tmp_path / "certificates"
    out_dir.mkdir()
    parent_payload = _spec_dict(certificate_id="fixture-parent")
    (out_dir / "parent.json").write_text(
        json.dumps({"spec": parent_payload}),
        encoding="utf-8",
    )
    child_payload = _spec_dict(
        certificate_id="fixture-child",
        evidence_revision=1,
        alpha=0.0125,
        parent_certificate_id="fixture-parent",
        parent_spec_hash="0" * 64,
    )
    spec_path = tmp_path / "child-spec.json"
    spec_path.write_text(json.dumps(child_payload), encoding="utf-8")

    with pytest.raises(LineageError, match="parent_spec_hash"):
        run_certificate(
            spec_path=spec_path,
            out_path=out_dir / "child.json",
            gate_results=_pass_gates(),
            preregistration_ok=True,
        )


def test_dirty_preregistration_is_rejected(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    with pytest.raises(PreregistrationError):
        run_certificate(
            spec_path=spec_path,
            out_path=out_path,
            gate_results=_pass_gates(),
            preregistration_ok=False,
            existing_ids=set(),
        )


def test_written_record_includes_spec_hash(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    payload = _spec_dict()
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    record = run_certificate(
        spec_path=spec_path,
        out_path=out_path,
        gate_results=_pass_gates(),
        preregistration_ok=True,
        existing_ids=set(),
    )
    spec = CertificateSpec.model_validate(payload)
    assert record.spec_hash == spec_hash(spec)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["spec_hash"] == record.spec_hash
    assert saved["schema_version"] == "certificate-v1"


def test_gate_artifact_predating_preregistration_is_rejected(tmp_path: Path) -> None:
    payload = _spec_dict()
    spec = CertificateSpec.model_validate(payload)
    digest = spec_hash(spec)
    artifact = tmp_path / "gate.json"
    artifact.write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00Z",
                "preregistration_commit": spec.preregistration_commit,
                "spec_hash": digest,
                "data_hashes": {"fixture": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    gate = _pass_gates()[0].model_copy(update={"artifacts": ["gate.json"]})

    with pytest.raises(PreregistrationError, match="predates"):
        validate_gate_artifacts(
            [gate],
            repo=tmp_path,
            spec=spec,
            digest=digest,
            preregistered_at=datetime(2026, 2, 1, tzinfo=UTC),
        )


def test_cli_returns_nonzero_for_incomplete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    monkeypatch.setattr(cli_module, "evaluate_gates", lambda spec, repo: _pass_gates())
    monkeypatch.setattr(
        cli_module,
        "run_certificate",
        lambda **kwargs: SimpleNamespace(completeness=False),
    )

    exit_code = cli_module.main(["--spec", str(spec_path), "--out", str(out_path)])

    assert exit_code != 0


def test_cli_never_bypasses_live_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_certificate(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(completeness=True)

    monkeypatch.setattr(cli_module, "evaluate_gates", lambda spec, repo: _pass_gates())
    monkeypatch.setattr(cli_module, "run_certificate", fake_run_certificate)

    assert cli_module.main(["--spec", str(spec_path), "--out", str(out_path)]) == 0
    assert captured.get("preregistration_ok") is not True
    assert captured.get("repo") == cli_module.REPO_ROOT


def test_cli_rejects_external_gate_injection(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    gates_path = tmp_path / "gates.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    gates_path.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit):
        cli_module.main(
            [
                "--spec",
                str(spec_path),
                "--out",
                str(out_path),
                "--gates",
                str(gates_path),
            ]
        )


def test_cli_dispatches_registered_gate_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "evaluate_gates", lambda spec, repo: _pass_gates())

    def fake_run_certificate(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(completeness=True)

    monkeypatch.setattr(cli_module, "run_certificate", fake_run_certificate)

    assert cli_module.main(["--spec", str(spec_path), "--out", str(out_path)]) == 0
    assert captured["gate_results"] == _pass_gates()


def test_builtin_btc_workflow_loads_only_frozen_repository_evidence(tmp_path: Path) -> None:
    from certificate.gates import evaluate_gates

    spec = CertificateSpec.model_validate(_spec_dict(certificate_id="btc-workflow-fixture"))
    spec.detector = "btc_vr_recentered_q2"
    artifact = tmp_path / "evidence" / "gates" / spec.certificate_id / "evidence.json"
    artifact.parent.mkdir(parents=True)
    gates = [
        {
            "gate": gate,
            "state": "pass",
            "artifacts": [f"evidence/gates/{spec.certificate_id}/evidence.json"],
        }
        for gate in spec.required_gates
    ]
    artifact.write_text(json.dumps({"gate_results": gates}), encoding="utf-8")

    loaded = evaluate_gates(spec, tmp_path)

    assert [gate.gate for gate in loaded] == spec.required_gates


def test_builtin_eurusd_workflow_loads_only_frozen_repository_evidence(tmp_path: Path) -> None:
    from certificate.gates import evaluate_gates

    spec = CertificateSpec.model_validate(_spec_dict(certificate_id="eurusd-workflow-fixture"))
    spec.detector = "eurusd_rolling_quantile"
    artifact = tmp_path / "evidence" / "gates" / spec.certificate_id / "evidence.json"
    artifact.parent.mkdir(parents=True)
    gates = [
        {
            "gate": gate,
            "state": "fail",
            "artifacts": [f"evidence/gates/{spec.certificate_id}/evidence.json"],
        }
        for gate in spec.required_gates
    ]
    artifact.write_text(json.dumps({"gate_results": gates}), encoding="utf-8")

    loaded = evaluate_gates(spec, tmp_path)

    assert [gate.gate for gate in loaded] == spec.required_gates
