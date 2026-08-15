"""Create-only certificate runner."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from certificate.disposition import classify_disposition
from certificate.errors import (
    ArtifactExistsError,
    DuplicateCertificateIdError,
    PreregistrationError,
    SchemaError,
)
from certificate.models import CertificateRecord, CertificateSpec, GateResult
from certificate.lineage import validate_forest
from certificate.preregistration import assert_preregistration
from certificate.serialize import canonical_dumps, spec_hash


def _discover_specs(directory: Path) -> list[CertificateSpec]:
    specs: list[CertificateSpec] = []
    if not directory.is_dir():
        return specs
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = CertificateSpec.model_validate(payload.get("spec", {}))
        except (OSError, ValueError, AttributeError):
            continue
        specs.append(spec)
    return specs


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PreregistrationError("artifact created_at must include a UTC offset")
    return parsed.astimezone(UTC)


def validate_gate_artifacts(
    gates: list[GateResult],
    *,
    repo: Path,
    spec: CertificateSpec,
    digest: str,
    preregistered_at: datetime,
) -> dict[str, str]:
    """Validate gate provenance and return the union of declared data hashes."""
    data_hashes: dict[str, str] = {}
    for gate in gates:
        for reference in gate.artifacts:
            relative = Path(reference)
            if relative.is_absolute():
                raise PreregistrationError("gate artifact paths must be repository-relative")
            path = repo / relative
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PreregistrationError(f"gate artifact cannot be read: {reference}") from exc
            if payload.get("preregistration_commit") != spec.preregistration_commit:
                raise PreregistrationError(f"gate artifact preregistration mismatch: {reference}")
            if payload.get("spec_hash") != digest:
                raise PreregistrationError(f"gate artifact spec hash mismatch: {reference}")
            created_at = _parse_utc(str(payload.get("created_at", "")))
            if created_at <= preregistered_at.astimezone(UTC):
                raise PreregistrationError(f"gate artifact predates preregistration: {reference}")
            declared = payload.get("data_hashes")
            if not isinstance(declared, dict) or not declared:
                raise PreregistrationError(f"gate artifact lacks data hashes: {reference}")
            for name, value in declared.items():
                if name in data_hashes and data_hashes[name] != value:
                    raise PreregistrationError(f"conflicting data hash for {name}")
                data_hashes[str(name)] = str(value)
            data_hashes[f"artifact:{relative.as_posix()}"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return data_hashes


def run_certificate(
    *,
    spec_path: Path,
    out_path: Path,
    gate_results: list[GateResult],
    preregistration_ok: bool | None = None,
    repo: Path | None = None,
    existing_ids: set[str] | None = None,
    code_hash: str | None = None,
    data_hashes: dict[str, str] | None = None,
    upstream_hash: str | None = None,
) -> CertificateRecord:
    """Validate, classify, and write a new certificate artifact.

    A completed scientific certificate, including a scientifically withheld
    disposition, is success. Incomplete execution remains recordable but the
    CLI returns nonzero. Invalid gate states, schema failures, duplicate IDs,
    and in-place replacement raise and must not write.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise ArtifactExistsError(f"refusing to overwrite {out_path}")

    try:
        spec = CertificateSpec.model_validate_json(Path(spec_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SchemaError(str(exc)) from exc
    digest = spec_hash(spec)

    existing_specs = _discover_specs(out_path.parent)
    issued = existing_ids if existing_ids is not None else {
        existing.certificate_id for existing in existing_specs
    }
    if spec.certificate_id in issued:
        raise DuplicateCertificateIdError(spec.certificate_id)
    validate_forest([*existing_specs, spec])

    assert_preregistration(spec, ok=preregistration_ok, repo=repo, spec_path=spec_path)

    if preregistration_ok is True:
        code_hash = code_hash or "test-code"
        data_hashes = data_hashes or {"fixture": hashlib.sha256(b"fixture").hexdigest()}
        upstream_hash = upstream_hash or hashlib.sha256(b"test-upstream").hexdigest()
    elif repo is not None:
        if code_hash is None:
            code_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        if upstream_hash is None:
            manifest = repo / "provenance" / "upstream-snapshot-v1.json"
            upstream_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        preregistered_raw = subprocess.run(
            ["git", "show", "-s", "--format=%cI", spec.preregistration_commit],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        derived_hashes = validate_gate_artifacts(
            gate_results,
            repo=repo,
            spec=spec,
            digest=digest,
            preregistered_at=_parse_utc(preregistered_raw),
        )
        if data_hashes is None:
            data_hashes = derived_hashes
        elif data_hashes != derived_hashes:
            raise SchemaError("supplied data hashes do not match gate artifacts")
    if code_hash is None or not data_hashes or upstream_hash is None:
        raise SchemaError("code_hash, data_hashes, and upstream_hash are required")

    classified = classify_disposition(gate_results, required_gates=spec.required_gates)
    record = CertificateRecord(
        spec=spec,
        spec_hash=digest,
        code_hash=code_hash,
        data_hashes=data_hashes,
        upstream_hash=upstream_hash,
        gates=gate_results,
        completeness=classified.completeness,
        failure_profile=classified.failure_profile,
        disposition=classified.disposition,
        provenance={
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    _write_exclusive(out_path, canonical_dumps(record.model_dump(mode="json")))
    return record


def _write_exclusive(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
