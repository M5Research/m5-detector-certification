"""Revision-tree validation and spent-alpha accounting."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from certificate.alpha import ALPHA_BUDGET, evidentiary_alpha
from certificate.errors import LineageError
from certificate.models import CertificateSpec
from certificate.serialize import spec_hash

__all__ = [
    "ALPHA_BUDGET",
    "SCOPE_FIELDS",
    "evidentiary_alpha",
    "scope_payload",
    "spent_alpha",
    "validate_revision",
]

SCOPE_FIELDS: tuple[str, ...] = (
    "detector",
    "instrument",
    "venue",
    "target",
    "schemes",
    "horizon",
    "signal_family",
    "null_family",
    "margins",
    "costs",
    "required_gates",
    "thresholds",
    "protocol_id",
    "claim_id",
)


def scope_payload(spec: CertificateSpec) -> dict[str, object]:
    data = spec.model_dump(mode="json")
    return {field: data[field] for field in SCOPE_FIELDS}


def spent_alpha(specs: Sequence[CertificateSpec]) -> float:
    """Sum evidentiary allocations. Metadata patches consume none."""
    total = 0.0
    seen: set[int] = set()
    for spec in specs:
        if spec.patch_kind == "metadata":
            continue
        if spec.evidence_revision in seen:
            continue
        seen.add(spec.evidence_revision)
        total += spec.alpha
    return total


def validate_revision(child: CertificateSpec, parent: CertificateSpec | None) -> None:
    if child.parent_certificate_id and child.parent_certificate_id == child.certificate_id:
        raise LineageError("parent cycle: certificate cannot parent itself")
    if parent is None:
        if child.parent_certificate_id is not None:
            raise LineageError("parent_certificate_id set but parent spec is missing")
        if child.evidence_revision != 0:
            raise LineageError("first evidentiary revision must be 0")
        return

    if child.parent_certificate_id != parent.certificate_id:
        raise LineageError("parent_certificate_id does not match parent spec")
    if child.parent_spec_hash != spec_hash(parent):
        raise LineageError("parent_spec_hash does not match parent spec")
    if parent.parent_certificate_id == child.certificate_id:
        raise LineageError("parent cycle")

    if child.patch_kind == "metadata":
        if child.evidence_revision != parent.evidence_revision:
            raise LineageError("metadata patch must retain the evidence revision")
        if scope_payload(child) != scope_payload(parent):
            raise LineageError("metadata patch changed claim scope")
        if child.alpha != parent.alpha:
            raise LineageError("metadata patch must not change allocated alpha")
        return

    if child.evidence_revision != parent.evidence_revision + 1:
        raise LineageError(
            f"evidentiary child revision {child.evidence_revision} must be "
            f"{parent.evidence_revision + 1}"
        )
    expected = evidentiary_alpha(child.evidence_revision)
    if child.alpha != expected:
        raise LineageError(
            f"evidentiary alpha {child.alpha} does not match allocation {expected}"
        )


def validate_forest(specs: Iterable[CertificateSpec]) -> None:
    items = list(specs)
    by_id = {spec.certificate_id: spec for spec in items}
    if len(by_id) != len(items):
        raise LineageError("duplicate certificate IDs")
    for spec in items:
        parent = by_id.get(spec.parent_certificate_id) if spec.parent_certificate_id else None
        validate_revision(spec, parent)
