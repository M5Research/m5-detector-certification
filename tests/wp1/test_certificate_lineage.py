"""Immutable certificate lineage and geometric error spending."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from certificate.errors import LineageError
from certificate.lineage import (
    ALPHA_BUDGET,
    evidentiary_alpha,
    spent_alpha,
    validate_revision,
)
from certificate.models import CertificateSpec
from certificate.serialize import spec_hash


def _spec(**overrides: object) -> CertificateSpec:
    payload: dict[str, object] = {
        "certificate_id": "claim-r0",
        "claim_id": "claim",
        "evidence_revision": 0,
        "detector": "rolling_quantile",
        "instrument": "EURUSD",
        "venue": "histdata",
        "target": "next_hour_rv_exceeds_trailing_75",
        "schemes": ["calendar", "tick_count", "intrinsic_variance"],
        "horizon": 60,
        "signal_family": "six_hour_vol_burst",
        "null_family": "stationary_block",
        "margins": {"disagreement": 0.10, "extreme_disagreement": 0.05},
        "costs": {"spread_bp": 1.0},
        "required_gates": ["instrument", "target", "size", "transport", "power", "information", "value"],
        "preregistration_commit": "abc1234",
        "alpha": 0.025,
        "protocol_id": "protocol-v5",
        "thresholds": {},
    }
    payload.update(overrides)
    return CertificateSpec.model_validate(payload)


def test_revision_zero_alpha_is_half_budget() -> None:
    assert evidentiary_alpha(0) == pytest.approx(0.025)
    assert _spec(evidence_revision=0, alpha=0.025).alpha == 0.025


def test_geometric_alpha_sum_never_exceeds_budget() -> None:
    running = 0.0
    for k in range(0, 40):
        running += evidentiary_alpha(k)
        assert running <= ALPHA_BUDGET
    expected = ALPHA_BUDGET * (1.0 - 2.0 ** (-40))
    assert running == pytest.approx(expected)
    assert expected < ALPHA_BUDGET


def test_metadata_patch_retains_revision_and_consumes_no_budget() -> None:
    parent = _spec(
        certificate_id="claim-r0",
        evidence_revision=0,
        alpha=0.025,
    )
    child = _spec(
        certificate_id="claim-r0-meta",
        evidence_revision=0,
        parent_certificate_id=parent.certificate_id,
        parent_spec_hash=spec_hash(parent),
        patch_kind="metadata",
        alpha=0.025,
    )
    validate_revision(child, parent)
    assert spent_alpha([parent, child]) == pytest.approx(0.025)


def test_evidentiary_child_consumes_next_geometric_slice() -> None:
    parent = _spec(certificate_id="claim-r0", evidence_revision=0, alpha=0.025)
    child = _spec(
        certificate_id="claim-r1",
        evidence_revision=1,
        parent_certificate_id=parent.certificate_id,
        parent_spec_hash=spec_hash(parent),
        patch_kind="evidentiary",
        alpha=0.0125,
        horizon=120,
    )
    validate_revision(child, parent)
    assert spent_alpha([parent, child]) == pytest.approx(0.0375)


def test_scope_change_under_same_evidence_revision_is_rejected() -> None:
    parent = _spec(certificate_id="claim-r0", evidence_revision=0, alpha=0.025)
    child = _spec(
        certificate_id="claim-r0-bad",
        evidence_revision=0,
        parent_certificate_id=parent.certificate_id,
        parent_spec_hash=spec_hash(parent),
        patch_kind="metadata",
        alpha=0.025,
        horizon=120,
    )
    with pytest.raises(LineageError, match="scope"):
        validate_revision(child, parent)


def test_metadata_patch_cannot_change_alpha() -> None:
    with pytest.raises(ValidationError):
        _spec(
            certificate_id="claim-r0-meta",
            evidence_revision=0,
            patch_kind="metadata",
            alpha=0.0125,
        )


def test_parent_cycle_is_rejected() -> None:
    spec = _spec(
        certificate_id="claim-r0",
        parent_certificate_id="claim-r0",
        parent_spec_hash="a" * 64,
    )
    with pytest.raises(LineageError, match="cycle"):
        validate_revision(spec, spec)


def test_parent_spec_hash_must_match_parent() -> None:
    parent = _spec(certificate_id="claim-r0", evidence_revision=0, alpha=0.025)
    child = _spec(
        certificate_id="claim-r1",
        evidence_revision=1,
        alpha=0.0125,
        parent_certificate_id=parent.certificate_id,
        parent_spec_hash="0" * 64,
    )

    with pytest.raises(LineageError, match="parent_spec_hash"):
        validate_revision(child, parent)
