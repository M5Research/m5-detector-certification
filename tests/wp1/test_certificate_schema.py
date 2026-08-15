"""Deterministic certificate-v1 schema and canonical hashing."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from certificate.models import CertificateRecord, CertificateSpec, GateResult, json_schema
from certificate.serialize import canonical_dumps, spec_hash


def _spec_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "certificate_id": "btc-q2-successor-r0",
        "claim_id": "btc-vr-q2-serial-dependence",
        "evidence_revision": 0,
        "detector": "vr_cascade_recentered",
        "instrument": "BTCUSDT",
        "venue": "binance_usd_m",
        "target": "positive_short_horizon_serial_dependence",
        "schemes": ["calendar", "volume"],
        "horizon": 2,
        "signal_family": "ar1_injection",
        "null_family": "stationary_block",
        "margins": {"transport": 0.019748},
        "costs": {},
        "required_gates": [
            "instrument",
            "target",
            "size",
            "transport",
            "power",
            "information",
            "value",
        ],
        "preregistration_commit": "deadbeef",
        "alpha": 0.025,
        "protocol_id": "protocol-v5",
        "thresholds": {"size_alpha": 0.025},
    }
    payload.update(overrides)
    return payload


def test_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CertificateSpec.model_validate({**_spec_kwargs(), "unexpected": True})


def test_spec_rejects_duplicate_required_gates() -> None:
    with pytest.raises(ValidationError):
        CertificateSpec.model_validate(
            _spec_kwargs(required_gates=["size", "size", "power"])
        )


def test_spec_rejects_non_finite_margin() -> None:
    with pytest.raises(ValidationError):
        CertificateSpec.model_validate(
            _spec_kwargs(margins={"transport": math.inf})
        )


def test_spec_rejects_alpha_outside_geometric_allocation() -> None:
    with pytest.raises(ValidationError):
        CertificateSpec.model_validate(_spec_kwargs(alpha=0.05))


def test_canonical_json_is_sorted_and_rejects_nan() -> None:
    dumped = canonical_dumps({"b": 1, "a": {"z": 2, "y": 3}})
    assert dumped == '{"a":{"y":3,"z":2},"b":1}'
    with pytest.raises(ValueError):
        canonical_dumps({"x": math.nan})


def test_spec_hash_is_deterministic_and_format_invariant() -> None:
    left = CertificateSpec.model_validate(_spec_kwargs())
    right = CertificateSpec.model_validate(
        json.loads(left.model_dump_json())
    )
    assert spec_hash(left) == spec_hash(right)
    assert len(spec_hash(left)) == 64


def test_key_order_does_not_change_spec_hash() -> None:
    a = CertificateSpec.model_validate(_spec_kwargs())
    reordered = _spec_kwargs()
    reordered = {
        "alpha": reordered["alpha"],
        "certificate_id": reordered["certificate_id"],
        **{k: v for k, v in reordered.items() if k not in {"alpha", "certificate_id"}},
    }
    b = CertificateSpec.model_validate(reordered)
    assert spec_hash(a) == spec_hash(b)


def test_gate_result_requires_invalid_reason_for_nonfinite_estimand() -> None:
    with pytest.raises(ValidationError):
        GateResult(
            gate="information",
            state="pass",
            estimand=math.nan,
            threshold=0.0,
            uncertainty={},
            artifacts=[],
            reason="looks fine",
        )
    ok = GateResult(
        gate="information",
        state="invalid",
        estimand=math.nan,
        threshold=0.0,
        uncertainty={},
        artifacts=[],
        reason="unstable_reference",
    )
    assert ok.state == "invalid"
    assert ok.reason == "unstable_reference"


def test_exported_json_schema_is_committed_and_stable() -> None:
    schema = json_schema()
    assert schema["title"] == "CertificateRecord"
    committed = Path("schemas/certificate-v1.json")
    assert committed.is_file()
    assert json.loads(committed.read_text(encoding="utf-8")) == schema
    blob = json.dumps(schema)
    assert "permutation_p" not in blob


def test_record_requires_code_data_and_upstream_hashes() -> None:
    spec = CertificateSpec.model_validate(_spec_kwargs())
    with pytest.raises(ValidationError):
        CertificateRecord(
            spec=spec,
            spec_hash="a" * 64,
            gates=[],
            completeness=False,
            disposition="incomplete",
        )
