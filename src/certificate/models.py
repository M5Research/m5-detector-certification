"""Pydantic v2 certificate-v1 models. Detector-independent."""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from certificate.alpha import evidentiary_alpha

GateState = Literal["pass", "fail", "not_run", "invalid"]
PatchKind = Literal["evidentiary", "metadata"]

KNOWN_GATE_STATES: frozenset[str] = frozenset({"pass", "fail", "not_run", "invalid"})


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_finite_mapping(name: str, value: dict[str, float]) -> dict[str, float]:
    return {key: _require_finite(f"{name}.{key}", float(val)) for key, val in value.items()}


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: str = Field(min_length=1)
    state: str
    estimand: float | None = None
    threshold: float | None = None
    uncertainty: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("uncertainty")
    @classmethod
    def _finite_uncertainty(cls, value: dict[str, float]) -> dict[str, float]:
        return _require_finite_mapping("uncertainty", value)

    @field_validator("threshold")
    @classmethod
    def _finite_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _require_finite("threshold", value)

    @model_validator(mode="before")
    @classmethod
    def _estimand_and_state(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        estimand = data.get("estimand")
        if estimand is None:
            return data
        try:
            value = float(estimand)
        except (TypeError, ValueError):
            return data
        if math.isfinite(value):
            return data
        state = data.get("state")
        reason = str(data.get("reason") or "").strip()
        if state != "invalid" or not reason:
            raise ValueError(
                "non-finite estimand requires state='invalid' and an explicit reason"
            )
        updated = dict(data)
        updated["estimand"] = None
        return updated


class CertificateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["certificate-v1"] = "certificate-v1"
    certificate_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_revision: int = Field(ge=0)
    parent_certificate_id: str | None = None
    parent_spec_hash: str | None = None
    patch_kind: PatchKind = "evidentiary"
    detector: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    target: str = Field(min_length=1)
    schemes: list[str] = Field(min_length=1)
    horizon: int = Field(gt=0)
    signal_family: str = Field(min_length=1)
    null_family: str = Field(min_length=1)
    margins: dict[str, float] = Field(default_factory=dict)
    costs: dict[str, float] = Field(default_factory=dict)
    required_gates: list[str] = Field(min_length=1)
    preregistration_commit: str = Field(min_length=7)
    alpha: float = Field(gt=0, le=0.05)
    protocol_id: str = "protocol-v5"
    thresholds: dict[str, float] = Field(default_factory=dict)

    @field_validator("margins", "costs", "thresholds")
    @classmethod
    def _finite_maps(cls, value: dict[str, float], info: Any) -> dict[str, float]:
        return _require_finite_mapping(info.field_name, value)

    @field_validator("required_gates")
    @classmethod
    def _unique_gates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("required_gates must not contain duplicates")
        if any(not gate for gate in value):
            raise ValueError("required_gates must be non-empty names")
        return list(value)

    @field_validator("schemes")
    @classmethod
    def _nonempty_schemes(cls, value: list[str]) -> list[str]:
        if any(not scheme for scheme in value):
            raise ValueError("schemes must be non-empty names")
        return list(value)

    @model_validator(mode="after")
    def _alpha_matches_revision(self) -> CertificateSpec:
        expected = evidentiary_alpha(self.evidence_revision)
        if not math.isclose(self.alpha, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"alpha {self.alpha} does not match evidentiary allocation "
                f"{expected} for revision {self.evidence_revision}"
            )
        return self


class CertificateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["certificate-v1"] = "certificate-v1"
    spec: CertificateSpec
    spec_hash: str = Field(min_length=64, max_length=64)
    code_hash: str = Field(min_length=7)
    data_hashes: dict[str, str] = Field(min_length=1)
    upstream_hash: str = Field(min_length=64, max_length=64)
    gates: list[GateResult]
    completeness: bool
    failure_profile: list[dict[str, Any]] = Field(default_factory=list)
    disposition: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    superseded: bool = False
    supersession_status: str | None = None


def json_schema() -> dict[str, Any]:
    return CertificateRecord.model_json_schema()
