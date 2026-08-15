"""Detector-independent certificate-v1 core."""

from certificate.alpha import ALPHA_BUDGET, evidentiary_alpha
from certificate.disposition import DISPOSITION_PRECEDENCE, classify_disposition
from certificate.lineage import spent_alpha, validate_revision
from certificate.models import CertificateRecord, CertificateSpec, GateResult, json_schema
from certificate.runner import run_certificate
from certificate.serialize import canonical_dumps, spec_hash

__all__ = [
    "ALPHA_BUDGET",
    "CertificateRecord",
    "CertificateSpec",
    "DISPOSITION_PRECEDENCE",
    "GateResult",
    "canonical_dumps",
    "classify_disposition",
    "evidentiary_alpha",
    "json_schema",
    "run_certificate",
    "spec_hash",
    "spent_alpha",
    "validate_revision",
]
