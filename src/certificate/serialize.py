"""Canonical JSON serialization and SHA-256 spec hashing."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from certificate.models import CertificateSpec


def canonical_dumps(obj: Any) -> str:
    """UTF-8 JSON with sorted keys, compact separators, and no NaN/Infinity."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def spec_hash(spec: CertificateSpec) -> str:
    payload = spec.model_dump(mode="json")
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
