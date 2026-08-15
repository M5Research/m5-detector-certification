"""Certificate-grade stationary-block information for a frozen BTC VR trigger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import scripts._bootstrap  # noqa: F401

from certificate.information import DEFAULT_N_BOOT, stationary_block_information
from certificate.models import CertificateSpec
from certificate.serialize import spec_hash
from scripts.wp1.harmonized_benchmark import (
    _load_btc_window,
    forward_returns_from_close,
    vr_holm_trigger_labels,
)


def _calibrated_e_value(p_value: float) -> float:
    return float(0.5 * p_value**-0.5)


def stationary_trigger_information(
    labels: np.ndarray,
    target: np.ndarray,
    *,
    spec_hash: str,
    alpha: float,
    n_boot: int = DEFAULT_N_BOOT,
    gate_label: str,
) -> dict[str, Any]:
    result = stationary_block_information(
        labels,
        target,
        n_boot=n_boot,
        spec_hash=spec_hash,
        gate_label=gate_label,
        alpha=alpha,
    )
    threshold = float(1.0 / alpha)
    primary_e = _calibrated_e_value(float(result["p_value"]))
    half_e = _calibrated_e_value(float(result["sensitivity"]["half"]["p_value"]))
    double_e = _calibrated_e_value(float(result["sensitivity"]["double"]["p_value"]))
    conclusions = [value >= threshold for value in (primary_e, half_e, double_e)]
    if len(set(conclusions)) > 1:
        gate_state = "invalid"
        reason = "unstable_reference"
    elif conclusions[0]:
        gate_state = "pass"
        reason = "stationary-block e-value clears revision threshold"
    else:
        gate_state = "fail"
        reason = "stationary-block e-value below revision threshold"
    result.update(
        {
            "reference": "stationary_block",
            "e_value": primary_e,
            "e_value_threshold": threshold,
            "gate_state": gate_state,
            "reason": reason,
            "sensitivity_e_values": {"half": half_e, "double": double_e},
        }
    )
    assert "permutation_p" not in json.dumps(result)
    return result


def build_artifact(
    spec_path: Path,
    *,
    start: str,
    end: str,
    n_boot: int,
) -> dict[str, Any]:
    spec = CertificateSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    digest = spec_hash(spec)
    close, timestamps, span = _load_btc_window(spec.instrument, start, end)
    trigger = vr_holm_trigger_labels(
        close,
        W=int(spec.thresholds["W"]),
        target_q=spec.horizon,
        stride=int(spec.thresholds["W"]),
        alpha=spec.alpha,
    )
    indices = np.asarray(trigger["indices"], dtype=np.int64)
    labels = np.asarray(trigger["labels"], dtype=np.int8)
    forward = forward_returns_from_close(close, spec.horizon)[indices]
    finite = np.isfinite(forward)
    labels = labels[finite]
    target = (forward[finite] > 0.0).astype(np.int8)
    row = stationary_trigger_information(
        labels,
        target,
        spec_hash=digest,
        alpha=spec.alpha,
        n_boot=n_boot,
        gate_label=f"btc-vr-q{spec.horizon}-information",
    )
    return {
        "schema_version": 1,
        "artifact": "btc_vr_stationary_block_information",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "preregistration_commit": spec.preregistration_commit,
        "spec_hash": digest,
        "data_hashes": {
            "decision_labels": hashlib.sha256(labels.tobytes()).hexdigest(),
            "target_labels": hashlib.sha256(target.tobytes()).hexdigest(),
        },
        "data_span": span,
        "q": spec.horizon,
        "n_decision_epochs": int(len(labels)),
        "result": row,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    artifact = build_artifact(args.spec, start=args.start, end=args.end, n_boot=args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
