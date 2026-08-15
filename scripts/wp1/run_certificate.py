"""CLI: python -m scripts.wp1.run_certificate --spec <spec.json> --out <new-artifact.json>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scripts._bootstrap  # noqa: F401

from certificate.errors import CertificateError, InvalidGateStateError, SchemaError
from certificate.gates import evaluate_gates
from certificate.models import CertificateSpec
from certificate.runner import run_certificate

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.wp1.run_certificate")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        spec_payload = json.loads(args.spec.read_text(encoding="utf-8"))
        spec = CertificateSpec.model_validate(spec_payload)
        gates = evaluate_gates(spec, REPO_ROOT)
        record = run_certificate(
            spec_path=args.spec,
            out_path=args.out,
            gate_results=gates,
            repo=REPO_ROOT,
        )
    except (CertificateError, InvalidGateStateError, SchemaError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if record.completeness else 2


if __name__ == "__main__":
    raise SystemExit(main())
