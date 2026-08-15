"""Live preregistration chronology and immutable-spec checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from certificate.errors import PreregistrationError
from certificate.models import CertificateSpec
from certificate.preregistration import assert_preregistration


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo_with_frozen_spec(tmp_path: Path) -> tuple[Path, Path, CertificateSpec]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    protocol = repo / "prereg" / "PREREGISTRATION-v5.0-amendment.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    _git(repo, "add", "prereg/PREREGISTRATION-v5.0-amendment.md")
    _git(repo, "commit", "-m", "freeze protocol")
    prereg_commit = _git(repo, "rev-parse", "HEAD")

    payload = {
        "certificate_id": "fixture-r0",
        "claim_id": "fixture-claim",
        "evidence_revision": 0,
        "detector": "fixture",
        "instrument": "BTCUSDT",
        "venue": "binance_usd_m",
        "target": "fixture-target",
        "schemes": ["calendar"],
        "horizon": 2,
        "signal_family": "ar1",
        "null_family": "stationary_block",
        "required_gates": ["instrument"],
        "preregistration_commit": prereg_commit,
        "alpha": 0.025,
    }
    spec_path = repo / "specs" / "fixture.json"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _git(repo, "add", "specs/fixture.json")
    _git(repo, "commit", "-m", "freeze spec")
    return repo, spec_path, CertificateSpec.model_validate(payload)


def test_live_preregistration_accepts_immutable_committed_spec(tmp_path: Path) -> None:
    repo, spec_path, spec = _repo_with_frozen_spec(tmp_path)

    assert_preregistration(spec, repo=repo, spec_path=spec_path)


def test_live_preregistration_rejects_later_committed_spec_amendment(tmp_path: Path) -> None:
    repo, spec_path, spec = _repo_with_frozen_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["target"] = "amended-target"
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _git(repo, "add", "specs/fixture.json")
    _git(repo, "commit", "-m", "amend spec")
    amended = CertificateSpec.model_validate(payload)

    with pytest.raises(PreregistrationError, match="later amended"):
        assert_preregistration(amended, repo=repo, spec_path=spec_path)
