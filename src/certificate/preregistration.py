"""Preregistration chronology and working-tree guards."""
from __future__ import annotations

import subprocess
from pathlib import Path

from certificate.errors import PreregistrationError
from certificate.models import CertificateSpec


def assert_preregistration(
    spec: CertificateSpec,
    *,
    ok: bool | None = None,
    repo: Path | None = None,
    spec_path: Path | None = None,
) -> None:
    """Refuse missing, dirty, or spec-inconsistent preregistration state.

    Tests may inject ``ok=False`` without touching git. Production callers
    leave ``ok=None`` to inspect the repository.
    """
    if ok is False:
        raise PreregistrationError(
            "preregistration commit is absent, dirty, later amended without "
            "disclosure, or inconsistent with the spec hash"
        )
    if ok is True:
        if not spec.preregistration_commit:
            raise PreregistrationError("preregistration commit is missing from the spec")
        return
    if repo is None:
        raise PreregistrationError(
            "repository path is required for live preregistration checks"
        )
    if spec_path is None:
        raise PreregistrationError("spec path is required for live preregistration checks")
    _assert_live(spec, repo, spec_path)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_live(spec: CertificateSpec, repo: Path, spec_path: Path) -> None:
    commit = spec.preregistration_commit
    show = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if show.returncode != 0:
        raise PreregistrationError(f"preregistration commit {commit} is absent")
    ancestor = _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    if ancestor.returncode != 0:
        raise PreregistrationError(f"preregistration commit {commit} is not an ancestor of HEAD")

    dirty = _git(repo, "status", "--porcelain")
    if dirty.stdout.strip():
        raise PreregistrationError(
            "working tree is dirty; evidence runners require a frozen commit"
        )

    resolved_repo = repo.resolve()
    resolved_spec = spec_path.resolve()
    try:
        relative = resolved_spec.relative_to(resolved_repo).as_posix()
    except ValueError as exc:
        raise PreregistrationError("spec path is outside the repository") from exc

    additions = _git(repo, "log", "--diff-filter=A", "--format=%H", "--follow", "--", relative)
    commits = [line for line in additions.stdout.splitlines() if line.strip()]
    if not commits:
        raise PreregistrationError("spec has no committed preregistration snapshot")
    frozen_spec_commit = commits[-1]
    chronology = _git(repo, "merge-base", "--is-ancestor", commit, frozen_spec_commit)
    if chronology.returncode != 0:
        raise PreregistrationError("spec predates its governing preregistration")
    frozen_blob = _git(repo, "show", f"{frozen_spec_commit}:{relative}")
    if frozen_blob.returncode != 0:
        raise PreregistrationError("committed preregistration spec cannot be read")
    if resolved_spec.read_text(encoding="utf-8").strip() != frozen_blob.stdout.strip():
        raise PreregistrationError("spec was later amended without disclosure")

    if spec.protocol_id == "protocol-v5":
        protocol = "prereg/PREREGISTRATION-v5.0-amendment.md"
        frozen_protocol = _git(repo, "show", f"{commit}:{protocol}")
        current_protocol = repo / protocol
        if frozen_protocol.returncode != 0 or not current_protocol.is_file():
            raise PreregistrationError("governing protocol v5 amendment is absent")
        if current_protocol.read_text(encoding="utf-8").strip() != frozen_protocol.stdout.strip():
            raise PreregistrationError("governing protocol was later amended without disclosure")
