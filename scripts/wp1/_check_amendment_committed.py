"""Guard script: assert that the §14.5 HMM amendment bundle is present in
01-PREREGISTRATION.md AND that the commit containing it has been recorded in git.

Run BEFORE validate_hmm_detector.py as the no-HARKing ordering gate:

    .venv\Scripts\python.exe scripts/wp1/_check_amendment_committed.py

Exits 0 and prints OK on success.
Exits 1 and prints a failure description otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREREG_PATH = (
    _REPO_ROOT
    / ".planning"
    / "phases"
    / "01-detector-spec-diagnosis-pre-registration"
    / "01-PREREGISTRATION.md"
)

# Marker strings that MUST be present in the pre-registration after the amendment
# is appended.  All four must be present simultaneously.
_REQUIRED_MARKERS = [
    "14.5 HMM Amendment",
    "No-peeking",
    "169fc20",
    "D-02",
    "D-03",
    "D-06",
]


def _check_markers() -> list[str]:
    """Return list of markers NOT found in the pre-registration."""
    try:
        text = _PREREG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"FILE NOT FOUND: {_PREREG_PATH}"]
    return [m for m in _REQUIRED_MARKERS if m not in text]


def _check_git_commit() -> str | None:
    """Return None if the amendment commit is found; return error string otherwise."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1", "--", str(_PREREG_PATH)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return f"git log failed: {exc}"

    if result.returncode != 0:
        return f"git log returned non-zero ({result.returncode}): {result.stderr.strip()}"

    subject = result.stdout.strip()
    if not subject:
        return (
            "git log returned empty output — "
            f"{_PREREG_PATH.name} may not be committed yet."
        )

    # The commit subject must contain "14.5 hmm amendment" (case-insensitive)
    if "14.5 hmm amendment" not in subject.lower():
        return (
            f"Most recent commit for {_PREREG_PATH.name} does not appear to be "
            f"the HMM amendment commit.\n"
            f"  Found: {subject}\n"
            f"  Expected subject containing: '14.5 hmm amendment' (case-insensitive)"
        )
    return None


def main() -> int:
    """Run both checks; exit 0 on full success, 1 on any failure."""
    errors: list[str] = []

    # 1. Marker check
    missing = _check_markers()
    if missing:
        errors.append(
            "MARKER CHECK FAILED: The following required strings are missing from "
            f"{_PREREG_PATH.name}:\n  " + "\n  ".join(missing)
        )

    # 2. Git commit check
    git_err = _check_git_commit()
    if git_err is not None:
        errors.append(f"GIT COMMIT CHECK FAILED: {git_err}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("OK: §14.5 HMM amendment bundle is present and committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
