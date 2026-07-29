#!/usr/bin/env python3
"""Fail if a developer-machine absolute path appears in published artifacts.

Provenance fields in generated artifacts pick up whatever path the producing
machine used. Those paths leak a contributor's username and local directory
layout into a public repository, and they are not reproducible for anyone
else. Artifacts must carry repository-relative paths instead.

Run locally before committing generated artifacts:

    python scripts/check_no_absolute_paths.py

Exits non-zero and prints every offending file, line, and match.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories holding generated or published artifacts.
SCAN_DIRS = ("backtest_results", "data", "prereg", "paper")

SCAN_SUFFIXES = {".json", ".txt", ".md", ".csv", ".tex", ".yaml", ".yml"}

# Two guards against false positives. The drive letter must not follow another
# alphanumeric, or "https:" matches as drive "s:". And a genuine drive path
# carries a second separator (C:\Users\name), which LaTeX math such as
# "t:\text{admit}" does not.
PATTERNS = (
    # Windows drive paths: C:\Users\..., D:/projects/...
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9._ -]{1,60}[\\/]"),
    # POSIX home directories: /Users/<name>/..., /home/<name>/...
    re.compile(r"/(?:Users|home)/[^/\s\"'\\]+/"),
)


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, matched_text) for every offending line."""
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits

    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                hits.append((lineno, match.group(0)))
                break
    return hits


def main() -> int:
    offenders: list[tuple[Path, int, str]] = []

    for dirname in SCAN_DIRS:
        root = REPO_ROOT / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            for lineno, matched in scan_file(path):
                offenders.append((path.relative_to(REPO_ROOT), lineno, matched))

    if not offenders:
        print(f"OK: no absolute developer paths under {', '.join(SCAN_DIRS)}/")
        return 0

    print("Absolute developer paths found in published artifacts:\n")
    for relpath, lineno, matched in offenders:
        print(f"  {relpath}:{lineno}: {matched}")
    print(
        f"\n{len(offenders)} occurrence(s). Replace each with a "
        "repository-relative path before committing."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
