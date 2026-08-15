"""Local metadata capitalization and author-marker hygiene. No network."""
from __future__ import annotations

import json
from pathlib import Path

TITLE = "Certifying Regime Detectors Before Use"
ROOT = Path(__file__).resolve().parents[2]


def test_latex_title_is_exact() -> None:
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert rf"\title{{{TITLE}}}" in tex
    assert "AUTHOR DECISION REQUIRED" not in tex


def test_package_and_zenodo_title_capitalization() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    zenodo = (ROOT / "prereg" / ".zenodo.json").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert TITLE in pyproject
    assert TITLE in zenodo
    assert TITLE in readme


def test_readiness_records_the_six_reviewer_issues() -> None:
    payload = json.loads((ROOT / "paper" / "submission-readiness.json").read_text(encoding="utf-8"))
    assert payload["title"] == TITLE
    assert [item["id"] for item in payload["issues"]] == [1, 2, 3, 4, 5, 6]
