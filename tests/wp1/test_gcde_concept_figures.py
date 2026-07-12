"""Tests for GCDE conceptual figure generation."""
from __future__ import annotations

from pathlib import Path


def test_gcde_concept_figure_generator_writes_nonempty_pngs(tmp_path: Path) -> None:
    from scripts.wp1.gcde_concept_figures import generate_all

    paths = generate_all(tmp_path)

    names = {path.name for path in paths}
    assert names == {
        "fig00_gcde_pipeline.png",
        "fig05_failure_fingerprint_tree.png",
        "fig06_static_online_gcde.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000
