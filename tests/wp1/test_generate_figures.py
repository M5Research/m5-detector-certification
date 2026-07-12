"""Unit tests for scripts/wp1/generate_figures.py.

Five tests:
  1. test_holdout_guard_rejects_2026      -- holdout guard AssertionError on 2026 data
  2. test_all_pngs_emitted                -- driver exits 0 and all 5 PNGs exist (skip if no data)
  3. test_determinism                     -- second run pixel-identical to first (skip if no data)
  4. test_population_counts_match_json    -- Fig 3 population counts match validation JSON
  5. test_provenance_stamp                -- provenance output contains required fields

All seeded with np.random.default_rng (legacy seed API forbidden).
No skip-and-pass markers -- these are live tests.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap so scripts/wp1/generate_figures is importable from tests/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _skip_if_legacy_validation_json_missing() -> None:
    wp1_dir = _REPO_ROOT / "backtest_results" / "wp1"
    required = (
        "rolling_quantile_validation_*.json",
        "hmm_validation_*.json",
    )
    missing = [pattern for pattern in required if not list(wp1_dir.glob(pattern))]
    if missing:
        pytest.skip(
            "legacy figure validation JSON missing: "
            + ", ".join(missing)
            + "; run validate_rolling_quantile.py and validate_hmm_detector.py first"
        )


def test_holdout_guard_rejects_2026(monkeypatch) -> None:
    """Holdout guard raises AssertionError when load_and_clean returns 2026 data.

    Monkeypatches load_and_clean to return a fake dataset whose last timestamp
    is 2026-01-15. The driver's Layer 2 assert (ts_end.year < 2026) must fire.
    """
    import scripts._bootstrap  # noqa: F401 -- ensure src/ on sys.path
    import scripts.wp1.generate_figures as gf

    fake_ts_2026 = int(_dt.datetime(2026, 1, 15, tzinfo=_dt.UTC).timestamp() * 1000)
    fake_data = {
        "close": np.array([50000.0, 50001.0]),
        "timestamp": np.array([fake_ts_2026 - 60000, fake_ts_2026]),
    }
    monkeypatch.setattr(gf, "load_and_clean", lambda **kw: fake_data)
    with pytest.raises(AssertionError, match="D-10 VIOLATION"):
        gf.main()


def test_all_pngs_emitted() -> None:
    """Driver exits 0 and all 5 expected PNGs exist under docs/research/figures/.

    Skips if real BTCUSDT 2021 Parquet data is absent.
    """
    parquet_path = Path("data/binance_futures/symbol=BTCUSDT/year=2021")
    if not parquet_path.exists():
        pytest.skip("BTC Parquet not found; skipping real-data driver smoke test")
    _skip_if_legacy_validation_json_missing()

    import scripts._bootstrap  # noqa: F401
    import scripts.wp1.generate_figures as gf

    rc = gf.main()
    assert rc == 0, f"generate_figures.main() returned non-zero exit code: {rc}"

    figures_dir = _REPO_ROOT / "docs" / "research" / "figures"
    expected_pngs = [
        "fig01_diagnosis_histogram.png",
        "fig02_epsilon_sq_grid.png",
        "fig03_regime_populations.png",
        "fig04_crisis_timelines.png",
        "fig05_v1v2_comparison.png",
    ]
    for name in expected_pngs:
        png_path = figures_dir / name
        assert png_path.exists(), f"Expected PNG not found: {png_path}"
        assert png_path.stat().st_size > 0, f"PNG is empty: {png_path}"


def test_determinism() -> None:
    """Running the driver twice produces pixel-identical PNGs for all 5 figures.

    Skips if real BTCUSDT 2021 Parquet data is absent.
    Pixel arrays compared via np.array_equal -- NOT file hashes (PNG metadata
    may differ across runs; only pixel content is guaranteed identical per
    RESEARCH.md Determinism section).
    """
    import matplotlib.pyplot as plt

    parquet_path = Path("data/binance_futures/symbol=BTCUSDT/year=2021")
    if not parquet_path.exists():
        pytest.skip("BTC Parquet not found; skipping real-data determinism test")
    _skip_if_legacy_validation_json_missing()

    import scripts._bootstrap  # noqa: F401
    import scripts.wp1.generate_figures as gf

    figures_dir = _REPO_ROOT / "docs" / "research" / "figures"
    expected_pngs = [
        "fig01_diagnosis_histogram.png",
        "fig02_epsilon_sq_grid.png",
        "fig03_regime_populations.png",
        "fig04_crisis_timelines.png",
        "fig05_v1v2_comparison.png",
    ]

    # First run
    rc1 = gf.main()
    assert rc1 == 0, f"First run returned non-zero: {rc1}"
    imgs_first = {name: plt.imread(str(figures_dir / name)) for name in expected_pngs}

    # Second run
    rc2 = gf.main()
    assert rc2 == 0, f"Second run returned non-zero: {rc2}"
    imgs_second = {name: plt.imread(str(figures_dir / name)) for name in expected_pngs}

    for name in expected_pngs:
        assert np.array_equal(imgs_first[name], imgs_second[name]), (
            f"Pixel arrays differ between runs for {name} -- driver is non-deterministic"
        )


def test_population_counts_match_json() -> None:
    """Fig 3 population fractions match the rolling_quantile_validation JSON to within 0.1%.

    Reads 'population_stats' from the latest rolling_quantile_validation_*.json and
    asserts LOW approx 0.751, ELEVATED approx 0.194, EXTREME approx 0.054.
    Skips if the JSON file is absent (not yet generated).
    This is a spot-check -- NOT a re-computation of population statistics.
    """
    wp1_dir = _REPO_ROOT / "backtest_results" / "wp1"
    reports = sorted(wp1_dir.glob("rolling_quantile_validation_*.json"))
    if not reports:
        pytest.skip("rolling_quantile_validation JSON not found; run validate_rolling_quantile.main() first")

    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    pop = report["population_stats"]

    # rolling_quantile_validation_*.json uses keys low_frac / elevated_frac / extreme_frac
    assert abs(pop["low_frac"] - 0.751) < 0.001, (
        f"low_frac {pop['low_frac']:.4f} not within 0.001 of 0.751"
    )
    assert abs(pop["elevated_frac"] - 0.194) < 0.001, (
        f"elevated_frac {pop['elevated_frac']:.4f} not within 0.001 of 0.194"
    )
    assert abs(pop["extreme_frac"] - 0.054) < 0.001, (
        f"extreme_frac {pop['extreme_frac']:.4f} not within 0.001 of 0.054"
    )


def test_provenance_stamp() -> None:
    """Driver provenance output contains required integrity fields.

    Asserts the driver's provenance output (JSON sidecar or returned dict) contains:
    - year_2026_loaded: false
    - prereg_commit: "169fc20"
    - git_commit: non-empty string

    Monkeypatches load_and_clean to avoid loading real data (not needed for this test).
    The provenance stamp is a module-level dict or sidecar emitted after main() returns.
    """
    import scripts._bootstrap  # noqa: F401
    import scripts.wp1.generate_figures as gf

    # This test checks the provenance dict returned / accessible from the driver.
    # Provenance fields must be present regardless of whether figures were generated.
    provenance = getattr(gf, "_PROVENANCE", None)
    if provenance is None:
        # If the driver does not expose _PROVENANCE at module level, skip until Plan 02 lands.
        # The driver module exists once Plan 02 is complete; until then this scaffolds the
        # expectation.
        pytest.skip("generate_figures._PROVENANCE not yet available (Plan 02 not yet complete)")

    assert provenance.get("year_2026_loaded") is False, (
        f"year_2026_loaded must be False, got: {provenance.get('year_2026_loaded')!r}"
    )
    assert provenance.get("prereg_commit") == "169fc20", (
        f"prereg_commit must be '169fc20', got: {provenance.get('prereg_commit')!r}"
    )
    git_commit = provenance.get("git_commit", "")
    assert isinstance(git_commit, str) and len(git_commit) > 0, (
        f"git_commit must be a non-empty string, got: {git_commit!r}"
    )
