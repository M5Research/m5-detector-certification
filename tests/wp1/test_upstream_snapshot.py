"""Pinned upstream detector and market-clock snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "provenance" / "upstream-snapshot-v1.json"
EXPECTED = {
    "src/strategies/vol_regime_switch/defaults.py",
    "src/strategies/vol_regime_switch/hmm_detector.py",
    "src/strategies/vol_regime_switch/realized_vol.py",
    "src/strategies/vol_regime_switch/regime_detector.py",
    "src/strategies/vol_regime_switch/regime_engine.py",
    "src/strategies/vol_regime_switch/regime_population.py",
    "src/strategies/vol_regime_switch/rolling_quantile_detector.py",
    "src/strategies/vol_regime_switch/strategy_modules.py",
    "scripts/wp1/gauge_bars.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_upstream_snapshot_is_complete_and_matches_vendored_files() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["source_url"] == "https://github.com/M5Research/VolRegime-Engine.git"
    assert payload["release_tag"] == "jfds-detector-core-v1.0.0"
    assert payload["commit"] == "b6911cf9812e9eafad269a44c92d77a5c95fabc3"
    assert set(payload["files"]) == EXPECTED
    assert all(Path(path).as_posix() == path for path in payload["files"])
    for relative, expected_hash in payload["files"].items():
        assert _sha256(REPO_ROOT / relative) == expected_hash


def test_snapshot_contains_dependency_and_export_provenance() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "upstream-snapshot-v1"
    assert payload["dependencies"]["python"]
    assert payload["dependencies"]["numpy"]
    assert payload["dependencies"]["scipy"]
    assert payload["dependencies"]["statsmodels"]
    assert payload["exported_at"].endswith("Z")
    assert payload["export_tool"] == "scripts/wp1/export_upstream_snapshot.py@v1"
