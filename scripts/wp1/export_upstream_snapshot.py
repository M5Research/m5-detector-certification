"""Export the pinned shared detector/market-clock snapshot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import scripts._bootstrap  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://github.com/M5Research/VolRegime-Engine.git"
RELEASE_TAG = "jfds-detector-core-v1.0.0"
EXPECTED_COMMIT = "b6911cf9812e9eafad269a44c92d77a5c95fabc3"
SELECTED_FILES = (
    "src/strategies/vol_regime_switch/defaults.py",
    "src/strategies/vol_regime_switch/hmm_detector.py",
    "src/strategies/vol_regime_switch/realized_vol.py",
    "src/strategies/vol_regime_switch/regime_detector.py",
    "src/strategies/vol_regime_switch/regime_engine.py",
    "src/strategies/vol_regime_switch/regime_population.py",
    "src/strategies/vol_regime_switch/rolling_quantile_detector.py",
    "src/strategies/vol_regime_switch/strategy_modules.py",
    "scripts/wp1/gauge_bars.py",
)


def _run(*args: str, cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="strict").strip()


def _blob(engine_repo: Path, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{RELEASE_TAG}:{relative}"],
        cwd=engine_repo,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _dependency_versions(engine_python: Path) -> dict[str, str]:
    code = (
        "import json,platform,numpy,scipy,statsmodels;"
        "print(json.dumps({'python':platform.python_version(),"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'statsmodels':statsmodels.__version__},sort_keys=True))"
    )
    output = _run(str(engine_python), "-c", code, cwd=REPO_ROOT)
    return json.loads(output)


def build_manifest(engine_repo: Path, engine_python: Path) -> dict[str, object]:
    commit = _run("git", "rev-list", "-n", "1", RELEASE_TAG, cwd=engine_repo)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"release tag resolves to {commit}, expected {EXPECTED_COMMIT}")
    remote = _run(
        "git",
        "ls-remote",
        "--tags",
        "origin",
        f"refs/tags/{RELEASE_TAG}^{{}}",
        cwd=engine_repo,
    )
    if not remote.startswith(EXPECTED_COMMIT):
        raise RuntimeError("published release tag does not resolve to the expected commit")

    hashes: dict[str, str] = {}
    for relative in SELECTED_FILES:
        upstream = _blob(engine_repo, relative)
        local = (REPO_ROOT / relative).read_bytes()
        if local != upstream:
            raise RuntimeError(f"vendored file diverges byte-for-byte: {relative}")
        hashes[relative] = hashlib.sha256(upstream).hexdigest()

    return {
        "schema_version": "upstream-snapshot-v1",
        "source_url": SOURCE_URL,
        "release_tag": RELEASE_TAG,
        "commit": commit,
        "files": hashes,
        "dependencies": _dependency_versions(engine_python),
        "exported_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "export_tool": "scripts/wp1/export_upstream_snapshot.py@v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-repo", required=True, type=Path)
    parser.add_argument("--engine-python", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "provenance" / "upstream-snapshot-v1.json",
    )
    args = parser.parse_args(argv)
    payload = build_manifest(args.engine_repo.resolve(), args.engine_python.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
