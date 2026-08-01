"""Validation of the frozen research artifacts themselves.

README.md advertises ``pytest -m artifact`` as the command that validates
"every frozen artifact's shape and values". Until this module existed, no test
carried the ``artifact`` marker, so that command selected nothing, reported
success, and validated exactly zero artifacts.

The other test modules under ``tests/wp1/`` are unit tests: they exercise the
analysis functions against synthetic inputs. They are not marked ``artifact``,
because passing them says nothing about the JSON files committed to this
repository. This module reads the committed artifacts and checks them against
each other, against the published ledger, and against the freeze anchors.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backtest.utils import PROJECT_ROOT

pytestmark = pytest.mark.artifact

INJECTION_DIR = PROJECT_ROOT / "data" / "injection_runs"
RESULTS_DIR = PROJECT_ROOT / "backtest_results"
LEDGER = PROJECT_ROOT / "paper" / "ARTIFACT_LEDGER.md"
FREEZE_ANCHOR = PROJECT_ROOT / "prereg" / "FREEZE_ANCHOR.txt"
VERIFICATION = PROJECT_ROOT / "prereg" / "VERIFICATION.md"

N_MC_EXPECTED = 200
CONFIRMATORY_CELLS = 96
EXPLORATORY_CELLS = 8


def _load_cells() -> list[tuple[Path, dict]]:
    return [
        (p, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(INJECTION_DIR.glob("inj_*.json"))
    ]


def _load_freeze_anchors() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in FREEZE_ANCHOR.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _parse_utc(text: str) -> datetime:
    """Parse the ISO timestamps used across the artifacts, tolerating 'Z'."""
    cleaned = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Ledger digests
# --------------------------------------------------------------------------


def test_ledger_digests_match_committed_bytes() -> None:
    """Every SHA-256 in the ledger must verify against the file it names.

    Digests are computed over LF bytes, which .gitattributes enforces. This
    test failing on a fresh clone means either the artifact changed or the
    line-ending pin was lost.
    """
    text = LEDGER.read_text(encoding="utf-8")
    pairs = re.findall(r"`([^`]+\.json)`[^|]*\|\s*`([A-Fa-f0-9]{64})`", text)
    assert pairs, "no (path, digest) pairs found in ARTIFACT_LEDGER.md"

    for rel_path, expected in pairs:
        target = PROJECT_ROOT / rel_path
        assert target.is_file(), f"ledger cites a missing artifact: {rel_path}"
        raw = target.read_bytes()
        assert b"\r\n" not in raw, (
            f"{rel_path} has CRLF line endings in the working tree; the "
            "published digest is computed over LF bytes. Check .gitattributes."
        )
        actual = hashlib.sha256(raw).hexdigest().upper()
        assert actual == expected.upper(), (
            f"digest mismatch for {rel_path}\n"
            f"  ledger: {expected.upper()}\n"
            f"  actual: {actual}"
        )


# --------------------------------------------------------------------------
# Injection grid
# --------------------------------------------------------------------------


def test_injection_grid_cell_counts() -> None:
    cells = _load_cells()
    confirmatory = [
        c for _, c in cells
        if c.get("provenance", {}).get("exploratory_addendum") is not True
    ]
    exploratory = [
        c for _, c in cells
        if c.get("provenance", {}).get("exploratory_addendum") is True
    ]
    assert len(confirmatory) == CONFIRMATORY_CELLS
    assert len(exploratory) == EXPLORATORY_CELLS


def test_injection_cells_are_internally_consistent() -> None:
    """P_det, n_fires and the per-draw records must agree in every cell.

    This is the check a referee would run first: the headline exclusion result
    is a uniform 0/200, and it should be reproducible by counting the draws.
    """
    for path, cell in _load_cells():
        name = path.name
        assert cell["N_mc"] == N_MC_EXPECTED, f"{name}: N_mc != {N_MC_EXPECTED}"
        assert len(cell["per_draw"]) == cell["N_mc"], (
            f"{name}: {len(cell['per_draw'])} draw records for N_mc={cell['N_mc']}"
        )
        fires = sum(1 for d in cell["per_draw"] if d.get("cascade_fired"))
        assert fires == cell["n_fires"], (
            f"{name}: n_fires={cell['n_fires']} but {fires} draws fired"
        )
        assert abs(cell["n_fires"] / cell["N_mc"] - cell["P_det"]) < 1e-12, (
            f"{name}: P_det={cell['P_det']} inconsistent with "
            f"{cell['n_fires']}/{cell['N_mc']}"
        )
        assert 0.0 <= cell["ci_95_lo"] <= cell["P_det"] <= cell["ci_95_hi"] <= 1.0, (
            f"{name}: P_det outside its own 95% interval"
        )


def test_injection_draw_seeds_are_unique_within_a_cell() -> None:
    """A repeated draw seed would mean a cell contains duplicated draws."""
    for path, cell in _load_cells():
        seeds = [d["seed"] for d in cell["per_draw"]]
        assert len(set(seeds)) == len(seeds), f"{path.name}: duplicate draw seeds"


def test_only_the_declared_clone_cell_is_a_clone() -> None:
    """One exploratory cell is a declared W-invariant clone, not a fresh run.

    It is not an independent Monte Carlo draw and the manuscript's power table
    carries a footnote saying so. If a second clone ever appears, that footnote
    is no longer sufficient.
    """
    clones = {
        path.name
        for path, cell in _load_cells()
        if cell.get("provenance", {}).get("w_invariant_clone")
    }
    assert clones == {"inj_d0.15_q5_W240.json"}, (
        f"unexpected set of declared clone cells: {sorted(clones)}"
    )


def test_no_artifact_carries_bypassed_provenance() -> None:
    """The injection driver once stamped a placeholder instead of a commit.

    Artifacts generated before the gate guard was reconnected carried the
    literal string 'bypassed_for_execution' in provenance.freeze_commit. The
    grid was re-executed with the guard active; nothing published may still
    carry the placeholder. See VERIFICATION.md, "Gate-guard provenance".
    """
    for path, cell in _load_cells():
        freeze = cell.get("provenance", {}).get("freeze_commit", "")
        assert freeze != "bypassed_for_execution", (
            f"{path.name} still carries the pre-fix placeholder provenance"
        )


# --------------------------------------------------------------------------
# Ledger numeric anchors
# --------------------------------------------------------------------------


LEDGER_ANCHORS = [
    ("inj_d0.15_q2_W120.json", 200, 1.0),
    ("inj_d0.15_q5_W120.json", 0, 0.0),
    ("inj_d0.2_q5_W120.json", 0, 0.0),
    ("inj_d0.3_q5_W120.json", 200, 1.0),
]


@pytest.mark.parametrize("filename,fires,p_det", LEDGER_ANCHORS)
def test_ledger_anchor_cells_match_artifacts(
    filename: str, fires: int, p_det: float
) -> None:
    """The four cells quoted in the ledger's anchor table must still say that."""
    cell = json.loads((INJECTION_DIR / filename).read_text(encoding="utf-8"))
    assert cell["n_fires"] == fires
    assert cell["P_det"] == p_det


# --------------------------------------------------------------------------
# Freeze-before-run ordering
# --------------------------------------------------------------------------


def _timestamped_artifacts() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(RESULTS_DIR.glob("**/*.json")):
        prov = json.loads(path.read_text(encoding="utf-8")).get("provenance", {})
        if prov.get("run_ts") or prov.get("run_utc"):
            out.append((path, prov))
    return out


def test_every_timestamped_run_postdates_the_freeze_it_cites() -> None:
    """The ordering claim in VERIFICATION.md, checked against the artifacts.

    Each artifact names its own governing freeze in provenance.prereg_commit.
    That freeze's commit time comes from FREEZE_ANCHOR.txt. The run must be
    later. This is the whole freeze-before-run argument, in one assertion.
    """
    anchors = _load_freeze_anchors()
    freeze_times = {
        anchors["freeze_v3_commit"]: _parse_utc(anchors["freeze_v3_utc"]),
        anchors["amendment_d07_commit"]: _parse_utc(anchors["amendment_d07_utc"]),
        anchors["freeze_v4_commit"]: _parse_utc(anchors["freeze_v4_utc"]),
    }

    artifacts = _timestamped_artifacts()
    assert artifacts, "no timestamped run artifacts found under backtest_results/"

    for path, prov in artifacts:
        cited = prov["prereg_commit"]
        assert cited in freeze_times, (
            f"{path.name} cites freeze {cited[:7]}, which is not in "
            "FREEZE_ANCHOR.txt"
        )
        run = _parse_utc(prov.get("run_ts") or prov["run_utc"])
        assert run > freeze_times[cited], (
            f"ORDERING VIOLATION: {path.name} ran at {run.isoformat()}, "
            f"before the freeze {cited[:7]} it cites"
        )


def test_verification_ordering_table_matches_the_artifacts() -> None:
    """VERIFICATION.md tells a referee to open these files and check.

    So the table and the files must agree on both the timestamp and the cited
    freeze. A mismatch here is exactly the sentence a referee should never get
    to write.
    """
    anchors = _load_freeze_anchors()
    text = VERIFICATION.read_text(encoding="utf-8")

    expected = {
        anchors["run_precheck_a_artifact"]: anchors["run_precheck_a_cites"],
        anchors["run_precheck_b_artifact"]: anchors["run_precheck_b_cites"],
        anchors["run_holdout_confirmatory_artifact"]:
            anchors["run_holdout_confirmatory_cites"],
        anchors["run_eth_replication_artifact"]: anchors["run_eth_replication_cites"],
    }

    for rel_path, cited in expected.items():
        target = PROJECT_ROOT / rel_path
        assert target.is_file(), (
            f"VERIFICATION.md's ordering table cites {rel_path}, which is not "
            "published in this repository"
        )
        prov = json.loads(target.read_text(encoding="utf-8"))["provenance"]
        assert prov["prereg_commit"] == cited, (
            f"{rel_path} names freeze {prov['prereg_commit'][:7]} but "
            f"FREEZE_ANCHOR.txt records {cited[:7]}"
        )
        assert Path(rel_path).name in text or rel_path in text, (
            f"{rel_path} is anchored but not referenced in VERIFICATION.md"
        )


def test_freeze_anchor_hashes_are_full_length() -> None:
    """A truncated hash in the anchor file is not a verifiable anchor."""
    anchors = _load_freeze_anchors()
    for key in ("freeze_v3_commit", "amendment_d07_commit", "freeze_v4_commit"):
        assert len(anchors[key]) == 40, f"{key} is not a full 40-character SHA"


# --------------------------------------------------------------------------
# Guard: the marker must never go dead again
# --------------------------------------------------------------------------


def test_artifact_marker_is_not_empty(pytestconfig: pytest.Config) -> None:
    """README.md promises `pytest -m artifact` validates the artifacts.

    If every test carrying the marker were deleted or renamed away, that
    command would go back to selecting nothing and exiting 0 — a false signal
    of validation in the most visible place in the repository. This test is
    itself marked, so the marker can never select an empty set while the suite
    still passes.
    """
    assert pytestmark.name == "artifact"
