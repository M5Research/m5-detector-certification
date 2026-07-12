"""One-time 2026 holdout confirmatory check for the calibrated-detector paper."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1 import vr_significance  # noqa: E402
from scripts.wp1.precheck_b import _gate_guard  # noqa: E402

HOLDOUT_W = 120
HOLDOUT_Q = 5
FROZEN_DELTA_STAR_LOWER_BOUND = 0.10
HOLDOUT_START_MS = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
HOLDOUT_END_MS = int(datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp() * 1000)
MANUSCRIPT_DIR = PROJECT_ROOT / "docs" / "research" / "calibrated-detector-exclusion"
JEF_DRAFT_MARKER = MANUSCRIPT_DIR / ".jef_draft_complete"
HOLDOUT_DIR = PROJECT_ROOT / "backtest_results" / "holdout"

__all__ = [
    "HOLDOUT_Q",
    "HOLDOUT_W",
    "assert_holdout_gate",
    "compute_holdout_primary",
    "validate_holdout_timestamps",
]


def assert_holdout_gate(
    *,
    confirm_jef_draft: bool,
    marker_path: Path = JEF_DRAFT_MARKER,
) -> None:
    """Require explicit CLI confirmation and the JEF draft-complete marker."""
    if not confirm_jef_draft:
        raise PermissionError("Holdout requires --confirm-jef-draft")
    if not marker_path.exists():
        raise FileNotFoundError(f"JEF draft marker not found: {marker_path}")


def validate_holdout_timestamps(timestamps: np.ndarray) -> dict:
    """Require a nonempty timestamp array entirely inside calendar year 2026."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or len(ts) == 0:
        raise ValueError("timestamps must be a nonempty 1-D array")
    if int(np.min(ts)) < HOLDOUT_START_MS:
        raise ValueError("holdout driver received pre-2026 data")
    if int(np.max(ts)) > HOLDOUT_END_MS:
        raise ValueError("holdout driver received post-2026 data")
    start = datetime.fromtimestamp(int(ts[0]) / 1000, tz=UTC)
    end = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=UTC)
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "n_bars": int(len(ts)),
        "year_2026_loaded": True,
    }


def _all_bars_non_overlapping(series: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(series))
    mask = ((idx - (stride - 1)) % stride == 0) & (idx >= stride - 1)
    sample_mask = mask & np.isfinite(series)
    retained_idx = idx[sample_mask]
    return np.asarray(series[sample_mask], dtype=np.float64), retained_idx


def compute_holdout_primary(close: np.ndarray, timestamps: np.ndarray) -> dict:
    """Compute the primary-cell holdout median |VR(5)-1| and z summary."""
    close_arr = np.asarray(close, dtype=np.float64)
    ts_arr = np.asarray(timestamps, dtype=np.int64)
    if close_arr.ndim != 1 or len(close_arr) != len(ts_arr):
        raise ValueError("close and timestamps must be same-length 1-D arrays")
    validate_holdout_timestamps(ts_arr)

    vr_arr, z_arr = vr_significance.compute_rolling_vr_and_z(
        close_arr, W=HOLDOUT_W, q=HOLDOUT_Q
    )
    pred_nl, retained_idx = _all_bars_non_overlapping(vr_arr, stride=HOLDOUT_W)
    z_nl = z_arr[retained_idx]
    sig = vr_significance.compute_vr_significance(pred_nl, z_nl)
    return {
        "W": HOLDOUT_W,
        "q": HOLDOUT_Q,
        "n_nl": int(sig["n_nl"]),
        "observed_vr_dep": float(sig["median_vr_dep"]),
        "mean_vr_dep": float(sig["mean_vr_dep"]),
        "median_z_m2": float(sig["median_z_m2"]),
        "p_twotailed": float(sig["p_twotailed"]),
        "closed": bool(sig["closed"]),
        "frozen_delta_star_lower_bound": FROZEN_DELTA_STAR_LOWER_BOUND,
        "comparison": (
            "observed_exceeds_frozen_lower_bound"
            if float(sig["median_vr_dep"]) > FROZEN_DELTA_STAR_LOWER_BOUND
            else "observed_within_frozen_confirmatory_grid"
        ),
    }


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def run_holdout(*, confirm_jef_draft: bool, marker_path: Path = JEF_DRAFT_MARKER) -> Path:
    """Load 2026 BTCUSDT data, compute the primary holdout check, and write JSON."""
    from scripts.wp1.py_engine import load_and_clean  # noqa: PLC0415

    assert_holdout_gate(confirm_jef_draft=confirm_jef_draft, marker_path=marker_path)
    prereg_results = _gate_guard()
    prereg_commit = next(iter(prereg_results.values()), "")

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    ohlcv = load_and_clean(
        data_path=data_path,
        symbol="BTCUSDT",
        start_ms=HOLDOUT_START_MS,
        end_ms=HOLDOUT_END_MS,
        max_gap_allowed_mins=60,
    )
    data_span = validate_holdout_timestamps(ohlcv["timestamp"])
    primary = compute_holdout_primary(ohlcv["close"], ohlcv["timestamp"])

    run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id,
        "holdout_confirmatory": True,
        "year_2026_loaded": True,
        "data_span": data_span,
        "primary_cell": primary,
        "provenance": {
            "prereg_commit": prereg_commit,
            "code_commit": _code_commit(),
            "jef_draft_marker": str(marker_path),
            "run_utc": datetime.now(tz=UTC).isoformat(),
        },
    }

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HOLDOUT_DIR / f"holdout_confirmatory_{run_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out_path}")
    print(
        "Primary holdout: "
        f"median |VR(5)-1|={primary['observed_vr_dep']:.4f}, "
        f"median z={primary['median_z_m2']:.4f}, "
        f"comparison={primary['comparison']}"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-jef-draft", action="store_true")
    args = parser.parse_args()
    run_holdout(confirm_jef_draft=args.confirm_jef_draft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
