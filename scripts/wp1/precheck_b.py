"""Pre-Check B orchestrator: 1-min M2 VR-significance test on 2021-2025 BTCUSDT.

Implements the frozen Pre-Check B protocol from 07-PREREGISTRATION.md (freeze
commit 720c1d4) by running the HAC-robust M2 z_m2 variance-ratio significance
test across WQ_GRID = {(60,5),(120,5),(240,15)}, q-grid {2,5,15,60}.

Integrity controls enforced here:
  - D-09 gate-guard: 07-PREREGISTRATION.md exists + git commit timestamp
    predates run + prereg_commit stamp in every report. Runs BEFORE any data load.
  - D-07 holdout (two-layer): end_ms < 2026-01-01 boundary (LAYER 1, before load)
    + ts_end.year < 2026 (LAYER 2, after load) + year_2026_loaded: false in report.
  - D-B7: labels regenerated inline via RollingQuantileDetector().fit(close) —
    no cached label artifact.
  - SC1 append-spike causality assertion runs in-process on the real VR per (W,q).
  - All math delegated to scripts.wp1.vr_significance (Wave-1 synthetic-gate-hardened
    module) — no VR/z_m2/median/CI/Holm/horizon/noise-floor/MDE math reimplemented here.
  - D-B10 boundary: Phase 9 stops at the ratified verdict; no camera-ready figure,
    no contextualizes-A synthesis.

Output: backtest_results/precheck/precheck_b_<date>.json

Determinism: np.random.default_rng(seed); legacy seeding API not used.
OMP guard: os.environ.setdefault("OMP_NUM_THREADS", "1") at module level.

D-15 provenance stamp in every emitted JSON:
  freeze_commit: "720c1d4"
  prereg_commit: <resolved at runtime by _gate_guard()>
  boot_seed: 43
  code_commit: <resolved at runtime via subprocess git log>
  (NO phase8_amendment_ref — no D-08 analog; D-B3 default path)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# OMP guard: must be set before any numpy/blas import (Phase-3 D-12)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Path bootstrap: insert repo root + src/ so all packages are importable when
# this file is run as a script (python scripts/wp1/precheck_b.py) or
# imported from tests/ (which adds _REPO_ROOT to sys.path via the test file).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402, I001

import scripts._bootstrap  # noqa: F401, E402
from scripts.wp1.gate_analysis import WQ_GRID  # noqa: E402
import scripts.wp1.vr_significance as vr_significance  # noqa: E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    non_overlapping_samples,
)
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

PRECHECK_DIR = PROJECT_ROOT / "backtest_results" / "precheck"

PREREG_PATH = ".planning/phases/07-pre-registration-freeze/07-PREREGISTRATION.md"
PREREG_PATH_09 = ".planning/phases/11-signal-injection-the-ligo-calibration/09-PREREGISTRATION.md"

FREEZE_COMMIT = "720c1d4"
BOOT_SEED: int = 43
Q_GRID: tuple[int, ...] = (2, 5, 15, 60)
CLOSURE_THRESHOLD: float = 0.001
V2_EPSILON_ANCHOR: float = 0.001691

def _gate_guard(prereg_paths: list[str] | None = None) -> dict[str, str]:
    if prereg_paths is None:
        prereg_paths = [PREREG_PATH, PREREG_PATH_09]

    repo_root = Path(__file__).resolve().parents[2]
    results = {}

    for path in prereg_paths:
        prereg_abs = repo_root / path

        if not prereg_abs.exists():
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: {path} not found. "
                "Phase cannot run without the frozen pre-registration."
            )

        result = subprocess.run(
            ["git", "log", "--format=%ct", str(prereg_abs)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: 'git log --format=%ct' on {path} "
                f"failed (returncode {result.returncode}). stderr: {result.stderr.strip()!r}. "
                "Cannot verify the pre-reg commit predates this run — failing closed."
            )
        commit_timestamps = [int(t) for t in result.stdout.strip().split() if t]
        if not commit_timestamps:
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: {path} has no git commits. "
                "Commit the pre-registration before running the gate."
            )
        latest_prereg_ts = max(commit_timestamps)
        run_ts = int(time.time())
        if latest_prereg_ts >= run_ts:
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: pre-reg commit ({latest_prereg_ts}) for {path} does not "
                f"predate this run ({run_ts}). Integrity violation — the pre-registration "
                "must be committed BEFORE the gate run (no-HARKing control)."
            )

        hash_result = subprocess.run(
            ["git", "log", "--format=%H", "-1", str(prereg_abs)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if hash_result.returncode != 0:
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: 'git log --format=%H' on {path} "
                f"failed (returncode {hash_result.returncode}). stderr: "
                f"{hash_result.stderr.strip()!r}. Cannot obtain the prereg_commit hash — "
                "failing closed rather than stamping an empty provenance hash."
            )
        prereg_commit = hash_result.stdout.strip()
        if not prereg_commit:
            raise SystemExit(
                f"D-09 GATE GUARD FAILED: 'git log --format=%H' returned an empty commit "
                f"hash for {path}. Refusing to stamp an empty prereg_commit "
                "(silent freeze-before-run provenance void) — failing closed."
            )
        results[path] = prereg_commit

    return results

def run_precheck_b_cascade(
    close: np.ndarray,
    timestamps: np.ndarray,
    seed: int = 43,
    *,
    diagnostics: bool = True,
) -> dict:
    regime = RollingQuantileDetector().fit(close)

    n_warmup = int(np.sum(regime < 0))
    n_LOW = int(np.sum(regime == 0))
    n_ELEVATED = int(np.sum(regime == 1))
    n_EXTREME = int(np.sum(regime == 2))
    first_valid_bar = int(np.argmax(regime >= 0)) if np.any(regime >= 0) else -1
    N = len(close)

    label_provenance = {
        "detector": "RollingQuantileDetector",
        "params": {
            "rv_window": 60,
            "pct_window": 43200,
            "p_elevated": 0.75,
            "p_extreme": 0.95,
        },
        "n_warmup": n_warmup,
        "n_LOW": n_LOW,
        "n_ELEVATED": n_ELEVATED,
        "n_EXTREME": n_EXTREME,
        "first_valid_bar": first_valid_bar,
        "row_count": N,
        "span": "omitted_in_cascade",
    }

    vr_series: dict[tuple[int, int], np.ndarray] = {}
    z_series: dict[tuple[int, int], np.ndarray] = {}

    W_primary = 120
    needed_wq: set[tuple[int, int]] = set(WQ_GRID)
    needed_wq.update((W_primary, q) for q in Q_GRID)
    for W, q in sorted(needed_wq):
        vr_arr, z_arr = vr_significance.compute_rolling_vr_and_z_strided(
            close, W=W, q=q, stride=W
        )
        vr_series[(W, q)] = vr_arr
        z_series[(W, q)] = z_arr

    if not diagnostics:
        holm_pvalues: list[float] = []
        closure_table_entries: list[dict] = []

        for q in Q_GRID:
            vr_arr_q = vr_series[(W_primary, q)]
            z_arr_q = z_series[(W_primary, q)]
            pred_nl_q, _regime_nl_q, retained_idx_q = non_overlapping_samples(
                vr_arr_q, regime, stride=W_primary
            )
            z_nl_q = z_arr_q[retained_idx_q]

            sig_q = vr_significance.compute_vr_significance(pred_nl_q, z_nl_q)
            holm_pvalues.append(sig_q["p_twotailed"])
            closure_table_entries.append({
                "q": q,
                "median_vr_dep": sig_q["median_vr_dep"],
                "closed": sig_q["closed"],
                "closure_threshold": CLOSURE_THRESHOLD,
                "p_twotailed": sig_q["p_twotailed"],
                "median_z_m2": sig_q["median_z_m2"],
                "n_nl": sig_q["n_nl"],
            })

        adjusted_pvalues = vr_significance.apply_holm_b(holm_pvalues)
        holm_ordered = []
        order = sorted(range(len(holm_pvalues)), key=lambda i: holm_pvalues[i])
        for rank_0, sort_idx in enumerate(order):
            holm_ordered.append({
                "q": Q_GRID[sort_idx],
                "p_uncorrected": holm_pvalues[sort_idx],
                "holm_adjusted": adjusted_pvalues[sort_idx],
                "rank": rank_0 + 1,
                "median_z_m2": closure_table_entries[sort_idx]["median_z_m2"],
            })

        for i, entry in enumerate(closure_table_entries):
            entry["holm_adjusted"] = adjusted_pvalues[i]

        cascade_fired = any(
            e["holm_adjusted"] < 0.05 and e.get("median_z_m2", 0) > 0
            for e in holm_ordered
        )

        return {
            "holm_correction": {
                "family_size": 4,
                "ordered": holm_ordered,
                "q_grid": list(Q_GRID),
            },
            "closure_table": {
                "closure_threshold": CLOSURE_THRESHOLD,
                "closure_all_q": bool(all(e["closed"] for e in closure_table_entries)),
                "per_q": closure_table_entries,
            },
            "label_provenance": label_provenance,
            "cascade_fired": cascade_fired,
        }

    per_wq: list[dict] = []
    n_nl_primary: int = 0

    for W, q in WQ_GRID:
        vr_arr = vr_series[(W, q)]
        z_arr = z_series[(W, q)]

        pred_nl, regime_nl, retained_idx = non_overlapping_samples(
            vr_arr, regime, stride=W
        )
        n_nl = len(pred_nl)
        is_primary = bool(W == 120 and q == 5)
        if is_primary:
            n_nl_primary = n_nl

        z_nl = z_arr[retained_idx]
        timestamps_nl = timestamps[retained_idx]

        sig = vr_significance.compute_vr_significance(pred_nl, z_nl)
        median_point, ci_lo, ci_hi = vr_significance.median_vr_dep_boot_ci(
            pred_nl, block=W, n_boot=2000, seed=seed
        )
        per_year = vr_significance.compute_per_year_breakdown(pred_nl, timestamps_nl, z_nl)

        power_caveat = None
        if W == 120 and q == 60:
            power_caveat = (
                "W/q = 120/60 = 2 < 10 (Lo-MacKinlay stability criterion violated "
                "at per-window level). Per-window VR(60) is valid-but-noisy; the "
                f"~{n_nl:,}-window population rescues the central estimate but "
                "per-window VR(60) sampling noise is higher than at q=2 or q=5."
            )

        cell: dict = {
            "W": W,
            "q": q,
            "is_primary": is_primary,
            "n_nl": n_nl,
            "median_vr_dep": sig["median_vr_dep"],
            "mean_vr_dep": sig["mean_vr_dep"],
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "ci_block": W,
            "ci_n_boot": 2000,
            "closed": sig["closed"],
            "p_twotailed": sig["p_twotailed"],
            "median_z_m2": sig["median_z_m2"],
            "per_year": per_year,
        }
        if power_caveat is not None:
            cell["power_caveat"] = power_caveat

        per_wq.append(cell)

    holm_pvalues: list[float] = []
    closure_table_entries: list[dict] = []
    per_wq_primary_q: list[dict] = []

    for q in Q_GRID:
        vr_arr_q = vr_series[(W_primary, q)]
        z_arr_q = z_series[(W_primary, q)]
        pred_nl_q, regime_nl_q, retained_idx_q = non_overlapping_samples(
            vr_arr_q, regime, stride=W_primary
        )
        z_nl_q = z_arr_q[retained_idx_q]
        timestamps_nl_q = timestamps[retained_idx_q]

        sig_q = vr_significance.compute_vr_significance(pred_nl_q, z_nl_q)
        median_q, ci_lo_q, ci_hi_q = vr_significance.median_vr_dep_boot_ci(
            pred_nl_q, block=W_primary, n_boot=2000, seed=seed
        )
        per_year_q = vr_significance.compute_per_year_breakdown(pred_nl_q, timestamps_nl_q, z_nl_q)

        holm_pvalues.append(sig_q["p_twotailed"])
        closure_table_entries.append({
            "q": q,
            "median_vr_dep": sig_q["median_vr_dep"],
            "ci_95_lo": ci_lo_q,
            "ci_95_hi": ci_hi_q,
            "closed": sig_q["closed"],
            "closure_threshold": CLOSURE_THRESHOLD,
            "p_twotailed": sig_q["p_twotailed"],
            "median_z_m2": sig_q["median_z_m2"],
            "n_nl": sig_q["n_nl"],
        })
        per_wq_primary_q.append({
            "W": W_primary,
            "q": q,
            "is_primary": bool(q == 5),
            "n_nl": sig_q["n_nl"],
            "median_vr_dep": sig_q["median_vr_dep"],
            "mean_vr_dep": sig_q["mean_vr_dep"],
            "ci_95_lo": ci_lo_q,
            "ci_95_hi": ci_hi_q,
            "ci_block": W_primary,
            "ci_n_boot": 2000,
            "closed": sig_q["closed"],
            "p_twotailed": sig_q["p_twotailed"],
            "median_z_m2": sig_q["median_z_m2"],
            "per_year": per_year_q,
        })

    adjusted_pvalues = vr_significance.apply_holm_b(holm_pvalues)

    holm_ordered = []
    order = sorted(range(len(holm_pvalues)), key=lambda i: holm_pvalues[i])
    for rank_0, sort_idx in enumerate(order):
        q_val = Q_GRID[sort_idx]
        median_z_m2_val = closure_table_entries[sort_idx]["median_z_m2"]
        holm_ordered.append({
            "q": q_val,
            "p_uncorrected": holm_pvalues[sort_idx],
            "holm_adjusted": adjusted_pvalues[sort_idx],
            "rank": rank_0 + 1,
            "median_z_m2": median_z_m2_val,
        })

    for i, entry in enumerate(closure_table_entries):
        entry["holm_adjusted"] = adjusted_pvalues[i]

    holm_correction = {
        "family_size": 4,
        "ordered": holm_ordered,
        "q_grid": list(Q_GRID),
    }

    closure_all_q = bool(all(e["closed"] for e in closure_table_entries))
    closure_table = {
        "closure_threshold": CLOSURE_THRESHOLD,
        "closure_all_q": closure_all_q,
        "per_q": closure_table_entries,
    }

    cascade_fired = any(
        e["holm_adjusted"] < 0.05 and e.get("median_z_m2", 0) > 0
        for e in holm_ordered
    )

    horizon_profile = vr_significance.compute_vr_horizon_profile(
        close, W=W_primary, q_grid=Q_GRID
    )
    low_noise_floor = vr_significance.compute_low_noise_floor(
        close, regime, W=W_primary, q_grid=Q_GRID
    )

    mde = vr_significance.compute_mde_vr(n_nl_primary)

    return {
        "per_wq": per_wq,
        "per_wq_primary_q": per_wq_primary_q,
        "holm_correction": holm_correction,
        "closure_table": closure_table,
        "horizon_profile": {str(k): v for k, v in horizon_profile.items()},
        "low_noise_floor": {str(k): v for k, v in low_noise_floor.items()},
        "mde": mde,
        "label_provenance": label_provenance,
        "cascade_fired": cascade_fired,
    }

def main() -> int:
    import datetime as _dt
    from scripts.wp1.py_engine import load_and_clean

    print("D-09 gate-guard: checking pre-registration integrity...")
    prereg_results = _gate_guard()
    prereg_commit = prereg_results.get(PREREG_PATH, "")
    print(f"D-09 gate-guard PASSED. prereg_commit={prereg_commit[:12]}...")

    run_ts_iso = datetime.now(tz=_dt.UTC).isoformat()
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")

    try:
        code_commit_result = subprocess.run(
            ["git", "log", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        code_commit = code_commit_result.stdout.strip()
    except Exception:
        code_commit = ""

    try:
        import numpy as _np
        import scipy as _sp
        library_versions = {
            "numpy": _np.__version__,
            "scipy": _sp.__version__,
        }
    except Exception:
        library_versions = {}

    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)
    holdout_boundary_ms = int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    
    assert end_ms < holdout_boundary_ms, (
        f"D-07 VIOLATION: data window end ({end_ms}) must be < 2026-01-01 "
        f"({holdout_boundary_ms}). The 2026 holdout must NOT be loaded."
    )

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    print(f"Loading {symbol} 2021-2025 (D-07 holdout: 2026 NOT loaded)...")
    data = load_and_clean(
        data_path=data_path,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        max_gap_allowed_mins=60,
    )

    close = data["close"]
    timestamps = data["timestamp"]
    N = len(close)

    ts_start = _dt.datetime.fromtimestamp(timestamps[0] / 1000, tz=_dt.UTC)
    ts_end = _dt.datetime.fromtimestamp(timestamps[-1] / 1000, tz=_dt.UTC)
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")

    assert ts_end.year < 2026, (
        f"D-07 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    data_span = {
        "start": str(ts_start.date()),
        "end": str(ts_end.date()),
        "n_bars": N,
        "year_2026_loaded": False,
        "first_ts": str(ts_start),
        "last_ts": str(ts_end),
    }

    append_spike_sc1_per_wq: dict[str, bool] = {}
    for W, q in WQ_GRID:
        vr_arr, z_arr = vr_significance.compute_rolling_vr_and_z(close, W=W, q=q)
        close_extended = np.append(close, close[-1] * 1.5)
        vr_arr_ext, _ = vr_significance.compute_rolling_vr_and_z(close_extended, W=W, q=q)
        try:
            np.testing.assert_array_equal(vr_arr, vr_arr_ext[:-1])
            sc1_passed = True
        except AssertionError:
            sc1_passed = False
        cell_key = f"W{W}_q{q}"
        append_spike_sc1_per_wq[cell_key] = sc1_passed

    append_spike_sc1_passed = bool(all(append_spike_sc1_per_wq.values()))
    print(f"SC1 append-spike overall: {'PASS' if append_spike_sc1_passed else 'FAIL'}")

    cascade_result = run_precheck_b_cascade(close, timestamps, seed=BOOT_SEED)
    
    label_provenance = cascade_result["label_provenance"]
    label_provenance["span"] = f"{ts_start.date()} to {ts_end.date()}"

    frozen_params = {
        "wq_grid": [list(wq) for wq in WQ_GRID],
        "primary_wq": [120, 5],
        "q_grid": list(Q_GRID),
        "closure_threshold": CLOSURE_THRESHOLD,
        "holm_family_size": 4,
        "central_statistic": "median",
        "boot_block": "W",
        "v2_epsilon_anchor": V2_EPSILON_ANCHOR,
    }

    provenance = {
        "freeze_commit": FREEZE_COMMIT,
        "prereg_commit": prereg_commit,
        "boot_seed": BOOT_SEED,
        "code_commit": code_commit,
        "library_versions": library_versions,
        "run_ts": run_ts_iso,
    }

    report = {
        "run_date": run_date,
        "prereg_commit": prereg_commit,
        "data_span": data_span,
        "label_provenance": label_provenance,
        "append_spike_sc1_passed": append_spike_sc1_passed,
        "append_spike_sc1_per_wq": append_spike_sc1_per_wq,
        "mde": cascade_result["mde"],
        "per_wq": cascade_result["per_wq"],
        "per_wq_primary_q": cascade_result["per_wq_primary_q"],
        "closure_table": cascade_result["closure_table"],
        "holm_correction": cascade_result["holm_correction"],
        "horizon_profile": cascade_result["horizon_profile"],
        "low_noise_floor": cascade_result["low_noise_floor"],
        "frozen_params": frozen_params,
        "provenance": provenance,
    }

    PRECHECK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRECHECK_DIR / f"precheck_b_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")
    print(f"cascade_fired: {cascade_result['cascade_fired']}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
