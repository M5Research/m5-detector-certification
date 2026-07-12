"""Re-gate orchestrator: dual-detector provisional gate execution with D-09 guard.

Executes the frozen pre-registered protocol (01-PREREGISTRATION.md, commit 169fc20,
with the Phase-4 §14.5 amendment bundle, commit 04b667e) against the full 2021-2025
BTC dataset using BOTH completed detectors:
  - Primary:    RollingQuantileDetector (Phase 2)
  - Robustness: HMMDetector (Phase 3)

Integrity controls enforced here:
  - D-09 gate-guard: pre-reg exists + git commit timestamp predates run + prereg_commit
    stamp in every report. Runs BEFORE any data load.
  - D-07 holdout (two-layer): end_ms < 2026-01-01 boundary + ts_end.year < 2026 +
    year_2026_loaded: false in every report.
  - D-08 PROVISIONAL-only: no FINAL verdict is written autonomously.
  - D-11 output: backtest_results/regate/ subdir (NOT v1.0 backtest_results/gate/).
  - T-04-08: list-form subprocess.run; no shell=True; no f-string-interpolated paths.
  - T-04-11: np.random.default_rng only; PERM_SEED=42, BOOT_SEED=43 recorded.
  - T-04-12: run_gate verdict logic reused unchanged; confirmatory CI/permutation null
    added ALONGSIDE, never read by the PASS/FAIL branch.

Output: gate_report_<date>_quantile.json, gate_report_<date>_hmm.json,
        cross_detector_<date>.json in backtest_results/regate/.

Determinism: np.random.default_rng(seed); never np.random.seed().
OMP guard: os.environ.setdefault("OMP_NUM_THREADS", "1") at module level (Phase-3 D-12).
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
# this file is run as a script (python scripts/wp1/regate_analysis.py) or
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
from scripts.wp1.gate_analysis import (  # noqa: E402
    DEGEN_CV_FLOOR,
    DEGEN_IQR_FLOOR,
    DEGEN_RAIL_FRAC_CUTOFF,
    DEGEN_RAIL_LEVEL,
    EPSILON_SQ_FLOOR,
    WQ_GRID,
    run_gate,
)
from scripts.wp1.predictability import compute_rolling_predictability  # noqa: E402
from strategies.vol_regime_switch.hmm_detector import HMMDetector  # noqa: E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    epsilon_sq_boot_ci,
    non_overlapping_samples as _nl_sampler,
    regime_label_permutation_null,
    regime_population_stats,
)
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGATE_DIR = PROJECT_ROOT / "backtest_results" / "regate"

# Pre-registration path (relative to repo root) — the D-09 gate-guard checks this
PREREG_PATH = (
    ".planning/phases/01-detector-spec-diagnosis-pre-registration/01-PREREGISTRATION.md"
)

# Determinism seeds (recorded in provenance — T-04-11)
PERM_SEED: int = 42
BOOT_SEED: int = 43


# ---------------------------------------------------------------------------
# D-09 Gate-Guard (§12 frozen contract)
# ---------------------------------------------------------------------------


def _gate_guard() -> str:
    """Enforce §12 D-09: pre-reg exists, commit predates run, return commit hash.

    Returns
    -------
    str
        The full git commit hash (format=%H) of the most recent commit to
        01-PREREGISTRATION.md, to be stamped as 'prereg_commit' in every report.

    Raises
    ------
    SystemExit
        With 'D-09 GATE GUARD FAILED' message on any integrity violation (fails
        CLOSED — never returns an unverified or empty hash):
        - pre-reg file absent
        - a git invocation returns a non-zero returncode (broken git / not a repo)
        - file has no git commits
        - most recent commit timestamp >= run timestamp
        - the resolved prereg_commit hash is empty
    """
    repo_root = Path(__file__).resolve().parents[2]
    prereg_abs = repo_root / PREREG_PATH

    # Step 1: pre-reg existence check
    if not prereg_abs.exists():
        raise SystemExit(
            f"D-09 GATE GUARD FAILED: {PREREG_PATH} not found. "
            "Phase 4 cannot run without the frozen pre-registration."
        )

    # Step 2: commit timestamp predates run (list-form subprocess; no shell=True; T-04-08)
    result = subprocess.run(
        ["git", "log", "--format=%ct", str(prereg_abs)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    # Fail CLOSED on a broken git invocation: a non-zero returncode means we cannot
    # trust stdout, so do NOT fall through to the 'no git commits' branch (which would
    # misattribute a tooling failure to a genuinely uncommitted file).
    if result.returncode != 0:
        raise SystemExit(
            f"D-09 GATE GUARD FAILED: 'git log --format=%ct' on the pre-registration "
            f"failed (returncode {result.returncode}). stderr: {result.stderr.strip()!r}. "
            "Cannot verify the pre-reg commit predates this run — failing closed."
        )
    commit_timestamps = [int(t) for t in result.stdout.strip().split() if t]
    if not commit_timestamps:
        raise SystemExit(
            "D-09 GATE GUARD FAILED: pre-reg file has no git commits. "
            "Commit the pre-registration before running the gate."
        )
    latest_prereg_ts = max(commit_timestamps)  # most recent commit to the file
    run_ts = int(time.time())
    if latest_prereg_ts >= run_ts:
        raise SystemExit(
            f"D-09 GATE GUARD FAILED: pre-reg commit ({latest_prereg_ts}) does not "
            f"predate this run ({run_ts}). Integrity violation — the pre-registration "
            "must be committed BEFORE the gate run (no-HARKing control)."
        )

    # Step 3: resolve commit hash for the prereg_commit stamp in every report.
    # Fail CLOSED: a non-zero returncode or an empty hash must abort the run rather
    # than silently stamp an empty prereg_commit (which voids freeze-before-run
    # provenance in all emitted reports — CR-01).
    hash_result = subprocess.run(
        ["git", "log", "--format=%H", "-1", str(prereg_abs)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if hash_result.returncode != 0:
        raise SystemExit(
            f"D-09 GATE GUARD FAILED: 'git log --format=%H' on the pre-registration "
            f"failed (returncode {hash_result.returncode}). stderr: "
            f"{hash_result.stderr.strip()!r}. Cannot obtain the prereg_commit hash — "
            "failing closed rather than stamping an empty provenance hash."
        )
    prereg_commit = hash_result.stdout.strip()
    if not prereg_commit:
        raise SystemExit(
            "D-09 GATE GUARD FAILED: 'git log --format=%H' returned an empty commit "
            "hash for the pre-registration. Refusing to stamp an empty prereg_commit "
            "(silent freeze-before-run provenance void) — failing closed."
        )
    return prereg_commit


# ---------------------------------------------------------------------------
# Cross-detector Cohen kappa (D-09 / SC3)
# ---------------------------------------------------------------------------


def compute_kappa(
    regime_quant: np.ndarray,
    regime_hmm: np.ndarray,
) -> dict:
    """Compute Cohen kappa + fraction-agreement over the valid-both intersection.

    Computes kappa over the intersection of bars where BOTH detectors have valid
    (>= 0) labels.  This is required because the two detectors have different
    warmup lengths (quantile ~43259 bars vs HMM ~60 bars), so the intersection
    excludes quantile-warmup bars (the dominant constraint).

    Parameters
    ----------
    regime_quant : np.ndarray[int8]
        Quantile detector labels (-1=warmup, 0=LOW, 1=ELEVATED, 2=EXTREME).
    regime_hmm : np.ndarray[int8]
        HMM detector labels (-1=warmup, 0=LOW, 1=ELEVATED, 2=EXTREME).

    Returns
    -------
    dict with keys:
        n_bars_intersection : int   — bars where both valid (>= 0)
        cohens_kappa        : float — Cohen's kappa over the intersection
        fraction_agreement_per_regime : dict with LOW, ELEVATED, EXTREME keys
            fraction_agreement[r] = (quant==r and hmm==r).sum() / valid_both.sum()
        kappa_caveat        : str   — D-09 interpretive caveat (pre-stated)
    """
    regime_quant = np.asarray(regime_quant)
    regime_hmm = np.asarray(regime_hmm)

    # Intersection of valid bars (both >= 0)
    valid_both = (regime_quant >= 0) & (regime_hmm >= 0)
    n_intersection = int(np.sum(valid_both))

    if n_intersection < 2:
        return {
            "n_bars_intersection": n_intersection,
            "cohens_kappa": float("nan"),
            "fraction_agreement_per_regime": {"LOW": float("nan"), "ELEVATED": float("nan"), "EXTREME": float("nan")},
            "kappa_caveat": (
                "Insufficient intersection bars for kappa computation. "
                "D-09 caveat: low kappa reflects HMM degeneracy + label incommensurability."
            ),
        }

    q_valid = regime_quant[valid_both]
    h_valid = regime_hmm[valid_both]
    n = n_intersection

    # Cohen kappa via Appendix formula: kappa = (p_o - p_e) / (1 - p_e)
    # p_o = observed agreement fraction
    # p_e = expected chance agreement (marginal products)
    labels = (0, 1, 2)

    p_o = float(np.sum(q_valid == h_valid)) / n

    p_e = 0.0
    for label in labels:
        p_q = float(np.sum(q_valid == label)) / n
        p_h = float(np.sum(h_valid == label)) / n
        p_e += p_q * p_h

    if abs(1.0 - p_e) < 1e-12:
        kappa = float("nan")  # degenerate case: perfect chance agreement
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Per-regime fraction agreement
    label_names = {0: "LOW", 1: "ELEVATED", 2: "EXTREME"}
    fraction_agreement: dict[str, float] = {}
    for label, name in label_names.items():
        both_agree = np.sum((q_valid == label) & (h_valid == label))
        fraction_agreement[name] = float(both_agree) / n

    # D-09 kappa caveat (pre-stated in §14.5 Phase-4 amendment bundle)
    kappa_caveat = (
        "A low cross-detector kappa reflects HMM degeneracy + label incommensurability "
        "(quantile LOW 75% / ELEVATED 19% / EXTREME 5% vs HMM LOW ~51% / ELEVATED ~49% / "
        "EXTREME 0%, and the two 'ELEVATED' labels mean different things — a 75-95th-pct "
        "RV band vs a degenerate high-variance state). Low kappa is a SYMPTOM of "
        "instrument failure, NOT evidence the primary is wrong, NOT a post-hoc resolution "
        "of disagreement. Per-regime EXTREME fraction-agreement is structurally 0 (the HMM "
        "never emits label 2). Known from as-built populations (no epsilon-squared peeking). "
        "Source: §14.5 Phase-4 amendment bundle, D-09."
    )

    return {
        "n_bars_intersection": n_intersection,
        "cohens_kappa": float(kappa),
        "fraction_agreement_per_regime": fraction_agreement,
        "kappa_caveat": kappa_caveat,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load 2021-2025, run dual-detector gate, emit PROVISIONAL reports."""
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # D-09 gate-guard: MUST run BEFORE any data load (no-HARKing control)
    # -----------------------------------------------------------------------
    print("D-09 gate-guard: checking pre-registration integrity...")
    prereg_commit = _gate_guard()
    print(f"D-09 gate-guard PASSED. prereg_commit={prereg_commit[:12]}...")

    # -----------------------------------------------------------------------
    # Capture run timestamp and code git commit for provenance
    # -----------------------------------------------------------------------
    run_ts_iso = datetime.now(tz=_dt.UTC).isoformat()
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")

    # Code commit (best-effort; empty string if git not available)
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

    # Library versions for provenance
    try:
        import numpy as _np
        import scipy as _sp
        import statsmodels as _sm

        library_versions = {
            "numpy": _np.__version__,
            "scipy": _sp.__version__,
            "statsmodels": _sm.__version__,
        }
    except Exception:
        library_versions = {}

    # -----------------------------------------------------------------------
    # D-07 holdout: define data window and assert it ends before 2026-01-01
    # -----------------------------------------------------------------------
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

    # Confirm data span
    ts_start = _dt.datetime.fromtimestamp(timestamps[0] / 1000, tz=_dt.UTC)
    ts_end = _dt.datetime.fromtimestamp(timestamps[-1] / 1000, tz=_dt.UTC)
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")

    # D-07 second layer: assert last timestamp is before 2026
    assert ts_end.year < 2026, (
        f"D-07 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # Compute predictability DV ONCE per (W,q) -- shared across both detectors
    # This is the D-05 byte-identical-RV isolation: the DV is detector-independent.
    # -----------------------------------------------------------------------
    print(f"Computing predictability_t for {len(WQ_GRID)} (W,q) settings (shared DV)...")
    pred_series: dict[tuple[int, int], np.ndarray] = {}
    for W, q in WQ_GRID:
        print(f"  (W={W}, q={q})...")
        pred_series[(W, q)] = compute_rolling_predictability(close, W=W, q=q)

    # Strides for non-overlapping KW sampling (stride = W for each setting)
    strides = {(W, q): W for W, q in WQ_GRID}

    # -----------------------------------------------------------------------
    # Fit BOTH detectors ONCE on the same close series
    # -----------------------------------------------------------------------
    import time as _time

    print("Fitting RollingQuantileDetector (primary)...")
    regime_quant = RollingQuantileDetector().fit(close)

    # D-06 smoke-test: determine if 3-state HMM is feasible on this data.
    # Replicated verbatim from validate_hmm_detector.py (Phase 3 production logic).
    # The 3-state model took >30 min extrapolated on Phase 3 data -> 2-state fallback.
    print("Fitting HMMDetector (robustness) -- D-06 smoke-test first...")
    SMOKE_SLICE_START = N // 2 - 250_000
    SMOKE_SLICE_END = SMOKE_SLICE_START + 500_000
    SMOKE_SLICE_START = max(0, SMOKE_SLICE_START)
    SMOKE_SLICE_END = min(N, SMOKE_SLICE_END)
    smoke_slice = close[SMOKE_SLICE_START:SMOKE_SLICE_END]
    smoke_n = len(smoke_slice)
    print(
        f"  D-06 smoke-test: 3-state fit on bars [{SMOKE_SLICE_START}:{SMOKE_SLICE_END}] "
        f"({smoke_n:,} bars)..."
    )

    k_regimes_primary = 3
    hmm_fallback_val = "none"
    try:
        t0 = _time.perf_counter()
        _smoke_det = HMMDetector(k_regimes=3, em_iter=100, search_reps=3)
        _smoke_det.fit(smoke_slice)
        smoke_secs = _time.perf_counter() - t0
        extrapolated_secs = smoke_secs * (N / smoke_n)
        print(
            f"  Smoke-test timing: {smoke_secs:.1f}s -> extrapolated full-run: "
            f"{extrapolated_secs:.1f}s ({extrapolated_secs / 60:.1f} min)"
        )
        if extrapolated_secs > 30 * 60:
            print(
                "  D-06 INTRACTABILITY: extrapolated > 30 min. "
                "Activating frozen 2-state fallback (hmm_fallback='2-state')."
            )
            k_regimes_primary = 2
            hmm_fallback_val = "2-state"
        else:
            print(
                f"  D-06: 3-state feasible ({extrapolated_secs / 60:.1f} min < 30 min). "
                "Proceeding with 3-state primary."
            )
    except (MemoryError, RuntimeError) as _exc:
        print(
            f"  D-06 smoke-test failed ({type(_exc).__name__}: {_exc}). "
            "Activating frozen 2-state fallback (hmm_fallback='2-state')."
        )
        k_regimes_primary = 2
        hmm_fallback_val = "2-state"

    print(f"Fitting HMMDetector(k_regimes={k_regimes_primary}) on full {N:,}-bar span...")
    print("  This may take 10-40 minutes...")
    hmm_detector = HMMDetector(k_regimes=k_regimes_primary)
    regime_hmm = hmm_detector.fit(close)
    # If the detector's hmm_fallback_ attribute was set by the fit (degenerate 2-state
    # internal flag), use it; otherwise use the D-06-determined fallback value.
    if hmm_detector.hmm_fallback_ != "none":
        hmm_fallback_val = hmm_detector.hmm_fallback_
    print(f"HMMDetector fit complete. hmm_fallback={hmm_fallback_val}")

    # -----------------------------------------------------------------------
    # Data span dict (shared across all reports)
    # -----------------------------------------------------------------------
    data_span = {
        "start": str(ts_start.date()),
        "end": str(ts_end.date()),
        "n_bars": N,
        "year_2026_loaded": False,  # D-07 two-layer holdout guard echo
        "first_ts": str(ts_start),
        "last_ts": str(ts_end),
    }

    # -----------------------------------------------------------------------
    # Frozen parameters echo (from ratified pre-registration)
    # -----------------------------------------------------------------------
    frozen_params = {
        "epsilon_sq_floor": EPSILON_SQ_FLOOR,
        "cv_floor": DEGEN_CV_FLOOR,
        "iqr_floor": DEGEN_IQR_FLOOR,
        "rail_level": DEGEN_RAIL_LEVEL,
        "rail_frac_cutoff": DEGEN_RAIL_FRAC_CUTOFF,
        "wq_grid": [list(wq) for wq in WQ_GRID],
    }

    # -----------------------------------------------------------------------
    # Provenance (shared across all reports)
    # -----------------------------------------------------------------------
    provenance = {
        "freeze_commit": "169fc20",
        "phase4_amendment_ref": "01-PREREGISTRATION.md §14.5 Phase-4 bundle (commit 04b667e)",
        "perm_seed": PERM_SEED,
        "boot_seed": BOOT_SEED,
        "omp_num_threads": "1 (defensive, per Phase-3 D-12)",
        "code_commit": code_commit,
        "library_versions": library_versions,
        "run_ts": run_ts_iso,
    }

    # -----------------------------------------------------------------------
    # Run gate + attach confirmatory stats per detector
    # -----------------------------------------------------------------------
    REGATE_DIR.mkdir(parents=True, exist_ok=True)

    for detector_name, regime in [("quantile", regime_quant), ("hmm", regime_hmm)]:
        print(f"\nRunning gate for detector={detector_name}...")
        gate_result = run_gate(
            pred_series=pred_series,
            regime=regime,
            wq_grid=WQ_GRID,
            strides=strides,
        )

        # Attach confirmatory stats (CONFIRMATORY-ONLY; never gate-driving — T-04-12)
        enhanced_per_wq = []
        for row, (W, q) in zip(gate_result["per_wq"], WQ_GRID, strict=True):
            stride = W
            # Non-overlapping samples (shared WR-01-corrected sampler — D-06)
            pred_nl, regime_nl, _retained_idx = _nl_sampler(
                pred_series[(W, q)], regime, stride
            )

            # Bootstrap CI (CONFIRMATORY-ONLY)
            pt, ci_lo, ci_hi = epsilon_sq_boot_ci(
                pred_nl, regime_nl, block=10, n_boot=2000, seed=BOOT_SEED
            )

            # Permutation null (CONFIRMATORY-ONLY)
            perm_p = regime_label_permutation_null(
                pred_nl, regime_nl, n_perm=5000, seed=PERM_SEED
            )

            # sparse_extreme flag from regime_population_stats
            pop_stats = regime_population_stats(regime, n_min=50, stride=stride)
            sparse_extreme = bool(pop_stats["sparse"])

            # Per-(W,q) HMM note when EXTREME is structurally absent (D-10)
            row_extra: dict = {}
            if detector_name == "hmm":
                n_extreme = int(np.sum(regime_nl == 2))
                if n_extreme == 0:
                    row_extra["hmm_extreme_note"] = (
                        "HMM non-crisis contrast; EXTREME structurally absent "
                        "(2-state fallback: sigma2 ratio=1.06 does not represent a genuine "
                        "crisis regime per D-02). This contrast is uninformative on the "
                        "EXTREME question. D-01 carve-out applies: a degenerate HMM EXTREME "
                        "arm is instrument failure, not regime disagreement."
                    )
                    # permutation_p on an EXTREME-involving contrast is null
                    if not np.isfinite(perm_p):
                        row_extra["permutation_p_note"] = (
                            "permutation_p is NaN/null: EXTREME group has 0 samples; "
                            "KW collapses to 2-group (LOW vs ELEVATED). "
                            "Permutation null over a degenerate EXTREME arm is not meaningful."
                        )

            enhanced_row = {
                **row,
                "epsilon_sq_ci_lo": ci_lo if np.isfinite(ci_lo) else None,
                "epsilon_sq_ci_hi": ci_hi if np.isfinite(ci_hi) else None,
                "epsilon_sq_ci_block": 10,
                "epsilon_sq_ci_n_boot": 2000,
                "permutation_p": perm_p if np.isfinite(perm_p) else None,
                "n_perm": 5000,
                "sparse_extreme": sparse_extreme,
                **row_extra,
            }
            enhanced_per_wq.append(enhanced_row)

        # D-03 HMM soft-probability note (REPORTED observation only, not gate-driving)
        hmm_soft_prob_note = None
        if detector_name == "hmm":
            hmm_soft_prob_note = (
                "D-03: HMM filtered P(high-variance state) peaked at ~0.9999 during "
                "LUNA (2022-05-07..05-18) and FTX (2022-11-08..11-12) crises per "
                "03-VALIDATION.md §D-11. This is a REPORTED OBSERVATION only — it "
                "does not enter the gate logic or verdict. The 2-state fallback is "
                "ELEVATED(1), not EXTREME(2) per D-02."
            )

        # Assemble full report
        report = {
            "run_date": run_date,
            "prereg_commit": prereg_commit,
            "data_span": data_span,
            "detector": detector_name,
            "hmm_fallback": hmm_fallback_val if detector_name == "hmm" else None,
            "hmm_soft_prob_note": hmm_soft_prob_note,
            "verdict": gate_result["verdict"],
            "verdict_note": (
                "PROVISIONAL — computed against ratified pre-registered thresholds. "
                "FINAL verdict requires human ratification (D-07/D-08)."
            ),
            "per_wq": enhanced_per_wq,
            "reasons": gate_result["reasons"],
            "frozen_params": frozen_params,
            "provenance": provenance,
        }

        out_path = REGATE_DIR / f"gate_report_{run_date}_{detector_name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written: {out_path}")

    # -----------------------------------------------------------------------
    # Cross-detector kappa (D-09 / SC3)
    # -----------------------------------------------------------------------
    print("\nComputing cross-detector Cohen kappa...")
    cross = compute_kappa(regime_quant, regime_hmm)

    # D-01 carve-out statement
    d01_carve_out = (
        "D-01 degenerate-HMM carve-out: a degenerate / SPARSE HMM EXTREME arm is an "
        "instrument failure, treated as uninformative-on-EXTREME. It cannot manufacture "
        "an INCONCLUSIVE that overrides a clean primary (RollingQuantileDetector) result. "
        "The HMM LOW-vs-ELEVATED contrast (2-group KW) is the most faithful to §10 and "
        "is reported, but explicitly labeled uninformative on the EXTREME question."
    )

    cross_report = {
        "run_date": run_date,
        "prereg_commit": prereg_commit,
        "data_span": data_span,
        "cohens_kappa": cross["cohens_kappa"],
        "n_bars_intersection": cross["n_bars_intersection"],
        "fraction_agreement_per_regime": cross["fraction_agreement_per_regime"],
        "kappa_caveat": cross["kappa_caveat"],
        "d01_carve_out": d01_carve_out,
        "verdict_note": (
            "PROVISIONAL — cross-detector kappa is a reported observation per §11/D-09. "
            "FINAL verdict requires human ratification (D-07/D-08)."
        ),
        "provenance": provenance,
    }

    cross_path = REGATE_DIR / f"cross_detector_{run_date}.json"
    cross_path.write_text(json.dumps(cross_report, indent=2), encoding="utf-8")
    print(f"Cross-detector report written: {cross_path}")

    # -----------------------------------------------------------------------
    # PROVISIONAL summary (no FINAL verdict — D-08)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RE-GATE EXECUTION COMPLETE (PROVISIONAL)")
    print("=" * 70)
    print(f"Run date:        {run_date}")
    print(f"prereg_commit:   {prereg_commit[:12]}...")
    print(f"Data span:       {data_span['start']} -> {data_span['end']} ({N:,} bars)")
    print("2026 holdout:    NOT loaded (D-07 two-layer guard held)")
    print(f"HMM fallback:    {hmm_fallback_val}")
    print()
    print("Reports written to:", REGATE_DIR)
    print()
    print("IMPORTANT: Verdicts in these reports are PROVISIONAL.")
    print("FINAL verdict requires human ratification (D-07/D-08) in Plan 04-04.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
