"""Pre-Check A orchestrator: nested transition-vs-levels test on 2021-2025 BTCUSDT.

Implements the frozen Pre-Check A protocol from 07-PREREGISTRATION.md (freeze
commit 720c1d4, with the D-08 amendment commit dd44e7a) by running the
HAC-robust Wald chi2(1) nested test across WQ_GRID = {(60,5),(120,5),(240,15)}.

Integrity controls enforced here:
  - D-09 gate-guard: 07-PREREGISTRATION.md exists + git commit timestamp
    predates run + prereg_commit stamp in every report. Runs BEFORE any data load.
  - D-07 holdout (two-layer): end_ms < 2026-01-01 boundary (LAYER 1, before load)
    + ts_end.year < 2026 (LAYER 2, after load) + year_2026_loaded: false in report.
  - D-05: labels regenerated inline via RollingQuantileDetector().fit(close) —
    no cached label artifact.
  - SC1 append-spike causality assertion runs in-process on the real DV per (W,q).
  - All math delegated to scripts.wp1.nested_test (Wave-1 synthetic-gate-hardened
    module) — no Wald/deltaR2/bootstrap/Holm/MDE/disposition reimplemented here.
  - D-10 boundary: Phase 8 stops at the ratified verdict; no camera-ready
    table/figure, no positive-branch study.

Output: backtest_results/precheck/precheck_a_<date>.json

Determinism: np.random.default_rng(seed); never np.random.seed().
OMP guard: os.environ.setdefault("OMP_NUM_THREADS", "1") at module level.

D-15 provenance stamp in every emitted JSON:
  freeze_commit: "720c1d4"
  phase8_amendment_ref: "dd44e7a"
  prereg_commit: <resolved at runtime by _gate_guard()>
  boot_seed: 43
  code_commit: <resolved at runtime via subprocess git log>
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
# this file is run as a script (python scripts/wp1/precheck_a.py) or
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
from scripts.wp1.gate_analysis import WQ_GRID  # noqa: E402  -- frozen (60,5),(120,5),(240,15)
from scripts.wp1.predictability import compute_rolling_predictability  # noqa: E402
import scripts.wp1.nested_test as nested_test  # noqa: E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    non_overlapping_samples,
)
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRECHECK_DIR = PROJECT_ROOT / "backtest_results" / "precheck"

# Pre-registration path (relative to repo root) — the D-09 gate-guard checks this
# CHANGE-ONLY-THIS-LINE versus regate_analysis.py (which points at 01-PREREGISTRATION.md):
PREREG_PATH = (
    ".planning/phases/07-pre-registration-freeze/07-PREREGISTRATION.md"
)

# D-08 amendment: the no-peeking amendment commit that operationalizes D-07.
# This was committed as dd44e7a before any real-data run (08-02-SUMMARY.md).
PHASE8_AMENDMENT_REF = "dd44e7a16e79702a232bec30fa90428b158c7c3e"

# D-15 provenance stamp
FREEZE_COMMIT = "720c1d4"

# Determinism seed (recorded in provenance — T-04-11; frozen D-06)
BOOT_SEED: int = 43


# ---------------------------------------------------------------------------
# D-09 Gate-Guard (§12 frozen contract — body copied VERBATIM from regate_analysis.py)
# ---------------------------------------------------------------------------


def _gate_guard() -> str:
    """Enforce §12 D-09: pre-reg exists, commit predates run, return commit hash.

    Returns
    -------
    str
        The full git commit hash (format=%H) of the most recent commit to
        07-PREREGISTRATION.md, to be stamped as 'prereg_commit' in every report.

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
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> int:
    """Gate-guard -> load 2021-2025 -> re-fit detector inline -> DV + SC1 ->
    nested_test per (W,q) -> Holm + D-07 disposition -> emit JSON.

    Returns 0 on success. Guards under __main__ guard below.
    """
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # 1. D-09 gate-guard: MUST run BEFORE any data load (no-HARKing control)
    # -----------------------------------------------------------------------
    print("D-09 gate-guard: checking pre-registration integrity...")
    prereg_commit = _gate_guard()
    print(f"D-09 gate-guard PASSED. prereg_commit={prereg_commit[:12]}...")
    # The gate-guard resolves the latest commit to 07-PREREGISTRATION.md.
    # That commit is the D-08 amendment (dd44e7a) which operationalizes D-07.
    # Both the freeze anchor (720c1d4) and this amendment are stamped in the JSON.

    # -----------------------------------------------------------------------
    # 2. Capture run timestamp and code git commit for provenance
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
    # 3. D-07 holdout LAYER 1: define data window and assert it ends before
    #    2026-01-01 BEFORE loading any data (epoch-ms boundary check).
    # -----------------------------------------------------------------------
    start_ms = int(_dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    end_ms = int(_dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000)

    holdout_boundary_ms = int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000)
    assert end_ms < holdout_boundary_ms, (
        f"D-07 VIOLATION: data window end ({end_ms}) must be < 2026-01-01 "
        f"({holdout_boundary_ms}). The 2026 holdout must NOT be loaded."
    )

    # -----------------------------------------------------------------------
    # 4. Load 2021-2025 BTCUSDT (NEVER 2026)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 5. D-07 holdout LAYER 2: assert last timestamp is before 2026
    #    (post-load check on the ACTUAL last bar's year)
    # -----------------------------------------------------------------------
    assert ts_end.year < 2026, (
        f"D-07 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    data_span = {
        "start": str(ts_start.date()),
        "end": str(ts_end.date()),
        "n_bars": N,
        "year_2026_loaded": False,  # D-07 two-layer holdout guard echo
        "first_ts": str(ts_start),
        "last_ts": str(ts_end),
    }

    # -----------------------------------------------------------------------
    # 6. Fit RollingQuantileDetector INLINE (labels regenerated, not cached — D-05)
    # -----------------------------------------------------------------------
    print("Fitting RollingQuantileDetector inline (D-05: labels regenerated, not cached)...")
    regime = RollingQuantileDetector().fit(close)

    # Build label_provenance for the JSON
    n_warmup = int(np.sum(regime < 0))
    n_LOW = int(np.sum(regime == 0))
    n_ELEVATED = int(np.sum(regime == 1))
    n_EXTREME = int(np.sum(regime == 2))
    first_valid_bar = int(np.argmax(regime >= 0)) if np.any(regime >= 0) else -1

    # RollingQuantileDetector default params (from rolling_quantile_detector.py)
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
    }
    print(
        f"Detector fit: {n_warmup:,} warmup | {n_LOW:,} LOW | "
        f"{n_ELEVATED:,} ELEVATED | {n_EXTREME:,} EXTREME"
    )

    # -----------------------------------------------------------------------
    # 7. Compute rolling DV per (W,q) + in-process SC1 append-spike assertion
    # -----------------------------------------------------------------------
    print(f"Computing DV + SC1 causality assertion for {len(WQ_GRID)} (W,q) settings...")
    pred_series: dict[tuple[int, int], np.ndarray] = {}
    append_spike_sc1_per_wq: dict[tuple[int, int], bool] = {}

    for W, q in WQ_GRID:
        print(f"  (W={W}, q={q}): computing DV...")
        pred_orig = compute_rolling_predictability(close, W=W, q=q)

        # SC1 append-spike causality assertion (Q6): append one FUTURE bar, recompute,
        # assert all ORIGINAL bars' DV values are unchanged. This confirms no look-ahead bias.
        # pred_orig has length N; pred_extended has length N+1. The appended (future) bar
        # may only create a NEW value at index N — it must not alter indices 0..N-1. So
        # compare pred_orig (len N) against pred_extended[:-1] (first N values, len N).
        close_extended = np.append(close, close[-1] * 1.5)  # append a future bar
        pred_extended = compute_rolling_predictability(close_extended, W=W, q=q)

        try:
            np.testing.assert_array_equal(pred_orig, pred_extended[:-1])
            sc1_passed = True
        except AssertionError:
            sc1_passed = False
            print(f"  WARNING: SC1 append-spike assertion FAILED for (W={W}, q={q})!")

        append_spike_sc1_per_wq[(W, q)] = sc1_passed
        pred_series[(W, q)] = pred_orig
        print(f"  (W={W}, q={q}): DV computed, SC1={'PASS' if sc1_passed else 'FAIL'}")

    # Overall SC1 flag: True only if ALL cells pass
    append_spike_sc1_passed = bool(all(append_spike_sc1_per_wq.values()))
    print(f"SC1 append-spike overall: {'PASS' if append_spike_sc1_passed else 'FAIL'}")

    # -----------------------------------------------------------------------
    # 8. Run nested_test per (W,q): design matrix, degeneracy, min-sep,
    #    Wald HAC, delta_R2, beta_T bootstrap CI
    # -----------------------------------------------------------------------
    print("Running nested tests per (W,q)...")
    per_wq: list[dict] = []
    n_nl_primary: int = 0

    for W, q in WQ_GRID:
        print(f"  (W={W}, q={q}): non-overlapping sampling...")
        pred_nl, regime_nl, retained_idx = non_overlapping_samples(
            pred_series[(W, q)], regime, stride=W
        )
        n_nl = len(pred_nl)
        is_primary = bool(W == 120 and q == 5)
        if is_primary:
            n_nl_primary = n_nl

        # Newey-West lag formulas (frozen §4 / RESEARCH Q7)
        L_primary = int(4 * (n_nl / 100) ** (2 / 9))
        L_robustness = int(n_nl ** (1 / 3))

        print(f"    n_nl={n_nl:,}, L_primary={L_primary}, L_robustness={L_robustness}")

        # Build design matrix: TRANSITION model (includes T_t)
        X_trans, meta = nested_test.build_design_matrix(
            regime_nl=regime_nl,
            retained_idx=retained_idx,
            regime_full=regime,
        )
        X_level = meta["X_level"]
        col_T = meta["col_T"]
        y = pred_nl  # DV: |VR(q)-1| at non-overlapping samples

        # Degeneracy check BEFORE any Wald test (frozen §4)
        degen = nested_test.check_degeneracy(X_trans)
        print(
            f"    Degeneracy: rank={degen['rank']}, cond={degen['cond']:.3g}, "
            f"degenerate={degen['degenerate']}"
        )

        # Min-separation filter (frozen §5)
        min_sep_info = nested_test.apply_min_sep_filter(
            regime=regime,
            retained_idx=retained_idx,
            W=W,
        )
        print(
            f"    Min-sep: UP {min_sep_info['n_UP_before']}->{min_sep_info['n_UP_after']}, "
            f"DOWN {min_sep_info['n_DOWN_before']}->{min_sep_info['n_DOWN_after']}, "
            f"under_powered={min_sep_info['under_powered']}"
        )

        # -----------------------------------------------------------------------
        # Wald test routing: degenerate vs non-degenerate
        # -----------------------------------------------------------------------
        fallback_used = False
        skipped = False
        skip_reason = None
        wald_result = None
        wald_robustness_result = None
        delta_r2 = float("nan")
        beta_T = float("nan")
        beta_T_ci_lo = float("nan")
        beta_T_ci_hi = float("nan")

        if degen["degenerate"]:
            n_transitions_min = min(
                min_sep_info["n_transitions_UP"], min_sep_info["n_transitions_DOWN"]
            )
            if n_transitions_min >= 100:
                # Fallback: E_t/D_t split design, joint Wald chi2(2)
                print(f"    Degenerate + n_transitions>={n_transitions_min}: using E_t/D_t fallback...")
                # Build fallback design matrix: replace T_t with E_t (UP) and D_t (DOWN)
                n = len(regime_nl)
                R1_raw = np.where(
                    retained_idx > 0,
                    regime[np.maximum(retained_idx - 1, 0)].astype(np.int8),
                    np.int8(-1),
                )
                R_curr = regime_nl.astype(np.int8)
                valid_lag = retained_idx > 0
                E_t = np.where(valid_lag & (R_curr > R1_raw), 1.0, 0.0)
                D_t = np.where(valid_lag & (R_curr < R1_raw), 1.0, 0.0)
                const = np.ones(n, dtype=np.float64)
                R_elev = (regime_nl == 1).astype(np.float64)
                R_extr = (regime_nl == 2).astype(np.float64)
                R1_elev = (R1_raw == 1).astype(np.float64)
                R1_extr = (R1_raw == 2).astype(np.float64)
                X_fallback = np.column_stack([const, R_elev, R_extr, R1_elev, R1_extr, E_t, D_t])
                col_E = 5
                col_D = 6
                wald_result = nested_test.run_wald_hac_joint(y, X_fallback, col_E, col_D, L_primary)
                wald_robustness_result = nested_test.run_wald_hac_joint(y, X_fallback, col_E, col_D, L_robustness)
                fallback_used = True
                # delta_R2: compare level model vs fallback (E_t+D_t replaces T_t)
                X_fallback_level = X_fallback[:, :5]  # drop E_t and D_t
                delta_r2 = nested_test.compute_delta_r2(y, X_fallback_level, X_fallback)
                # beta_T not meaningful in joint fallback (already float('nan'))
                print(f"    Fallback Wald chi2(2): stat={wald_result['statistic']:.4f}, p={wald_result['pvalue']:.4f}")
            else:
                # Cannot run: degenerate + insufficient transitions
                skipped = True
                skip_reason = "degeneracy+insufficient_transitions"
                print(f"    Skipped: degenerate + n_transitions < 100 per direction")
        else:
            # Primary path: HAC Wald chi2(1) on T_t
            wald_result = nested_test.run_wald_hac(y, X_trans, col_T, L_primary)
            wald_robustness_result = nested_test.run_wald_hac(y, X_trans, col_T, L_robustness)
            delta_r2 = nested_test.compute_delta_r2(y, X_level, X_trans)
            beta_T, beta_T_ci_lo, beta_T_ci_hi = nested_test.beta_t_boot_ci(
                y_nl=y,
                X_nl=X_trans,
                col_T=col_T,
                block=W,
                n_boot=2000,
                seed=BOOT_SEED,
            )
            print(
                f"    Wald chi2(1): stat={wald_result['statistic']:.4f}, "
                f"p={wald_result['pvalue']:.4f}, delta_r2={delta_r2:.6f}, "
                f"beta_T={beta_T:.4f} [{beta_T_ci_lo:.4f}, {beta_T_ci_hi:.4f}]"
            )

        # Effect-size thresholds (frozen §7)
        delta_r2_gte_0005 = bool(delta_r2 >= 0.005)
        abs_beta_T_gte_001 = bool(abs(beta_T) >= 0.01)
        thresholds_both = bool(delta_r2_gte_0005 and abs_beta_T_gte_001)

        # Full conjunction: Wald rejects (uncorrected p < 0.05) AND both thresholds met.
        # Holm adjustment applied later; cell_passes_full_conjunction uses uncorrected p
        # (apply_disposition_rule reads cell_passes_full_conjunction, which the caller sets).
        if skipped or wald_result is None:
            wald_rejects = False
        else:
            wald_rejects = bool(wald_result["pvalue"] < 0.05)
        cell_passes_full_conjunction = bool(wald_rejects and thresholds_both)

        # Assemble per-(W,q) entry per Q8 schema
        cell: dict = {
            "W": W,
            "q": q,
            "is_primary": is_primary,
            "n_nl": n_nl,
            "L_newey_west_primary": L_primary,
            "L_newey_west_robustness": L_robustness,
            "degeneracy": degen,
            "fallback_used": fallback_used,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "min_sep_filter": min_sep_info,
            "delta_r2": delta_r2 if np.isfinite(delta_r2) else None,
            "beta_T": beta_T if np.isfinite(beta_T) else None,
            "beta_T_ci_lo": beta_T_ci_lo if np.isfinite(beta_T_ci_lo) else None,
            "beta_T_ci_hi": beta_T_ci_hi if np.isfinite(beta_T_ci_hi) else None,
            "beta_T_ci_block": W,
            "beta_T_ci_n_boot": 2000,
            "effect_size_thresholds_met": {
                "delta_r2_gte_0005": delta_r2_gte_0005,
                "abs_beta_T_gte_001": abs_beta_T_gte_001,
                "both": thresholds_both,
            },
            "cell_passes_full_conjunction": cell_passes_full_conjunction,
        }

        # Attach Wald results
        if wald_result is not None:
            if fallback_used:
                cell["wald_chi2_joint"] = {
                    "statistic": wald_result["statistic"],
                    "pvalue_uncorrected": wald_result["pvalue"],
                    "df": wald_result["df"],
                    "lag_L": L_primary,
                    "fallback_type": "E_t/D_t joint chi2(2)",
                }
                if wald_robustness_result is not None:
                    cell["wald_chi2_joint_robustness"] = {
                        "statistic": wald_robustness_result["statistic"],
                        "pvalue_uncorrected": wald_robustness_result["pvalue"],
                        "df": wald_robustness_result["df"],
                        "lag_L": L_robustness,
                    }
            else:
                cell["wald_chi2_1"] = {
                    "statistic": wald_result["statistic"],
                    "pvalue_uncorrected": wald_result["pvalue"],
                    "df": wald_result["df"],
                    "lag_L": L_primary,
                }
                if wald_robustness_result is not None:
                    cell["wald_chi2_1_robustness"] = {
                        "statistic": wald_robustness_result["statistic"],
                        "pvalue_uncorrected": wald_robustness_result["pvalue"],
                        "df": wald_robustness_result["df"],
                        "lag_L": L_robustness,
                    }
                # R2 components for non-fallback path
                cell["r2_level"] = None  # not separately computed; delta_r2 is the key metric
                cell["r2_transition"] = None

        per_wq.append(cell)

    # -----------------------------------------------------------------------
    # 9. Apply Holm correction (family_size=3, frozen §8)
    #    + compute MDE for the primary cell (D-03 -> Phase 10 SC2)
    # -----------------------------------------------------------------------
    print("Applying Holm correction (family_size=3)...")

    # Collect uncorrected p-values in WQ_GRID order for Holm
    pvalues_uncorrected: list[float] = []
    for cell in per_wq:
        if cell["skipped"] or (cell.get("wald_chi2_1") is None and cell.get("wald_chi2_joint") is None):
            # Degenerate/skipped cell: assign 1.0 (conservative) for Holm
            pvalues_uncorrected.append(1.0)
        elif cell.get("wald_chi2_1") is not None:
            pvalues_uncorrected.append(cell["wald_chi2_1"]["pvalue_uncorrected"])
        else:
            pvalues_uncorrected.append(cell["wald_chi2_joint"]["pvalue_uncorrected"])

    adjusted_pvalues = nested_test.apply_holm(pvalues_uncorrected, family_size=3)

    # Build Holm correction block per Q8 schema
    holm_ordered = []
    for rank_0, sort_idx in enumerate(sorted(range(len(pvalues_uncorrected)),
                                              key=lambda i: pvalues_uncorrected[i])):
        W_cell, q_cell = WQ_GRID[sort_idx]
        holm_ordered.append({
            "wq": [W_cell, q_cell],
            "p_uncorrected": pvalues_uncorrected[sort_idx],
            "holm_adjusted": adjusted_pvalues[sort_idx],
            "rank": rank_0 + 1,
        })

    # Primary cell (120,5) Holm-adjusted p-value
    primary_idx = next(i for i, (W, q) in enumerate(WQ_GRID) if W == 120 and q == 5)
    primary_wq_holm_adjusted = adjusted_pvalues[primary_idx]

    holm_correction = {
        "family_size": 3,
        "ordered": holm_ordered,
        "primary_wq_holm_adjusted": primary_wq_holm_adjusted,
    }

    # MDE for the primary cell (D-03)
    print(f"Computing MDE for primary cell (n_nl_primary={n_nl_primary:,})...")
    mde = nested_test.compute_mde(n_nl_primary)

    # -----------------------------------------------------------------------
    # 10. D-07 disposition rule (PROPOSED — human ratifies in Task 3)
    # -----------------------------------------------------------------------
    print("Applying D-07 disposition rule...")
    disp_result = nested_test.apply_disposition_rule(per_wq)
    disposition = disp_result["disposition"]
    print(
        f"Proposed disposition: {disposition} "
        f"(primary_passes={disp_result['primary_passes']}, "
        f"n_cells_passing={disp_result['n_cells_passing']})"
    )

    # -----------------------------------------------------------------------
    # 11. Frozen parameters echo (consumed verbatim from 07-PREREGISTRATION.md)
    # -----------------------------------------------------------------------
    frozen_params = {
        "wq_grid": [list(wq) for wq in WQ_GRID],
        "primary_wq": [120, 5],
        "lag_L_primary_formula": "int(4 * (n_nl / 100) ** (2/9))",
        "lag_L_robustness_formula": "int(n_nl ** (1/3))",
        "lag_L_primary_at_n_nl_primary": int(4 * (n_nl_primary / 100) ** (2 / 9)),
        "lag_L_robustness_at_n_nl_primary": int(n_nl_primary ** (1 / 3)),
        "delta_r2_threshold": 0.005,
        "beta_t_threshold": 0.01,
        "holm_family_size": 3,
        "min_sep_W": True,
        "under_powered_n": 50,
        "degeneracy_rank_check": True,
        "degeneracy_cond_threshold": 1e10,
    }

    # -----------------------------------------------------------------------
    # 12. D-15 provenance stamp
    # -----------------------------------------------------------------------
    provenance = {
        "freeze_commit": FREEZE_COMMIT,
        "phase8_amendment_ref": (
            f"07-PREREGISTRATION.md §D-07 operationalization "
            f"(D-08 commit {PHASE8_AMENDMENT_REF})"
        ),
        "prereg_commit": prereg_commit,
        "boot_seed": BOOT_SEED,
        "code_commit": code_commit,
        "library_versions": library_versions,
        "run_ts": run_ts_iso,
    }

    # -----------------------------------------------------------------------
    # 13. Assemble full report (Q8 schema) and write JSON
    # -----------------------------------------------------------------------
    report = {
        "run_date": run_date,
        "prereg_commit": prereg_commit,
        "phase8_amendment_ref": (
            f"07-PREREGISTRATION.md §D-07 operationalization "
            f"(D-08 commit {PHASE8_AMENDMENT_REF})"
        ),
        "data_span": data_span,
        "label_provenance": label_provenance,
        "append_spike_sc1_passed": append_spike_sc1_passed,
        "mde": mde,
        "per_wq": per_wq,
        "holm_correction": holm_correction,
        "disposition": disposition,
        "disposition_rule_applied": (
            "D-07: primary (120,5) passes full conjunction AND >= 2/3 cells pass; "
            "PROPOSED only — human ratification required (Task 3)"
        ),
        "disposition_detail": disp_result,
        "frozen_params": frozen_params,
        "provenance": provenance,
    }

    PRECHECK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRECHECK_DIR / f"precheck_a_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")
    print(f"Disposition (PROPOSED): {disposition}")
    print("Awaiting human ratification (Task 3 blocking checkpoint).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
