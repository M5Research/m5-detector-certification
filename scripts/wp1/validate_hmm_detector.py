"""HMM detector SC4 validation driver: 2021-2025 BTCUSDT.

Loads the full 2021-2025 span via py_engine.load_and_clean, fits the causal
HMMDetector (frozen §7.2 defaults + §14.5 HMM amendment bundle D-02/D-03/D-06),
and reports:
  - EM convergence diagnostics (log-likelihood, top-5 LL spread, convergence flag,
    variance-separation ratios, hmm_fallback flag, floored-bar count).
  - Per-crisis table (LUNA / FTX): EXTREME bar count, EXTREME fraction, total bars,
    peak and mean filtered P(top regime: EXTREME in 3-state, ELEVATED in the 2-state fallback), WR-01-corrected non-overlapping EXTREME count
    at stride=60, n_min=50 check (D-11).
  - Whole-span population statistics (LOW/ELEVATED/EXTREME fractions + SPARSE flag).
  - D-15 provenance stamp (data span, prereg commit 169fc20, §14.5 HMM amendment ref,
    frozen-param echo, library versions, statsmodels version, git commit).

Integrity controls:
  - OMP_NUM_THREADS=1 for determinism (D-12); set BEFORE any statsmodels import.
  - D-06: smoke-test on ~500K-bar middle slice; routes to frozen 2-state fallback
    if extrapolated full-run > 30 min or OOM (downsampling the endog is FORBIDDEN).
  - Two-layer 2026 holdout guard (epoch-ms assert + ts_end.year < 2026).
  - gate_analysis and predictability are NOT imported (no ε²/VR — Phase 4 boundary).
  - SC4 fail-loud: if LUNA or FTX nonoverlap count == 0, print SC4 FAILURE + return 1.
  - Determinism check (D-07/D-12): double-fit and record regime_label_identical.

Output:
    backtest_results/wp1/hmm_validation_<timestamp>.json
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# D-12: Force single-threaded BLAS BEFORE any statsmodels / numpy heavy imports
# This maximises in-environment determinism (does not guarantee cross-platform).
# ---------------------------------------------------------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors validate_rolling_quantile.py lines 39-44)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402, I001
from strategies.vol_regime_switch.hmm_detector import HMMDetector  # noqa: E402, I001
from strategies.vol_regime_switch.regime_population import (  # noqa: E402, I001
    count_nonoverlapping_extreme,
    regime_population_stats,
)
import scripts._bootstrap  # noqa: F401, E402, I001

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
WP1_DIR = PROJECT_ROOT / "backtest_results" / "wp1"


# ---------------------------------------------------------------------------
# Git commit helper (D-15; mirrors validate_rolling_quantile.py lines 89-103)
# ---------------------------------------------------------------------------


def _get_git_commit() -> str:
    """Return the short HEAD hash, or a safe fallback string if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load 2021-2025 BTC, fit HMMDetector, validate SC4, emit JSON."""
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # D-12 / D-07 holdout guard Layer 1 (epoch-ms boundary)
    # Mirrors validate_rolling_quantile.py lines 121-133
    # -----------------------------------------------------------------------
    start_ms = int(
        _dt.datetime(2021, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    end_ms = int(
        _dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000
    )
    holdout_boundary_ms = int(
        _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000
    )
    assert end_ms < holdout_boundary_ms, (
        f"D-10 VIOLATION: data window end ({end_ms}) must be < 2026-01-01 "
        f"({holdout_boundary_ms}). The 2026 holdout must NOT be loaded."
    )

    data_path = str(PROJECT_ROOT / "data" / "binance_futures")
    symbol = "BTCUSDT"

    print(f"Loading {symbol} 2021-2025 (holdout: 2026 NOT loaded, year_2026_loaded=false)...")
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
    # Holdout guard Layer 2: last loaded bar must not be in 2026
    # -----------------------------------------------------------------------
    assert ts_end.year < 2026, (
        f"D-10 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # D-06 Timing smoke-test on a contiguous ~500K-bar MIDDLE slice
    # Route to frozen 2-state fallback if extrapolated full-run > 30 min or OOM.
    # The actual fit uses the FULL span — the slice is timing-only.
    # Silently downsampling the endog or shrinking the 2021-2025 span is FORBIDDEN.
    # -----------------------------------------------------------------------
    SMOKE_SLICE_START = N // 2 - 250_000  # centre of the series
    SMOKE_SLICE_END = SMOKE_SLICE_START + 500_000
    SMOKE_SLICE_START = max(0, SMOKE_SLICE_START)
    SMOKE_SLICE_END = min(N, SMOKE_SLICE_END)
    smoke_slice = close[SMOKE_SLICE_START:SMOKE_SLICE_END]
    smoke_n = len(smoke_slice)

    print(
        f"D-06 smoke-test: 3-state fit on bars [{SMOKE_SLICE_START}:{SMOKE_SLICE_END}] "
        f"({smoke_n:,} bars) to estimate full-run feasibility..."
    )

    k_regimes_primary = 3
    hmm_fallback = "none"
    timing_seconds: float = float("nan")
    extrapolated_full_run_seconds: float = float("nan")

    try:
        t0 = time.perf_counter()
        _smoke_det = HMMDetector(k_regimes=3, em_iter=100, search_reps=3)  # fewer reps for speed
        _smoke_det.fit(smoke_slice)
        timing_seconds = time.perf_counter() - t0

        # Linear extrapolation: full run scales with N (EM is O(T) per iteration)
        extrapolated_full_run_seconds = timing_seconds * (N / smoke_n)
        print(
            f"  Smoke-test timing: {timing_seconds:.1f}s on {smoke_n:,} bars -> "
            f"extrapolated full-run: {extrapolated_full_run_seconds:.1f}s "
            f"({extrapolated_full_run_seconds / 60:.1f} min)"
        )

        if extrapolated_full_run_seconds > 30 * 60:  # 30-minute threshold (D-06)
            print(
                "  D-06 INTRACTABILITY: extrapolated > 30 min. "
                "Activating frozen 2-state fallback (hmm_fallback='2-state')."
            )
            k_regimes_primary = 2
            hmm_fallback = "2-state"
        else:
            print(
                f"  D-06: 3-state feasible (extrapolated {extrapolated_full_run_seconds / 60:.1f} min "
                f"< 30 min threshold). Proceeding with 3-state primary."
            )

    except (MemoryError, RuntimeError) as exc:
        # D-06: OOM or all-restarts-failed on the smoke-test slice also triggers fallback.
        # A RuntimeError ("All HMM EM restarts failed") on a 500K-bar slice indicates
        # the 3-state model is degenerate/intractable on this data — route to 2-state.
        print(
            f"  D-06 smoke-test failed ({type(exc).__name__}: {exc}). "
            "Activating frozen 2-state fallback (hmm_fallback='2-state')."
        )
        k_regimes_primary = 2
        hmm_fallback = "2-state"

    # -----------------------------------------------------------------------
    # Fit HMMDetector on the FULL 2021-2025 span (NOT the smoke-test slice)
    # The k_regimes may have been overridden to 2 by the D-06 fallback above.
    # -----------------------------------------------------------------------
    print(f"Fitting HMMDetector(k_regimes={k_regimes_primary}) on full {N:,}-bar span...")
    print("  OMP_NUM_THREADS=1 (D-12 determinism). This may take 10-30 minutes...")

    fit_start = time.perf_counter()
    detector = HMMDetector(k_regimes=k_regimes_primary)
    regime = detector.fit(close)
    fit_duration = time.perf_counter() - fit_start

    print(
        f"Fit complete in {fit_duration:.1f}s. "
        f"Regime shape: {regime.shape}. Valid bars: {int(np.sum(regime >= 0)):,}"
    )
    print(
        f"  convergence_ok={detector.convergence_ok_}, "
        f"hmm_fallback_='{detector.hmm_fallback_}', "
        f"floored_bar_count={detector.floored_bar_count_}"
    )
    print(
        f"  sigma2_sorted={detector.sigma2_sorted_.tolist()}, "
        f"ratio_elev_low={detector.ratio_elev_low_:.4f}, "
        f"ratio_ext_elev={detector.ratio_ext_elev_:.4f}"
    )

    # -----------------------------------------------------------------------
    # D-07/D-12 Determinism check: double-fit -> identical int8 labels
    # Note: if all restarts fail on second fit (degenerate model), record as
    # regime_label_identical=false (diagnostic, not a pass/fail gate for SC4).
    # -----------------------------------------------------------------------
    print("D-07/D-12 determinism check: second fit (same close)...")
    try:
        detector2 = HMMDetector(k_regimes=k_regimes_primary)
        regime2 = detector2.fit(close)
        regime_label_identical = bool(np.array_equal(regime.astype(np.int32), regime2.astype(np.int32)))
        print(f"  regime_label_identical={regime_label_identical}")
    except RuntimeError as exc:
        print(f"  WARNING: second fit failed ({exc}). Recording regime_label_identical=false.")
        regime_label_identical = False

    # -----------------------------------------------------------------------
    # Top-5 LL spread calculation for JSON (D-07)
    # Re-derive from the detector's per-restart LLs (approximated via llf_ + sigma2)
    # We use the detector's exposed llf_ as the best-restart LLF.
    # The spread and fraction are computed inside the detector; expose them via
    # an attribute computed at fit-time.  For the driver JSON, we collect the
    # stored llf_ (best restart LL) and compute a conservative spread from
    # what's available.
    # -----------------------------------------------------------------------
    # Use the stored top5_spread_ and spread_fraction_ attributes exposed by HMMDetector.fit()
    top5_spread_approx: float = detector.top5_spread_
    spread_fraction_approx: float = detector.spread_fraction_

    # -----------------------------------------------------------------------
    # Crisis-window epoch-ms masks (mirrors validate_rolling_quantile.py lines 192-199)
    # LUNA: 2022-05-07 .. 2022-05-18; FTX: 2022-11-08 .. 2022-11-12
    # -----------------------------------------------------------------------
    luna_start_ms = int(_dt.datetime(2022, 5, 7, tzinfo=_dt.UTC).timestamp() * 1000)
    luna_end_ms = int(
        _dt.datetime(2022, 5, 18, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000
    )
    ftx_start_ms = int(_dt.datetime(2022, 11, 8, tzinfo=_dt.UTC).timestamp() * 1000)
    ftx_end_ms = int(
        _dt.datetime(2022, 11, 12, 23, 59, 59, tzinfo=_dt.UTC).timestamp() * 1000
    )

    def _crisis_stats(name: str, t_start: int, t_end: int) -> dict:
        """Compute D-11 crisis table for one window."""
        window_mask = (timestamps >= t_start) & (timestamps <= t_end)
        window_regime = regime[window_mask]

        total_bars = int(np.sum(window_mask))
        extreme_bar_count = int(np.sum(window_regime == 2))
        extreme_frac = (
            float(extreme_bar_count) / total_bars if total_bars > 0 else float("nan")
        )

        # D-11 HMM-natural diagnostics: peak and mean filtered P(top / highest-variance regime).
        # filtered_probs_ has NaN warmup rows. The top regime is EXTREME (column 2) in 3-state mode,
        # but ELEVATED (column 1) in the 2-state fallback. The JSON key 'peak_filtered_p_extreme' is a
        # legacy 3-state identifier; in 2-state mode it holds the ELEVATED (top) state's probability.
        window_fprobs = detector.filtered_probs_[window_mask]  # shape (W, k_regimes)
        if window_fprobs.size > 0 and detector.k_regimes >= 3:
            prob_extreme_col = window_fprobs[:, 2]  # col 2 = EXTREME
            peak_p_extreme = float(np.nanmax(prob_extreme_col))
            mean_p_extreme = float(np.nanmean(prob_extreme_col))
        elif window_fprobs.size > 0 and detector.k_regimes == 2:
            # 2-state fallback: col 1 = highest-variance state = ELEVATED (label 1), NOT a true EXTREME
            prob_extreme_col = window_fprobs[:, 1]
            peak_p_extreme = float(np.nanmax(prob_extreme_col))
            mean_p_extreme = float(np.nanmean(prob_extreme_col))
        else:
            peak_p_extreme = mean_p_extreme = float("nan")

        # WR-01-corrected non-overlapping EXTREME count (D-11, stride=60)
        stride = 60
        nonoverlap_count = count_nonoverlapping_extreme(window_regime, stride)

        # n_min=50 check (D-11)
        n_min = 50
        populated_str = "populated" if nonoverlap_count >= n_min else "SPARSE"

        print(
            f"  {name}: total_bars={total_bars}, extreme_bars={extreme_bar_count}, "
            f"extreme_frac={extreme_frac:.4f}, nonoverlap_count={nonoverlap_count} "
            f"({populated_str}), peak_p_extreme={peak_p_extreme:.4f}, "
            f"mean_p_extreme={mean_p_extreme:.4f}"
        )

        # SC4 check — fail loudly if EITHER crisis shows 0
        if nonoverlap_count == 0:
            print(
                f"  SC4 FAILURE: {name} non-overlapping EXTREME count = 0. "
                "The HMM detector is instrument-broken. STOP — do NOT tune frozen parameters."
            )

        return {
            "window": (
                f"{_dt.datetime.fromtimestamp(t_start / 1000, tz=_dt.UTC).date()} "
                f"to {_dt.datetime.fromtimestamp(t_end / 1000, tz=_dt.UTC).date()} UTC"
            ),
            "total_bars": total_bars,
            "extreme_bar_count": extreme_bar_count,
            "extreme_frac": extreme_frac,
            "peak_filtered_p_extreme": peak_p_extreme,
            "mean_filtered_p_extreme": mean_p_extreme,
            "nonoverlapping_extreme_count_stride60": nonoverlap_count,
            "n_min": n_min,
            "n_min_check": populated_str,
        }

    print("Computing crisis-window statistics...")
    luna_stats = _crisis_stats("LUNA", luna_start_ms, luna_end_ms)
    ftx_stats = _crisis_stats("FTX", ftx_start_ms, ftx_end_ms)

    # -----------------------------------------------------------------------
    # Whole-span population statistics (via the imported src/ helper)
    # -----------------------------------------------------------------------
    print("Computing whole-span population statistics...")
    pop_stats = regime_population_stats(regime, n_min=50, stride=60)
    sparse_flag = pop_stats["sparse"]
    print(
        f"  Population: LOW={pop_stats['low_frac']:.4f}, "
        f"ELEVATED={pop_stats['elevated_frac']:.4f}, "
        f"EXTREME={pop_stats['extreme_frac']:.4f}, "
        f"n_valid={pop_stats['n_valid']:,}, "
        f"n_extreme_nonoverlap={pop_stats['n_extreme_nonoverlap']}, "
        f"sparse={sparse_flag}"
    )

    # -----------------------------------------------------------------------
    # D-15 provenance stamp: library versions + git commit
    # -----------------------------------------------------------------------
    try:
        polars_version = importlib.metadata.version("polars")
    except importlib.metadata.PackageNotFoundError:
        polars_version = "unavailable"
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = str(np.__version__)
    try:
        statsmodels_version = importlib.metadata.version("statsmodels")
    except importlib.metadata.PackageNotFoundError:
        statsmodels_version = "unavailable"

    python_version = sys.version
    git_commit = _get_git_commit()

    # -----------------------------------------------------------------------
    # Assemble and emit JSON report (mirrors Phase-2 schema + HMM-specific fields)
    # -----------------------------------------------------------------------
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")

    sc4_pass = (
        luna_stats["nonoverlapping_extreme_count_stride60"] > 0
        and ftx_stats["nonoverlapping_extreme_count_stride60"] > 0
    )

    # Sigma2 sorted values (list form for JSON)
    sigma2_sorted_list = detector.sigma2_sorted_.tolist()

    report = {
        "run_date": run_date,
        "year_2026_loaded": False,
        "data_span": {
            "start": str(ts_start.date()),
            "end": str(ts_end.date()),
            "n_bars": N,
            "first_timestamp_utc": ts_start.isoformat(),
            "last_timestamp_utc": ts_end.isoformat(),
        },
        "preregistration": {
            "source": "01-PREREGISTRATION.md",
            "prereg_commit": "169fc20",
            "amendment_ref": (
                "01-PREREGISTRATION.md §14.5 HMM amendment bundle (D-02/D-03/D-06, 2026-06-02)"
            ),
            "k_regimes": detector.k_regimes,
            "rv_window": detector.rv_window,
            "switching_variance": detector.switching_variance,
            "trend": detector.trend,
            "switching_trend": detector.switching_trend,
            "em_iter": detector.em_iter,
            "search_reps": detector.search_reps,
            "pinned_seed": 42,
            "ewma": detector.ewma,
        },
        "hmm_diagnostics": {
            "llf": detector.llf_,
            "top5_spread": top5_spread_approx,
            "spread_fraction": spread_fraction_approx,
            "convergence_ok": detector.convergence_ok_,
            "hmm_fallback": hmm_fallback,
            "floored_bar_count": detector.floored_bar_count_,
            "sigma2_sorted": sigma2_sorted_list,
            "ratio_elev_low": detector.ratio_elev_low_,
            "ratio_ext_elev": detector.ratio_ext_elev_,
            "timing_seconds": fit_duration,
            "smoke_test_timing_seconds": timing_seconds,
            "smoke_test_extrapolated_full_run_seconds": extrapolated_full_run_seconds,
            "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "regime_label_identical": regime_label_identical,
        },
        "causal_test": {
            "sc1_status": (
                "PROVEN by pytest suite "
                "(test_filtered_lags_smoother_causality + test_fixed_param_append_causal)"
            ),
            "note": (
                "SC1 (filter-lags-smoother causality) and SC2 (fixed-θ prefix unchanged "
                "on append) are enforced by the synthetic pytest suite (Plan 03-02); "
                "no recomputation here."
            ),
        },
        "crisis_validation": {
            "luna": luna_stats,
            "ftx": ftx_stats,
            "sc2_pass": sc4_pass,
        },
        "population_stats": {
            "low_frac": pop_stats["low_frac"],
            "elevated_frac": pop_stats["elevated_frac"],
            "extreme_frac": pop_stats["extreme_frac"],
            "n_valid": pop_stats["n_valid"],
            "n_extreme_nonoverlap_stride60": pop_stats["n_extreme_nonoverlap"],
            "n_min": 50,
            "sparse": sparse_flag,
        },
        "library_versions": {
            "statsmodels": statsmodels_version,
            "polars": polars_version,
            "numpy": numpy_version,
            "python": python_version,
        },
        "git_commit": git_commit,
    }

    WP1_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WP1_DIR / f"hmm_validation_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"HMM VALIDATION REPORT: {out_path}")
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")
    print("2026 holdout: NOT loaded (guard held — both layers)")
    print(f"hmm_fallback: {hmm_fallback}")
    print(f"k_regimes used: {k_regimes_primary}")
    print(f"convergence_ok: {detector.convergence_ok_}")
    print(f"fit_duration: {fit_duration:.1f}s")
    print()
    print(f"SC4 PASS: {sc4_pass}")
    print(
        f"  LUNA non-overlapping EXTREME count (stride=60): "
        f"{luna_stats['nonoverlapping_extreme_count_stride60']}"
    )
    print(
        f"  FTX  non-overlapping EXTREME count (stride=60): "
        f"{ftx_stats['nonoverlapping_extreme_count_stride60']}"
    )
    print()
    print("Population statistics:")
    print(f"  LOW:      {pop_stats['low_frac']:.4f}")
    print(f"  ELEVATED: {pop_stats['elevated_frac']:.4f}")
    print(f"  EXTREME:  {pop_stats['extreme_frac']:.4f}")
    print(f"  SPARSE:   {sparse_flag}")
    print()
    print("Variance-separation ratios (D-10):")
    print(f"  sigma2_sorted: {sigma2_sorted_list}")
    print(f"  ratio_elev/low: {detector.ratio_elev_low_:.4f}")
    print(f"  ratio_ext/elev: {detector.ratio_ext_elev_:.4f}")
    print()
    print(f"regime_label_identical (D-07/D-12): {regime_label_identical}")
    print(f"git commit: {git_commit}")
    print("=" * 70)

    if not sc4_pass:
        print(
            "\nSC4 FAILURE: LUNA or FTX non-overlapping EXTREME count = 0. "
            "The HMM detector is instrument-broken. Returning exit code 1."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
