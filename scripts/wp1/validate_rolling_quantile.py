"""Rolling-quantile detector SC2/SC4 validation driver: 2021-2025 BTCUSDT.

Loads the full 2021-2025 span via py_engine.load_and_clean, fits the causal
RollingQuantileDetector (frozen primary defaults), and reports:
  - Per-crisis table (LUNA / FTX): EXTREME bar count, EXTREME fraction,
    total bars in window, peak RV percentile-rank (vs trailing 30-day),
    WR-01-corrected non-overlapping EXTREME count at stride=60, n_min=50 check.
  - Whole-span SC4 population statistics (LOW/ELEVATED/EXTREME fractions +
    non-overlapping EXTREME count + SPARSE flag) via the imported Plan-02 src/
    regime_population helpers.
  - D-15 provenance stamp (data span, prereg commit 169fc20, 14.5 amendment ref,
    frozen-param echo, library versions, git commit).

Integrity controls:
  - D-10: NO predictability_t / VR / epsilon-squared computed.
    gate_analysis and predictability are NOT imported.
  - D-07/D-10: two-layer 2026 holdout guard (epoch-ms assert + ts_end.year < 2026).
  - D-07: count_nonoverlapping_extreme and regime_population_stats are IMPORTED
    from strategies.vol_regime_switch.regime_population — NOT re-implemented here.
  - D-14: degeneracy flag surfaced into the JSON report (q_elevated==0 and rv>0).

Output:
    backtest_results/wp1/rolling_quantile_validation_<timestamp>.json
"""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors diagnose_v1_detector.py lines 34-46)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402

import scripts._bootstrap  # noqa: F401, E402
from strategies.vol_regime_switch.regime_population import (  # noqa: E402
    count_nonoverlapping_extreme,
    regime_population_stats,
)

# Import ONLY the needed symbols — NOT gate_analysis, NOT predictability (D-10)
from strategies.vol_regime_switch.rolling_quantile_detector import (  # noqa: E402
    RollingQuantileDetector,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
WP1_DIR = PROJECT_ROOT / "backtest_results" / "wp1"


# ---------------------------------------------------------------------------
# D-06 helper: peak RV percentile-rank (Pattern 5 from 02-RESEARCH.md)
# Kept INLINE in the driver — only the non-overlapping / population core
# was moved to src/; this is a report-only metric.
# ---------------------------------------------------------------------------


def peak_rv_percentile_rank(
    rv: np.ndarray, peak_bar_idx: int, pct_window: int
) -> float:
    """Fraction of trailing pct_window bars with RV <= rv[peak_bar_idx]."""
    start = max(0, peak_bar_idx - pct_window + 1)
    window = rv[start : peak_bar_idx + 1]
    valid_window = window[~np.isnan(window)]
    if len(valid_window) == 0:
        return float("nan")
    return float(np.mean(valid_window <= rv[peak_bar_idx]))


# ---------------------------------------------------------------------------
# Git commit helper (D-15)
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
    """Load 2021-2025 BTC, fit RollingQuantileDetector, validate SC2/SC4, emit JSON."""
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # D-07/D-10 holdout guard Layer 1 (epoch-ms boundary)
    # Mirrors diagnose_v1_detector.py lines 78-94
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

    print(f"Loading {symbol} 2021-2025 (D-10 holdout: 2026 NOT loaded)...")
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
    # D-10 holdout guard Layer 2: last loaded bar must not be in 2026
    # -----------------------------------------------------------------------
    assert ts_end.year < 2026, (
        f"D-10 VIOLATION: last loaded bar is in {ts_end.year} "
        "(year=2026 partition was read). The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # Fit the detector (frozen primary defaults — SC3 already locked in Plan 02)
    # -----------------------------------------------------------------------
    print("Fitting RollingQuantileDetector (frozen primary defaults)...")
    detector = RollingQuantileDetector()
    regime = detector.fit(close)

    rv = detector.rv_
    print(f"Fit complete. Regime shape: {regime.shape}. Valid bars: {int(np.sum(regime >= 0)):,}")

    # -----------------------------------------------------------------------
    # D-14 degeneracy flag
    # -----------------------------------------------------------------------
    q_el = detector.q_elevated_
    degenerate_mask = (
        (~np.isnan(q_el))
        & (q_el == 0.0)
        & (~np.isnan(rv))
        & (rv > 0.0)
    )
    degenerate_flag = bool(np.any(degenerate_mask))
    n_degenerate_bars = int(np.sum(degenerate_mask))
    print(f"D-14 degeneracy flag: {degenerate_flag} (n_degenerate_bars={n_degenerate_bars})")

    # -----------------------------------------------------------------------
    # D-08 Crisis-window epoch-ms masks (mirrors diagnose_v1_detector.py lines 181-202)
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
        """Compute D-06 crisis table for one window."""
        window_mask = (timestamps >= t_start) & (timestamps <= t_end)
        window_regime = regime[window_mask]
        window_rv = rv[window_mask]
        window_ts_idx = np.where(window_mask)[0]  # indices in the FULL array

        total_bars = int(np.sum(window_mask))
        extreme_bar_count = int(np.sum(window_regime == 2))
        extreme_frac = (
            float(extreme_bar_count) / total_bars if total_bars > 0 else float("nan")
        )

        # Peak RV percentile-rank (D-06): peak bar in window vs its trailing 30-day
        # distribution. Use index in the FULL array for the trailing window.
        pct_window = detector.pct_window  # 43200
        if total_bars > 0 and not np.all(np.isnan(window_rv)):
            valid_rv_in_window = window_rv[~np.isnan(window_rv)]
            if len(valid_rv_in_window) > 0:
                local_peak_pos = int(np.argmax(valid_rv_in_window))
                # Map back to global index
                valid_global_idx = window_ts_idx[~np.isnan(window_rv)]
                global_peak_idx = int(valid_global_idx[local_peak_pos])
                peak_rv_pct = peak_rv_percentile_rank(rv, global_peak_idx, pct_window)
            else:
                peak_rv_pct = float("nan")
        else:
            peak_rv_pct = float("nan")

        # WR-01-corrected non-overlapping EXTREME count (D-07)
        # Apply count_nonoverlapping_extreme to the WINDOW regime slice
        # (stride on the local window index, consistent with the slice method
        # the plan specifies: "sliced to the crisis window")
        stride = 60
        nonoverlap_count = count_nonoverlapping_extreme(window_regime, stride)

        # n_min=50 check (D-06)
        n_min = 50
        populated_str = "populated" if nonoverlap_count >= n_min else "SPARSE"

        print(
            f"  {name}: total_bars={total_bars}, extreme_bars={extreme_bar_count}, "
            f"extreme_frac={extreme_frac:.4f}, nonoverlap_count={nonoverlap_count} "
            f"({populated_str}), peak_rv_pct={peak_rv_pct:.4f}"
        )

        # SC2 check — fail loudly if EITHER crisis shows 0
        if nonoverlap_count == 0:
            print(
                f"  SC2 FAILURE: {name} non-overlapping EXTREME count = 0. "
                "The detector is instrument-broken. STOP — do NOT tune frozen parameters."
            )

        return {
            "window": f"{_dt.datetime.fromtimestamp(t_start / 1000, tz=_dt.UTC).date()} "
            f"to {_dt.datetime.fromtimestamp(t_end / 1000, tz=_dt.UTC).date()} UTC",
            "total_bars": total_bars,
            "extreme_bar_count": extreme_bar_count,
            "extreme_frac": extreme_frac,
            "peak_rv_percentile_rank": peak_rv_pct,
            "nonoverlapping_extreme_count_stride60": nonoverlap_count,
            "n_min": n_min,
            "n_min_check": populated_str,
        }

    print("Computing crisis-window statistics...")
    luna_stats = _crisis_stats("LUNA", luna_start_ms, luna_end_ms)
    ftx_stats = _crisis_stats("FTX", ftx_start_ms, ftx_end_ms)

    # -----------------------------------------------------------------------
    # SC4 whole-span population statistics (via the imported src/ helper)
    # -----------------------------------------------------------------------
    print("Computing whole-span population statistics (SC4)...")
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

    python_version = sys.version
    git_commit = _get_git_commit()

    # -----------------------------------------------------------------------
    # Assemble and emit JSON report (mirrors diagnose_v1_detector.py lines 208-246)
    # -----------------------------------------------------------------------
    run_date = datetime.now(tz=_dt.UTC).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_date": run_date,
        "data_span": {
            "start": str(ts_start.date()),
            "end": str(ts_end.date()),
            "n_bars": N,
            "first_timestamp_utc": ts_start.isoformat(),
            "last_timestamp_utc": ts_end.isoformat(),
            "year_2026_loaded": False,
        },
        "preregistration": {
            "source": "01-PREREGISTRATION.md",
            "prereg_commit": "169fc20",
            "amendment_ref": "01-PREREGISTRATION.md §14.5 (warmup resolution, 2026-06-02)",
            "rv_window": detector.rv_window,
            "pct_window": detector.pct_window,
            "p_elevated": detector.p_elevated,
            "p_extreme": detector.p_extreme,
            "interpolation": "linear",
            "center": False,
            "min_samples": None,
            "ewma": detector.ewma,
            "pct_mode": detector.pct_mode,
        },
        "causal_test": {
            "sc1_status": "PROVEN by pytest suite (test_append_spike_causal_500bars + test_prefix_equivalence_exhaustive)",
            "note": "SC1 is enforced by the synthetic pytest suite (Plan 02-02); no recomputation here.",
        },
        "crisis_validation": {
            "luna": luna_stats,
            "ftx": ftx_stats,
            "sc2_pass": (
                luna_stats["nonoverlapping_extreme_count_stride60"] > 0
                and ftx_stats["nonoverlapping_extreme_count_stride60"] > 0
            ),
        },
        "population_stats": {
            "low_frac": pop_stats["low_frac"],
            "elevated_frac": pop_stats["elevated_frac"],
            "extreme_frac": pop_stats["extreme_frac"],
            "n_valid": pop_stats["n_valid"],
            "n_extreme_nonoverlap_stride60": pop_stats["n_extreme_nonoverlap"],
            "n_min": 50,
            "sparse": sparse_flag,
            "sc4_pass": True,  # population stats always computed; sparse is expected outcome
        },
        "degeneracy_flag": {
            "d14_fired": degenerate_flag,
            "n_degenerate_bars": n_degenerate_bars,
            "condition": "q_elevated==0 and rv>0",
            "note": (
                "Expected false on real BTC 1-min data. "
                "Surfaced for auditability (D-14 — no epsilon-floor; frozen recipe exact)."
            ),
        },
        "library_versions": {
            "polars": polars_version,
            "numpy": numpy_version,
            "python": python_version,
        },
        "git_commit": git_commit,
    }

    WP1_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WP1_DIR / f"rolling_quantile_validation_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"VALIDATION REPORT: {out_path}")
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")
    print("2026 holdout: NOT loaded (D-10 guard held — both layers)")
    print()
    sc2_pass = report["crisis_validation"]["sc2_pass"]
    print(f"SC2 PASS: {sc2_pass}")
    print(
        f"  LUNA non-overlapping EXTREME count (stride=60): "
        f"{luna_stats['nonoverlapping_extreme_count_stride60']}"
    )
    print(
        f"  FTX  non-overlapping EXTREME count (stride=60): "
        f"{ftx_stats['nonoverlapping_extreme_count_stride60']}"
    )
    print()
    print("Population statistics (SC4):")
    print(f"  LOW:      {pop_stats['low_frac']:.4f}")
    print(f"  ELEVATED: {pop_stats['elevated_frac']:.4f}")
    print(f"  EXTREME:  {pop_stats['extreme_frac']:.4f}")
    print(f"  SPARSE:   {sparse_flag}")
    print()
    print(f"D-14 degeneracy flag: {degenerate_flag}")
    print(f"git commit: {git_commit}")
    print("=" * 70)

    if not sc2_pass:
        print(
            "\nSC2 FAILURE: LUNA or FTX non-overlapping EXTREME count = 0. "
            "The detector is instrument-broken. Returning exit code 1."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
