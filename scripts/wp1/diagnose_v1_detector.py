"""V1.0 static-detector failure diagnosis: reproduce the exact fast20/slow100 RMS ratio
over 2021-2025 and emit quantitative evidence explaining why EXTREME (ratio > 2.5) never
fired (n=0).

Integrity controls enforced here:
    - D-07 holdout: data window end must be < 2026-01-01 (two-layer assert in main).
    - D-04/D-08 boundary: this script does NOT compute epsilon-squared.
      gate_analysis.run_gate and predictability.compute_rolling_predictability are
      NOT imported. The diagnosis quantifies the v1.0 instrument failure only.
    - Unsmoothed static path: uses rolling_std_from_returns (RMS, no mean subtraction)
      directly from regime_detector. Does NOT apply ema_jit to vr — that is the
      quantile branch only (regime_detector.py lines 169-184); the static path
      (lines 185-193) uses the raw vr.

Output:
    backtest_results/gate/diagnosis_report_<timestamp>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors gate_analysis.py lines 33-38)
# When run as `python scripts/wp1/diagnose_v1_detector.py`, Python adds the
# script directory to sys.path[0] rather than the repo root. Insert repo root
# and src/ so that `backtest`, `scripts`, and `strategies` packages are
# importable regardless of invocation style.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from backtest.utils import PROJECT_ROOT  # noqa: E402
import scripts._bootstrap  # noqa: F401, E402  (prepends repo src/ to sys.path)

# Import ONLY the RMS function from regime_detector — NOT detect_regime, NOT gate_analysis,
# NOT predictability (D-04/D-08: no epsilon-squared in the diagnosis).
from strategies.vol_regime_switch.regime_detector import rolling_std_from_returns  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory (same as gate_analysis.py line 49)
# ---------------------------------------------------------------------------
GATE_DIR = PROJECT_ROOT / "backtest_results" / "gate"

# ---------------------------------------------------------------------------
# V1.0 static-path constants (the thresholds under diagnosis — NOT pre-registering)
# These match DEFAULT_REGIME_CONFIG in regime_detector.py and gate_analysis.py.
# ---------------------------------------------------------------------------
FAST_WINDOW: int = 20
SLOW_WINDOW: int = 100
EXTREME_VOL_THRESHOLD: float = 2.5   # The threshold that never fired (n=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Reproduce the v1.0 unsmoothed fast20/slow100 RMS ratio over 2021-2025,
    compute the ratio-distribution stats, and emit a timestamped diagnosis JSON."""
    import datetime as _dt

    from scripts.wp1.py_engine import load_and_clean

    # -----------------------------------------------------------------------
    # D-07 holdout: define the 2021-2025 data window
    # Layer 1 — epoch-ms boundary assertion (mirrors gate_analysis.py lines 355-370)
    # -----------------------------------------------------------------------
    # 2021-01-01 00:00:00 UTC in epoch milliseconds
    start_ms = int(
        _dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    # 2025-12-31 23:59:59 UTC in epoch milliseconds
    end_ms = int(
        _dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )

    # D-07 guard Layer 1: the window end must be strictly before 2026-01-01
    holdout_boundary_ms = int(
        _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
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
    ts_start = _dt.datetime.fromtimestamp(timestamps[0] / 1000, tz=_dt.timezone.utc)
    ts_end = _dt.datetime.fromtimestamp(timestamps[-1] / 1000, tz=_dt.timezone.utc)
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")

    # D-07 guard Layer 2: belt-and-suspenders — last bar must not be in 2026
    assert ts_end.year < 2026, (
        f"D-07 VIOLATION: last loaded bar is in {ts_end.year} (year=2026 partition was read). "
        "The 2026 holdout must remain UNTOUCHED."
    )

    # -----------------------------------------------------------------------
    # Reproduce the EXACT v1.0 static-path UNSMOOTHED fast20/slow100 RMS ratio
    # Source: regime_detector.py lines 149-161 (static path, no ema_jit)
    #
    # CRITICAL: Use rolling_std_from_returns (RMS, no mean subtraction) imported
    # directly from regime_detector. Do NOT use np.std. Do NOT apply ema_jit.
    # The static path (lines 185-193 of regime_detector.py) uses the raw vr
    # directly; ema_jit is only in the quantile branch (lines 169-184).
    # -----------------------------------------------------------------------
    print("Computing v1.0 static-path UNSMOOTHED fast20/slow100 RMS ratio...")

    log_close = np.log(close)
    r = np.diff(log_close, prepend=log_close[0])

    # RMS with NO mean subtraction (rolling_std_from_returns: mean_r2 = rolling_mean(r**2, w))
    fast_vol = rolling_std_from_returns(r, FAST_WINDOW)
    slow_vol = rolling_std_from_returns(r, SLOW_WINDOW)

    # Raw VR ratio — UNSMOOTHED (the static path never calls ema_jit)
    vr = np.full_like(fast_vol, np.nan)
    valid_slow = (~np.isnan(slow_vol)) & (slow_vol > 0.0)
    vr[valid_slow] = fast_vol[valid_slow] / slow_vol[valid_slow]

    # -----------------------------------------------------------------------
    # Warmup filter: first slow_window=100 bars have vr == NaN
    # Filter to valid (non-NaN) vr before computing statistics
    # -----------------------------------------------------------------------
    valid_mask = ~np.isnan(vr)
    vr_valid = vr[valid_mask]
    ts_valid = timestamps[valid_mask]
    n_valid = len(vr_valid)
    print(f"Valid vr bars (post-warmup): {n_valid:,} / {N:,}")

    # -----------------------------------------------------------------------
    # D-03 quantitative deliverable set: ratio distribution over 2021-2025
    # -----------------------------------------------------------------------
    ratio_mean = float(np.mean(vr_valid))
    ratio_std = float(np.std(vr_valid, ddof=1))
    ratio_max = float(np.max(vr_valid))

    # Percentile rank of threshold 2.5 in the empirical distribution
    # (fraction of valid bars with ratio <= 2.5)
    threshold_25_percentile_rank = float(np.mean(vr_valid <= EXTREME_VOL_THRESHOLD))

    # Sigma above mean for threshold 2.5
    threshold_25_sigma_above_mean = float(
        (EXTREME_VOL_THRESHOLD - ratio_mean) / ratio_std
    )

    # Count of bars with ratio > 2.5 — MUST be 0 to match gate_report n=0
    n_bars_ratio_gt_25 = int(np.sum(vr_valid > EXTREME_VOL_THRESHOLD))

    # -----------------------------------------------------------------------
    # D-03 / D-19: Crisis-window max ratios
    # LUNA: 2022-05-07 00:00:00 UTC to 2022-05-18 23:59:59 UTC
    # FTX:  2022-11-08 00:00:00 UTC to 2022-11-12 23:59:59 UTC
    # Filter by timestamp (epoch ms) against valid vr bars
    # -----------------------------------------------------------------------
    luna_start_ms = int(
        _dt.datetime(2022, 5, 7, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    luna_end_ms = int(
        _dt.datetime(2022, 5, 18, 23, 59, 59, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    luna_mask = (ts_valid >= luna_start_ms) & (ts_valid <= luna_end_ms)
    if np.any(luna_mask):
        luna_window_max_ratio = float(np.max(vr_valid[luna_mask]))
    else:
        luna_window_max_ratio = float("nan")

    ftx_start_ms = int(
        _dt.datetime(2022, 11, 8, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    ftx_end_ms = int(
        _dt.datetime(2022, 11, 12, 23, 59, 59, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    ftx_mask = (ts_valid >= ftx_start_ms) & (ts_valid <= ftx_end_ms)
    if np.any(ftx_mask):
        ftx_window_max_ratio = float(np.max(vr_valid[ftx_mask]))
    else:
        ftx_window_max_ratio = float("nan")

    # -----------------------------------------------------------------------
    # Assemble and emit JSON report (mirrors gate_analysis.py lines 443-472)
    # -----------------------------------------------------------------------
    run_date = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_date": run_date,
        "data_span": {
            "start": str(ts_start.date()),
            "end": str(ts_end.date()),
            "n_bars": N,
            "year_2026_loaded": False,
        },
        "methodology": {
            "vol_estimator": "rolling_std_from_returns (RMS, no mean subtraction)",
            "fast_window": FAST_WINDOW,
            "slow_window": SLOW_WINDOW,
            "smoothing": "NONE — static path; ema_jit NOT applied",
            "source_function": "strategies.vol_regime_switch.regime_detector.rolling_std_from_returns",
            "note": (
                "Reproduces the EXACT v1.0 static-path unsmoothed ratio. "
                "No epsilon-squared computed (D-04/D-08 boundary)."
            ),
        },
        "diagnosis": {
            "n_bars_valid_vr": n_valid,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_max": ratio_max,
            "extreme_vol_threshold": EXTREME_VOL_THRESHOLD,
            "threshold_25_percentile_rank": threshold_25_percentile_rank,
            "threshold_25_sigma_above_mean": threshold_25_sigma_above_mean,
            "n_bars_ratio_gt_25": n_bars_ratio_gt_25,
            "luna_window": "2022-05-07 to 2022-05-18 UTC",
            "luna_window_max_ratio": luna_window_max_ratio,
            "ftx_window": "2022-11-08 to 2022-11-12 UTC",
            "ftx_window_max_ratio": ftx_window_max_ratio,
        },
    }

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GATE_DIR / f"diagnosis_report_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Human-readable summary for copy-paste into 01-DIAGNOSIS.md
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"DIAGNOSIS REPORT: {out_path}")
    print(f"Data span: {ts_start.date()} -> {ts_end.date()} ({N:,} bars)")
    print(f"2026 holdout: NOT loaded (D-07 guard held — both layers)")
    print()
    print("--- Ratio Distribution (2021-2025, fast20/slow100 RMS, unsmoothed) ---")
    print(f"  Mean:                   {ratio_mean:.4f}")
    print(f"  Std:                    {ratio_std:.4f}")
    print(f"  Max:                    {ratio_max:.4f}")
    print(f"  Threshold 2.5:")
    print(f"    Percentile rank:      {threshold_25_percentile_rank*100:.2f}%")
    print(f"    Sigma above mean:     {threshold_25_sigma_above_mean:.2f} sigma")
    print(f"  n_bars_ratio_gt_25:     {n_bars_ratio_gt_25}  (gate_report regime-2 n=0 CONFIRMED)")
    print()
    print("--- Crisis-Window Max Ratios (D-19) ---")
    print(f"  LUNA (2022-05-07..05-18 UTC): max ratio = {luna_window_max_ratio:.4f}  (< 2.5: {luna_window_max_ratio < 2.5})")
    print(f"  FTX  (2022-11-08..11-12 UTC): max ratio = {ftx_window_max_ratio:.4f}  (< 2.5: {ftx_window_max_ratio < 2.5})")
    print()
    print("Reconciliation: n_bars_ratio_gt_25 == 0:", n_bars_ratio_gt_25 == 0)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
